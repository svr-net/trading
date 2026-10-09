// Composite model and composite strategy: do they add value over the library's defaults?
//
//   composite_study [markets per set=6] [days=2520] [--csv data.csv]
//   composite_study 0 --csv data.csv     (real data only)
//
// The default pool's models (logistic regression, random forest) are trained walk-forward
// once per market. Then:
//  1. Composite model: the members' forecasts combined by the equal-weight average, a
//     stacked logistic meta-learner and Bernstein Online Aggregation; each composite is
//     scored by AUC and traded by the self-adaptive selector, alongside or instead of the
//     members.
//  2. Composite strategy: every combination of signals (ML forecast, momentum, trailing
//     Sharpe), signal weighting (equal, adaptive), holdings (5, 10) and overlays (none,
//     volatility target, HMM regime gate, both). Settings are ranked on one set of synthetic
//     markets (selection) and the winner is checked on an unseen set (validation) and, with
//     --csv, on real data; the deflated Sharpe ratio charges for every variant tried.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "sat/sat.hpp"

using namespace sat;

namespace {

constexpr double kCostBps = 10.0;
const std::vector<CompositeMethod> kMethods = {CompositeMethod::Average, CompositeMethod::Stacked, CompositeMethod::Online};

struct Market {
  std::string name;
  MarketData data;
  PredictionSet p;                          // the members
  std::vector<CompositePredictions> composites;  // per kMethods
  std::size_t first = 0;                    // first traded date, the same for every variant
};

Market prepare(const std::string& name, MarketData data) {
  ExperimentSpec exp;
  Market m{name, std::move(data), {}, {}, 0};
  m.p = runPredictions(m.data, exp);
  for (auto method : kMethods) {
    CompositeSpec cs;
    cs.method = method;
    m.composites.push_back(compositePredictions(m.p.models, m.p.labels, m.p.labelEnds, cs));
  }
  std::size_t start = 0;
  for (const auto& mp : m.p.models) start = std::max(start, mp.start);
  m.first = start + SelectorSpec{}.lookback;
  std::fprintf(stderr, "  %s ready\n", name.c_str());
  return m;
}

struct Run {
  PerformanceMetrics metrics;
  std::vector<double> net;
};

// The default self-adaptive selector over the default rules on the given models.
Run selector(const Market& m, const std::vector<ModelPredictions>& models) {
  const CandidateBook book(models, ExperimentSpec::defaultStrategies(), m.p.nextReturns, kCostBps);
  const auto a = runSelector(book, SelectorSpec{}, m.first - book.start());
  return {a.metrics, a.net};
}

std::vector<ModelPredictions> pool(const Market& m, int composite, bool members) {
  std::vector<ModelPredictions> out;
  if (members) out = m.p.models;
  if (composite >= 0) out.push_back(m.composites[static_cast<std::size_t>(composite)].model);
  return out;
}

struct Variant {
  std::string name;
  algo::MultiSignalSpec spec;
};

std::vector<Variant> strategyGrid() {
  std::vector<Variant> out;
  const std::vector<std::pair<std::string, std::vector<bool>>> signals = {
      {"ML", {true, false, false}},        {"Mom", {false, true, false}},     {"TS", {false, false, true}},
      {"ML+Mom", {true, true, false}},     {"ML+Mom+TS", {true, true, true}},
  };
  for (const auto& [sname, use] : signals)
    for (auto weighting : {algo::SignalWeighting::Equal, algo::SignalWeighting::Adaptive})
      for (std::size_t holdings : {5u, 10u})
        for (int overlay = 0; overlay < 4; ++overlay) {
          if (weighting == algo::SignalWeighting::Adaptive && sname.find('+') == std::string::npos) continue;
          Variant v;
          v.spec.useMl = use[0];
          v.spec.useMomentum = use[1];
          v.spec.useTrailingSharpe = use[2];
          v.spec.weighting = weighting;
          v.spec.holdings = holdings;
          v.spec.costBps = kCostBps;
          v.spec.targetVol = 0.0;
          v.spec.maxLeverage = 1.0;
          if (overlay & 1) v.spec.targetVol = 0.15, v.spec.maxLeverage = 1.5;
          v.spec.regimeGate = (overlay & 2) != 0;
          static const char* names[] = {"", " +vol", " +regime", " +vol+regime"};
          v.name = sname + (weighting == algo::SignalWeighting::Adaptive ? " adapt" : "") + " k" + std::to_string(holdings) + names[overlay];
          out.push_back(v);
        }
  return out;
}

Run strategy(const Market& m, const algo::MultiSignalSpec& spec) {
  // The ML signal is the composite model of the members (equal-weight average).
  const auto r = algo::multiSignalSelection(m.data, &m.composites[0].model.probability, spec, m.first, m.composites[0].model.end);
  return {r.metrics, r.returns};
}

struct Row {
  std::string name;
  double ret = 0, sharpe = 0, worst = 1e9, dd = 0, dsr = 0;
};

Row summarise(const std::string& name, const std::vector<Run>& runs, double trials, double trialVariance) {
  Row row{name};
  const double n = static_cast<double>(runs.size());
  for (const auto& r : runs) {
    row.ret += r.metrics.annualReturn / n;
    row.sharpe += r.metrics.sharpe / n;
    row.worst = std::min(row.worst, r.metrics.sharpe);
    row.dd += r.metrics.maxDrawdown / n;
    const double sr = afml::periodSharpe(r.net);
    row.dsr += afml::deflatedSharpe(sr, static_cast<double>(r.net.size()), skewness(r.net), kurtosis(r.net), trials, std::max(1e-12, trialVariance)) / n;
  }
  return row;
}

void print(const char* title, const std::vector<Row>& rows) {
  std::printf("\n%s\n  %-34s %8s %8s %8s %8s %6s\n", title, "", "ann.ret", "Sharpe", "worst", "max DD", "DSR");
  for (const auto& r : rows)
    std::printf("  %-34s %7.1f%% %8.2f %8.2f %7.1f%% %5.0f%%\n", r.name.c_str(), 100 * r.ret, r.sharpe, r.worst, 100 * r.dd, 100 * r.dsr);
}

// Variance of per-day Sharpe ratios across a family of variants: the trials of the DSR.
double trialVariance(const std::vector<std::vector<Run>>& family) {
  std::vector<double> s;
  for (const auto& runs : family)
    for (const auto& r : runs) s.push_back(afml::periodSharpe(r.net));
  const double sd = s.size() > 1 ? stdev(s) : 0.0;
  return sd * sd;
}

void compositeModels(const char* title, const std::vector<Market>& set) {
  std::printf("\n%s: forecast quality (mean AUC / log loss)\n", title);
  const double n = static_cast<double>(set.size());
  for (std::size_t k = 0; k < set[0].p.models.size(); ++k) {
    double auc = 0, ll = 0;
    for (const auto& m : set) auc += m.p.models[k].oos.auc / n, ll += m.p.models[k].oos.logLoss / n;
    std::printf("  %-34s %8.4f %8.4f\n", set[0].p.models[k].name.c_str(), auc, ll);
  }
  for (std::size_t c = 0; c < kMethods.size(); ++c) {
    double auc = 0, ll = 0;
    std::vector<double> w(set[0].p.models.size(), 0.0);
    for (const auto& m : set) {
      auc += m.composites[c].model.oos.auc / n, ll += m.composites[c].model.oos.logLoss / n;
      for (std::size_t k = 0; k < w.size(); ++k) w[k] += m.composites[c].weights.back()[k] / n;
    }
    std::printf("  %-34s %8.4f %8.4f   final weights", set[0].composites[c].model.name.c_str(), auc, ll);
    for (double v : w) std::printf(" %.2f", v);
    std::printf("\n");
  }
  std::vector<std::vector<Run>> runs;
  std::vector<std::string> names;
  auto add = [&](const std::string& name, int composite, bool members) {
    std::vector<Run> r;
    for (const auto& m : set) r.push_back(selector(m, pool(m, composite, members)));
    runs.push_back(r);
    names.push_back(name);
  };
  add("members (library default)", -1, true);
  for (std::size_t c = 0; c < kMethods.size(); ++c) {
    add("members + " + compositeMethodName(kMethods[c]), static_cast<int>(c), true);
    add(compositeMethodName(kMethods[c]) + " only", static_cast<int>(c), false);
  }
  const double tv = trialVariance(runs);
  std::vector<Row> rows;
  for (std::size_t k = 0; k < runs.size(); ++k) rows.push_back(summarise(names[k], runs[k], static_cast<double>(runs.size()), tv));
  print((std::string(title) + ": self-adaptive selector on each pool").c_str(), rows);
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::size_t markets = 6, days = 2520;
  std::string csv;
  std::vector<std::string> pos;
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv" && k + 1 < argc) csv = argv[++k];
    else pos.push_back(a);
  }
  if (pos.size() > 0) markets = std::strtoul(pos[0].c_str(), nullptr, 10);
  if (pos.size() > 1) days = std::strtoul(pos[1].c_str(), nullptr, 10);
  auto readCsv = [&]() {
    std::ifstream f(csv);
    std::stringstream ss;
    ss << f.rdbuf();
    return parseCsv(ss.str());
  };
  auto build = [&](std::uint64_t base) {
    std::vector<Market> set;
    for (std::size_t k = 0; k < markets; ++k) {
      SyntheticMarketSpec ms;
      ms.seed = base + k;
      ms.numDates = days;
      set.push_back(prepare("seed " + std::to_string(ms.seed), generateSyntheticMarket(ms)));
    }
    return set;
  };
  const auto grid = strategyGrid();
  const double trials = static_cast<double>(grid.size());
  auto evaluate = [&](const std::vector<Market>& set, std::vector<std::vector<Run>>& runs) {
    runs.clear();
    for (const auto& v : grid) {
      std::vector<Run> r;
      for (const auto& m : set) r.push_back(strategy(m, v.spec));
      runs.push_back(r);
    }
  };
  // The composite strategy (library defaults: runCompositeStrategy) against its parts, the
  // library's default selector on the separate members, and the market, over the same days.
  auto references = [&](const std::vector<Market>& set, double tv) {
    std::vector<std::vector<Run>> runs(5);
    for (const auto& m : set) {
      ExperimentSpec exp;
      exp.costBps = kCostBps;
      const auto c = algo::runCompositeStrategy(m.data, m.p, exp, algo::CompositeStrategySpec{});
      // The default selector on the members, trimmed to the composite strategy's days.
      const auto s = selector(m, m.p.models);
      const std::size_t skip = c.start - m.first;
      const std::vector<double> members(s.net.begin() + static_cast<std::ptrdiff_t>(skip), s.net.end());
      runs[0].push_back({c.metrics[3], c.returns[3]});
      runs[1].push_back({evaluatePerformance(members), members});
      for (std::size_t k = 0; k < 3; ++k) runs[2 + k].push_back({c.metrics[k], c.returns[k]});
    }
    return std::vector<Row>{summarise("equal-weight market", runs[0], 1, tv), summarise("default selector (members)", runs[1], 1, tv),
                            summarise("selector on the composite model", runs[2], 1, tv),
                            summarise("multi-signal selection (default)", runs[3], trials, tv),
                            summarise("COMPOSITE STRATEGY", runs[4], trials, tv)};
  };

  if (markets == 0) {
    if (csv.empty()) throw std::invalid_argument("composite_study 0 needs --csv");
    const std::vector<Market> real = {prepare(csv, readCsv())};
    compositeModels(("Real data: " + csv).c_str(), real);
    std::vector<std::vector<Run>> runs;
    evaluate(real, runs);
    const double tv = trialVariance(runs);
    std::vector<Row> rows;
    for (std::size_t k = 0; k < grid.size(); ++k) rows.push_back(summarise(grid[k].name, runs[k], trials, tv));
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.sharpe > b.sharpe; });
    print(("Real data: " + csv + ": composite strategies, best first").c_str(), rows);
    print("Real data: composite strategy (library defaults) over the same days", references(real, tv));
    return 0;
  }

  std::fprintf(stderr, "selection markets\n");
  const auto selection = build(101);
  compositeModels("Selection markets", selection);
  std::vector<std::vector<Run>> selRuns;
  evaluate(selection, selRuns);
  const double selTv = trialVariance(selRuns);
  std::vector<std::pair<double, std::size_t>> order;
  for (std::size_t k = 0; k < grid.size(); ++k) {
    double s = 0;
    for (const auto& r : selRuns[k]) s += r.metrics.sharpe / static_cast<double>(selection.size());
    order.push_back({s, k});
  }
  std::sort(order.rbegin(), order.rend());
  std::vector<Row> rows;
  for (std::size_t q = 0; q < order.size(); ++q) rows.push_back(summarise(grid[order[q].second].name, selRuns[order[q].second], trials, selTv));
  print("Selection markets: composite strategies, best first", rows);
  const Variant& best = grid[order[0].second];
  std::printf("\nChosen on the selection markets: %s\n", best.name.c_str());
  print("Selection markets: composite strategy (library defaults) over the same days", references(selection, selTv));

  std::fprintf(stderr, "validation markets\n");
  const auto validation = build(201);
  compositeModels("Validation markets (unseen)", validation);
  std::vector<std::vector<Run>> valRuns;
  evaluate(validation, valRuns);
  const double valTv = trialVariance(valRuns);
  // The selection ranking's top ten on the unseen markets, and the library default.
  rows.clear();
  for (std::size_t q = 0; q < std::min<std::size_t>(10, order.size()); ++q)
    rows.push_back(summarise(std::to_string(q + 1) + ". " + grid[order[q].second].name, valRuns[order[q].second], trials, valTv));
  print("Validation markets: the selection's top ten", rows);
  std::vector<double> sel, val;
  for (std::size_t k = 0; k < grid.size(); ++k) {
    double a = 0, b = 0;
    for (const auto& r : selRuns[k]) a += r.metrics.sharpe;
    for (const auto& r : valRuns[k]) b += r.metrics.sharpe;
    sel.push_back(a), val.push_back(b);
  }
  std::printf("\nRank correlation of variant Sharpe ratios, selection vs validation: %.2f\n", rankCorrelation(sel, val));
  print("Validation markets: composite strategy (library defaults) over the same days", references(validation, valTv));
  if (!csv.empty()) {
    const std::vector<Market> real = {prepare(csv, readCsv())};
    compositeModels(("Real data: " + csv).c_str(), real);
    std::vector<std::vector<Run>> runs;
    evaluate(real, runs);
    const double tv = trialVariance(runs);
    rows.clear();
    for (std::size_t q = 0; q < std::min<std::size_t>(10, order.size()); ++q)
      rows.push_back(summarise(std::to_string(q + 1) + ". " + grid[order[q].second].name, runs[order[q].second], trials, tv));
    print(("Real data: " + csv + ": the selection's top ten").c_str(), rows);
    print("Real data: composite strategy (library defaults) over the same days", references(real, tv));
  }
  return 0;
}
