// Which forecast should the self-adaptive selector trade?
//
//   forecast_study [markets per set=6] [days=2520] [--csv data.csv]
//   forecast_study 0 --csv data.csv     (real data only)
//
// The default models (logistic regression, random forest) are trained walk-forward once per
// market. The self-adaptive selector (default rules and settings) then trades:
//  - the models as separate candidates (the paper's pool),
//  - their equal-weight average,
//  - the self-adaptive forecast: every 21 days, the model or average with the best rank
//    correlation with the labels over the last 63 resolved days (the selector's own rule,
//    one level down; nothing tuned),
//  - the models and the self-adaptive forecast together.
// Forecast quality (AUC, log loss, information coefficient) and the selector's results are
// reported on synthetic markets (selection), unseen ones (validation) and, with --csv, real bars.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sat/sat.hpp"

using namespace sat;

namespace {

constexpr double kCostBps = 10.0;

// Scoring variants of the self-adaptive forecast: how long and how a candidate's record is
// remembered, and how the record picks the forecast. Ranked on the selection markets only.
struct Variant {
  std::string name;
  CompositeSpec spec;
};

std::vector<Variant> variants() {
  std::vector<Variant> out;
  auto add = [&](const std::string& name, ScoringWindow w, double halfLife) {
    for (auto d : {ScoringDecision::Best, ScoringDecision::Evidence}) {
      Variant v;
      v.spec.method = CompositeMethod::Adaptive;
      v.spec.window = w;
      v.spec.halfLife = halfLife;
      v.spec.decision = d;
      v.name = name + (d == ScoringDecision::Best ? ", best" : ", evidence t>2");
      out.push_back(v);
    }
  };
  add("fixed 63 days, every 21", ScoringWindow::Fixed, 0);
  add("exponential, half-life 63, daily", ScoringWindow::Exponential, 63);
  add("exponential, half-life 252, daily", ScoringWindow::Exponential, 252);
  add("expanding, daily", ScoringWindow::Exponential, 0);
  add("ADWIN on each forecast's record, daily", ScoringWindow::Adwin, 0);
  add("ADWIN on the record and the market state", ScoringWindow::MarketAdwin, 0);
  add("similar market state (vol + trend)", ScoringWindow::SimilarState, 0);
  return out;
}

struct Market {
  std::string name;
  PredictionSet p;
  ModelPredictions average, adaptive;
  std::vector<int> choice;  // adaptive: candidate per date (-1 = average)
  std::vector<ModelPredictions> variant;   // per variants()
  std::vector<double> variantMemory;       // mean scoring memory (dates) per variant
  std::vector<double> variantMemberShare;  // share of dates a member (not the average) was used
  std::size_t first = 0;
  // The generalists plus logistic regressions trained on one market state each, and the
  // forecasts built from them (per specialistVariants()).
  std::vector<ModelPredictions> pool6;
  std::vector<ModelPredictions> specialistForecast;
  std::vector<double> specialistMemberShare;
};

// Forecasts over the generalists and the state specialists.
std::vector<Variant> specialistVariants() {
  std::vector<Variant> out;
  auto add = [&](const std::string& name, CompositeMethod method, ScoringWindow w, ScoringDecision d) {
    Variant v;
    v.spec.method = method;
    v.spec.window = w;
    v.spec.halfLife = 0;
    v.spec.decision = d;
    v.name = name;
    out.push_back(v);
  };
  add("average of all six", CompositeMethod::Average, ScoringWindow::Fixed, ScoringDecision::Best);
  add("fixed 63 days, best", CompositeMethod::Adaptive, ScoringWindow::Fixed, ScoringDecision::Best);
  add("expanding, evidence", CompositeMethod::Adaptive, ScoringWindow::Exponential, ScoringDecision::Evidence);
  add("market-state ADWIN, best", CompositeMethod::Adaptive, ScoringWindow::MarketAdwin, ScoringDecision::Best);
  add("market-state ADWIN, evidence", CompositeMethod::Adaptive, ScoringWindow::MarketAdwin, ScoringDecision::Evidence);
  add("similar state, best", CompositeMethod::Adaptive, ScoringWindow::SimilarState, ScoringDecision::Best);
  add("similar state, evidence", CompositeMethod::Adaptive, ScoringWindow::SimilarState, ScoringDecision::Evidence);
  return out;
}

Market prepare(const std::string& name, const MarketData& data) {
  Market m{name, runPredictions(data, ExperimentSpec{}), {}, {}, {}, {}, {}, {}, 0, {}, {}, {}};
  CompositeSpec cs;
  cs.method = CompositeMethod::Average;
  m.average = compositePredictions(m.p.models, m.p.labels, m.p.labelEnds, cs).model;
  cs.method = CompositeMethod::Adaptive;
  auto a = compositePredictions(m.p.models, m.p.labels, m.p.labelEnds, cs);
  m.adaptive = std::move(a.model);
  m.choice = std::move(a.choice);
  // State specialists: logistic regression trained only on calm, turbulent, rising or falling
  // market days (each judged from information available that day).
  const ExperimentSpec exp;
  m.pool6 = m.p.models;
  for (auto side : {StateSide::Calm, StateSide::Turbulent, StateSide::Rising, StateSide::Falling}) {
    ModelSpec logistic;
    logistic.type = "logistic";
    const Panel mask = marketStateMask(m.p.nextReturns, side);
    auto mp = walkForward(logistic, m.p.features, m.p.labels, exp.label, exp.walkForward, 0, &m.p.labelEnds, &mask);
    mp.name = "Logistic (" + stateSideName(side) + ")";
    m.pool6.push_back(std::move(mp));
  }
  for (const auto& v : specialistVariants()) {
    auto r = compositePredictions(m.pool6, m.p.labels, m.p.labelEnds, v.spec, &m.p.nextReturns);
    double member = 0;
    for (int c : r.choice) member += (c >= 0 ? 1.0 : 0.0) / static_cast<double>(r.choice.size());
    m.specialistForecast.push_back(std::move(r.model));
    m.specialistMemberShare.push_back(member);
  }
  for (const auto& v : variants()) {
    auto r = compositePredictions(m.p.models, m.p.labels, m.p.labelEnds, v.spec, &m.p.nextReturns);
    double mem = 0, member = 0;
    for (double x : r.memory) mem += x / static_cast<double>(r.memory.size());
    for (int c : r.choice) member += (c >= 0 ? 1.0 : 0.0) / static_cast<double>(r.choice.size());
    m.variant.push_back(std::move(r.model));
    m.variantMemory.push_back(mem);
    m.variantMemberShare.push_back(member);
  }
  for (const auto& mp : m.p.models) m.first = std::max(m.first, mp.start);
  m.first += SelectorSpec{}.lookback;
  std::fprintf(stderr, "  %s ready\n", name.c_str());
  return m;
}

// Mean daily cross-sectional rank correlation of a forecast with the labels.
double meanIc(const ModelPredictions& f, const Panel& labels) {
  double sum = 0;
  std::size_t n = 0;
  for (std::size_t t = f.start; t < f.end; ++t) {
    std::vector<double> x, y;
    for (std::size_t i = 0; i < labels.assets(); ++i)
      if (std::isfinite(f.probability(t, i)) && std::isfinite(labels(t, i))) x.push_back(f.probability(t, i)), y.push_back(labels(t, i));
    if (x.size() < 4) continue;
    const double r = correlation(averageRanks(x), averageRanks(y));
    if (std::isfinite(r)) sum += r, ++n;
  }
  return n ? sum / static_cast<double>(n) : 0.0;
}

struct Run {
  PerformanceMetrics metrics;
  std::vector<double> net;
  double trials = 1, trialVariance = 0;
};

Run selector(const Market& m, const std::vector<ModelPredictions>& models) {
  const CandidateBook book(models, ExperimentSpec::defaultStrategies(), m.p.nextReturns, kCostBps);
  const std::size_t from = m.first - book.start();
  const auto a = runSelector(book, SelectorSpec{}, from);
  std::vector<double> sharpes;
  for (std::size_t c = 0; c < book.size(); ++c) sharpes.push_back(afml::periodSharpe(book.netSeries(c, from)));
  const double sd = sharpes.size() > 1 ? stdev(sharpes) : 0.0;
  return {a.metrics, a.net, static_cast<double>(book.size()), sd * sd};
}

void study(const char* title, const std::vector<Market>& set) {
  const double n = static_cast<double>(set.size());
  std::printf("\n%s: forecast quality (mean over markets)\n  %-34s %8s %8s %8s\n", title, "forecast", "AUC", "logloss", "IC");
  auto quality = [&](const std::string& name, auto get) {
    double auc = 0, ll = 0, ic = 0;
    for (const auto& m : set) {
      const ModelPredictions& f = get(m);
      auc += f.oos.auc / n, ll += f.oos.logLoss / n, ic += meanIc(f, m.p.labels) / n;
    }
    std::printf("  %-34s %8.4f %8.4f %8.4f\n", name.c_str(), auc, ll, ic);
  };
  for (std::size_t k = 0; k < set[0].p.models.size(); ++k) quality(set[0].p.models[k].name, [k](const Market& m) -> const ModelPredictions& { return m.p.models[k]; });
  quality("Equal-weight average", [](const Market& m) -> const ModelPredictions& { return m.average; });
  quality("Self-adaptive forecast", [](const Market& m) -> const ModelPredictions& { return m.adaptive; });
  // What the self-adaptive forecast used.
  std::vector<double> share(set[0].p.models.size() + 1, 0.0);
  for (const auto& m : set)
    for (int c : m.choice) share[c < 0 ? share.size() - 1 : static_cast<std::size_t>(c)] += 1.0 / (n * static_cast<double>(m.choice.size()));
  std::printf("  self-adaptive forecast used:");
  for (std::size_t k = 0; k + 1 < share.size(); ++k) std::printf(" %s %.0f%%,", set[0].p.models[k].name.c_str(), 100 * share[k]);
  std::printf(" average %.0f%%\n", 100 * share.back());

  std::printf("\n%s: self-adaptive selector on each pool\n  %-34s %8s %8s %8s %8s %6s\n", title, "pool", "ann.ret", "Sharpe", "worst", "max DD", "DSR");
  auto pool = [&](const std::string& name, auto models) {
    double ret = 0, sharpe = 0, worst = 1e9, dd = 0, dsr = 0;
    for (const auto& m : set) {
      const auto r = selector(m, models(m));
      ret += r.metrics.annualReturn / n, sharpe += r.metrics.sharpe / n, dd += r.metrics.maxDrawdown / n;
      worst = std::min(worst, r.metrics.sharpe);
      dsr += afml::deflatedSharpe(afml::periodSharpe(r.net), static_cast<double>(r.net.size()), skewness(r.net), kurtosis(r.net), r.trials,
                                  std::max(1e-12, r.trialVariance)) / n;
    }
    std::printf("  %-34s %7.1f%% %8.2f %8.2f %7.1f%% %5.0f%%\n", name.c_str(), 100 * ret, sharpe, worst, 100 * dd, 100 * dsr);
  };
  pool("separate models (paper)", [](const Market& m) { return m.p.models; });
  pool("equal-weight average", [](const Market& m) { return std::vector<ModelPredictions>{m.average}; });
  pool("self-adaptive forecast", [](const Market& m) { return std::vector<ModelPredictions>{m.adaptive}; });
  pool("models + self-adaptive forecast", [](const Market& m) {
    auto v = m.p.models;
    v.push_back(m.adaptive);
    return v;
  });
}

// The state specialists: forecast quality of each, and the selector on each forecast built from
// all six models, against the default (the average of the two generalists).
void specialists(const char* title, const std::vector<Market>& set) {
  const double n = static_cast<double>(set.size());
  std::printf("\n%s: state specialists (mean AUC, IC)\n", title);
  for (std::size_t k = 0; k < set[0].pool6.size(); ++k) {
    double auc = 0, ic = 0;
    for (const auto& m : set) auc += m.pool6[k].oos.auc / n, ic += meanIc(m.pool6[k], m.p.labels) / n;
    std::printf("  %-34s %8.4f %8.4f\n", set[0].pool6[k].name.c_str(), auc, ic);
  }
  std::vector<double> base;
  for (const auto& m : set) base.push_back(selector(m, {m.average}).metrics.sharpe);
  double b = 0;
  for (double x : base) b += x / n;
  std::printf("\n%s: selector on forecasts from all six models (beats = markets where it beats the default)\n"
              "  %-40s %8s %8s %6s %8s\n  %-40s %8s %8.2f\n", title, "forecast", "AUC", "Sharpe", "beats", "member", "default: average of the two generalists", "", b);
  const auto vs = specialistVariants();
  for (std::size_t v = 0; v < vs.size(); ++v) {
    double auc = 0, sharpe = 0, member = 0;
    int beats = 0;
    for (std::size_t k = 0; k < set.size(); ++k) {
      const double sr = selector(set[k], {set[k].specialistForecast[v]}).metrics.sharpe;
      sharpe += sr / n, auc += set[k].specialistForecast[v].oos.auc / n, member += set[k].specialistMemberShare[v] / n;
      beats += sr > base[k] + 1e-9 ? 1 : 0;
    }
    std::printf("  %-40s %8.4f %8.2f %3d/%-2zu %7.0f%%\n", vs[v].name.c_str(), auc, sharpe, beats, set.size(), 100 * member);
  }
}

// Every scoring variant: the selector's Sharpe ratio on each market against the average's.
// Returns the mean Sharpe ratio per variant.
std::vector<double> scoring(const char* title, const std::vector<Market>& set, const std::vector<std::size_t>& order) {
  const auto vs = variants();
  const double n = static_cast<double>(set.size());
  std::vector<double> avg;
  for (const auto& m : set) avg.push_back(selector(m, {m.average}).metrics.sharpe);
  std::printf("\n%s: scoring variants of the self-adaptive forecast (selector Sharpe; beats = markets where it beats the average)\n"
              "  %-46s %8s %8s %6s %8s %8s\n", title, "variant", "AUC", "Sharpe", "beats", "memory", "member");
  double a = 0;
  for (double x : avg) a += x / n;
  std::printf("  %-46s %8s %8.2f %6s %8s %8s\n", "equal-weight average (reference)", "", a, "", "", "");
  std::vector<double> mean(vs.size(), 0.0);
  for (std::size_t q = 0; q < vs.size(); ++q) {
    const std::size_t v = order.empty() ? q : order[q];
    double auc = 0, mem = 0, member = 0;
    int beats = 0;
    for (std::size_t k = 0; k < set.size(); ++k) {
      const double sr = selector(set[k], {set[k].variant[v]}).metrics.sharpe;
      mean[v] += sr / n;
      beats += sr > avg[k] + 1e-9 ? 1 : 0;
      auc += set[k].variant[v].oos.auc / n, mem += set[k].variantMemory[v] / n, member += set[k].variantMemberShare[v] / n;
    }
    std::printf("  %-46s %8.4f %8.2f %3d/%-2zu %8.0f %7.0f%%\n", vs[v].name.c_str(), auc, mean[v], beats, set.size(), mem, 100 * member);
  }
  return mean;
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
  std::vector<std::size_t> ranked;  // variants in the selection markets' order
  auto real = [&]() {
    std::ifstream f(csv);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::vector<Market> set = {prepare(csv, parseCsv(ss.str()))};
    study(("Real data: " + csv).c_str(), set);
    scoring(("Real data: " + csv).c_str(), set, ranked);
    specialists(("Real data: " + csv).c_str(), set);
  };
  if (markets == 0) {
    if (csv.empty()) throw std::invalid_argument("forecast_study 0 needs --csv");
    real();
    return 0;
  }
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
  std::fprintf(stderr, "selection markets\n");
  const auto selection = build(101);
  study("Selection markets", selection);
  const auto sel = scoring("Selection markets", selection, {});
  specialists("Selection markets", selection);
  for (std::size_t v = 0; v < sel.size(); ++v) ranked.push_back(v);
  std::stable_sort(ranked.begin(), ranked.end(), [&](std::size_t a, std::size_t b) { return sel[a] > sel[b]; });
  std::printf("\nChosen on the selection markets: %s\n", variants()[ranked[0]].name.c_str());
  std::fprintf(stderr, "validation markets\n");
  const auto validation = build(201);
  study("Validation markets (unseen)", validation);
  scoring("Validation markets (unseen), in the selection markets' order", validation, ranked);
  specialists("Validation markets (unseen)", validation);
  if (!csv.empty()) real();
  return 0;
}
