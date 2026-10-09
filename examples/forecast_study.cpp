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

struct Market {
  std::string name;
  PredictionSet p;
  ModelPredictions average, adaptive;
  std::vector<int> choice;  // adaptive: candidate per date (-1 = average)
  std::size_t first = 0;
};

Market prepare(const std::string& name, const MarketData& data) {
  Market m{name, runPredictions(data, ExperimentSpec{}), {}, {}, {}, 0};
  CompositeSpec cs;
  cs.method = CompositeMethod::Average;
  m.average = compositePredictions(m.p.models, m.p.labels, m.p.labelEnds, cs).model;
  cs.method = CompositeMethod::Adaptive;
  auto a = compositePredictions(m.p.models, m.p.labels, m.p.labelEnds, cs);
  m.adaptive = std::move(a.model);
  m.choice = std::move(a.choice);
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
  auto real = [&]() {
    std::ifstream f(csv);
    std::stringstream ss;
    ss << f.rdbuf();
    study(("Real data: " + csv).c_str(), {prepare(csv, parseCsv(ss.str()))});
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
  study("Selection markets", build(101));
  std::fprintf(stderr, "validation markets\n");
  study("Validation markets (unseen)", build(201));
  if (!csv.empty()) real();
  return 0;
}
