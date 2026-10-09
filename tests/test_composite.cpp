#include <cmath>

#include "sat/adaptive/composite.hpp"
#include "sat/adaptive/experiment.hpp"
#include "sat/algo/composite.hpp"
#include "sat/core/random.hpp"
#include "sat/data/synthetic_market.hpp"
#include "test_framework.hpp"

using namespace sat;

namespace {

// Labels on 300 days x 20 stocks; a "good" member knows the label (0.7 / 0.3), a "noise"
// member is random around 0.5.
struct Members {
  Panel labels, ends;
  std::vector<ModelPredictions> models;
  explicit Members(std::size_t T = 300, std::size_t N = 20) : labels(T, N), ends(T, N) {
    Rng rng(5);
    ModelPredictions good, noise;
    good.name = "good";
    noise.name = "noise";
    good.probability = Panel(T, N);
    noise.probability = Panel(T, N);
    for (std::size_t t = 0; t + 1 < T; ++t)
      for (std::size_t i = 0; i < N; ++i) {
        labels(t, i) = rng.uniform() < 0.5 ? 1.0 : 0.0;
        ends(t, i) = static_cast<double>(t + 1);
        good.probability(t, i) = (labels(t, i) > 0.5 ? 0.6 : 0.4) + 0.2 * (rng.uniform() - 0.5);
        noise.probability(t, i) = 0.5 + 0.3 * (rng.uniform() - 0.5);
      }
    good.start = noise.start = 20;
    good.end = noise.end = T - 1;
    models = {good, noise};
  }
};

// A market of N stocks: stock i drifts by drift[i] a day plus noise.
MarketData trendingMarket(const std::vector<double>& drift, std::size_t T, double noise, std::uint64_t seed = 3) {
  const std::size_t N = drift.size();
  MarketData d;
  for (std::size_t i = 0; i < N; ++i) d.tickers.push_back("S" + std::to_string(i));
  for (std::size_t t = 0; t < T; ++t) d.dates.push_back(std::to_string(t));
  d.close = Panel(T, N);
  Rng rng(seed);
  for (std::size_t i = 0; i < N; ++i) {
    double p = 100;
    for (std::size_t t = 0; t < T; ++t) {
      d.close(t, i) = p;
      p *= std::exp(drift[i] + noise * rng.normal());
    }
  }
  d.open = d.high = d.low = d.vwap = d.close;
  d.volume = Panel(T, N, 1e6);
  d.regime.assign(T, -1);
  return d;
}

}  // namespace

TEST(composite_average_is_the_mean_of_members) {
  Members w;
  CompositeSpec spec;
  spec.method = CompositeMethod::Average;
  const auto c = compositePredictions(w.models, w.labels, w.ends, spec);
  CHECK(c.model.start == 20 && c.model.end == 299);
  for (std::size_t t = 20; t < 299; t += 37)
    for (std::size_t i = 0; i < 20; i += 7)
      CHECK_NEAR(c.model.probability(t, i), 0.5 * (w.models[0].probability(t, i) + w.models[1].probability(t, i)), 1e-12);
  CHECK(std::isnan(c.model.probability(10, 0)));
  CHECK(c.weights.size() == 279);
}

TEST(composite_stacked_and_online_favour_the_informative_member) {
  Members w;
  for (auto method : {CompositeMethod::Stacked, CompositeMethod::Online}) {
    CompositeSpec spec;
    spec.method = method;
    spec.window = 60;
    const auto c = compositePredictions(w.models, w.labels, w.ends, spec);
    const auto& last = c.weights.back();
    CHECK(last[0] > 3 * last[1]);
    CHECK(last[1] >= 0.0);
    CHECK(c.model.oos.auc > 0.75);
    CHECK(c.model.oos.auc >= w.models[0].oos.auc - 0.02 || w.models[0].oos.count == 0);
  }
}

TEST(composite_uses_only_resolved_labels) {
  for (auto method : {CompositeMethod::Stacked, CompositeMethod::Online}) {
    Members a, b;
    // Scramble every label from day 150 on: predictions up to day 150 must not change.
    for (std::size_t t = 150; t < 299; ++t)
      for (std::size_t i = 0; i < 20; ++i) b.labels(t, i) = 1.0 - b.labels(t, i);
    CompositeSpec spec;
    spec.method = method;
    spec.window = 40;
    spec.refitEvery = 5;
    const auto ca = compositePredictions(a.models, a.labels, a.ends, spec);
    const auto cb = compositePredictions(b.models, b.labels, b.ends, spec);
    for (std::size_t t = 20; t <= 151; ++t)
      for (std::size_t i = 0; i < 20; ++i) CHECK_NEAR(ca.model.probability(t, i), cb.model.probability(t, i), 1e-12);
    bool differs = false;
    for (std::size_t t = 152; t < 299 && !differs; ++t) differs = ca.model.probability(t, 0) != cb.model.probability(t, 0);
    CHECK(differs);
  }
}

TEST(composite_rejects_bad_input) {
  Members w;
  CompositeSpec spec;
  CHECK_THROWS(compositePredictions(w.models, w.labels, w.ends, spec));  // method none
  spec.method = CompositeMethod::Average;
  CHECK_THROWS(compositePredictions({}, w.labels, w.ends, spec));
  CHECK_THROWS(compositePredictions(w.models, Panel(10, 20), w.ends, spec));
  CHECK(parseCompositeMethod("boa") == CompositeMethod::Online);
  CHECK_THROWS(parseCompositeMethod("median"));
}

TEST(experiment_appends_the_composite_model) {
  SyntheticMarketSpec ms;
  ms.numAssets = 12;
  ms.numDates = 420;
  ExperimentSpec exp;
  exp.walkForward.trainWindow = 150;
  exp.walkForward.retrainEvery = 60;
  exp.composite.method = CompositeMethod::Online;
  const auto p = runPredictions(generateSyntheticMarket(ms), exp);
  CHECK(p.models.size() == 3);
  CHECK(p.models.back().name == "Composite (online)");
  CHECK(p.compositeMembers.size() == 2);
  CHECK(p.compositeWeights.size() == p.models.back().end - p.models.back().start);
  exp.composite.keepMembers = false;
  const auto q = runPredictions(generateSyntheticMarket(ms), exp);
  CHECK(q.models.size() == 1);
  const auto e = runStrategies(q, exp);
  CHECK(e.book.size() == ExperimentSpec::defaultStrategies().size());
}

TEST(multi_signal_momentum_holds_the_strongest_trends) {
  std::vector<double> drift = {0.002, 0.0015, -0.001, 0.0, -0.002, 0.001, 0.0005, -0.0005};
  const auto d = trendingMarket(drift, 600, 0.002);
  algo::MultiSignalSpec s;
  s.useMl = false;
  s.useTrailingSharpe = false;
  s.weighting = algo::SignalWeighting::Equal;
  s.targetVol = 0;
  s.regimeGate = false;
  s.holdings = 2;
  s.costBps = 0;
  const auto r = algo::multiSignalSelection(d, nullptr, s, 300);
  CHECK(r.signals.size() == 1 && r.signals[0] == "Momentum");
  CHECK(r.returns.size() == 599 - 300);
  CHECK(r.holdings[0] > 0 && r.holdings[1] > 0);
  double held = 0;
  for (double v : r.holdings) held += v;
  CHECK_NEAR(held, 1.0, 1e-9);
  CHECK(r.metrics.annualReturn > 0.3);
}

TEST(multi_signal_volatility_target_and_regime_gate) {
  std::vector<double> drift(10, 0.0003);
  const auto d = trendingMarket(drift, 900, 0.02);
  algo::MultiSignalSpec s;
  s.useMl = false;
  s.targetVol = 0.05;
  s.maxLeverage = 1.5;
  s.regimeGate = true;
  s.regimes.window = 252;
  const auto r = algo::multiSignalSelection(d, nullptr, s, 300);
  for (double e : r.exposure) CHECK(e >= 0.0 && e <= 1.5 + 1e-12);
  // A 5% target on stocks with ~30% volatility de-levers.
  double avg = 0;
  for (std::size_t k = 100; k < r.exposure.size(); ++k) avg += r.exposure[k] / static_cast<double>(r.exposure.size() - 100);
  CHECK(avg < 0.5);
  CHECK(r.metrics.annualVolatility < 0.12);
}

TEST(multi_signal_adaptive_weights_follow_the_working_signal) {
  // The ML "forecast" is tomorrow's realised return: a perfect signal; momentum is noise here.
  std::vector<double> drift(15, 0.0);
  const auto d = trendingMarket(drift, 700, 0.015);
  Panel prob = d.forwardReturns(1);
  algo::MultiSignalSpec s;
  s.useTrailingSharpe = false;
  s.weighting = algo::SignalWeighting::Adaptive;
  s.targetVol = 0;
  s.regimeGate = false;
  s.holdings = 3;
  const auto r = algo::multiSignalSelection(d, &prob, s, 300);
  CHECK(r.signalWeights.back()[0] > 0.9);
  CHECK(r.metrics.sharpe > 3);
  CHECK_THROWS(algo::multiSignalSelection(d, nullptr, s, 300));
}

TEST(multi_signal_has_no_lookahead) {
  std::vector<double> drift = {0.001, -0.001, 0.0005, 0.0, 0.0002, -0.0003};
  const auto a = trendingMarket(drift, 500, 0.01);
  auto b = a;
  for (std::size_t t = 400; t < 500; ++t)
    for (std::size_t i = 0; i < 6; ++i) b.close(t, i) = a.close(t, i) * (1.0 + 0.3 * static_cast<double>(i % 3));
  b.open = b.high = b.low = b.vwap = b.close;
  algo::MultiSignalSpec s;
  s.useMl = false;
  s.weighting = algo::SignalWeighting::Adaptive;
  s.targetVol = 0.1;
  s.regimeGate = false;
  s.holdings = 2;
  const auto ra = algo::multiSignalSelection(a, nullptr, s, 260);
  const auto rb = algo::multiSignalSelection(b, nullptr, s, 260);
  // Returns earned up to date 399 (index 398 - 260 + 1) are identical.
  for (std::size_t k = 0; k + 260 < 399; ++k) CHECK_NEAR(ra.returns[k], rb.returns[k], 1e-12);
}

TEST(composite_strategy_allocates_between_the_sleeves) {
  SyntheticMarketSpec ms;
  ms.numAssets = 15;
  ms.numDates = 700;
  const auto d = generateSyntheticMarket(ms);
  ExperimentSpec exp;
  exp.walkForward.trainWindow = 150;
  exp.walkForward.retrainEvery = 60;
  const auto p = runPredictions(d, exp);
  algo::CompositeStrategySpec cs;
  cs.books[0].regimes.window = 200;
  const auto r = algo::runCompositeStrategy(d, p, exp, cs);
  CHECK(r.names.size() == 4 && r.returns.size() == 4 && r.metrics.size() == 4 && r.sleeves == 2);
  CHECK(r.names[0] == "Self-adaptive selector (Composite (average))");
  CHECK(r.names[1] == "Multi-signal book (ML + momentum + trailing Sharpe)" && r.names[2] == "Composite strategy");
  for (const auto& x : r.returns) CHECK(x.size() == r.returns[2].size());
  CHECK(r.weights.size() == 2);
  for (std::size_t k = 0; k < r.cash.size(); k += 50) CHECK_NEAR(r.weights[0][k] + r.weights[1][k] + r.cash[k], 1.0, 1e-9);
  CHECK(r.selectorCandidates.size() == ExperimentSpec::defaultStrategies().size());
  CHECK(r.start + r.returns[0].size() == p.models[0].end);
  // More books and the market as sleeves.
  algo::MultiSignalSpec mom;
  mom.useMl = mom.useTrailingSharpe = false;
  mom.regimes.window = 200;
  cs.books.push_back(mom);
  cs.marketSleeve = true;
  const auto r2 = algo::runCompositeStrategy(d, p, exp, cs);
  CHECK(r2.sleeves == 4 && r2.names.size() == 6 && r2.weights.size() == 4);
  CHECK(r2.names[2] == "Momentum book" && r2.names[3] == "Equal-weight market");
  CHECK_NEAR(r2.metrics[0].sharpe, r.metrics[0].sharpe, 1e-12);
}
