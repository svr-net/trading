#include <cmath>

#include "sat/adaptive/experiment.hpp"
#include "sat/adaptive/self_adaptive.hpp"
#include "sat/data/synthetic_market.hpp"
#include "test_framework.hpp"

using namespace sat;

namespace {

// Two "models" on four stocks over 200 days. Stock 0 rises in the first half and falls in
// the second, stock 1 the opposite. Model A always prefers stock 0, model B stock 1, so A
// wins the first half and B the second: no fixed choice is right all the time.
struct TwoRegimeWorld {
  Panel next;
  std::vector<ModelPredictions> models;
  TwoRegimeWorld() : next(200, 4, 0.0) {
    for (std::size_t t = 0; t < 200; ++t) {
      const double s = t < 100 ? 1.0 : -1.0;
      next(t, 0) = 0.004 * s + (t % 3 == 0 ? 0.002 : -0.001);
      next(t, 1) = -0.004 * s + (t % 4 == 0 ? 0.002 : -0.0005);
    }
    for (int m = 0; m < 2; ++m) {
      ModelPredictions p;
      p.name = m == 0 ? "A" : "B";
      p.probability = Panel(200, 4, 0.5);
      for (std::size_t t = 0; t < 200; ++t) {
        p.probability(t, m == 0 ? 0 : 1) = 0.9;
        p.probability(t, m == 0 ? 1 : 0) = 0.1;
        p.probability(t, 2) = 0.4;
        p.probability(t, 3) = 0.3;
      }
      p.start = 10;
      p.end = 199;
      models.push_back(p);
    }
  }
};

}  // namespace

TEST(candidate_book_matches_single_backtests) {
  TwoRegimeWorld w;
  const std::vector<StrategySpec> strategies = {{StrategyKind::LongTopK, 1, 1}, {StrategyKind::LongShort, 1, 3}};
  const CandidateBook book(w.models, strategies, w.next, 10.0);
  CHECK(book.size() == 4 && book.start() == 10 && book.days() == 189);
  CHECK(book.candidates()[3].label == "B · Long-short 1 /3d");
  const auto ranks = rankTable(w.models[1].probability);
  const auto r = backtest(strategies[1], w.models[1].probability, ranks, w.next, 10, 199, 10.0);
  for (std::size_t d = 0; d < book.days(); ++d) CHECK_NEAR(book.net(3, d), r.series.net[d], 1e-15);
  double wt[4];
  book.weights(3, 7, wt);  // rebalanced on day 6
  CHECK(wt[1] == 1.0 && wt[0] == -1.0);
}

TEST(selector_switches_to_the_winning_model) {
  TwoRegimeWorld w;
  const CandidateBook book(w.models, {{StrategyKind::LongTopK, 1, 1}}, w.next, 10.0);
  SelectorSpec s;
  s.lookback = 20;
  s.adaptEvery = 5;
  s.metric = ScoreMetric::Return;
  const AdaptiveResult a = runSelector(book, s);
  CHECK(a.evalFrom == 20 && a.net.size() == book.days() - 20);
  // Holds A before the regime change (date 100 = day 90) and B well after it.
  CHECK(a.selection[50 - 20] == 0);
  CHECK(a.selection[150 - 20] == 1);
  CHECK(a.switches >= 1 && a.switches <= 3);
  // It beats both fixed candidates over the same days.
  const auto fixedA = evaluatePerformance(book.netSeries(0, 20)), fixedB = evaluatePerformance(book.netSeries(1, 20));
  CHECK(a.metrics.totalReturn > fixedA.totalReturn && a.metrics.totalReturn > fixedB.totalReturn);
  // The switch pays for selling one stock and buying the other.
  for (std::size_t k = 1; k < a.selection.size(); ++k)
    if (a.selection[k] != a.selection[k - 1]) CHECK_NEAR(a.turnover[k], 2.0, 1e-12);
  CHECK_NEAR(a.turnover[0], 1.0, 1e-12);  // initial purchase
  double share = 0;
  for (double v : a.share) share += v;
  CHECK_NEAR(share, 1.0, 1e-12);
}

TEST(selector_risk_control_goes_to_cash) {
  TwoRegimeWorld w;
  // Long-only in a stock that only falls: nothing ever scores above zero.
  for (std::size_t t = 0; t < 200; ++t) w.next(t, 0) = w.next(t, 1) = -0.002;
  const CandidateBook book(w.models, {{StrategyKind::LongTopK, 1, 1}}, w.next, 10.0);
  SelectorSpec s;
  s.lookback = 20;
  s.adaptEvery = 10;
  AdaptiveResult a = runSelector(book, s);
  for (int sel : a.selection) CHECK(sel == -1);
  CHECK(a.metrics.totalReturn == 0.0 && a.share.back() == 1.0);
  s.allowCash = false;
  a = runSelector(book, s);
  for (int sel : a.selection) CHECK(sel >= 0);
  CHECK(a.metrics.totalReturn < 0.0);
}

TEST(selector_switching_bar_holds_until_the_lead_is_significant) {
  TwoRegimeWorld w;
  const CandidateBook book(w.models, {{StrategyKind::LongTopK, 1, 1}}, w.next, 10.0);
  SelectorSpec s;
  s.lookback = 20;
  s.adaptEvery = 5;
  s.metric = ScoreMetric::Return;
  s.allowCash = false;
  const AdaptiveResult free = runSelector(book, s);
  CHECK(free.heldBack == 0);
  // An unreachable bar: the first candidate is kept, whatever the record says.
  s.switchBar = 1e6;
  AdaptiveResult a = runSelector(book, s);
  CHECK(a.switches == 0 && a.heldBack > 0);
  for (int sel : a.selection) CHECK(sel == a.selection[0]);
  // A reachable bar still switches after the regime change, no sooner than without one.
  s.switchBar = 2.0;
  a = runSelector(book, s);
  CHECK(a.selection[150 - 20] == 1 && a.switches >= 1 && a.switches <= free.switches);
  std::size_t firstFree = 0, firstBar = 0;
  while (free.selection[firstFree] == free.selection[0]) ++firstFree;
  while (a.selection[firstBar] == a.selection[0]) ++firstBar;
  CHECK(firstBar >= firstFree);
  // The cost hurdle: a lead that cannot pay for the switch keeps the held candidate.
  s.switchBar = 0;
  s.switchCost = 1e6;
  a = runSelector(book, s);
  CHECK(a.switches == 0);
  s.switchCost = -1;
  CHECK_THROWS(runSelector(book, s));
}

TEST(selector_mixes_top_candidates) {
  TwoRegimeWorld w;
  const CandidateBook book(w.models, {{StrategyKind::LongTopK, 1, 1}, {StrategyKind::LongTopK, 2, 1}}, w.next, 0.0);
  SelectorSpec s;
  s.lookback = 30;
  s.adaptEvery = 10;
  s.topM = 2;
  s.allowCash = false;
  const AdaptiveResult a = runSelector(book, s);
  // The mix earns the average of its two members' gross returns.
  CHECK(a.net.size() == book.days() - 30);
  CHECK(s.label().find("top 2") != std::string::npos);
  CHECK_THROWS(runSelector(book, SelectorSpec{1, 5, ScoreMetric::Sharpe, 1, true, 0.0}));
}

TEST(window_score_metrics) {
  CHECK(windowScore(ScoreMetric::Sharpe, 1, 0.01, 0.0001, 0) < -1e29);
  CHECK_NEAR(windowScore(ScoreMetric::Return, 4, 0.04, 0.0, 0.0), 0.01, 1e-15);
  CHECK(windowScore(ScoreMetric::Sharpe, 4, 0.04, 0.0004, 0.0) > 999.0);  // no variance: mean / 1e-5
  CHECK(windowScore(ScoreMetric::Sortino, 4, -0.04, 0.0004, 0.0004) < 0);
  CHECK(parseScoreMetric("sortino") == ScoreMetric::Sortino);
  CHECK_THROWS(parseScoreMetric("alpha"));
}

TEST(grid_evaluation_matches_runs) {
  TwoRegimeWorld w;
  const CandidateBook book(w.models, {{StrategyKind::LongTopK, 1, 1}, {StrategyKind::LongShort, 1, 2}}, w.next, 5.0);
  const std::vector<SelectorSpec> sel = {{20, 5, ScoreMetric::Sharpe, 1, true, 0.0}, {40, 10, ScoreMetric::Return, 1, false, 0.0}};
  const GridResult g = evaluateGrid(book, sel, 40);
  CHECK(g.candidates.size() == 4 && g.selectors.size() == 2);
  const AdaptiveResult a = runSelector(book, sel[1], 40);
  CHECK_NEAR(g.selectors[1].sharpe, a.metrics.sharpe, 1e-12);
  CHECK(g.switches(1) == a.switches);
  CHECK_NEAR(g.candidates[2].totalReturn, evaluatePerformance(book.netSeries(2, 40)).totalReturn, 1e-12);
}

TEST(full_experiment_runs_end_to_end) {
  SyntheticMarketSpec ms;
  ms.numAssets = 12;
  ms.numDates = 520;
  const MarketData d = generateSyntheticMarket(ms);
  ExperimentSpec spec;
  spec.alphaIds = {2, 6, 12, 33, 41};
  spec.walkForward.trainWindow = 250;
  spec.walkForward.retrainEvery = 63;
  spec.models.resize(2);
  spec.models[0].type = "logistic";
  spec.models[1].type = "xgboost";
  spec.models[1].trees = 20;
  spec.selector.lookback = 42;
  const PredictionSet p = runPredictions(d, spec);
  CHECK(p.models.size() == 2 && p.models[0].oos.count > 1000);
  const Experiment e = runStrategies(p, spec);
  CHECK(e.book.size() == 2 * ExperimentSpec::defaultStrategies().size());
  CHECK(e.evalFrom == 42 && e.adaptive.net.size() == e.book.days() - 42);
  CHECK(e.benchmark.series.net.size() == e.adaptive.net.size());
  CHECK(e.candidateMetrics.size() == e.book.size() && e.bestFixed < e.book.size());
  CHECK(std::isfinite(e.adaptive.metrics.sharpe));
}
