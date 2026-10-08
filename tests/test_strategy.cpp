#include <cmath>
#include <numeric>

#include "sat/strategy/performance.hpp"
#include "sat/strategy/strategy.hpp"
#include "test_framework.hpp"

using namespace sat;

TEST(strategy_weights) {
  const double p[6] = {0.60, 0.40, 0.70, 0.52, 0.30, 0.55};
  const auto rank = rankRow(p, 6);
  CHECK(rank[2] == 0 && rank[0] == 1 && rank[5] == 2 && rank[4] == 5);
  double w[6];
  strategyWeights({StrategyKind::LongTopK, 2, 1}, p, rank.data(), 6, w);
  CHECK(w[2] == 0.5 && w[0] == 0.5 && w[5] == 0.0);
  strategyWeights({StrategyKind::LongShort, 2, 1}, p, rank.data(), 6, w);
  CHECK(w[2] == 0.5 && w[0] == 0.5 && w[4] == -0.5 && w[1] == -0.5 && w[3] == 0.0);
  strategyWeights({StrategyKind::Threshold, 0.54, 1}, p, rank.data(), 6, w);
  CHECK_NEAR(w[0] + w[2] + w[5], 1.0, 1e-12);
  CHECK(w[3] == 0.0 && std::fabs(w[0] - 1.0 / 3) < 1e-12);
  strategyWeights({StrategyKind::Threshold, 0.9, 1}, p, rank.data(), 6, w);
  CHECK(std::accumulate(w, w + 6, 0.0) == 0.0);  // nothing qualifies: cash
  strategyWeights({StrategyKind::ProbabilityWeighted, 0.5, 1}, p, rank.data(), 6, w);
  CHECK_NEAR(std::accumulate(w, w + 6, 0.0), 1.0, 1e-12);
  CHECK_NEAR(w[2] / w[0], 2.0, 1e-12);
  CHECK(StrategySpec({StrategyKind::LongShort, 3, 5}).label() == "Long-short 3 /5d");
  CHECK(parseStrategyKind("threshold") == StrategyKind::Threshold);
  CHECK_THROWS(parseStrategyKind("magic"));
}

TEST(backtest_books_returns_and_costs) {
  // Two stocks, the model always prefers stock 0; stock 0 earns 1% a day, stock 1 loses 1%.
  const std::size_t T = 6;
  Panel prob(T, 2), next(T, 2);
  for (std::size_t t = 0; t < T; ++t) {
    prob(t, 0) = 0.6;
    prob(t, 1) = 0.4;
    next(t, 0) = 0.01;
    next(t, 1) = -0.01;
  }
  const auto ranks = rankTable(prob);
  const auto top = backtest({StrategyKind::LongTopK, 1, 1}, prob, ranks, next, 1, 5, 10.0);
  CHECK(top.series.gross.size() == 4);
  CHECK_NEAR(top.series.gross[0], 0.01, 1e-15);
  CHECK_NEAR(top.series.turnover[0], 1.0, 1e-15);
  CHECK_NEAR(top.series.net[0], 0.01 - 0.001, 1e-15);
  CHECK_NEAR(top.series.turnover[1], 0.0, 1e-15);
  const auto ls = backtest({StrategyKind::LongShort, 1, 1}, prob, ranks, next, 1, 5, 0.0);
  CHECK_NEAR(ls.series.gross[2], 0.02, 1e-15);
  CHECK_NEAR(ls.series.netExposure[2], 0.0, 1e-15);
  CHECK_NEAR(ls.metrics.totalReturn, std::pow(1.02, 4) - 1, 1e-12);
  CHECK_THROWS(backtest({StrategyKind::LongTopK, 1, 0}, prob, ranks, next, 1, 5, 0.0));
  CHECK_THROWS(backtest({StrategyKind::LongTopK, 1, 1}, prob, ranks, next, 5, 5, 0.0));
}

TEST(holding_period_rebalances_on_schedule) {
  const std::size_t T = 12;
  Panel prob(T, 2), next(T, 2, 0.0);
  for (std::size_t t = 0; t < T; ++t) {  // the preferred stock alternates every day
    prob(t, 0) = t % 2 ? 0.4 : 0.6;
    prob(t, 1) = t % 2 ? 0.6 : 0.4;
  }
  const auto ranks = rankTable(prob);
  const auto daily = backtest({StrategyKind::LongTopK, 1, 1}, prob, ranks, next, 0, 10, 0.0);
  const auto weekly = backtest({StrategyKind::LongTopK, 1, 5}, prob, ranks, next, 0, 10, 0.0);
  const double dailyTurnover = std::accumulate(daily.series.turnover.begin(), daily.series.turnover.end(), 0.0);
  const double weeklyTurnover = std::accumulate(weekly.series.turnover.begin(), weekly.series.turnover.end(), 0.0);
  CHECK_NEAR(dailyTurnover, 1.0 + 9 * 2.0, 1e-12);
  CHECK_NEAR(weeklyTurnover, 1.0 + 2.0, 1e-12);  // buys on day 0, switches on day 5
  CHECK(rebalanceOffset(7, 5) == 5 && rebalanceOffset(7, 1) == 7);
}

TEST(performance_metrics) {
  const std::vector<double> r = {0.01, -0.02, 0.03, 0.0, -0.01};
  const auto m = evaluatePerformance(r, {1, 0, 0, 0, 0});
  CHECK(m.days == 5);
  double w = 1;
  for (double x : r) w *= 1 + x;
  CHECK_NEAR(m.totalReturn, w - 1, 1e-12);
  CHECK_NEAR(m.annualReturn, std::pow(w, 252.0 / 5) - 1, 1e-9);
  const double mean = 0.002, sd = std::sqrt((0.008 * 0.008 + 0.022 * 0.022 + 0.028 * 0.028 + 0.002 * 0.002 + 0.012 * 0.012) / 4);
  CHECK_NEAR(m.sharpe, mean / sd * std::sqrt(252.0), 1e-9);
  CHECK_NEAR(m.sortino, mean / std::sqrt((0.0004 + 0.0001) / 5) * std::sqrt(252.0), 1e-9);
  CHECK_NEAR(m.maxDrawdown, 1 - 1.01 * 0.98 / 1.01, 1e-12);
  CHECK_NEAR(m.winRate, 0.4, 1e-12);
  CHECK_NEAR(m.averageTurnover, 0.2, 1e-12);
  const auto dd = drawdownCurve(r);
  CHECK(dd.size() == 6 && dd[0] == 0 && std::fabs(dd[2] - 0.02) < 1e-12);
  CHECK(equityCurve(r).back() == w);
  CHECK(evaluatePerformance({}).days == 0);
}

TEST(equal_weight_benchmark) {
  Panel next(4, 2);
  for (std::size_t t = 0; t < 4; ++t) {
    next(t, 0) = 0.02;
    next(t, 1) = 0.0;
  }
  const auto b = equalWeightBenchmark(next, 0, 4, 0.0);
  CHECK_NEAR(b.metrics.totalReturn, std::pow(1.01, 4) - 1, 1e-12);
}
