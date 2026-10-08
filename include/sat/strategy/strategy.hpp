#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sat/core/panel.hpp"
#include "sat/strategy/performance.hpp"

namespace sat {

/// How predicted probabilities become portfolio weights.
enum class StrategyKind : int {
  /// Equal-weight long position in the k stocks with the highest P(up).
  LongTopK = 0,
  /// Long the top k and short the bottom k, each leg with weight 1/k: dollar neutral,
  /// so the market move cancels and only the cross-sectional forecast is traded.
  LongShort = 1,
  /// Equal-weight long position in every stock with P(up) > threshold; all cash if none
  /// qualifies (market timing).
  Threshold = 2,
  /// Long weights proportional to P(up) - threshold where positive, summing to 1.
  ProbabilityWeighted = 3,
  /// Bet sizing from the forecast's confidence (Lopez de Prado, 2018, ch. 10): each stock's
  /// signed size 2 N(z) - 1 with z = (p - 1/2) / sqrt(p (1 - p)); sizes smaller in absolute
  /// value than the threshold are dropped, the rest scaled to a gross exposure of 1.
  BetSized = 4,
};

/// A fixed trading rule. Weights are set on rebalance dates (every `holding` dates from the
/// start of the backtest) from that date's predictions and held until the next rebalance.
struct StrategySpec {
  StrategyKind kind = StrategyKind::LongTopK;
  double param = 5.0;  ///< k (stocks per leg), the probability threshold, or the minimum bet size
  std::size_t holding = 1;
  std::string label() const;
};

StrategyKind parseStrategyKind(const std::string& name);
std::string strategyKindName(StrategyKind kind);

/// Position of each stock when the date's probabilities are sorted descending (0 = most
/// likely to rise); ties keep the lower stock index first, NaN sorts last.
std::vector<std::uint32_t> rankRow(const double* prob, std::size_t n);

/// Rank table of a probability panel (dates x assets, row-major).
std::vector<std::uint32_t> rankTable(const Panel& prob);

/// Target weights of one date. `rank` comes from rankRow.
void strategyWeights(const StrategySpec& s, const double* prob, const std::uint32_t* rank, std::size_t n, double* w);

/// Daily strategy returns over dates [start, end): the weights set at the close of t earn
/// the return from t to t + 1. Costs are `costBps` basis points per unit of turnover,
/// turnover being the sum of absolute weight changes (the first date buys the book).
struct DailySeries {
  std::size_t start = 0;
  std::vector<double> gross, turnover, net;
  std::vector<double> netExposure;  ///< sum of weights
};

/// Rebalance date of offset d for a holding period h (offsets from the start).
inline std::size_t rebalanceOffset(std::size_t d, std::size_t h) { return h <= 1 ? d : (d / h) * h; }

struct BacktestResult {
  std::string label;
  DailySeries series;
  PerformanceMetrics metrics;
};

BacktestResult backtest(const StrategySpec& s, const Panel& prob, const std::vector<std::uint32_t>& ranks,
                        const Panel& nextReturns, std::size_t start, std::size_t end, double costBps);

/// Equal-weight portfolio of the whole universe, rebalanced daily without costs after the
/// initial purchase: the market benchmark.
BacktestResult equalWeightBenchmark(const Panel& nextReturns, std::size_t start, std::size_t end, double costBps);

}  // namespace sat
