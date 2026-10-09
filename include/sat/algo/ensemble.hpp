#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "sat/core/panel.hpp"

namespace sat::algo {

/// Adaptive pairs book: every `rescanEvery` days all pairs of the universe are tested for
/// cointegration (Engle-Granger) on the past `scanWindow` days, the `pairs` most strongly
/// cointegrated are traded with the Kalman-filter strategy until the next scan, with equal
/// capital, and the book sits in cash when no pair passes the test.
struct PairsBookSpec {
  std::size_t scanWindow = 252, rescanEvery = 63, pairs = 5;
  double delta = 1e-7, observationVariance = 1.0, entryZ = 1.0, exitZ = 0.0, costBps = 5.0;
};

struct PairsBookResult {
  std::vector<double> returns;  ///< returns[k] is earned from date start + k to start + k + 1
  std::vector<double> activePairs;  ///< pairs traded each day
  std::size_t start = 0, scans = 0;
};

PairsBookResult pairsBook(const Panel& close, const PairsBookSpec& spec);

/// How the meta-allocator spreads capital over the strategies ("sleeves").
enum class AllocationMethod {
  Equal,              ///< 1/S, rebalanced
  Best,               ///< all in the `topN` sleeves with the highest trailing Sharpe ratio
  SharpeWeighted,     ///< in proportion to the positive trailing Sharpe ratios
  InverseVolatility,  ///< in proportion to 1 / trailing volatility (risk parity without correlations)
  RiskAdjustedSharpe, ///< in proportion to positive Sharpe / volatility (mean-variance with a diagonal covariance)
  ExponentialWeights, ///< multiplicative weights (the Hedge algorithm): exp(eta x trailing annual Sharpe)
};

AllocationMethod parseAllocationMethod(const std::string& name);
std::string allocationName(AllocationMethod m);
std::vector<AllocationMethod> allAllocationMethods();

/// Self-adaptive allocation across strategies: every `rebalanceEvery` days the weights are
/// recomputed from each sleeve's returns over the past `lookback` days (strictly before the
/// day traded); `costBps` is charged on the change of weights. With `allowCash`, methods that
/// score by Sharpe hold cash when no sleeve has a positive trailing Sharpe ratio.
///
/// The defaults are the setting chosen by examples/strategy_tournament: ranked on one set of
/// synthetic markets and confirmed on an unseen set (exponential weights over a 63-day
/// look-back, re-weighted weekly, eta = 4).
struct AllocationSpec {
  AllocationMethod method = AllocationMethod::ExponentialWeights;
  std::size_t lookback = 63, rebalanceEvery = 5, topN = 1;
  double costBps = 2.0, eta = 4.0;
  bool allowCash = true;
};

struct AllocationResult {
  std::vector<double> returns;               ///< from day `lookback` on
  std::vector<std::vector<double>> weights;  ///< per sleeve, from day `lookback` on
  std::vector<double> cash;
  std::size_t rebalances = 0;
  double averageTurnover = 0.0;              ///< per rebalance
};

/// sleeves[s][t]: daily net return of strategy s on day t (all the same length).
AllocationResult allocate(const std::vector<std::vector<double>>& sleeves, const AllocationSpec& spec);

}  // namespace sat::algo
