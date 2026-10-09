#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "sat/core/panel.hpp"

namespace sat::algo {

/// A systematic trend-following system in the style of Carver's Systematic Trading.
///
/// Each rule is an exponentially weighted moving-average crossover (fast, slow) whose raw
/// forecast is (EMA_fast - EMA_slow) / price volatility. Forecasts are scaled so that their
/// average absolute value is 10 (the scalar estimated from past data only) and capped at
/// +-20. The rules are combined with weights and a forecast diversification multiplier;
/// positions size each instrument to an equal share of the target portfolio volatility,
/// scaled up by the instrument diversification multiplier (from the average correlation of
/// the instruments' past returns) and by the combined forecast / 10. A buffer skips trades smaller than `buffer` x the average
/// position, which saves costs.
///
/// Self-adaptation: with `adaptiveWeights`, every `reweightEvery` days the rule weights are
/// reset in proportion to each rule's positive Sharpe ratio over the past `reweightWindow`
/// days (equal weights when none is positive), so the system leans towards the speeds that
/// work in the current market.
struct TrendSpec {
  std::vector<std::pair<std::size_t, std::size_t>> rules = {{2, 8}, {4, 16}, {8, 32}, {16, 64}, {32, 128}, {64, 256}};
  double targetVol = 0.15;       ///< annual portfolio volatility target
  double volSpan = 36.0;         ///< span of the instrument volatility estimate (days)
  double forecastCap = 20.0;
  double buffer = 0.1;
  double maxIdm = 2.5;           ///< cap on the instrument diversification multiplier
  double costBps = 5.0;
  bool longOnly = false;
  bool adaptiveWeights = true;
  std::size_t reweightEvery = 63, reweightWindow = 252;
  std::size_t warmup = 260;      ///< days before trading (slowest EMA, forecast scalars)
};

struct TrendResult {
  std::vector<double> returns, turnover, grossLeverage, idm;
  std::vector<std::vector<double>> ruleReturns;   ///< per rule, each traded alone with the same sizing
  std::vector<std::vector<double>> weights;       ///< per rule, over time
  std::vector<double> combinedForecastMean;       ///< average |combined forecast| across instruments per day
  std::vector<double> forecastScalars;            ///< final scalar per rule
  std::size_t start = 0;                          ///< date index of returns[0] (earned to start + 1)
};

TrendResult trendFollowing(const Panel& close, const TrendSpec& spec);

}  // namespace sat::algo
