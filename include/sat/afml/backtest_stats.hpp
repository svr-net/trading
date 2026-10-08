#pragma once

#include <cstddef>
#include <vector>

#include "sat/core/matrix.hpp"

namespace sat::afml {

/// Per-period Sharpe ratio (mean / standard deviation, not annualised).
double periodSharpe(const std::vector<double>& returns);

/// Probabilistic Sharpe ratio: the probability that the true per-period Sharpe ratio
/// exceeds `benchmark`, given an estimate `sharpe` from n observations with the given
/// skewness and (non-excess) kurtosis. Fat tails and negative skew widen the estimate's
/// standard error and lower the probability.
double probabilisticSharpe(double sharpe, double benchmark, double n, double skew, double kurt);

/// Expected maximum of `trials` independent Sharpe ratio estimates with variance
/// `variance` when the true Sharpe ratios are all zero (the false-discovery benchmark).
double expectedMaxSharpe(double trials, double variance);

/// Deflated Sharpe ratio: the probabilistic Sharpe ratio against the expected maximum
/// Sharpe ratio of the trials that were run to find the strategy.
double deflatedSharpe(double sharpe, double n, double skew, double kurt, double trials, double trialVariance);

/// Probability of backtest overfitting by combinatorially symmetric cross-validation.
///
/// The T x N matrix of the N strategies' returns is cut into S blocks of rows. For each of
/// the C(S, S/2) ways to pick half the blocks as in-sample, the strategy with the best
/// in-sample Sharpe ratio is ranked out of sample; the logit of its relative rank is
/// negative when it falls below the out-of-sample median. PBO is the share of such cases.
struct PboResult {
  double pbo = 0.0;
  std::size_t combinations = 0;
  std::vector<double> logits;
  std::vector<double> inSampleSharpe, outOfSampleSharpe;  ///< of the in-sample winner, per combination
  double probabilityOfLoss = 0.0;                          ///< share of combinations whose winner loses money out of sample
  double degradationSlope = 0.0;                           ///< regression slope of OOS on IS Sharpe
};

PboResult probabilityOfBacktestOverfitting(const Matrix& returns, std::size_t blocks = 16);

/// Drawdown episodes of a return series: the deepest drawdown, the longest time under water
/// (periods from a peak until it is regained, or to the end), and the 95% quantile of the
/// depths of all episodes.
struct DrawdownStats {
  double maxDrawdown = 0.0, drawdown95 = 0.0;
  std::size_t longestUnderWater = 0, episodes = 0;
};

DrawdownStats drawdownStats(const std::vector<double>& returns);

/// Concentration of positive (or negative) returns: the normalised Herfindahl index of the
/// return shares, 0 when every period contributes equally and 1 when one period is all.
double returnConcentration(const std::vector<double>& returns, bool positive = true);

}  // namespace sat::afml
