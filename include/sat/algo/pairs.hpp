#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sat::algo {

/// Engle-Granger cointegration test of two price series: regress y on x (with a constant),
/// then test the residual spread for a unit root with the ADF statistic. The pair is
/// cointegrated at 5% when the statistic is below -3.34 (asymptotic critical value for two
/// series; stricter than the plain ADF value because the hedge ratio is estimated).
struct Cointegration {
  double hedgeRatio = 0.0, intercept = 0.0;
  double adf = 0.0;
  static constexpr double critical5 = -3.34;
  double halfLife = 0.0;  ///< of the spread, in days
  bool cointegrated() const { return adf < critical5; }
};

Cointegration engleGranger(const std::vector<double>& y, const std::vector<double>& x);

/// Half-life of mean reversion of a series, -ln 2 / lambda from the regression
/// ds_t = c + lambda s_{t-1} (an Ornstein-Uhlenbeck fit). Infinite if lambda >= 0.
double halfLife(const std::vector<double>& spread);

/// Two prices whose spread y - beta x is an Ornstein-Uhlenbeck process with the given
/// half-life: a cointegrated pair to test the strategy on.
struct PairSpec {
  std::size_t days = 1000;
  double beta = 1.5, halfLifeDays = 10.0, spreadVol = 0.01, priceVol = 0.015;
  std::uint64_t seed = 21;
};

void generatePair(const PairSpec& spec, std::vector<double>& y, std::vector<double>& x);

/// Mean-reversion pairs trading with a Kalman-filter hedge ratio. The filter tracks
/// y = beta x + alpha; its forecast error divided by its standard deviation is the z-score
/// of the spread. Enter long the spread (long y, short beta x) below -entryZ, short above
/// entryZ, exit when |z| falls under exitZ. Positions are set at the close and earn the next
/// day's move; returns are per unit of gross capital |y| + |beta x|, net of `costBps` per
/// unit of traded notional. The filter noises are in price units: the state noise must be
/// small enough that beta x cannot soak up the spread itself (beta's daily drift times the
/// price level well below the spread's swing), and the observation variance close to the
/// spread's variance.
struct PairsSpec {
  double delta = 1e-7, observationVariance = 1.0;
  double entryZ = 1.0, exitZ = 0.0;
  double costBps = 5.0;
  std::size_t warmup = 20;  ///< days before the first trade (filter convergence)
};

struct PairsResult {
  std::vector<double> returns, beta, zscore;
  std::vector<int> position;  ///< +1 long spread, -1 short, 0 flat
  std::size_t trades = 0;
};

PairsResult kalmanPairs(const std::vector<double>& y, const std::vector<double>& x, const PairsSpec& spec);

}  // namespace sat::algo
