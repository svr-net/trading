#pragma once

#include <cstddef>
#include <vector>

namespace sat::afml {

/// Weights of the fractional difference operator (1 - B)^d: w_0 = 1,
/// w_k = -w_{k-1} (d - k + 1) / k, truncated once |w_k| falls below `threshold`
/// (the fixed-width window method) or after `maxWidth` terms.
std::vector<double> fracDiffWeights(double d, double threshold = 1e-4, std::size_t maxWidth = 5000);

/// Fixed-width-window fractional differentiation of x. The first width - 1 values are NaN.
/// d = 1 is the ordinary first difference and d = 0 the series itself; values in between
/// keep part of the series' memory while removing enough of its trend to make it stationary.
std::vector<double> fracDiff(const std::vector<double>& x, double d, double threshold = 1e-4);

/// Augmented Dickey-Fuller test with a constant:
///   dx_t = a + g x_{t-1} + sum_{i=1..lags} p_i dx_{t-i} + e_t.
/// The statistic is the t-ratio of g; the unit-root hypothesis is rejected (the series is
/// stationary) when it lies below the critical value. Critical values are the asymptotic
/// ones for a regression with a constant.
struct AdfResult {
  double statistic = 0.0;
  std::size_t lags = 0, observations = 0;
  static constexpr double critical1 = -3.43, critical5 = -2.86, critical10 = -2.57;
  bool stationaryAt5() const { return statistic < critical5; }
};

AdfResult adfTest(const std::vector<double>& x, std::size_t lags = 1);

/// ADF statistic and correlation with the original series for d = 0, step, 2 step, ..., 1,
/// and the smallest d whose fractionally differentiated series passes the 5% test.
struct FracDiffScan {
  std::vector<double> d, adf, correlation;
  std::vector<std::size_t> width;
  double minimumD = 1.0;
};

FracDiffScan scanFracDiff(const std::vector<double>& x, double step = 0.1, double threshold = 1e-4, std::size_t lags = 1);

}  // namespace sat::afml
