#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace sat {

inline bool finite(double x) { return std::isfinite(x); }

/// Mean of the finite entries (NaN if there are none).
double mean(const std::vector<double>& x);

/// Sample standard deviation of the finite entries (NaN with fewer than two).
double stdev(const std::vector<double>& x);

/// Pearson correlation over the pairs where both entries are finite.
double correlation(const std::vector<double>& x, const std::vector<double>& y);

/// Spearman rank correlation over the pairs where both entries are finite.
double rankCorrelation(const std::vector<double>& x, const std::vector<double>& y);

/// Ranks 1..n of the finite entries with ties averaged; NaN stays NaN.
std::vector<double> averageRanks(const std::vector<double>& x);

/// Linear-interpolation quantile (q in [0, 1]) of the finite entries.
double quantile(std::vector<double> x, double q);

/// Indices that sort x descending; ties keep the lower index first, NaN goes last.
std::vector<std::size_t> argsortDescending(const double* x, std::size_t n);

}  // namespace sat
