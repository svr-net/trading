#pragma once

#include <cstddef>

#include "sat/core/panel.hpp"

namespace sat::ops {

/// The operator algebra in which the formulaic alphas are written (Kakushadze, 2016).
///
/// Cross-sectional operators act on one date across stocks; time-series operators act on
/// one stock over the past d dates, including today. A window that is not yet full, or
/// that contains NaN, gives NaN. Results never look ahead.

// Element-wise arithmetic (NaN propagates).
Panel operator+(const Panel& a, const Panel& b);
Panel operator-(const Panel& a, const Panel& b);
Panel operator*(const Panel& a, const Panel& b);
Panel operator/(const Panel& a, const Panel& b);  ///< x / 0 is NaN
Panel operator+(const Panel& a, double b);
Panel operator-(const Panel& a, double b);
Panel operator*(const Panel& a, double b);
Panel operator*(double a, const Panel& b);
Panel operator/(const Panel& a, double b);
Panel operator-(double a, const Panel& b);
Panel operator-(const Panel& a);

/// 1 where a < b, else 0 (NaN if either is NaN). Likewise lessThan(a, scalar) etc.
Panel lessThan(const Panel& a, const Panel& b);
Panel lessThan(const Panel& a, double b);
Panel greaterThan(double a, const Panel& b);  ///< a > b, i.e. b < a
/// cond ? a : b, element-wise (NaN where cond is NaN).
Panel where(const Panel& cond, const Panel& a, const Panel& b);
Panel where(const Panel& cond, const Panel& a, double b);

Panel abs(const Panel& x);
Panel sign(const Panel& x);
Panel log(const Panel& x);                     ///< NaN for x <= 0
Panel signedPower(const Panel& x, double a);   ///< sign(x) |x|^a
Panel power(const Panel& x, double a);         ///< x^a (NaN where undefined)

/// Cross-sectional percentile rank in (0, 1], ties averaged.
Panel rank(const Panel& x);
/// Rescaled so that the absolute values sum to `a` on each date.
Panel scale(const Panel& x, double a = 1.0);

Panel delay(const Panel& x, std::size_t d);
Panel delta(const Panel& x, std::size_t d);
Panel tsSum(const Panel& x, std::size_t d);
Panel tsMean(const Panel& x, std::size_t d);
Panel tsProduct(const Panel& x, std::size_t d);
Panel tsStddev(const Panel& x, std::size_t d);  ///< sample standard deviation
Panel tsMin(const Panel& x, std::size_t d);
Panel tsMax(const Panel& x, std::size_t d);
/// Position of the window maximum, 1 (oldest) .. d (today).
Panel tsArgMax(const Panel& x, std::size_t d);
/// Rank of today's value within the window, scaled to (0, 1] (ties averaged).
Panel tsRank(const Panel& x, std::size_t d);
/// Rolling Pearson correlation; NaN when either series is constant over the window.
Panel tsCorr(const Panel& x, const Panel& y, std::size_t d);
Panel tsCov(const Panel& x, const Panel& y, std::size_t d);

}  // namespace sat::ops
