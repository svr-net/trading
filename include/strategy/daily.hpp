// Helpers shared by the daily strategies (strategies/*.cpp): moving averages, the 2-day RSI, a share's beta to the
// market, its relative volume and the bar's internal strength. Each reads only the View, so only days up to its close.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "strategy/strategy.hpp"

namespace strat::daily {

// The mean close of the n sessions to day s (NaN when any is missing).
inline double sma(const View& v, std::size_t s, std::size_t n, std::size_t i) {
  if (s + 1 < n) return kNaN;
  double m = 0;
  for (std::size_t q = s + 1 - n; q <= s; ++q) {
    const double c = v.close(q, i);
    if (!(c > 0)) return kNaN;
    m += c;
  }
  return m / double(n);
}

// Wilder's RSI of length n at day s, its averages seeded `seed` sessions before (a fixed window: the same value
// whatever the history before it). 0: every move down; 100: every move up.
inline double rsi(const View& v, std::size_t s, std::size_t n, std::size_t i, std::size_t seed = 30) {
  if (s < seed + 1) return kNaN;
  double up = 0, dn = 0;
  const double a = 1.0 / double(n);
  for (std::size_t q = s - seed + 1; q <= s; ++q) {
    const double x = v.close(q - 1, i), y = v.close(q, i);
    if (!(x > 0 && y > 0)) return kNaN;
    const double d = y - x;
    up = (1 - a) * up + a * std::max(d, 0.0), dn = (1 - a) * dn + a * std::max(-d, 0.0);
  }
  return up + dn > 0 ? 100 * up / (up + dn) : 50;
}

// The share's beta to the market's daily return over the B sessions to day s (1 when it cannot be measured).
inline double beta(const View& v, std::size_t s, std::size_t B, std::size_t i) {
  if (s + 1 < B) return 1.0;
  double sxy = 0, sxx = 0, sx = 0, sy = 0, n = 0;
  for (std::size_t q = s + 1 - B; q <= s; ++q) {
    const double y = v.ret(q, i), x = v.market(q);
    if (std::isfinite(y) && std::isfinite(x)) sxy += x * y, sxx += x * x, sx += x, sy += y, n += 1;
  }
  if (n < 0.8 * double(B)) return 1.0;
  const double vx = sxx / n - (sx / n) * (sx / n);
  return vx > 0 ? (sxy / n - (sx / n) * (sy / n)) / vx : 1.0;
}

// The share's return net of beta times the market's over the W sessions to day s, in its daily volatilities
// times sqrt(W) (a z-score: -2 is a fall two volatilities more than the market explains). NaN when a day is missing.
inline double residual(const View& v, std::size_t s, std::size_t W, std::size_t B, std::size_t i) {
  if (s + 1 < std::max(W, B)) return kNaN;
  const double b = beta(v, s, B, i);
  double r = 0;
  for (std::size_t q = s + 1 - W; q <= s; ++q) {
    const double y = v.ret(q, i), x = v.market(q);
    if (!std::isfinite(y) || !std::isfinite(x)) return kNaN;
    r += y - b * x;
  }
  const double sg = v.sigma(s, i);
  return sg > 0 ? r / (sg * std::sqrt(double(W))) : kNaN;
}

// The mean volume of the W sessions to day s over the median volume of the B sessions before them (NaN without
// volumes: a market whose bars carry none, such as currencies).
inline double relVolume(const View& v, std::size_t s, std::size_t W, std::size_t B, std::size_t i) {
  if (s + 1 < W + B) return kNaN;
  double now = 0;
  for (std::size_t q = s + 1 - W; q <= s; ++q) {
    const double x = v.volume(q, i);
    if (!(x > 0)) return kNaN;
    now += x;
  }
  std::vector<double> past;
  past.reserve(B);
  for (std::size_t q = s + 1 - W - B; q <= s - W; ++q)
    if (v.volume(q, i) > 0) past.push_back(v.volume(q, i));
  if (past.size() < B / 2) return kNaN;
  std::nth_element(past.begin(), past.begin() + std::ptrdiff_t(past.size() / 2), past.end());
  const double med = past[past.size() / 2];
  return med > 0 ? now / double(W) / med : kNaN;
}

// The bar's internal strength: where the day closed in its range, 0 at the low, 1 at the high (NaN: no range).
inline double ibs(const View& v, std::size_t s, std::size_t i) {
  const double h = v.high(s, i), l = v.low(s, i), c = v.close(s, i);
  return v.bar(s, i) && h > l && c > 0 ? std::clamp((c - l) / (h - l), 0.0, 1.0) : kNaN;
}

// The highest and lowest close of the n sessions to day s.
inline double maxClose(const View& v, std::size_t s, std::size_t n, std::size_t i) {
  double m = kNaN;
  for (std::size_t q = s + 1 - std::min(n, s + 1); q <= s; ++q)
    if (v.close(q, i) > 0) m = std::isfinite(m) ? std::max(m, v.close(q, i)) : v.close(q, i);
  return m;
}
inline double minClose(const View& v, std::size_t s, std::size_t n, std::size_t i) {
  double m = kNaN;
  for (std::size_t q = s + 1 - std::min(n, s + 1); q <= s; ++q)
    if (v.close(q, i) > 0) m = std::isfinite(m) ? std::min(m, v.close(q, i)) : v.close(q, i);
  return m;
}

}  // namespace strat::daily
