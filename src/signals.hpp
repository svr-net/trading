#pragma once

// One signal of one stock on one day, in the arithmetic type R (double for the reference, float
// for the emulated GPU). The WGSL zscore kernel (kernels.cpp) computes the same expressions.
// at(table, t) reads the stock's table entry; tables: 0 log close, 1 prefix of log returns,
// 2 prefix of squared log returns, 3 prefix of centred log traded value, 7 log high, 8 log low,
// 9 traded value (all carried forward over missing days).
//
// Families: 0 return, 1 low volatility, 2 nearness to the high, 3 trend quality, 4 small size,
// 5 RSI (share of up moves in the absolute moves), 6 Bollinger z-score (close against the mean
// and spread of the previous h closes), 7 stochastic %K (close within the low-high range), 8 money flow
// (share of traded value on up days), 9 intraday range (negative mean high-low span).

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ofm::detail {

template <class R, class At>
R signal(std::size_t fam, std::size_t h, std::size_t t, At at) {
  const R fh = static_cast<R>(h);
  const R lc = at(0, t);
  switch (fam) {
    case 0: return lc - at(0, t - h);
    case 1:
    case 3: {
      const R d1 = at(1, t) - at(1, t - h), d2 = at(2, t) - at(2, t - h);
      const R vol = std::sqrt(std::max(R(0), (d2 - d1 * d1 / fh) / (fh - R(1))));
      if (fam == 1) return -vol;
      return vol > R(0) ? (lc - at(0, t - h)) / (vol * std::sqrt(fh)) : R(0);
    }
    case 2: {
      R mx = lc;
      for (std::size_t u = 1; u < h; ++u) mx = std::max(mx, at(0, t - u));
      return lc - mx;
    }
    case 4: return -(at(3, t) - at(3, t - h)) / fh;
    case 5: {
      R up = 0, tot = 0;
      for (std::size_t u = 0; u < h; ++u) {
        const R d = at(0, t - u) - at(0, t - u - 1);
        up = up + std::max(d, R(0));
        tot = tot + std::fabs(d);
      }
      return tot > R(0) ? up / tot : R(0.5);
    }
    case 6: {
      // Against the previous h closes (with today's close in the window, a 2-day z-score is
      // always +-0.707: only the sign of the last move).
      R s = 0;
      for (std::size_t u = 1; u <= h; ++u) s = s + at(0, t - u);
      const R m = s / fh;
      R v = 0;
      for (std::size_t u = 1; u <= h; ++u) v = v + (at(0, t - u) - m) * (at(0, t - u) - m);
      const R sd = std::sqrt(v / (fh - R(1)));
      return sd > R(0) ? (lc - m) / sd : R(0);
    }
    case 7: {
      R lo = at(8, t), hi = at(7, t);
      for (std::size_t u = 1; u < h; ++u) lo = std::min(lo, at(8, t - u)), hi = std::max(hi, at(7, t - u));
      return hi > lo ? (lc - lo) / (hi - lo) : R(0.5);
    }
    case 8: {
      R up = 0, tot = 0;
      for (std::size_t u = 0; u < h; ++u) {
        const R v = at(9, t - u);
        if (at(0, t - u) > at(0, t - u - 1)) up = up + v;
        tot = tot + v;
      }
      return tot > R(0) ? up / tot : R(0.5);
    }
    default: {
      R s = 0;
      for (std::size_t u = 0; u < h; ++u) s = s + (at(7, t - u) - at(8, t - u));
      return -s / fh;
    }
  }
}

}  // namespace ofm::detail
