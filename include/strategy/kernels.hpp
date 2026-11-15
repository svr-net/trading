// The fused kernels of apps/ofm_strategy (run by the chosen back end; include/strategy/backend.hpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "strategy/backend.hpp"
#include "strategy/strategy.hpp"

namespace strat {

struct PrepareSpec {
  std::size_t roll = 5;        // a close carried over at most this many missing bars
  std::size_t volWindow = 60;  // the volatility's sessions
  std::size_t volMin = 40;     // the fewest returns a volatility needs
  std::size_t valWindow = 20;  // the median traded value's bars
  double unit = 1.0;           // prices per unit of the currency (the UK's pence: 100)
};

// One share's sweep over its days, fused: the carried-forward close, its log return, the rolling volatility
// (running sums over the window), the median traded value of the last bars, and the first bar. Each output
// at day t uses days <= t only.
inline void fusedShare(Data& d, std::size_t i, const PrepareSpec& s) {
  const std::size_t T = d.T, N = d.N;
  double last = kNaN;
  std::size_t gap = 0;
  d.first[i] = T;
  std::vector<double> win;  // the volatility's returns, a ring
  win.reserve(s.volWindow);
  std::size_t head = 0;
  double s1 = 0, s2 = 0;
  std::vector<double> vals, tmp;  // the traded values of the last bars
  vals.reserve(s.valWindow);
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t k = t * N + i;
    const double c = d.C[k];
    double cf = kNaN;
    if (std::isfinite(c) && c > 0) {
      cf = c, gap = 0;
      if (d.first[i] == T) d.first[i] = t;
      const double v = d.V[k];
      if (std::isfinite(v) && v >= 0) {
        if (vals.size() == s.valWindow) vals.erase(vals.begin());
        vals.push_back(c * v / s.unit);
      }
    } else if (std::isfinite(last) && ++gap <= s.roll) {
      cf = last;
    }
    d.Cf[k] = cf;
    double r = kNaN;
    if (std::isfinite(cf) && std::isfinite(last)) r = std::log(cf / last);
    d.ret[k] = r;
    if (std::isfinite(r)) {
      if (win.size() < s.volWindow) {
        win.push_back(r);
      } else {
        s1 -= win[head], s2 -= win[head] * win[head];
        win[head] = r, head = (head + 1) % s.volWindow;
      }
      s1 += r, s2 += r * r;
    }
    const double n = double(win.size());
    d.sigma[k] = win.size() >= s.volMin ? std::sqrt(std::max(0.0, s2 / n - (s1 / n) * (s1 / n))) : kNaN;
    if (vals.size() == s.valWindow && std::isfinite(c)) {
      tmp = vals;
      std::nth_element(tmp.begin(), tmp.begin() + long(tmp.size() / 2), tmp.end());
      d.value[k] = tmp[tmp.size() / 2];
    } else {
      d.value[k] = kNaN;
    }
    if (std::isfinite(cf)) last = cf;
  }
}

struct UniverseSpec {
  double minPrice = 0, minValue = 0;
  std::size_t warmup = 260;  // sessions since a share's first bar
  bool needValue = true;     // false: a market without volumes (FOREX, INDEX)
  std::size_t top = 0;       // the day's most traded only (by the median traded value of the 20 bars before; 0: all)
};

// One day's universe and market return, fused: each share tradable after the close of t (a real bar, a price
// of at least minPrice, a median traded value of at least minValue, warmup sessions of history and a
// volatility), and the day's mean return over the tradable shares.
inline void fusedDay(Data& d, std::size_t t, const UniverseSpec& u, std::vector<char>& in) {
  const std::size_t N = d.N;
  std::vector<double> vals;
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t k = t * N + i;
    const bool ok = std::isfinite(d.C[k]) && d.C[k] >= u.minPrice && d.first[i] < d.T && t >= d.first[i] + u.warmup &&
                    std::isfinite(d.sigma[k]) && d.sigma[k] > 0 && (!u.needValue || (std::isfinite(d.value[k]) && d.value[k] >= u.minValue));
    in[k] = ok ? 1 : 0;
    if (ok && u.top && u.needValue) vals.push_back(d.value[k]);
  }
  // As of the day: only its `top` most traded (a list as it stood then, not today's most traded applied backwards).
  if (u.top && u.needValue && vals.size() > u.top) {
    std::nth_element(vals.begin(), vals.begin() + long(u.top - 1), vals.end(), std::greater<double>());
    const double cut = vals[u.top - 1];
    std::size_t kept = 0;
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t k = t * N + i;
      if (in[k] && !(d.value[k] >= cut && kept < u.top)) in[k] = 0;
      else if (in[k]) ++kept;
    }
  }
  double s = 0, n = 0;
  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t k = t * N + i;
    if (in[k] && std::isfinite(d.ret[k])) s += d.ret[k], n += 1;
  }
  d.mkt[t] = n > 0 ? s / n : 0.0;
}

// The kernels over the whole panel on a back end: shares in parallel, then days in parallel.
inline void prepare(Data& d, const PrepareSpec& p, const UniverseSpec& u, std::vector<char>& in, const Backend& b) {
  const std::size_t T = d.T, N = d.N;
  d.Cf.assign(T * N, kNaN), d.ret.assign(T * N, kNaN), d.sigma.assign(T * N, kNaN), d.value.assign(T * N, kNaN);
  d.first.assign(N, T), d.mkt.assign(T, 0.0), in.assign(T * N, 0);
  b.parallelFor(N, [&](std::size_t a, std::size_t e) {
    for (std::size_t i = a; i < e; ++i) fusedShare(d, i, p);
  });
  b.parallelFor(T, [&](std::size_t a, std::size_t e) {
    for (std::size_t t = a; t < e; ++t) fusedDay(d, t, u, in);
  });
}

}  // namespace strat
