// The nested Monte Carlo of apps/ofm_strategy (mc=1).
//
//   outer  resets on the real market: at each rebalance t0 the simulation is reset to the real daily OHLCV up to
//          t0's close: the real prices, the real tradable shares and the strategy's real list for the period.
//   inner  `paths` futures of H sessions from t0, each built from blocks of `block` whole real days drawn from the
//          `window` sessions up to t0 (never later): every share takes its own bar of the drawn day (open, high, low
//          and close relative to its close before), so the cross-section (a market fall on the same day for all)
//          and the short runs of volatility are kept. On each path the book runs as it does live: each listed
//          share bought at the next open under its cap, its stop (fixed or the Parabolic SAR) hit on the day's low,
//          bought back after a stop by the same rule, marked at the last close.
//   real   the one real future after t0 through the same simulator: where it falls among the simulated ones
//          (its percentile, and whether inside the 5-95% band) says whether the simulation is honest.
//
// The fused kernel: one path's draw and every share's trade on it in one pass (simPath), run in parallel over the
// paths by the chosen back end. The random numbers are counter-based (a hash of the seed, the reset and the path),
// so every back end draws the same paths and gives the same results.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "strategy/backend.hpp"
#include "strategy/strategy.hpp"

namespace strat {

struct McSpec {
  std::size_t paths = 500, horizon = 21, block = 5, window = 750;
  std::uint64_t seed = 1;
  bool psar = false;
  double stopLoss = 0, rebuy = -1, afStep = 0.02, afMax = 0.2;
  // vstop: the volatility stop, vstop times the share's daily volatility at the reset under the price paid, then
  // under the highest close (a chandelier); a close under it sells at the next open, and the price touching 1.5 times
  // that distance under it sells at once (the catastrophic stop at the broker); bought back once a close is rebuy
  // over the level it left at. 0: none.
  double vstop = 0;
  // psarVol: the volatility-adjusted SAR (with psar): it starts psarVol times the share's daily volatility under the
  // price paid, and rises as the SAR does but never closer than that under the close. 0: the plain SAR.
  double psarVol = 0;
  // drift: 0 each share's days as they were (its own past trend kept); 1 each share's own mean daily move over the
  // window taken out and the market's put in (the futures carry the share's risk, not its past trend)
  int drift = 1;
  // skip: the window ends this many sessions before t0 (the days drawn from [t0 - window, t0 - skip]): a
  // momentum book skips the last month (its short-term reversal)
  std::size_t skip = 0;
};

// A share's drift adjustment for a reset (a factor on each simulated day's bar): exp(market's mean daily log move
// over the window - the share's own), both from the window before t0 only; 1 with drift 0.
inline double mcAdjust(const Data& d, std::size_t t0, std::size_t i, const McSpec& s) {
  if (!s.drift) return 1.0;
  const std::size_t hi = t0 > s.skip + 1 ? t0 - s.skip : 1, lo = hi > s.window ? hi - s.window : 1;
  double si = 0, ni = 0, sm = 0, nm = 0;
  for (std::size_t q = lo; q <= hi; ++q) {
    const double r = d.ret[q * d.N + i];
    if (std::isfinite(r)) si += r, ni += 1;
    sm += d.mkt[q], nm += 1;
  }
  return ni > 20 && nm > 0 ? std::exp(sm / nm - si / ni) : 1.0;
}

// A position's state at the reset (a share the book holds already): its stop and the price it is marked at.
struct McHeld {
  bool held = false;
  double mark = 0, sar = 0, ep = 0, af = 0;  // vstop: the catastrophic level, the close's level, the distance
  bool exitNext = false;                    // vstop: a close under its level: sold at the first open
};

struct McTrade {
  double ret = 0;  // the slot's return over the horizon (cash: 0)
  bool stopped = false;
};

inline std::uint64_t mcMix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// The days of a simulated path: blocks of whole real days from [t0 - window, t0], each block's start drawn by the
// counter-based hash; or, for the real path (key 0 with real = true), the days after t0.
inline void mcDays(std::size_t t0, std::size_t T, const McSpec& s, std::uint64_t key, bool real, std::vector<std::size_t>& out) {
  out.clear();
  if (real) {
    for (std::size_t j = 1; j <= s.horizon && t0 + j < T; ++j) out.push_back(t0 + j);
    return;
  }
  const std::size_t hi = t0 > s.skip + 1 ? t0 - s.skip : 1;  // the last day that may be drawn (never after t0)
  const std::size_t lo = hi > s.window ? hi - s.window : 1, B = std::max<std::size_t>(1, s.block);
  const std::size_t span = hi + 1 > lo + B ? hi + 1 - lo - B + 1 : 1;
  std::uint64_t h = key;
  while (out.size() < s.horizon) {
    h = mcMix(h);
    const std::size_t b = lo + std::size_t(h % span);
    for (std::size_t j = 0; j < B && out.size() < s.horizon; ++j) out.push_back(std::min(hi, b + j));
  }
}

// One share's slot on a path: from the real close of t0 (and its real lows and highs before), the day bars of `days`
// applied in turn; the book's rules for the buy (the cap), the stop and the buy-back. The fused trade kernel.
// scale: the network's volatility scale (mcnn), each drawn day's relative bar raised to it (1: the real days as they were).
inline McTrade mcSlot(const Data& d, std::size_t t0, std::size_t i, const std::vector<std::size_t>& days, double cap, const McSpec& s, const McHeld& h, double adj = 1.0,
                      double scale = 1.0) {
  const std::size_t N = d.N;
  McTrade out;
  double last = d.Cf[t0 * N + i];
  if (!(last > 0)) return out;
  // the recent lows and highs (the SAR's start and its caps): the real ones before t0, then the path's
  double lows[8], highs[8];
  std::size_t nl = 0;
  for (std::size_t q = t0 >= 4 ? t0 - 4 : 0; q <= t0; ++q)
    if (d.L[q * N + i] > 0) lows[nl % 8] = d.L[q * N + i], highs[nl % 8] = d.H[q * N + i], ++nl;
  const auto lowOf = [&](std::size_t n) {  // the lowest of the last n lows
    double m = 0;
    for (std::size_t j = 0; j < std::min(n, std::min<std::size_t>(nl, 8)); ++j) {
      const double x = lows[(nl - 1 - j) % 8];
      m = m > 0 ? std::min(m, x) : x;
    }
    return m;
  };
  const auto highOf = [&](std::size_t n) {
    double m = 0;
    for (std::size_t j = 0; j < std::min(n, std::min<std::size_t>(nl, 8)); ++j) m = std::max(m, highs[(nl - 1 - j) % 8]);
    return m;
  };
  const bool stops = s.psar || s.stopLoss > 0;
  bool in = h.held, pending = !h.held, stopped = false, turned = false;
  double w = 1, mark = h.held ? h.mark : 0, sar = h.sar, ep = h.ep, af = h.af;
  // the volatility stop's state: the distance (a fraction of the price), the highest close, a sale due at the open
  const double sig0 = std::isfinite(d.sigma[t0 * N + i]) ? d.sigma[t0 * N + i] : 0.02;
  double vd = h.held ? h.af : 0, hic = h.held && h.af < 1 ? h.ep / (1 - h.af) : 0;
  bool exitNext = h.exitNext;
  double dsar = 0, dep = 0, daf = 0, level = 0;  // a stopped share's SAR on the way down, and the level it was stopped at
  for (std::size_t q : days) {
    const double pc = q > 0 ? d.Cf[(q - 1) * N + i] : 0.0, C = d.C[q * N + i];
    if (!(pc > 0 && C > 0 && d.O[q * N + i] > 0 && d.H[q * N + i] > 0 && d.L[q * N + i] > 0)) continue;  // no bar that day: flat
    const double base = last * adj;  // the day's bar on the path's price (adj: the drift adjustment, real days only)
    const auto bar = [&](double x) { return scale == 1.0 ? base * x / pc : base * std::pow(x / pc, scale); };
    const double o = bar(d.O[q * N + i]), hi = bar(d.H[q * N + i]), lo = bar(d.L[q * N + i]), c = bar(C);
    if (s.vstop > 0) {  // the volatility stop: its own day (the buy, the stop, the buy-back)
      if (!in && pending && o <= last * (1 + cap)) {
        in = true, pending = false, mark = o, vd = std::clamp(s.vstop * sig0, 0.02, 0.5);
        ep = o * (1 - vd), sar = o * (1 - 1.5 * vd), hic = o, exitNext = false;
      }
      if (in) {
        if (exitNext) {  // a close under its level yesterday: sold at this open
          w *= o / mark, in = false, out.stopped = stopped = true, exitNext = false, level = ep;
        } else if (lo <= sar) {  // the catastrophic stop: at its level, or the open through a gap
          w *= std::min(o, sar) / mark, in = false, out.stopped = stopped = true, level = ep;
        } else {
          w *= c / mark, mark = c;
          if (c < ep) {
            exitNext = true;
          } else {
            hic = std::max(hic, c), ep = std::max(ep, hic * (1 - vd)), sar = std::max(sar, ep * (1 - 0.5 * vd));
          }
        }
      } else if (stopped && s.rebuy >= 0 && c >= level * (1 + s.rebuy)) {
        pending = true, stopped = false;
      }
      last = c;
      continue;
    }
    if (!in && pending && o <= last * (1 + cap)) {  // bought at the open, under the cap
      in = true, pending = false, mark = o;
      if (s.psar && s.psarVol > 0) {
        sar = o * (1 - std::clamp(s.psarVol * sig0, 0.02, 0.5)), ep = o, af = s.afStep;
      } else if (s.psar) {
        const double l5 = lowOf(5);
        sar = l5 > 0 && l5 < o ? l5 : 0.95 * o, ep = o, af = s.afStep;
      } else {
        sar = s.stopLoss > 0 ? o * (1 - s.stopLoss) : 0.0;
      }
    }
    if (in) {
      if (stops && sar > 0 && lo <= sar) {  // stopped: at the stop, or the open through a gap
        const double px = std::min(o, sar);
        w *= px / mark, in = false, out.stopped = stopped = true, turned = false, level = px;
        if (s.psar) dsar = std::max(ep, px), dep = px, daf = s.afStep;
      } else {
        w *= c / mark, mark = c;
        if (s.psar) {
          if (hi > ep) ep = hi, af = std::min(s.afMax, af + s.afStep);
          double next = sar + af * (ep - sar);
          lows[nl % 8] = lo, highs[nl % 8] = hi, ++nl;
          next = std::min(next, lowOf(2));
          if (s.psarVol > 0) next = std::min(next, c * (1 - std::clamp(s.psarVol * sig0, 0.02, 0.5)));  // the volatility band
          sar = std::max(sar, next);
          last = c;
          continue;
        }
      }
    } else if (stopped && s.rebuy >= 0) {  // the buy-back: its SAR turned up (a close above it), or recovered
      if (s.psar) {
        if (c > dsar) {
          turned = true;
        } else {
          if (lo < dep) dep = lo, daf = std::min(s.afMax, daf + s.afStep);
          lows[nl % 8] = lo, highs[nl % 8] = hi, ++nl;
          dsar = std::min(dsar, std::max(dsar + daf * (dep - dsar), highOf(2)));
          last = c;
          continue;
        }
      } else {
        turned = c >= level * (1 + s.rebuy);
      }
      if (turned) pending = true, stopped = false;
    }
    lows[nl % 8] = lo, highs[nl % 8] = hi, ++nl;
    last = c;
  }
  out.ret = w - 1;
  return out;
}

// One reset's simulated book returns (a slot each for the listed shares, 1/k each; cash 0), path by path on the
// back end, and each share's slot returns (for the orders): ret[path], share[s * paths + path], stop[...].
struct McReset {
  std::vector<double> book, share;
  std::vector<char> stop;
};

// mu, scale: each listed share's shape from the network (mcnn; empty: the plain Monte Carlo): a daily log drift
// added to the drawn days and their scale.
inline void mcRun(const Data& d, std::size_t t0, const std::vector<std::size_t>& list, const std::vector<double>& caps, const std::vector<McHeld>& held,
                  std::size_t k, const McSpec& s, std::uint64_t resetKey, const Backend& b, McReset& r, const std::vector<double>& mu = {},
                  const std::vector<double>& scale = {}) {
  const std::size_t P = s.paths, S = list.size();
  r.book.assign(P, 0.0), r.share.assign(S * P, 0.0), r.stop.assign(S * P, 0);
  std::vector<double> adj(S), sc(S, 1.0);
  for (std::size_t j = 0; j < S; ++j) adj[j] = mcAdjust(d, t0, list[j], s) * (j < mu.size() ? std::exp(mu[j]) : 1.0), sc[j] = j < scale.size() ? scale[j] : 1.0;
  b.parallelFor(P, [&](std::size_t a, std::size_t e) {
    std::vector<std::size_t> days;
    for (std::size_t p = a; p < e; ++p) {
      mcDays(t0, d.T, s, mcMix(s.seed ^ mcMix(resetKey * 1000003ULL + p)), false, days);
      double sum = 0;
      for (std::size_t j = 0; j < S; ++j) {
        const McTrade x = mcSlot(d, t0, list[j], days, caps[j], s, held[j], adj[j], sc[j]);
        r.share[j * P + p] = x.ret, r.stop[j * P + p] = x.stopped ? 1 : 0, sum += x.ret;
      }
      r.book[p] = sum / double(std::max<std::size_t>(1, k));
    }
  });
}

// The real future after t0 through the same simulator: the book's return.
inline double mcReal(const Data& d, std::size_t t0, const std::vector<std::size_t>& list, const std::vector<double>& caps, std::size_t k, const McSpec& s) {
  std::vector<std::size_t> days;
  mcDays(t0, d.T, s, 0, true, days);
  double sum = 0;
  const McHeld fresh;
  for (std::size_t j = 0; j < list.size(); ++j) sum += mcSlot(d, t0, list[j], days, caps[j], s, fresh).ret;
  return sum / double(std::max<std::size_t>(1, k));
}

// The book the simulation chooses: of the candidates (their slot returns on the same paths, share[j * paths + p]),
// the k that together give the most expected log growth, mean over the paths of log(1 + book), the book's slots
// 1/k each (an unbought slot cash). Greedy: each step adds the candidate that raises it most; ties to the earlier
// candidate (the strategy's higher score). The paths keep the days the market moved together, so shares that fall
// together add less than their means. No costs (they come off the final orders only). Returns candidate positions.
// mean: true chooses by the expected return alone (the k highest simulated means), no penalty for risk.
inline std::vector<std::size_t> mcChoose(const McReset& r, std::size_t candidates, std::size_t paths, std::size_t k, bool mean = false) {
  std::vector<std::size_t> chosen;
  std::vector<char> used(candidates, 0);
  std::vector<double> sum(paths, 0.0);
  const double kk = double(std::max<std::size_t>(1, k));
  for (std::size_t step = 0; step < std::min(k, candidates); ++step) {
    std::size_t best = candidates;
    double bestV = -1e300;
    for (std::size_t j = 0; j < candidates; ++j) {
      if (used[j]) continue;
      double v = 0;
      for (std::size_t p = 0; p < paths; ++p) v += mean ? r.share[j * paths + p] : std::log(std::max(1e-9, 1.0 + (sum[p] + r.share[j * paths + p]) / kk));
      if (v > bestV + 1e-12) bestV = v, best = j;
    }
    if (best == candidates) break;
    used[best] = 1, chosen.push_back(best);
    for (std::size_t p = 0; p < paths; ++p) sum[p] += r.share[best * paths + p];
  }
  return chosen;
}

inline double mcQuantile(std::vector<double> x, double q) {
  if (x.empty()) return kNaN;
  std::sort(x.begin(), x.end());
  const double at = q * double(x.size() - 1);
  const std::size_t lo = std::size_t(at);
  return lo + 1 < x.size() ? x[lo] + (at - double(lo)) * (x[lo + 1] - x[lo]) : x[lo];
}

}  // namespace strat
