// A synthetic market from the real one, by the nested Monte Carlo's own draw (apps/ofm_strategy --resample SEED).
//
// The real market's daily bars become one alternative history of the same length: blocks of whole real days (every
// share's own bar of the drawn day, so the days the market moved together stay together), drawn in the market's
// hidden states (regime.hpp: the number of states learnt by BIC, 1 to 4, fitted to the whole history; each block from
// past days in the block's state, the states moving by the fitted chances), so calm and turbulent spells come in
// runs as they did. Each share keeps its real calendar (it trades on the days it really traded: listings, delistings
// and gaps as they were) and its real volume (its liquidity, so the day's universe is the real one's); its prices
// start from its real first bar and then move by the drawn days' moves (open, high, low and close over the day's
// previous close). A drawn day on which the share had no bar (before it listed, after it delisted, a gap) is replaced
// for that share by another of its own real days, in the drawn day's state where one can be found, so a late listing
// keeps all its bars.
// Deterministic for a seed: the draws come from the counter-based hash.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "strategy/montecarlo.hpp"
#include "strategy/regime.hpp"
#include "strategy/strategy.hpp"

namespace strat {

struct ResampleInfo {
  std::size_t states = 0, block = 0;
};

// d: the raw bars (O, H, L, C, V; before prepare), replaced by the synthetic history. block: the blocks' length in days.
inline ResampleInfo resampleMarket(Data& d, std::uint64_t seed, std::size_t block) {
  const std::size_t T = d.T, N = d.N, B = std::max<std::size_t>(1, block);
  ResampleInfo info;
  info.block = B;
  if (T < 3 * B + 2 || N == 0) return info;
  // each real day's bar of each share over its previous close (the last close before it), and the market's move
  std::vector<double> prev(N, kNaN), r4(4 * T * N, kNaN);
  d.mkt.assign(T, 0.0);
  for (std::size_t q = 0; q < T; ++q) {
    double s = 0, n = 0;
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t k = q * N + i;
      const double c = d.C[k];
      if (!(c > 0)) continue;
      const double pc = prev[i];
      if (pc > 0 && d.O[k] > 0 && d.H[k] > 0 && d.L[k] > 0) {
        r4[4 * k] = d.O[k] / pc, r4[4 * k + 1] = d.H[k] / pc, r4[4 * k + 2] = d.L[k] / pc, r4[4 * k + 3] = c / pc;
        const double lr = std::log(c / pc);
        if (std::fabs(lr) < 0.5) s += lr, n += 1;  // a split or a bad bar is not the market's move
      }
      prev[i] = c;
    }
    d.mkt[q] = n > 0 ? s / n : 0.0;
  }
  // the market's states over the whole history: 1 to 4, the lowest BIC
  RegimeFit best;
  for (std::size_t K = 1; K <= 4; ++K) {
    RegimeFit f;
    fitRegime(d, T - 1, T, 0, B, B, K, nullptr, f);
    if (best.K == 0 || f.bic < best.bic) best = std::move(f);
  }
  info.states = best.K;
  // each share's own days with a bar (its first and last), for the drawn days on which it had none
  std::vector<std::size_t> fi(N, T), li(N, 0);
  for (std::size_t q = 1; q < T; ++q)
    for (std::size_t i = 0; i < N; ++i)
      if (r4[4 * (q * N + i) + 3] > 0) fi[i] = std::min(fi[i], q), li[i] = q;
  const auto stateOf = [&](std::size_t q) {
    return q >= best.lo && q - best.lo < best.label.size() ? best.label[q - best.lo] : -1;
  };
  // the synthetic calendar: day 0 as it was, then T - 1 days drawn in states from the real days 1 .. T-1
  std::uint64_t h = mcMix(seed * 0x9E3779B97F4A7C15ULL + 12345);
  std::vector<std::size_t> drawn;
  RegimeFit f = best;
  f.now.assign(best.K, 0.0);  // the first state from the states' share of the days
  for (int lab : best.label) f.now[std::size_t(lab)] += 1.0 / double(best.label.size());
  regimeDays(f, T - 1, 1, B, T - 1, [&] { h = mcMix(h); return double(h >> 11) / 9007199254740992.0; }, drawn);
  // the shares' prices along the drawn days, on their own real calendar, with their real volume
  std::vector<double> O(T * N, kNaN), H(T * N, kNaN), L(T * N, kNaN), C(T * N, kNaN);
  std::vector<double> last(N, kNaN);
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t k = t * N + i;
      if (!(d.C[k] > 0)) continue;  // no real bar that day: none here either
      if (!(last[i] > 0)) {  // its first bar: as it really was
        O[k] = d.O[k], H[k] = d.H[k], L[k] = d.L[k], C[k] = d.C[k], last[i] = d.C[k];
        continue;
      }
      std::size_t q = t == 0 ? 0 : drawn[t - 1];
      if (!(r4[4 * (q * N + i) + 3] > 0)) {  // the drawn day had no bar of it: another of its own days
        if (fi[i] > li[i]) continue;  // it never had a move to draw
        const int z = stateOf(q);
        const std::size_t span = li[i] - fi[i] + 1;
        std::size_t any = T;
        bool found = false;
        for (int tries = 0; tries < 32 && !found; ++tries) {
          h = mcMix(h);
          const std::size_t c = fi[i] + std::size_t(double(h >> 11) / 9007199254740992.0 * double(span)) % span;
          if (!(r4[4 * (c * N + i) + 3] > 0)) continue;
          if (any == T) any = c;
          if (z < 0 || stateOf(c) == z) q = c, found = true;
        }
        if (!found) {
          if (any == T)  // the nearest of its days to the drawn one
            for (std::size_t dq = 0; dq < span && any == T; ++dq) {
              const std::size_t c0 = std::clamp(q, fi[i], li[i]);
              if (c0 >= fi[i] + dq && r4[4 * ((c0 - dq) * N + i) + 3] > 0) any = c0 - dq;
              else if (c0 + dq <= li[i] && r4[4 * ((c0 + dq) * N + i) + 3] > 0) any = c0 + dq;
            }
          q = any;
        }
      }
      const std::size_t kq = q * N + i;
      const double p = last[i];
      O[k] = p * r4[4 * kq], H[k] = p * r4[4 * kq + 1], L[k] = p * r4[4 * kq + 2], C[k] = p * r4[4 * kq + 3];
      last[i] = C[k];
    }
  }
  d.O.swap(O), d.H.swap(H), d.L.swap(L), d.C.swap(C);
  d.mkt.clear();  // prepare computes the synthetic market's own
  return info;
}

}  // namespace strat
