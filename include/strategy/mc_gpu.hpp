// The nested Monte Carlo's trade kernel as one batch for a GPU (the Lab's webgpu back end, backends/webgpu.cpp and
// web/lab/mc.wgsl), with its reference here.
//
//   batch    G stop settings x S shares x P paths, every one independent: the days each path draws (H a path, as
//            indices into the reset's window), each share's day bars over that window as ratios to the day's previous
//            close (raised to the network's scale; 0: no bar that day), each share's start (its close, volatility,
//            drift factor, cap, the real lows and highs of the 5 sessions before, its held position) and the settings
//   kernel   one invocation per (setting, share, path): the path's bars and the book's rules on them (mcPath, then
//            mcSlotPath, in montecarlo.hpp), in one pass over the days without storing the bars
//   result   each one's return over the horizon and whether a stop sold it
//
// mcBatchRef<T> is the kernel in C++, statement for statement the shader's (T = float: the GPU's arithmetic, WebGPU
// has no doubles); T = double checks the batch's logic against montecarlo.hpp. A back end that can run the batch sets
// mcOffload(); the engine then packs its resets and stop searches (mcPack) and hands them over, and runs them itself
// whenever the offload declines (a day outside the window, a setting it does not carry), so results never depend on
// whether a GPU was there, only on its single precision.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "strategy/montecarlo.hpp"

namespace strat {

struct McGpuSpec {  // a stop setting (montecarlo.hpp's McSpec, as the shader reads it)
  float stopLoss = 0, rebuy = -1, afStep = 0.02f, afMax = 0.2f, vstop = 0, psarVol = 0, rebuyStop = 0;
  std::uint32_t flags = 0;  // 1 psar, 2 rebuyTrail, 4 trip (each buy-back pays the share's round trip)
};
struct McGpuShare {  // a share at the reset
  float last0 = 0, sig0 = 0.02f, adj = 1, cap = 0, trip = 0;
  float lows[5] = {0, 0, 0, 0, 0}, highs[5] = {0, 0, 0, 0, 0};
  std::uint32_t nl = 0;  // how many of lows/highs are real (the latest last)
  float mark = 0, sar = 0, ep = 0, af = 0;  // a held position (McHeld)
  std::uint32_t held = 0;                   // 1 held, 2 exitNext
  float bandMin = 0.02f, bandMax = 0.5f, slip = 0;  // McSpec's (the same for every setting of a batch)
};
constexpr std::uint32_t kMcNoDay = 0xFFFFFFFFu;
struct McBatch {
  std::uint32_t P = 0, H = 0, S = 0, G = 0, D = 0;  // paths, days a path, shares, settings, days in the window
  std::vector<std::uint32_t> days;                  // P * H: index into the window (kMcNoDay: none)
  std::vector<float> table;                         // S * D * 4: open, high, low, close over the previous close (0: no bar)
  std::vector<double> table64;                      // the same in doubles (the f64 reference's; never sent to a GPU)
  std::vector<McGpuShare> shares;
  std::vector<McGpuSpec> specs;
  std::size_t out(std::size_t g, std::size_t s, std::size_t p) const { return (g * S + s) * P + p; }
};

// A back end's batch runner: fills ret and stopped (G * S * P each); false: not run (the engine runs it itself).
using McOffload = bool (*)(const McBatch&, std::vector<float>& ret, std::vector<std::uint32_t>& stopped);
inline McOffload& mcOffload() {
  static McOffload f = nullptr;
  return f;
}

inline McGpuSpec mcGpuSpec(const McSpec& s) {
  McGpuSpec g;
  g.stopLoss = float(s.stopLoss), g.rebuy = float(s.rebuy), g.afStep = float(s.afStep), g.afMax = float(s.afMax);
  g.vstop = float(s.vstop), g.psarVol = float(s.psarVol), g.rebuyStop = float(s.rebuyStop);
  g.flags = (s.psar ? 1u : 0u) | (s.rebuyTrail ? 2u : 0u) | (s.trip ? 4u : 0u);
  return g;
}

// The batch for reset t0: the shares `list` (caps, held, drift factors adj, scales sc), the settings `specs` (sharing
// one trip table, the window and the horizon), the days each path draws (`paths`, H each). False: it cannot be a
// batch (a day outside the window, settings with different trip tables): the engine runs it itself.
inline bool mcPack(const Data& d, std::size_t t0, const std::vector<std::size_t>& list, const std::vector<double>& caps, const std::vector<McHeld>& held,
                   const std::vector<double>& adj, const std::vector<double>& sc, const std::vector<McSpec>& specs,
                   const std::vector<std::vector<std::size_t>>& paths, McBatch& b) {
  if (specs.empty() || paths.empty()) return false;
  const McSpec& s0 = specs[0];
  for (const auto& s : specs)
    if (s.trip != s0.trip || s.window != s0.window || s.skip != s0.skip || s.horizon != s0.horizon || s.bandMin != s0.bandMin || s.bandMax != s0.bandMax ||
        s.slip != s0.slip)
      return false;
  const std::size_t N = d.N;
  const std::size_t hi = t0 > s0.skip + 1 ? t0 - s0.skip : 1, lo = hi > s0.window ? hi - s0.window : 1;
  b.P = std::uint32_t(paths.size()), b.S = std::uint32_t(list.size()), b.G = std::uint32_t(specs.size()), b.D = std::uint32_t(hi - lo + 1);
  b.H = 0;
  for (const auto& p : paths) b.H = std::max<std::uint32_t>(b.H, std::uint32_t(p.size()));
  b.days.assign(std::size_t(b.P) * b.H, kMcNoDay);
  for (std::size_t p = 0; p < paths.size(); ++p)
    for (std::size_t k = 0; k < paths[p].size(); ++k) {
      const std::size_t q = paths[p][k];
      if (q < lo || q > hi) return false;  // only the window is in the table
      b.days[p * b.H + k] = std::uint32_t(q - lo);
    }
  b.table.assign(std::size_t(b.S) * b.D * 4, 0.0f);
  b.table64.assign(b.table.size(), 0.0);
  b.shares.assign(b.S, McGpuShare());
  const RegimeFit* rf = mcFit(s0, t0);
  for (std::size_t j = 0; j < list.size(); ++j) {
    const std::size_t i = list[j];
    const double scale = j < sc.size() ? sc[j] : 1.0;
    for (std::size_t q = lo; q <= hi; ++q) {
      const double pc = q > 0 ? d.Cf[(q - 1) * N + i] : 0.0, C = d.C[q * N + i];
      if (!(pc > 0 && C > 0 && d.O[q * N + i] > 0 && d.H[q * N + i] > 0 && d.L[q * N + i] > 0)) continue;  // no bar that day: flat (zeros)
      const std::size_t at = (j * b.D + (q - lo)) * 4;
      const double x[4] = {d.O[q * N + i], d.H[q * N + i], d.L[q * N + i], C};
      for (int f = 0; f < 4; ++f) b.table64[at + f] = scale == 1.0 ? x[f] / pc : std::pow(x[f] / pc, scale), b.table[at + f] = float(b.table64[at + f]);
    }
    McGpuShare& sh = b.shares[j];
    sh.last0 = float(d.Cf[t0 * N + i]);
    sh.sig0 = float(rf && !rf->sigma.empty() && std::isfinite(rf->sigma[i]) ? rf->sigma[i] : std::isfinite(d.sigma[t0 * N + i]) ? d.sigma[t0 * N + i] : 0.02);
    sh.adj = float(j < adj.size() ? adj[j] : 1.0), sh.cap = float(caps[j]), sh.trip = s0.trip ? float(s0.trip[i]) : 0.0f;
    for (std::size_t q = t0 >= 4 ? t0 - 4 : 0; q <= t0; ++q)
      if (d.L[q * N + i] > 0 && sh.nl < 5) sh.lows[sh.nl] = float(d.L[q * N + i]), sh.highs[sh.nl] = float(d.H[q * N + i]), ++sh.nl;
    const McHeld& h = held[j];
    sh.held = (h.held ? 1u : 0u) | (h.exitNext ? 2u : 0u);
    sh.mark = float(h.mark), sh.sar = float(h.sar), sh.ep = float(h.ep), sh.af = float(h.af);
    sh.bandMin = float(s0.bandMin), sh.bandMax = float(s0.bandMax), sh.slip = float(s0.slip);
  }
  b.specs.clear();
  for (const auto& s : specs) b.specs.push_back(mcGpuSpec(s));
  return true;
}

// The kernel for one (setting g, share s, path p): mcPath and mcSlotPath in one pass, in T arithmetic. The shader is
// this function statement for statement (T = f32).
template <class T>
inline void mcBatchRef(const McBatch& b, std::size_t g, std::size_t sIdx, std::size_t p, T& retOut, bool& stoppedOut) {
  const McGpuShare& sh = b.shares[sIdx];
  const McGpuSpec& s = b.specs[g];
  retOut = T(0), stoppedOut = false;
  const T last0 = T(sh.last0);
  if (!(last0 > T(0))) return;
  const bool psar = s.flags & 1u, trail = s.flags & 2u, tripOn = s.flags & 4u;
  const T sig0 = T(sh.sig0), cap = T(sh.cap), adj = T(sh.adj);
  const T stopLoss = T(s.stopLoss), rebuy = T(s.rebuy), afStep = T(s.afStep), afMax = T(s.afMax), vstop = T(s.vstop), psarVol = T(s.psarVol),
          rebuyStop = T(s.rebuyStop);
  // the recent lows and highs (the last 5 real ones, then the path's), newest last
  T lows[5], highs[5];
  std::uint32_t nl = 0;
  for (std::uint32_t k = 0; k < sh.nl; ++k) lows[k] = T(sh.lows[k]), highs[k] = T(sh.highs[k]);
  nl = sh.nl;
  const auto lowOf = [&](std::uint32_t n) {
    T m = T(0);
    for (std::uint32_t j = nl - std::min(n, nl); j < nl; ++j) m = m > T(0) ? std::min(m, lows[j]) : lows[j];
    return m;
  };
  const auto highOf = [&](std::uint32_t n) {
    T m = T(0);
    for (std::uint32_t j = nl - std::min(n, nl); j < nl; ++j) m = std::max(m, highs[j]);
    return m;
  };
  const bool stops = psar || stopLoss > T(0);
  bool in = sh.held & 1u, pending = !(sh.held & 1u), stopped = false, turned = false, outStopped = false;
  T w = T(1), mark = in ? T(sh.mark) : T(0), sar = T(sh.sar), ep = T(sh.ep), af = T(sh.af);
  T vd = in ? T(sh.af) : T(0), hic = in && T(sh.af) < T(1) ? T(sh.ep) / (T(1) - T(sh.af)) : T(0);
  bool exitNext = sh.held & 2u;
  T dsar = T(0), dep = T(0), daf = T(0), level = T(0), low = T(0);
  const T band = psarVol > T(0) ? std::clamp(psarVol * sig0, T(sh.bandMin), T(sh.bandMax)) : T(0);
  const T slip = T(1) - T(sh.slip) * sig0;
  T last = last0;
  for (std::uint32_t k = 0; k < b.H; ++k) {
    const std::uint32_t q = b.days[std::size_t(p) * b.H + k];
    if (q == kMcNoDay) continue;
    const std::size_t at = (std::size_t(sIdx) * b.D + q) * 4;
    T r[4];
    for (int f = 0; f < 4; ++f) r[f] = sizeof(T) == sizeof(double) && !b.table64.empty() ? T(b.table64[at + f]) : T(b.table[at + f]);
    if (r[3] == T(0)) continue;  // no bar that day: flat
    const T base = last * adj;
    const T o = base * r[0], hi = base * r[1], lo = base * r[2], c = base * r[3];
    const T l5 = lowOf(5);
    if (nl == 5) {  // keep the last 4, then add the day's (only the last 5 are ever read)
      for (std::uint32_t j = 0; j < 4; ++j) lows[j] = lows[j + 1], highs[j] = highs[j + 1];
      nl = 4;
    }
    lows[nl] = lo, highs[nl] = hi, ++nl;
    const T l2 = lowOf(2), h2 = highOf(2);
    const T prev = last;
    last = c;
    // the book's rules on the day (mcSlotPath)
    if (vstop > T(0)) {
      if (!in && pending && o <= prev * (T(1) + cap)) {
        if (outStopped && tripOn) w *= T(1) - T(sh.trip);
        in = true, pending = false, mark = o, vd = std::clamp(vstop * sig0, T(sh.bandMin), T(sh.bandMax));
        ep = o * (T(1) - vd), sar = o * (T(1) - T(1.5) * vd), hic = o, exitNext = false;
      }
      if (in) {
        if (exitNext) {
          w *= o * slip / mark, in = false, outStopped = stopped = true, exitNext = false, level = ep;
        } else if (lo <= sar) {
          w *= std::min(o, sar) * slip / mark, in = false, outStopped = stopped = true, level = ep;
        } else {
          w *= c / mark, mark = c;
          if (c < ep) {
            exitNext = true;
          } else {
            hic = std::max(hic, c), ep = std::max(ep, hic * (T(1) - vd)), sar = std::max(sar, ep * (T(1) - T(0.5) * vd));
          }
        }
      } else if (stopped && rebuy >= T(0) && c >= level * (T(1) + rebuy)) {
        pending = true, stopped = false;
      }
      continue;
    }
    const auto enter = [&](T px) {
      if (outStopped && tripOn) w *= T(1) - T(sh.trip);
      in = true, pending = false, stopped = false, mark = px;
      if (psar && psarVol > T(0)) {
        sar = px * (T(1) - band), ep = px, af = afStep;
      } else if (psar) {
        sar = l5 > T(0) && l5 < px ? l5 : T(0.95) * px, ep = px, af = afStep;
      } else {
        sar = stopLoss > T(0) ? px * (T(1) - stopLoss) : T(0);
      }
    };
    if (!in && pending && o <= prev * (T(1) + cap)) enter(o);
    bool intraday = false;
    if (!in && stopped && rebuyStop > T(0)) {
      const T trig = (trail ? low : level) * (T(1) + rebuyStop);
      if (o >= trig) enter(o);
      else if (hi >= trig) enter(trig), intraday = true;
      else if (lo < low) low = lo;
    }
    if (in) {
      if (stops && sar > T(0) && lo <= sar) {
        const T px = (intraday ? sar : std::min(o, sar)) * slip;
        w *= px / mark, in = false, outStopped = stopped = true, turned = false, level = px, low = std::min(px, lo);
        if (psar) dsar = std::max(ep, px), dep = px, daf = afStep;
        if (rebuyStop > T(0) && !intraday && c >= px * (T(1) + rebuyStop)) {
          enter(px * (T(1) + rebuyStop));
          w *= c / mark, mark = c;
        }
      } else {
        w *= c / mark, mark = c;
        if (psar) {
          if (hi > ep) ep = hi, af = std::min(afMax, af + afStep);
          T next = sar + af * (ep - sar);
          next = std::min(next, l2);
          if (psarVol > T(0)) next = std::min(next, c * (T(1) - band));
          sar = std::max(sar, next);
        }
      }
    } else if (stopped && rebuy >= T(0) && rebuyStop <= T(0)) {
      if (psar) {
        if (c > dsar) {
          turned = true;
        } else {
          if (lo < dep) dep = lo, daf = std::min(afMax, daf + afStep);
          dsar = std::min(dsar, std::max(dsar + daf * (dep - dsar), h2));
          continue;
        }
      } else {
        turned = c >= level * (T(1) + rebuy);
      }
      if (turned) pending = true, stopped = false;
    }
  }
  retOut = w - T(1), stoppedOut = outStopped;
}

// The whole batch on the CPU in T arithmetic (the reference the shader is checked against).
template <class T>
inline void mcBatchRun(const McBatch& b, std::vector<float>& ret, std::vector<std::uint32_t>& stopped) {
  ret.assign(std::size_t(b.G) * b.S * b.P, 0.0f), stopped.assign(ret.size(), 0);
  for (std::size_t g = 0; g < b.G; ++g)
    for (std::size_t s = 0; s < b.S; ++s)
      for (std::size_t p = 0; p < b.P; ++p) {
        T r;
        bool st;
        mcBatchRef<T>(b, g, s, p, r, st);
        ret[b.out(g, s, p)] = float(r), stopped[b.out(g, s, p)] = st ? 1u : 0u;
      }
}

// The GPU's arithmetic on the CPU (X_MCGPU_EMULATE=1): every batch through mcBatchRef<float>, as the shader runs it,
// so the engine's choices under single precision can be checked natively (tools/check/mcgpu_check.sh).
inline bool mcOffloadEmulate(const McBatch& b, std::vector<float>& ret, std::vector<std::uint32_t>& stopped) {
  mcBatchRun<float>(b, ret, stopped);
  return true;
}

// ------------------------------------------------------------------------------------------------ the engine's entry points
// X_MCGPU_CHECK=1: every batch also run by the references and compared with the CPU's own results (montecarlo.hpp);
// the agreement printed at exit and checked (tools/check/mcgpu_check.sh). X_MCGPU_DUMP=DIR: the first batches written
// with their f32 reference results, for the shader's own test in a browser (web/tests/mcgpu.mjs).
struct McGpuCheck {
  bool on = std::getenv("X_MCGPU_CHECK") != nullptr;
  std::string dump = std::getenv("X_MCGPU_DUMP") ? std::getenv("X_MCGPU_DUMP") : "";
  std::size_t batches = 0, slots = 0, dumped = 0;
  double maxDiff64 = 0, maxDiff32 = 0, sumDiff32 = 0, sumDiff64 = 0;
  std::size_t stopDiff64 = 0, stopDiff32 = 0, off64 = 0, off32 = 0;  // off: slots more than 1e-6 (f64: its inputs are the GPU's f32) / 1e-4 (f32) apart
  ~McGpuCheck() {
    if (!on || !batches) return;
    std::printf("mcgpu: %zu batches, %zu slots; f64 batch vs CPU: mean |diff| %.3g, %zu slots apart over 1e-6 (max %.3g), stops differ %zu; f32 batch vs CPU: mean |diff| %.3g, "
                "%zu slots apart over 1e-4 (%.4f%%, max %.3g), stops differ %zu\n",
                batches, slots, sumDiff64 / double(slots), off64, maxDiff64, stopDiff64, sumDiff32 / double(slots), off32, 100.0 * double(off32) / double(slots), maxDiff32, stopDiff32);
  }
};
inline McGpuCheck& mcGpuCheck() {
  static McGpuCheck c;
  return c;
}

inline void mcGpuDump(const McBatch& b, const std::vector<float>& ret, const std::vector<std::uint32_t>& st) {
  McGpuCheck& c = mcGpuCheck();
  if (c.dump.empty() || c.dumped >= 8 || std::size_t(b.G) * b.S * b.P < 64) return;
  const std::string f = c.dump + "/batch" + std::to_string(c.dumped++) + ".bin";
  if (FILE* o = std::fopen(f.c_str(), "wb")) {  // header, days, table, shares, specs, then the f32 reference's results
    const std::uint32_t h[6] = {0x4D434731u, b.P, b.H, b.S, b.G, b.D};
    std::fwrite(h, 4, 6, o);
    std::fwrite(b.days.data(), 4, b.days.size(), o);
    std::fwrite(b.table.data(), 4, b.table.size(), o);
    std::fwrite(b.shares.data(), sizeof(McGpuShare), b.shares.size(), o);
    std::fwrite(b.specs.data(), sizeof(McGpuSpec), b.specs.size(), o);
    std::fwrite(ret.data(), 4, ret.size(), o);
    std::fwrite(st.data(), 4, st.size(), o);
    std::fclose(o);
  }
}

// Compares a batch's references with the CPU's results (cpu(g, s, p) -> {ret, stopped}).
template <class Cpu>
inline void mcGpuCompare(const McBatch& b, Cpu cpu) {
  McGpuCheck& c = mcGpuCheck();
  std::vector<float> r32;
  std::vector<std::uint32_t> s32;
  mcBatchRun<float>(b, r32, s32);
  std::vector<double> r64(r32.size());  // the f64 reference kept in doubles
  std::vector<std::uint32_t> s64(r32.size());
  for (std::size_t g = 0; g < b.G; ++g)
    for (std::size_t s = 0; s < b.S; ++s)
      for (std::size_t p = 0; p < b.P; ++p) {
        bool st;
        mcBatchRef<double>(b, g, s, p, r64[b.out(g, s, p)], st);
        s64[b.out(g, s, p)] = st ? 1u : 0u;
      }
  for (std::size_t g = 0; g < b.G; ++g)
    for (std::size_t s = 0; s < b.S; ++s)
      for (std::size_t p = 0; p < b.P; ++p) {
        const std::size_t o = b.out(g, s, p);
        const auto x = cpu(g, s, p);
        const double d64 = std::fabs(r64[o] - x.first), d32 = std::fabs(double(r32[o]) - x.first);
        c.maxDiff64 = std::max(c.maxDiff64, d64), c.maxDiff32 = std::max(c.maxDiff32, d32), c.sumDiff32 += d32;
        c.stopDiff64 += (s64[o] != 0) != x.second, c.stopDiff32 += (s32[o] != 0) != x.second;
        c.off64 += d64 > 1e-6, c.off32 += d32 > 1e-4, c.sumDiff64 += d64;
        ++c.slots;
      }
  ++c.batches;
  mcGpuDump(b, r32, s32);
}

// mcRun through the offload when a back end carries one (the same results' layout); the CPU's mcRun otherwise.
inline void mcRunAny(const Data& d, std::size_t t0, const std::vector<std::size_t>& list, const std::vector<double>& caps, const std::vector<McHeld>& held,
                     std::size_t k, const McSpec& s, std::uint64_t resetKey, const Backend& be, McReset& r, const std::vector<double>& mu = {},
                     const std::vector<double>& scale = {}) {
  McOffload f = mcOffload();
  McGpuCheck& chk = mcGpuCheck();
  if (!f && !chk.on) return mcRun(d, t0, list, caps, held, k, s, resetKey, be, r, mu, scale);
  const std::size_t P = s.paths, S = list.size();
  std::vector<double> adj(S), sc(S, 1.0);
  for (std::size_t j = 0; j < S; ++j) adj[j] = mcAdjust(d, t0, list[j], s) * (j < mu.size() ? std::exp(mu[j]) : 1.0), sc[j] = j < scale.size() ? scale[j] : 1.0;
  std::vector<std::vector<std::size_t>> paths(P);
  for (std::size_t p = 0; p < P; ++p) mcDays(t0, d.T, s, mcMix(s.seed ^ mcMix(resetKey * 1000003ULL + p)), false, paths[p]);
  McBatch b;
  const bool packed = mcPack(d, t0, list, caps, held, adj, sc, {s}, paths, b);
  if (chk.on || !packed) {
    mcRun(d, t0, list, caps, held, k, s, resetKey, be, r, mu, scale);
    if (chk.on && packed) mcGpuCompare(b, [&](std::size_t, std::size_t j, std::size_t p) { return std::make_pair(r.share[j * P + p], r.stop[j * P + p] != 0); });
    if (chk.on || !f) return;
  }
  std::vector<float> ret;
  std::vector<std::uint32_t> st;
  if (!f(b, ret, st)) return mcRun(d, t0, list, caps, held, k, s, resetKey, be, r, mu, scale);
  r.book.assign(P, 0.0), r.share.assign(S * P, 0.0), r.stop.assign(S * P, 0);
  for (std::size_t p = 0; p < P; ++p) {
    double sum = 0;
    for (std::size_t j = 0; j < S; ++j) {
      const std::size_t o = b.out(0, j, p);
      r.share[j * P + p] = ret[o], r.stop[j * P + p] = st[o] ? 1 : 0, sum += ret[o];
    }
    r.book[p] = sum / double(std::max<std::size_t>(1, k));
  }
}

// One share's stop settings on the same futures (the engine's stop learning): out[g * paths + p] the setting's return
// on path p, through the offload when a back end carries one; false: not run (the engine runs its own loop).
inline bool mcSettingsAny(const Data& d, std::size_t t, std::size_t i, double adj, double cap, double scale, const McSpec& mc, const std::vector<McSpec>& specs,
                          std::vector<double>& out, const std::vector<double>* cpu = nullptr, const std::vector<char>* cpuStop = nullptr) {
  McOffload f = mcOffload();
  McGpuCheck& chk = mcGpuCheck();
  if (!f && !(chk.on && cpu)) return false;
  const std::size_t P = mc.paths;
  std::vector<std::vector<std::size_t>> paths(P);
  for (std::size_t p = 0; p < P; ++p) mcDays(t, d.T, mc, mcMix(mc.seed ^ mcMix(t * 1000003ULL + p)), false, paths[p]);
  McBatch b;
  if (!mcPack(d, t, {i}, {cap}, {McHeld{}}, {adj}, {scale}, specs, paths, b)) return false;
  if (chk.on && cpu) {
    // check mode: the CPU's returns and stop flags (the caller's own loop) against the references
    mcGpuCompare(b, [&](std::size_t g, std::size_t, std::size_t p) { return std::make_pair((*cpu)[g * P + p], cpuStop && (*cpuStop)[g * P + p] != 0); });
    return false;
  }
  std::vector<float> ret;
  std::vector<std::uint32_t> st;
  if (!f(b, ret, st)) return false;
  out.assign(specs.size() * P, 0.0);
  for (std::size_t g = 0; g < specs.size(); ++g)
    for (std::size_t p = 0; p < P; ++p) out[g * P + p] = ret[b.out(g, 0, p)];
  return true;
}

}  // namespace strat
