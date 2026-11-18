// The differential neural overlay on the nested Monte Carlo (mcnn=1, apps/ofm_strategy).
//
// At every reset t0 the nested Monte Carlo builds each share's futures from blocks of its own real days (the
// resets overlaid on real OHLCV). This network learns, for every share at every reset, how those futures should be
// shaped: a daily drift mu and a volatility scale s applied to the real days drawn (each day's log move x becomes
// mu + s * x), from what is known at t0 only:
//   features  the share's 12-1, 6-1 and 1-month returns, its 63- and 252-session volatility and its distance from
//             the 52-week high (each as its rank among the day's tradable shares, -0.5..0.5), the market's 12-month
//             return and 63-session volatility; with mcpool, every strategy's score of the share (its rank that
//             day): the network learns which strategy to trust for which share, in which market
//   network   17 -> 16 tanh -> 2; mu = 0.002 tanh(o0) (at most 0.2% a day), s = exp(0.7 tanh(o1))
//   loss      the CRPS of the share's simulated futures against its real return over the horizon: the mean of
//             |X_p - y| less half the mean of |X_p - X_q|, X_p = mu h + s S_p (S_p: the path's sum of the drawn
//             real days' moves, the share's own trend taken out and the market's put in)
//   training  differential: the exact gradients through the paths, dL/dmu = h mean sign(X_p - y) and
//             dL/ds = mean sign(X_p - y) S_p - mean|S_p - S_q| / 2, back through the network (Adam); walk-forward
//             only, at each reset on the samples whose horizon had ended by then (never a later day)
// Deterministic: a fixed start, the samples in a fixed order, one thread; every back end gives the same.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "strategy/montecarlo.hpp"
#include "strategy/strategy.hpp"

namespace strat {

constexpr std::size_t kNS = 8;            // the strategies' scores a sample can carry (mcpool: every strategy's rank)
constexpr std::size_t kNF = 9 + kNS, kNH = 16, kNP = 32;  // features, hidden units, the paths a training sample keeps

struct NnShape {
  double mu = 0, scale = 1;
};

struct NnSample {
  float f[kNF];
  float S[kNP];
  float y, D;  // the real log return over the horizon; half the mean |S_p - S_q|
  std::size_t ready;  // the day its horizon ends (usable only by a reset after it)
};

class DiffNet {
 public:
  explicit DiffNet(std::uint64_t seed = 7) {
    std::uint64_t h = seed;
    const auto u = [&] { h = mcMix(h); return (double(h >> 11) / 9007199254740992.0 - 0.5); };
    for (auto& r : W1) for (double& w : r) w = u() * 2 * std::sqrt(1.0 / kNF);
    for (auto& r : W2) for (double& w : r) w = u() * 0.02;  // the outputs start near 0: mu 0, s 1 (the plain Monte Carlo)
    zero(m), zero(v);
  }

  NnShape shape(const double* f) const {
    double hid[kNH], o[2];
    forward(f, hid, o);
    return {kMu * std::tanh(o[0]), std::exp(kS * std::tanh(o[1]))};
  }

  // One pass over the samples usable at day t0 (ready <= t0), the newest `limit` of them, in minibatches.
  // b: the back end the samples' gradients are computed on (each its own, then summed in their order: the same
  // numbers on every back end); null: one thread.
  void train(const std::vector<NnSample>& all, std::size_t t0, std::size_t horizon, std::size_t limit, int epochs, const Backend* b = nullptr) {
    std::vector<std::size_t> idx;
    for (std::size_t j = 0; j < all.size(); ++j)
      if (all[j].ready <= t0) idx.push_back(j);  // its horizon ended by t0's close: known then
    if (idx.size() < 500) return;
    if (idx.size() > limit) idx.erase(idx.begin(), idx.end() - long(limit));
    for (int e = 0; e < epochs; ++e) {
      std::uint64_t h = mcMix(0x51ED ^ (t0 * 131 + std::uint64_t(e)));
      for (std::size_t j = idx.size(); j > 1; --j) {  // a fixed shuffle (Fisher-Yates on the hash)
        h = mcMix(h);
        std::swap(idx[j - 1], idx[h % j]);
      }
      for (std::size_t a = 0; a < idx.size(); a += kBatch) {
        Grad g;
        zero(g);
        const std::size_t end = std::min(idx.size(), a + kBatch);
        if (b) {  // each sample's gradient on its own (in parallel), then summed in the samples' order
          per_.resize(kBatch);
          b->parallelFor(end - a, [&](std::size_t x, std::size_t y) {
            for (std::size_t q = x; q < y; ++q) zero(per_[q]), accumulate(all[idx[a + q]], horizon, per_[q]);
          });
          // then each parameter's sum over the samples, in their order (in parallel over the parameters)
          constexpr std::size_t nP = sizeof(Grad) / sizeof(double);
          static_assert(sizeof(Grad) == nP * sizeof(double), "a gradient is its parameters' doubles, nothing else");
          double* gp = reinterpret_cast<double*>(&g);
          const std::size_t nS = end - a;
          b->parallelFor(nP, [&](std::size_t x, std::size_t y) {
            for (std::size_t k = x; k < y; ++k) {
              double sum = gp[k];
              for (std::size_t q = 0; q < nS; ++q) sum += reinterpret_cast<const double*>(&per_[q])[k];
              gp[k] = sum;
            }
          });
        } else {
          for (std::size_t q = a; q < end; ++q) accumulate(all[idx[q]], horizon, g);
        }
        step(g, double(end - a));
      }
    }
    ++trained;
  }

  // The mean CRPS of the samples usable at t0 that are newer than `from` (a held-out check of the network: the last
  // ones it has not trained on yet, against the plain Monte Carlo's).
  std::pair<double, double> score(const std::vector<NnSample>& all, std::size_t from, std::size_t t0, std::size_t horizon) const {
    double a = 0, b = 0, n = 0;
    for (const auto& s : all) {
      if (s.ready > t0 || s.ready < from) continue;
      double f[kNF];
      for (std::size_t k = 0; k < kNF; ++k) f[k] = s.f[k];
      const NnShape sh = shape(f);
      a += crps(s, sh.mu, sh.scale, horizon), b += crps(s, 0, 1, horizon), n += 1;
    }
    return {n ? a / n : 0, n ? b / n : 0};
  }

  std::size_t trained = 0;

 private:
  static constexpr double kMu = 0.002, kS = 0.7, kLr = 0.003, kB1 = 0.9, kB2 = 0.999, kDecay = 1e-4;
  static constexpr std::size_t kBatch = 128;
  double W1[kNH][kNF], b1[kNH] = {}, W2[2][kNH], b2[2] = {};
  struct Grad {
    double W1[kNH][kNF], b1[kNH], W2[2][kNH], b2[2];
  };
  Grad m, v;
  std::vector<Grad> per_;  // a batch's per-sample gradients (train on a back end)
  std::size_t t = 0;



  template <class G>
  static void zero(G& g) {
    for (auto& r : g.W1) for (double& w : r) w = 0;
    for (double& w : g.b1) w = 0;
    for (auto& r : g.W2) for (double& w : r) w = 0;
    for (double& w : g.b2) w = 0;
  }

  void forward(const double* f, double* hid, double* o) const {
    for (std::size_t j = 0; j < kNH; ++j) {
      double z = b1[j];
      for (std::size_t k = 0; k < kNF; ++k) z += W1[j][k] * f[k];
      hid[j] = std::tanh(z);
    }
    for (std::size_t r = 0; r < 2; ++r) {
      double z = b2[r];
      for (std::size_t j = 0; j < kNH; ++j) z += W2[r][j] * hid[j];
      o[r] = z;
    }
  }

  static double crps(const NnSample& s, double mu, double sc, std::size_t h) {
    double a = 0;
    for (std::size_t p = 0; p < kNP; ++p) a += std::fabs(mu * double(h) + sc * s.S[p] - s.y);
    return a / kNP - sc * s.D;
  }

  // The sample's gradient through the paths (the differential), back through the network.
  void accumulate(const NnSample& s, std::size_t h, Grad& g) const {
    double f[kNF], hid[kNH], o[2];
    for (std::size_t k = 0; k < kNF; ++k) f[k] = s.f[k];
    forward(f, hid, o);
    const double t0 = std::tanh(o[0]), t1 = std::tanh(o[1]);
    const double mu = kMu * t0, sc = std::exp(kS * t1), H = double(h);
    double dmu = 0, ds = 0;
    for (std::size_t p = 0; p < kNP; ++p) {
      const double x = mu * H + sc * s.S[p] - s.y, sg = x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0;
      dmu += sg * H, ds += sg * s.S[p];
    }
    dmu /= kNP, ds = ds / kNP - s.D;
    const double d0 = dmu * kMu * (1 - t0 * t0), d1 = ds * sc * kS * (1 - t1 * t1);
    const double dout[2] = {d0, d1};
    for (std::size_t r = 0; r < 2; ++r) {
      g.b2[r] += dout[r];
      for (std::size_t j = 0; j < kNH; ++j) g.W2[r][j] += dout[r] * hid[j];
    }
    for (std::size_t j = 0; j < kNH; ++j) {
      const double dh = (dout[0] * W2[0][j] + dout[1] * W2[1][j]) * (1 - hid[j] * hid[j]);
      g.b1[j] += dh;
      for (std::size_t k = 0; k < kNF; ++k) g.W1[j][k] += dh * f[k];
    }
  }

  void step(const Grad& g, double n) {
    ++t;
    const double c1 = 1 - std::pow(kB1, double(t)), c2 = 1 - std::pow(kB2, double(t));
    const auto upd = [&](double& w, double gr, double& mm, double& vv, bool decay) {
      gr = gr / n + (decay ? kDecay * w : 0.0);
      mm = kB1 * mm + (1 - kB1) * gr, vv = kB2 * vv + (1 - kB2) * gr * gr;
      w -= kLr * (mm / c1) / (std::sqrt(vv / c2) + 1e-8);
    };
    for (std::size_t j = 0; j < kNH; ++j) {
      for (std::size_t k = 0; k < kNF; ++k) upd(W1[j][k], g.W1[j][k], m.W1[j][k], v.W1[j][k], true);
      upd(b1[j], g.b1[j], m.b1[j], v.b1[j], false);
    }
    for (std::size_t r = 0; r < 2; ++r) {
      for (std::size_t j = 0; j < kNH; ++j) upd(W2[r][j], g.W2[r][j], m.W2[r][j], v.W2[r][j], true);
      upd(b2[r], g.b2[r], m.b2[r], v.b2[r], false);
    }
  }
};

// The features of every tradable share at day t (known after its close), ranked across the day's shares; false
// for a share without a full year. out[i * kNF + k].
inline void nnFeatures(const Data& d, std::size_t t, const char* in, std::vector<double>& out, std::vector<char>& ok, const Backend* be = nullptr) {
  const std::size_t N = d.N;
  out.assign(N * kNF, 0.0), ok.assign(N, 0);
  if (t < 260) return;
  const auto lr = [&](std::size_t a, std::size_t b, std::size_t i) {
    const double x = d.Cf[a * N + i], y = d.Cf[b * N + i];
    return x > 0 && y > 0 ? std::log(y / x) : kNaN;
  };
  std::vector<std::vector<double>> raw(6, std::vector<double>(N, kNaN));
  // One fused pass over each share's last year (in parallel on the back end): the year's high close and the 63- and
  // 252-session volatilities together.
  const auto share = [&](std::size_t a, std::size_t e) {
    for (std::size_t i = a; i < e; ++i) {
      if (!in[i]) continue;
      double hi = 0, s63 = 0, q63 = 0, c63 = 0, s252 = 0, q252 = 0, c252 = 0;
      for (std::size_t q = t - 251; q <= t; ++q) {
        const double c = d.Cf[q * N + i], r = d.ret[q * N + i];
        hi = std::max(hi, c > 0 ? c : 0.0);
        if (std::isfinite(r)) {
          s252 += r, q252 += r * r, c252 += 1;
          if (q + 63 > t) s63 += r, q63 += r * r, c63 += 1;
        }
      }
      const auto sd = [](double s1, double s2, double c, double n) { return c > std::floor(n / 2) ? std::sqrt(std::max(0.0, s2 / c - (s1 / c) * (s1 / c))) : kNaN; };
      raw[0][i] = lr(t - 252, t - 21, i), raw[1][i] = lr(t - 126, t - 21, i), raw[2][i] = lr(t - 21, t, i);
      raw[3][i] = sd(s63, q63, c63, 63), raw[4][i] = sd(s252, q252, c252, 252), raw[5][i] = hi > 0 && d.Cf[t * N + i] > 0 ? std::log(d.Cf[t * N + i] / hi) : kNaN;
      bool all = true;
      for (std::size_t k = 0; k < 6; ++k) all = all && std::isfinite(raw[k][i]);
      ok[i] = all;
    }
  };
  if (be) be->parallelFor(N, share);
  else share(0, N);
  for (std::size_t k = 0; k < 6; ++k) {  // ranks -0.5..0.5 among the shares with every feature
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t i = 0; i < N; ++i)
      if (ok[i]) v.emplace_back(raw[k][i], i);
    std::sort(v.begin(), v.end());
    for (std::size_t r = 0; r < v.size(); ++r) out[v[r].second * kNF + k] = v.size() > 1 ? double(r) / double(v.size() - 1) - 0.5 : 0.0;
  }
  double m12 = 0, s = 0, s2 = 0;
  for (std::size_t q = t - 251; q <= t; ++q) m12 += d.mkt[q];
  for (std::size_t q = t - 62; q <= t; ++q) s += d.mkt[q], s2 += d.mkt[q] * d.mkt[q];
  const double mv = std::sqrt(std::max(0.0, s2 / 63 - (s / 63) * (s / 63)));
  for (std::size_t i = 0; i < N; ++i)
    if (ok[i]) out[i * kNF + 6] = std::clamp(2 * m12, -1.5, 1.5), out[i * kNF + 7] = std::clamp(50 * mv - 0.5, -1.5, 1.5), out[i * kNF + 8] = 1.0;
}

// A training sample for share i at reset t: its features, kNP futures' sums of drawn real days (the share's own
// trend out, the market's in: as the Monte Carlo draws them), and its real log return over the horizon (known only
// at t + h: the sample is ready then).
inline bool nnSample(const Data& d, std::size_t t, std::size_t i, const double* f, const McSpec& s, NnSample& out) {
  const std::size_t N = d.N, h = s.horizon;
  if (t + h >= d.T) return false;
  const double a = d.Cf[t * N + i], b = d.Cf[(t + h) * N + i];
  if (!(a > 0 && b > 0)) return false;
  const std::size_t hi = t > s.skip + 1 ? t - s.skip : 1, lo = hi > s.window ? hi - s.window : 1;
  double si = 0, ni = 0, sm = 0;
  for (std::size_t q = lo; q <= hi; ++q) {
    const double r = d.ret[q * N + i];
    if (std::isfinite(r)) si += r, ni += 1;
    sm += d.mkt[q];
  }
  if (ni < 60) return false;
  const double shift = sm / double(hi - lo + 1) - si / ni;
  std::vector<std::size_t> days;
  for (std::size_t p = 0; p < kNP; ++p) {
    mcDays(t, d.T, s, mcMix(0xD1FF ^ mcMix(t * 1000003ULL + i * 7919ULL + p)), false, days);
    double S = 0;
    for (std::size_t q : days) {
      const double r = d.ret[q * N + i];
      S += (std::isfinite(r) ? r : 0.0) + shift;
    }
    out.S[p] = float(S);
  }
  double D = 0;
  for (std::size_t p = 0; p < kNP; ++p)
    for (std::size_t q = 0; q < kNP; ++q) D += std::fabs(double(out.S[p]) - out.S[q]);
  out.D = float(D / (2.0 * kNP * kNP));
  for (std::size_t k = 0; k < kNF; ++k) out.f[k] = float(f[k]);
  out.y = float(std::log(b / a)), out.ready = t + h;
  return true;
}

}  // namespace strat
