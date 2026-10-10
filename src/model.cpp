#include "ofm/model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "ofm/kernels.hpp"

namespace ofm {

const char* familyName(std::size_t f) {
  static const char* names[] = {"return", "low volatility", "near high", "trend quality", "small size",
                                "RSI", "Bollinger z", "stochastic %K", "money flow", "intraday range"};
  return f < kFamilies ? names[f] : "?";
}

std::string Plan::signalName(std::size_t k) const {
  return std::string(familyName(k / H)) + " " + std::to_string(horizons[k % H]) + "d";
}

std::size_t Plan::chunkLength(std::size_t c) const { return std::min(chunkDays, T - chunkStart(c)); }

std::vector<std::uint32_t> Plan::header(std::size_t c) const {
  std::vector<std::uint32_t> h(32, 0);
  h[0] = static_cast<std::uint32_t>(T), h[1] = static_cast<std::uint32_t>(N), h[2] = static_cast<std::uint32_t>(K),
  h[3] = static_cast<std::uint32_t>(H);
  h[4] = static_cast<std::uint32_t>(chunkStart(c)), h[5] = static_cast<std::uint32_t>(chunkLength(c));
  h[6] = static_cast<std::uint32_t>(pairs()), h[7] = static_cast<std::uint32_t>(stride);
  for (std::size_t j = 0; j < H; ++j) h[16 + j] = horizons[j];
  return h;
}

Plan compilePlan(const Market& m, std::size_t budget) {
  Plan p;
  p.T = m.T(), p.N = m.N();
  std::uint32_t hmax = 2;
  while (hmax * 2 <= p.T / 8) hmax *= 2;
  if (p.T / 8 < 2) throw std::invalid_argument("too little history: need at least 16 days");
  // As many horizons as the universe supports: the signals must stay fewer than the stocks for
  // the orthonormalisation to exist (the shortest horizons are kept; longer ones are dropped).
  const std::size_t maxH = p.N > kFamilies + 2 ? (p.N - 2) / kFamilies : 0;
  if (maxH == 0) throw std::invalid_argument("too few stocks: need more stocks than signal families");
  for (std::uint32_t h = 2; h <= hmax && p.horizons.size() < std::min(kMaxHorizons, maxH); h *= 2) p.horizons.push_back(h);
  p.H = p.horizons.size();
  p.K = kFamilies * p.H;
  p.first = p.horizons.back();
  p.stride = p.pairs() + p.K + 2;
  const std::size_t perDay = p.N * p.K * 4;
  p.chunkDays = std::max<std::size_t>(1, std::min(p.T - p.first, budget / std::max<std::size_t>(perDay, 1)));

  const std::size_t T = p.T, N = p.N, TN = T * N;
  p.tables.assign(kTables * TN, 0.0f);
  float *lc = &p.tables[0], *s1 = &p.tables[TN], *s2 = &p.tables[2 * TN], *lv = &p.tables[3 * TN], *el = &p.tables[4 * TN],
        *tg = &p.tables[5 * TN], *tm = &p.tables[6 * TN], *lh = &p.tables[7 * TN], *ll = &p.tables[8 * TN], *tv = &p.tables[9 * TN];
  // Traded value is centred on one constant for all stocks and days: it keeps the prefix sums
  // small in single precision and shifts every stock alike, so no rank changes.
  // The constant is the average over the warm-up days only, so later data never changes it.
  double centre = 0, cnt = 0;
  for (std::size_t k = 0; k < p.first * N; ++k)
    if (std::isfinite(m.close.v[k])) centre += std::log1p(m.close.v[k] * m.volume.v[k]), cnt += 1;
  centre = cnt > 0 ? centre / cnt : 0.0;
  for (std::size_t i = 0; i < N; ++i) {
    std::size_t firstValid = T;
    double last = kNaN, lastH = kNaN, lastL = kNaN, a1 = 0, a2 = 0, av = 0;
    for (std::size_t t = 0; t < T; ++t) {
      const double c = m.close(t, i);
      if (std::isfinite(c) && firstValid == T) firstValid = t;
      double r = 0;
      if (std::isfinite(c)) {
        const double l = std::log(c);
        if (std::isfinite(last)) r = l - last;
        last = l;
      }
      a1 += r, a2 += r * r;
      av += (std::isfinite(c) ? std::log1p(c * m.volume(t, i)) : 0.0) - (firstValid <= t ? centre : 0.0);
      if (std::isfinite(c)) {
        const double hi = m.high(t, i), lo = m.low(t, i);
        lastH = std::log(hi > 0 ? std::max(hi, c) : c), lastL = std::log(lo > 0 ? std::min(lo, c) : c);
      }
      lc[t * N + i] = std::isfinite(last) ? static_cast<float>(last) : 0.0f;
      lh[t * N + i] = std::isfinite(lastH) ? static_cast<float>(lastH) : 0.0f;
      ll[t * N + i] = std::isfinite(lastL) ? static_cast<float>(lastL) : 0.0f;
      tv[t * N + i] = std::isfinite(c) ? static_cast<float>(c * m.volume(t, i)) : 0.0f;
      s1[t * N + i] = static_cast<float>(a1), s2[t * N + i] = static_cast<float>(a2), lv[t * N + i] = static_cast<float>(av);
      const bool e = std::isfinite(c) && firstValid != T && firstValid + p.first <= t;
      el[t * N + i] = e ? 1.0f : 0.0f;
    }
    for (std::size_t t = 0; t + 2 < T; ++t) {
      const double o1 = m.open(t + 1, i), o2 = m.open(t + 2, i);
      if (el[t * N + i] > 0 && o1 > 0 && o2 > 0) tg[t * N + i] = static_cast<float>(o2 / o1 - 1.0), tm[t * N + i] = 1.0f;
    }
  }
  return p;
}

namespace {

// Eigen decomposition of a symmetric matrix: Householder reduction to tridiagonal form, then the
// implicit QL algorithm (EISPACK tred2 / tql2). a (n x n, row-major) is overwritten; on return
// the columns of V are the eigenvectors and w the eigenvalues.
void eigenSymmetric(std::vector<double>& a, std::size_t n, std::vector<double>& w, std::vector<double>& V) {
  std::vector<double> d(n), e(n);
  V = a;
  auto v = [&](std::size_t i, std::size_t j) -> double& { return V[i * n + j]; };
  for (std::size_t j = 0; j < n; ++j) d[j] = v(n - 1, j);
  for (std::size_t i = n - 1; i > 0; --i) {
    double scale = 0, h = 0;
    for (std::size_t k = 0; k < i; ++k) scale += std::fabs(d[k]);
    if (scale == 0) {
      e[i] = d[i - 1];
      for (std::size_t j = 0; j < i; ++j) d[j] = v(i - 1, j), v(i, j) = 0, v(j, i) = 0;
    } else {
      for (std::size_t k = 0; k < i; ++k) d[k] /= scale, h += d[k] * d[k];
      double f = d[i - 1], g = std::sqrt(h);
      if (f > 0) g = -g;
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (std::size_t j = 0; j < i; ++j) e[j] = 0;
      for (std::size_t j = 0; j < i; ++j) {
        f = d[j];
        v(j, i) = f;
        g = e[j] + v(j, j) * f;
        for (std::size_t k = j + 1; k <= i - 1; ++k) g += v(k, j) * d[k], e[k] += v(k, j) * f;
        e[j] = g;
      }
      f = 0;
      for (std::size_t j = 0; j < i; ++j) e[j] /= h, f += e[j] * d[j];
      const double hh = f / (h + h);
      for (std::size_t j = 0; j < i; ++j) e[j] -= hh * d[j];
      for (std::size_t j = 0; j < i; ++j) {
        f = d[j], g = e[j];
        for (std::size_t k = j; k <= i - 1; ++k) v(k, j) -= (f * e[k] + g * d[k]);
        d[j] = v(i - 1, j);
        v(i, j) = 0;
      }
    }
    d[i] = h;
  }
  for (std::size_t i = 0; i + 1 < n; ++i) {
    v(n - 1, i) = v(i, i);
    v(i, i) = 1;
    const double h = d[i + 1];
    if (h != 0) {
      for (std::size_t k = 0; k <= i; ++k) d[k] = v(k, i + 1) / h;
      for (std::size_t j = 0; j <= i; ++j) {
        double g = 0;
        for (std::size_t k = 0; k <= i; ++k) g += v(k, i + 1) * v(k, j);
        for (std::size_t k = 0; k <= i; ++k) v(k, j) -= g * d[k];
      }
    }
    for (std::size_t k = 0; k <= i; ++k) v(k, i + 1) = 0;
  }
  for (std::size_t j = 0; j < n; ++j) d[j] = v(n - 1, j), v(n - 1, j) = 0;
  v(n - 1, n - 1) = 1;
  e[0] = 0;
  // Implicit QL.
  for (std::size_t i = 1; i < n; ++i) e[i - 1] = e[i];
  e[n - 1] = 0;
  double f = 0, tst1 = 0;
  const double eps = std::ldexp(1.0, -52);
  for (std::size_t l = 0; l < n; ++l) {
    tst1 = std::max(tst1, std::fabs(d[l]) + std::fabs(e[l]));
    std::size_t m = l;
    while (m < n) {
      if (std::fabs(e[m]) <= eps * tst1) break;
      ++m;
    }
    if (m > l) {
      for (int iter = 0; iter < 60; ++iter) {
        double g = d[l];
        double p = (d[l + 1] - g) / (2 * e[l]);
        double r = std::hypot(p, 1.0);
        if (p < 0) r = -r;
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const double dl1 = d[l + 1];
        double h = g - d[l];
        for (std::size_t i = l + 2; i < n; ++i) d[i] -= h;
        f += h;
        p = d[m];
        double c = 1, c2 = c, c3 = c, el1 = e[l + 1], s = 0, s2 = 0;
        for (std::size_t ii = m; ii-- > l;) {
          c3 = c2, c2 = c, s2 = s;
          g = c * e[ii], h = c * p;
          r = std::hypot(p, e[ii]);
          e[ii + 1] = s * r;
          s = e[ii] / r, c = p / r;
          p = c * d[ii] - s * g;
          d[ii + 1] = h + s * (c * g + s * d[ii]);
          for (std::size_t k = 0; k < n; ++k) {
            h = v(k, ii + 1);
            v(k, ii + 1) = s * v(k, ii) + c * h;
            v(k, ii) = c * v(k, ii) - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * e[l] / dl1;
        e[l] = s * p;
        d[l] = c * p;
        if (!(std::fabs(e[l]) > eps * tst1)) break;
      }
    }
    d[l] += f;
    e[l] = 0;
  }
  w = d;
}

}  // namespace

void eigenSym(std::vector<double>& a, std::size_t n, std::vector<double>& w, std::vector<double>& V) { eigenSymmetric(a, n, w, V); }

namespace {

// (G)^(-1/2) of a symmetric positive semi-definite matrix; directions with no variance are dropped.
std::vector<double> invSqrt(std::vector<double> G, std::size_t n) {
  std::vector<double> w, V;
  eigenSymmetric(G, n, w, V);
  const double top = *std::max_element(w.begin(), w.end());
  std::vector<double> out(n * n, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    if (!(w[k] > top * 1e-12)) continue;
    const double s = 1.0 / std::sqrt(w[k]);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) out[i * n + j] += V[i * n + k] * s * V[j * n + k];
  }
  return out;
}

}  // namespace

Forecaster::Forecaster(const Plan& plan) : plan_(plan), s1_(plan.K, 0.0), s2_(plan.K, 0.0) {
  mean.assign(plan.K, 0.0), tstat.assign(plan.K, 0.0), premium.assign(plan.K, 0.0);
  familyReturns.assign(plan.T, {});
}

std::vector<double> Forecaster::absorb(std::size_t t, const double* g) {
  const std::size_t K = plan_.K, P = plan_.pairs();
  // Records two or more days old are known at this close.
  for (std::size_t j = 0; j < pending_.size();) {
    if (pendingDay_[j] + 2 <= t) {
      for (std::size_t k = 0; k < K; ++k) s1_[k] += pending_[j][k], s2_[k] += pending_[j][k] * pending_[j][k];
      ++n_;
      pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(j));
      pendingDay_.erase(pendingDay_.begin() + static_cast<std::ptrdiff_t>(j));
    } else {
      ++j;
    }
  }
  const double eligible = g[P + K], targets = g[P + K + 1];
  if (targets > 0) {
    // Each family's return on day t (its z-weighted next-day return, averaged over horizons).
    auto& fr = familyReturns[t];
    fr.assign(kFamilies, 0.0);
    for (std::size_t k = 0; k < K; ++k) fr[k / plan_.H] += g[P + k] / (targets * static_cast<double>(plan_.H));
  }
  if (!(eligible > static_cast<double>(K) + 1)) return {};
  std::vector<double> G(K * K);
  for (std::size_t k = 0, p = 0; k < K; ++k)
    for (std::size_t l = k; l < K; ++l, ++p) G[k * K + l] = G[l * K + k] = g[p];
  const std::vector<double> M = invSqrt(G, K);
  if (targets > static_cast<double>(K)) {
    std::vector<double> f(K, 0.0);
    for (std::size_t k = 0; k < K; ++k)
      for (std::size_t l = 0; l < K; ++l) f[k] += M[k * K + l] * g[P + l];
    pending_.push_back(std::move(f));
    pendingDay_.push_back(t);
  }
  std::vector<double> v(K, 0.0);
  const double n = static_cast<double>(n_);
  for (std::size_t k = 0; k < K; ++k) {
    mean[k] = tstat[k] = premium[k] = 0.0;
    if (n_ < 3) continue;
    const double m = s1_[k] / n, var = std::max(0.0, (s2_[k] - n * m * m) / (n - 1));
    mean[k] = m;
    if (var <= 0) continue;
    tstat[k] = m / std::sqrt(var / n);
    premium[k] = m * std::max(0.0, 1.0 - 1.0 / (tstat[k] * tstat[k]));
  }
  for (std::size_t k = 0; k < K; ++k)
    for (std::size_t l = 0; l < K; ++l) v[k] += M[k * K + l] * premium[l];
  return v;
}

Forecast runModel(const Market& m, const std::string& engine) {
  const auto t0 = std::chrono::steady_clock::now();
  const Plan p = compilePlan(m);
  Forecaster fc(p);
  Forecast out;
  out.engine = engine;
  out.E = Panel(p.T, p.N);
  double kernelMs = 0;
  // Both CPU engines run the kernels' translation (kernels.hpp): float emulates the GPU, double is the reference.
  auto run = [&](auto zero) {
    using F = decltype(zero);
    const std::vector<F> tables(p.tables.begin(), p.tables.end());
    for (std::size_t c = 0; c < p.numChunks(); ++c) {
      const std::size_t cd = p.chunkLength(c), c0 = p.chunkStart(c);
      const auto k0 = std::chrono::steady_clock::now();
      std::vector<F> z, g, e;
      kernels::zscore<F>(p, c, tables.data(), z);
      kernels::gram<F>(p, c, tables.data(), z, g);
      kernelMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - k0).count();
      const std::vector<double> gd(g.begin(), g.end());
      std::vector<std::vector<double>> v(cd);
      std::vector<F> vf(cd * p.K, F(0));
      for (std::size_t td = 0; td < cd; ++td) {
        v[td] = fc.absorb(c0 + td, &gd[td * p.stride]);
        for (std::size_t k = 0; k < v[td].size(); ++k) vf[td * p.K + k] = static_cast<F>(v[td][k]);
      }
      const auto k1 = std::chrono::steady_clock::now();
      kernels::expect<F>(p, c, z, vf, e);
      kernelMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - k1).count();
      for (std::size_t td = 0; td < cd; ++td) {
        if (v[td].empty()) continue;
        for (std::size_t a = 0; a < p.N; ++a)
          if (p.tables[4 * p.T * p.N + (c0 + td) * p.N + a] > 0.5f) out.E(c0 + td, a) = static_cast<double>(e[td * p.N + a]);
      }
    }
  };
  if (engine == "reference") run(0.0);
  else if (engine == "emulated") run(0.0f);
  else throw std::invalid_argument("engine: reference or emulated");
  out.mean = fc.mean, out.tstat = fc.tstat, out.premium = fc.premium, out.records = fc.records();
  out.familyReturns = fc.familyReturns;
  out.horizons = p.horizons;
  out.kernelMs = engine == "reference" ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() : kernelMs;
  return out;
}

}  // namespace ofm
