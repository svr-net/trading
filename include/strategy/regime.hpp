// The market's hidden states for the nested Monte Carlo (mcregime=K, apps/ofm_strategy).
//
// Each cycle (every reset t0) runs one round of reverse engineering, then one of forward learning:
//   reverse  a hidden Markov model of the market's daily move (the day's mean return of the tradable shares) with K
//            states, fitted by Baum-Welch to the window up to t0 only. Each state has its own mean and volatility
//            (ordered calm first, turbulent last), and there is a matrix of the chances of moving between them. Two
//            starts, the better fit kept: the last cycle's fit (each cycle refines the one before) and afresh (so a
//            state the last fit lost can come back). From the fit: each past day's most likely state, the chances of
//            each state today (filtered: what t0 knew) and over the coming horizon (carried forward by the matrix).
//   forward  the futures are drawn in states: a path of states from today's chances through the matrix, each
//            block of real days drawn only from past days in the block's state, so a calm market's futures come from
//            calm days and a falling market's from falling ones. On these futures the book learns as before (the
//            candidates, each share's stop and buy-back). Each share's stop is sized by its volatility in the states
//            expected over the horizon (its past days weighted by their chances of each state), not by its last 60
//            sessions alone. The real future is then scored against the simulated ones (the check), and the next
//            cycle refits on the window that now includes it.
// Deterministic: fixed starts, fixed iterations, the futures from the counter-based hash.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include "strategy/backend.hpp"
#include "strategy/strategy.hpp"

namespace strat {

struct RegimeFit {
  std::size_t K = 0, lo = 0, hi = 0;             // the days fitted: [lo, hi] (hi = t0)
  std::vector<double> A, mean, var;              // the matrix A[from * K + to], each state's daily mean and variance
  std::vector<double> now, ahead;                // the chances of each state at t0, and their mean over the horizon after
  std::vector<int> label;                        // each fitted day's most likely state (day q at q - lo)
  std::vector<std::vector<std::size_t>> starts;  // each state's days a block may start at (the drawable ones)
  std::vector<double> gamma;                     // each fitted day's chances of each state (smoothed), [q * K + k]
  std::vector<double> sigma;                     // each share's daily volatility in the states expected (NaN: too few days; regimeSigma)
  double loglik = 0, bic = 0;                    // bic: -2 loglik + the parameters' count times log(days)
};

using RegimeFits = std::map<std::size_t, RegimeFit>;

// Fits the K-state model to the market's days [lo, t0] (lo: the drawable window's start, as mcDays draws from:
// [t0 - skip - window, t0 - skip]); warm: the last cycle's fit (null: afresh only). horizon, block: the futures'
// length and block (the chances over the horizon, the block starts).
inline void fitRegime(const Data& d, std::size_t t0, std::size_t window, std::size_t skip, std::size_t horizon, std::size_t block, std::size_t K,
                      const RegimeFit* warm, RegimeFit& f) {
  const std::size_t hiD = t0 > skip + 1 ? t0 - skip : 1, loD = hiD > window ? hiD - window : 1;
  f = RegimeFit{};
  f.K = K, f.lo = loD, f.hi = t0;
  const std::size_t n = t0 + 1 - loD;
  std::vector<double> x(n);
  for (std::size_t q = 0; q < n; ++q) x[q] = std::isfinite(d.mkt[loD + q]) ? d.mkt[loD + q] : 0.0;
  double floor = 0;  // the variances' floor: a hundredth of the moves' own (no state from a handful of quiet days)
  {
    double s1 = 0, s2 = 0;
    for (double y : x) s1 += y, s2 += y * y;
    floor = std::max(1e-12, 0.01 * (s2 / double(n) - (s1 / double(n)) * (s1 / double(n))));
  }
  std::vector<double> A(K * K), mu(K, 0.0), v(K), pi(K, 1.0 / double(K));
  const auto init = [&](bool fromWarm) {
    std::fill(mu.begin(), mu.end(), 0.0), std::fill(pi.begin(), pi.end(), 1.0 / double(K));
    if (fromWarm) {
      A = warm->A, mu = warm->mean, v = warm->var;
      return;
    }
    std::vector<double> x2(n);
    for (std::size_t q = 0; q < n; ++q) x2[q] = x[q] * x[q];
    std::sort(x2.begin(), x2.end());
    for (std::size_t k = 0; k < K; ++k) v[k] = std::max(floor, x2[std::min(n - 1, std::size_t((double(k) + 0.5) / double(K) * double(n)))]);
    for (std::size_t a = 0; a < K; ++a)
      for (std::size_t c = 0; c < K; ++c) A[a * K + c] = a == c ? 0.9 : 0.1 / double(K > 1 ? K - 1 : 1);
  };
  std::vector<double> al(n * K), be(n * K), em(n * K), sc(n), ga(n * K);
  const auto forwardBackward = [&] {  // scaled; the log-likelihood
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t k = 0; k < K; ++k) {
        const double z = x[q] - mu[k];
        em[q * K + k] = std::exp(-0.5 * z * z / v[k]) / std::sqrt(2 * M_PI * v[k]) + 1e-300;
      }
    double ll = 0;
    for (std::size_t q = 0; q < n; ++q) {
      double s = 0;
      for (std::size_t k = 0; k < K; ++k) {
        double p = 0;
        if (q == 0) p = pi[k];
        else
          for (std::size_t a = 0; a < K; ++a) p += al[(q - 1) * K + a] * A[a * K + k];
        al[q * K + k] = p * em[q * K + k], s += al[q * K + k];
      }
      sc[q] = s > 0 ? s : 1e-300, ll += std::log(sc[q]);
      for (std::size_t k = 0; k < K; ++k) al[q * K + k] /= sc[q];
    }
    for (std::size_t k = 0; k < K; ++k) be[(n - 1) * K + k] = 1;
    for (std::size_t q = n - 1; q-- > 0;)
      for (std::size_t a = 0; a < K; ++a) {
        double s = 0;
        for (std::size_t c = 0; c < K; ++c) s += A[a * K + c] * em[(q + 1) * K + c] * be[(q + 1) * K + c];
        be[q * K + a] = s / sc[q + 1];
      }
    for (std::size_t q = 0; q < n; ++q) {
      double s = 0;
      for (std::size_t k = 0; k < K; ++k) s += ga[q * K + k] = al[q * K + k] * be[q * K + k];
      for (std::size_t k = 0; k < K; ++k) ga[q * K + k] /= s > 0 ? s : 1;
    }
    return ll;
  };
  const auto baumWelch = [&] {  // 50 rounds from the start set; the fit's log-likelihood
    for (int it = 0; it < 50; ++it) {
      forwardBackward();
      std::vector<double> xi(K * K, 0.0);
      for (std::size_t q = 0; q + 1 < n; ++q)
        for (std::size_t a = 0; a < K; ++a)
          for (std::size_t c = 0; c < K; ++c) xi[a * K + c] += al[q * K + a] * A[a * K + c] * em[(q + 1) * K + c] * be[(q + 1) * K + c] / sc[q + 1];
      for (std::size_t a = 0; a < K; ++a) {
        double s = 0;
        for (std::size_t c = 0; c < K; ++c) s += xi[a * K + c];
        for (std::size_t c = 0; c < K; ++c) A[a * K + c] = s > 0 ? xi[a * K + c] / s : (a == c ? 1.0 : 0.0);
      }
      for (std::size_t k = 0; k < K; ++k) {
        double w = 0, s1 = 0, s2 = 0;
        for (std::size_t q = 0; q < n; ++q) w += ga[q * K + k], s1 += ga[q * K + k] * x[q];
        mu[k] = w > 0 ? s1 / w : 0.0;
        for (std::size_t q = 0; q < n; ++q) s2 += ga[q * K + k] * (x[q] - mu[k]) * (x[q] - mu[k]);
        v[k] = std::max(floor, w > 0 ? s2 / w : floor);
        pi[k] = ga[k];
      }
    }
    return forwardBackward();
  };
  init(false);
  const double fresh = baumWelch();
  if (warm && warm->K == K) {
    const std::vector<double> A0 = A, mu0 = mu, v0 = v, pi0 = pi;
    init(true);
    if (!(baumWelch() > fresh + 1e-9)) A = A0, mu = mu0, v = v0, pi = pi0;
  }
  // the states ordered calm first (by their variance), so the labels mean the same from one cycle to the next
  std::vector<std::size_t> ord(K);
  for (std::size_t k = 0; k < K; ++k) ord[k] = k;
  std::sort(ord.begin(), ord.end(), [&](std::size_t a, std::size_t c) { return v[a] < v[c] || (v[a] == v[c] && a < c); });
  {
    std::vector<double> A2(K * K), mu2(K), v2(K), pi2(K);
    for (std::size_t a = 0; a < K; ++a) {
      mu2[a] = mu[ord[a]], v2[a] = v[ord[a]], pi2[a] = pi[ord[a]];
      for (std::size_t c = 0; c < K; ++c) A2[a * K + c] = A[ord[a] * K + ord[c]];
    }
    A = A2, mu = mu2, v = v2, pi = pi2;
  }
  f.loglik = forwardBackward();
  f.bic = -2 * f.loglik + double(K * K + 2 * K - 1) * std::log(double(n));  // means, variances, the matrix's free entries, the start
  f.A = A, f.mean = mu, f.var = v, f.gamma = ga;
  f.now.assign(al.end() - std::ptrdiff_t(K), al.end());  // filtered at t0: only what t0 knew
  f.ahead.assign(K, 0.0);
  {
    std::vector<double> p = f.now;
    const std::size_t H = std::max<std::size_t>(1, horizon);
    for (std::size_t j = 0; j < H; ++j) {
      std::vector<double> nx(K, 0.0);
      for (std::size_t a = 0; a < K; ++a)
        for (std::size_t c = 0; c < K; ++c) nx[c] += p[a] * A[a * K + c];
      p = nx;
      for (std::size_t k = 0; k < K; ++k) f.ahead[k] += p[k] / double(H);
    }
  }
  f.label.assign(n, 0);
  for (std::size_t q = 0; q < n; ++q) {
    const auto row = ga.begin() + std::ptrdiff_t(q * K);
    f.label[q] = int(std::max_element(row, row + std::ptrdiff_t(K)) - row);
  }
  // the block starts of each state: the days mcDays may start a block at, in that state
  const std::size_t B = std::max<std::size_t>(1, block);
  f.starts.assign(K, {});
  for (std::size_t q = loD; q + B <= hiD + 1; ++q) f.starts[std::size_t(f.label[q - loD])].push_back(q);
}

// Each share's volatility in the states a fit expects over its horizon: its days in the fitted window weighted by their
// chances of each state (a state with fewer than 20 days' weight: the share's own over the window).
inline void regimeSigma(const Data& d, const Backend& b, RegimeFit& f) {
  const std::size_t N = d.N, K = f.K, n = f.hi + 1 - f.lo, loD = f.lo;
  const auto& ga = f.gamma;
  f.sigma.assign(N, kNaN);
  b.parallelFor(N, [&](std::size_t a0, std::size_t e0) {
    std::vector<double> w(K), s2(K);
    for (std::size_t i = a0; i < e0; ++i) {
      std::fill(w.begin(), w.end(), 0.0), std::fill(s2.begin(), s2.end(), 0.0);
      double wAll = 0, sAll = 0;
      for (std::size_t q = 0; q < n; ++q) {
        const double r = d.ret[(loD + q) * N + i];
        if (!std::isfinite(r)) continue;
        for (std::size_t k = 0; k < K; ++k) w[k] += ga[q * K + k], s2[k] += ga[q * K + k] * r * r;
        wAll += 1, sAll += r * r;
      }
      if (wAll < 60) continue;  // the 60 sessions the plain volatility needs
      double v0 = 0;
      for (std::size_t k = 0; k < K; ++k) v0 += f.ahead[k] * (w[k] >= 20 ? s2[k] / w[k] : sAll / wAll);
      f.sigma[i] = std::sqrt(v0);
    }
  });
}

// A path's days drawn in states (mcDays with a fit for t0): the first state from today's chances, each block from
// the past days in its state (from all the window's days when the state has too few), the state moving day by day
// by the matrix. u(): the path's next uniform number.
template <class U>
inline void regimeDays(const RegimeFit& f, std::size_t hiD, std::size_t loD, std::size_t B, std::size_t H, U u, std::vector<std::size_t>& out) {
  const std::size_t K = f.K;
  const auto drawFrom = [&](const double* p) {
    const double r = u();
    double c = 0;
    for (std::size_t k = 0; k < K; ++k)
      if (r < (c += p[k])) return k;
    return K - 1;
  };
  std::size_t z = drawFrom(f.now.data());
  const std::size_t span = hiD + 1 > loD + B ? hiD + 1 - loD - B + 1 : 1;
  while (out.size() < H) {
    const auto& st = f.starts[z];
    std::size_t b0;
    if (st.size() >= 2 * B) b0 = st[std::min(st.size() - 1, std::size_t(u() * double(st.size())))];
    else b0 = loD + std::min(span - 1, std::size_t(u() * double(span)));
    for (std::size_t j = 0; j < B && out.size() < H; ++j) {
      out.push_back(std::min(hiD, b0 + j));
      z = drawFrom(&f.A[z * K]);
    }
  }
}

}  // namespace strat
