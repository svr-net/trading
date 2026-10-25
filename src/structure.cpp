#include "ofm/structure.hpp"

#include "ofm/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ofm {

namespace {

// Symmetric matrix function f applied through the eigen decomposition.
template <class F>
std::vector<double> apply(std::vector<double> A, std::size_t n, F f) {
  std::vector<double> w, V;
  eigenSym(A, n, w, V);
  std::vector<double> out(n * n, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    const double s = f(w[k]);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) out[i * n + j] += V[i * n + k] * s * V[j * n + k];
  }
  return out;
}

}  // namespace

MarketStructure marketStructure(const Market& m, const std::vector<std::uint32_t>& hl, const std::vector<std::vector<double>>& fam) {
  const std::size_t T = m.T(), N = m.N(), H = hl.size(), F = kFamilies;
  MarketStructure ms;
  ms.halfLives = hl;
  ms.treeLength = ms.dimension = ms.geodesic = Panel(T, H);
  if (H == 0) return ms;
  std::vector<std::vector<double>> FC(H, std::vector<double>(F * F, 0.0));
  std::vector<double> lam(H), famSeen(H, 0.0);
  for (std::size_t j = 0; j < H; ++j) lam[j] = std::pow(2.0, -1.0 / static_cast<double>(hl[j]));
  // Each day's returns (0 for a stock without a bar today or yesterday); the covariances, the
  // effective dimension and the spanning tree in one fused kernel, a workgroup per half-life.
  std::vector<double> R(T * N, 0.0), last(N, kNaN);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t i = 0; i < N; ++i) {
      const double c = m.close(t, i);
      R[t * N + i] = (c > 0 && last[i] > 0) ? c / last[i] - 1 : 0.0;
      if (c > 0) last[i] = c;
    }
  std::vector<double> out(3 * T * H, kNaN);
  std::fill(out.begin() + std::ptrdiff_t(2 * T * H), out.end(), 0.0);
  kernels::structure(R, T, N, hl, out);
  for (std::size_t t = 0; t < T; ++t) {
    // Family returns of day t - 2 are known at this close.
    const std::vector<double>* fr = (t >= 2 && t - 2 < fam.size() && !fam[t - 2].empty()) ? &fam[t - 2] : nullptr;
    for (std::size_t j = 0; j < H; ++j) {
      const double l = lam[j], g = 1 - l;
      if (fr) {
        for (std::size_t a = 0; a < F; ++a)
          for (std::size_t b = 0; b < F; ++b) FC[j][a * F + b] = l * FC[j][a * F + b] + g * (*fr)[a] * (*fr)[b];
        famSeen[j] += 1;
      }
      if (!(out[(2 * T + t) * H + j] > 0)) continue;  // not yet half real data, or too few stocks
      ms.dimension(t, j) = out[t * H + j];
      ms.treeLength(t, j) = out[(T + t) * H + j];
      // Geodesic distance to the longest half-life's family covariance.
      if (j + 1 < H && famSeen[j] >= hl[j] && famSeen[H - 1] >= hl[H - 1]) {
        std::vector<double> L = FC[H - 1];
        double tr = 0;
        for (std::size_t a = 0; a < F; ++a) tr += L[a * F + a];
        if (tr > 0) {
          for (std::size_t a = 0; a < F; ++a) L[a * F + a] += 1e-12 * tr;  // numerical floor only
          const auto Li = apply(L, F, [](double x) { return x > 0 ? 1 / std::sqrt(x) : 0.0; });
          std::vector<double> M(F * F, 0.0), tmp(F * F, 0.0);
          for (std::size_t a = 0; a < F; ++a)
            for (std::size_t b = 0; b < F; ++b)
              for (std::size_t c = 0; c < F; ++c) tmp[a * F + b] += Li[a * F + c] * FC[j][c * F + b];
          for (std::size_t a = 0; a < F; ++a)
            for (std::size_t b = 0; b < F; ++b)
              for (std::size_t c = 0; c < F; ++c) M[a * F + b] += tmp[a * F + c] * Li[c * F + b];
          std::vector<double> w, V;
          eigenSym(M, F, w, V);
          double s = 0;
          for (double x : w)
            if (x > 0) s += std::log(x) * std::log(x);
          ms.geodesic(t, j) = std::sqrt(s);
        }
      }
    }
  }
  return ms;
}

}  // namespace ofm
