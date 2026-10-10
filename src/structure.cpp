#include "ofm/structure.hpp"

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
  std::vector<std::vector<double>> C(H, std::vector<double>(N * N, 0.0)), FC(H, std::vector<double>(F * F, 0.0));
  std::vector<double> lam(H), seen(H, 0.0), famSeen(H, 0.0);
  for (std::size_t j = 0; j < H; ++j) lam[j] = std::pow(2.0, -1.0 / static_cast<double>(hl[j]));
  std::vector<double> r(N), last(N, kNaN), key(N), dist(N);
  std::vector<char> in(N), active(N);
  for (std::size_t t = 0; t < T; ++t) {
    // Today's returns (0 for a stock without a bar today or yesterday).
    std::size_t n = 0;
    for (std::size_t i = 0; i < N; ++i) {
      const double c = m.close(t, i);
      r[i] = (c > 0 && last[i] > 0) ? c / last[i] - 1 : 0.0;
      if (c > 0) last[i] = c;
    }
    // Family returns of day t - 2 are known at this close.
    const std::vector<double>* fr = (t >= 2 && t - 2 < fam.size() && !fam[t - 2].empty()) ? &fam[t - 2] : nullptr;
    for (std::size_t j = 0; j < H; ++j) {
      auto& Cj = C[j];
      const double l = lam[j], g = 1 - l;
      for (std::size_t a = 0; a < N; ++a) {
        const double ra = g * r[a];
        double* row = &Cj[a * N];
        for (std::size_t b = a; b < N; ++b) row[b] = l * row[b] + ra * r[b];
      }
      seen[j] += 1;
      if (fr) {
        for (std::size_t a = 0; a < F; ++a)
          for (std::size_t b = 0; b < F; ++b) FC[j][a * F + b] = l * FC[j][a * F + b] + g * (*fr)[a] * (*fr)[b];
        famSeen[j] += 1;
      }
      if (seen[j] < hl[j]) continue;  // until half the weight is real data
      // Stocks with variance, correlation distances, Prim's minimum spanning tree.
      n = 0;
      for (std::size_t i = 0; i < N; ++i) active[i] = Cj[i * N + i] > 0, n += active[i] ? 1 : 0;
      if (n < 3) continue;
      double sumRho2 = 0;
      auto rho = [&](std::size_t a, std::size_t b) {
        const std::size_t x = std::min(a, b), y = std::max(a, b);
        return Cj[x * N + y] / std::sqrt(Cj[x * N + x] * Cj[y * N + y]);
      };
      for (std::size_t a = 0; a < N; ++a)
        if (active[a])
          for (std::size_t b = 0; b < N; ++b)
            if (active[b]) {
              const double p = a == b ? 1.0 : rho(a, b);
              sumRho2 += p * p;
            }
      ms.dimension(t, j) = static_cast<double>(n) * static_cast<double>(n) / sumRho2 / static_cast<double>(n);
      std::fill(in.begin(), in.end(), 0);
      std::fill(key.begin(), key.end(), std::numeric_limits<double>::infinity());
      std::size_t start = 0;
      while (!active[start]) ++start;
      key[start] = 0;
      double total = 0;
      for (std::size_t added = 0; added < n; ++added) {
        std::size_t u = N;
        for (std::size_t i = 0; i < N; ++i)
          if (active[i] && !in[i] && (u == N || key[i] < key[u])) u = i;
        in[u] = 1;
        total += key[u];
        for (std::size_t v = 0; v < N; ++v)
          if (active[v] && !in[v]) {
            const double d = std::sqrt(std::max(0.0, 2.0 * (1.0 - rho(u, v))));
            if (d < key[v]) key[v] = d;
          }
      }
      ms.treeLength(t, j) = total / static_cast<double>(n - 1);
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
