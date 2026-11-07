// The walk-forward ridge's pieces shared by the single-share models (ofm_single, ofm_hold): a Cholesky
// solve for several targets at once and a record of results scored by its mean shrunk by its t
// statistic.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace ridge {

// Cholesky solve of (G + add I) X = B for several right-hand sides (the intercept, 0, not shrunk).
inline bool solve(std::vector<double> G, std::size_t n, double add, const std::vector<double>& B, std::size_t nb, std::vector<double>& X) {
  for (std::size_t k = 1; k < n; ++k) G[k * n + k] += add;
  for (std::size_t j = 0; j < n; ++j) {
    double s = G[j * n + j];
    for (std::size_t k = 0; k < j; ++k) s -= G[j * n + k] * G[j * n + k];
    if (!(s > 1e-12)) return false;
    G[j * n + j] = std::sqrt(s);
    for (std::size_t r = j + 1; r < n; ++r) {
      double v = G[r * n + j];
      for (std::size_t k = 0; k < j; ++k) v -= G[r * n + k] * G[j * n + k];
      G[r * n + j] = v / G[j * n + j];
    }
  }
  X.assign(n * nb, 0.0);
  std::vector<double> y(n);
  for (std::size_t c = 0; c < nb; ++c) {
    for (std::size_t r = 0; r < n; ++r) {
      double v = B[r * nb + c];
      for (std::size_t k = 0; k < r; ++k) v -= G[r * n + k] * y[k];
      y[r] = v / G[r * n + r];
    }
    for (std::size_t r = n; r-- > 0;) {
      double v = y[r];
      for (std::size_t k = r + 1; k < n; ++k) v -= G[k * n + r] * X[k * nb + c];
      X[r * nb + c] = v / G[r * n + r];
    }
  }
  return true;
}

struct Rec {  // a record of results
  double n = 0, s = 0, s2 = 0;
  void add(double x) { n += 1, s += x, s2 += x * x; }
  double mean() const { return n > 0 ? s / n : 0; }
  double sd() const { return n > 1 ? std::sqrt(std::max(0.0, s2 / n - mean() * mean())) : 0; }
  double t() const { const double d = sd(); return d > 0 ? mean() / d * std::sqrt(n) : 0; }
  double shrunk() const { const double tt = t(); return n > 1 && tt != 0 ? mean() * std::max(0.0, 1.0 - 1.0 / (tt * tt)) : 0; }
};

}  // namespace ridge
