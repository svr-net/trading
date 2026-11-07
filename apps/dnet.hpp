#pragma once
// A differential network (research; Huge and Savine, "Differential Machine Learning", 2020): a
// two-hidden-layer network of softplus units f(x, d) of a sample's inputs x and one input d (here
// the limit's depth), trained on the value y and on its slope dy/dd: the loss of a sample is
// (f - y)^2 + lambda (df/dd - y')^2. The slope labels come from each sample's own path, so the
// network learns the whole curve along d (its shape, not only its points) from fewer samples.
// Softplus (smooth) units: the slope df/dd is smooth too (rectified units make it a step).
// The gradients are exact and hand-derived (the value's and the slope's, through both layers);
// tests/tests.cpp checks them against finite differences.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dnet {

struct Net {
  std::size_t In = 0, H = 0, D = 0;  // inputs (the last-but-none index D is the differential input), hidden units
  std::vector<double> w, m, v;        // weights: W1 [H][In], b1 [H], W2 [H][H], b2 [H], w3 [H], b3; Adam's moments
  double lr = 1e-3, lambda = 1.0, steps = 0;
  Net() = default;
  Net(std::size_t in, std::size_t hidden, std::size_t diffInput, std::uint64_t seed = 0x9E3779B97F4A7C15ull) : In(in), H(hidden), D(diffInput) {
    w.assign(size(), 0.0), m.assign(size(), 0.0), v.assign(size(), 0.0);
    std::uint64_t s = seed;
    const auto u = [&] { s ^= s << 13, s ^= s >> 7, s ^= s << 17; return double(s >> 11) * 0x1.0p-53 - 0.5; };
    for (std::size_t k = 0; k < H * In; ++k) w[k] = u() * 2 * std::sqrt(3.0 / double(In));
    for (std::size_t k = 0; k < H * H; ++k) w[oW2() + k] = u() * 2 * std::sqrt(3.0 / double(H));
    for (std::size_t k = 0; k < H; ++k) w[ow3() + k] = u() * 2 * std::sqrt(3.0 / double(H));
  }
  std::size_t size() const { return H * In + H + H * H + H + H + 1; }
  std::size_t ob1() const { return H * In; }
  std::size_t oW2() const { return ob1() + H; }
  std::size_t ob2() const { return oW2() + H * H; }
  std::size_t ow3() const { return ob2() + H; }
  std::size_t ob3() const { return ow3() + H; }
  static double sp(double z) { return z > 30 ? z : std::log1p(std::exp(z)); }
  static double sg(double z) { return 1 / (1 + std::exp(-z)); }

  /// The value f and its slope g = df/dx[D] for one sample.
  void eval(const double* x, double& f, double& g) const {
    std::vector<double> z1(H), h1(H), s1(H), z2(H);
    for (std::size_t j = 0; j < H; ++j) {
      double z = w[ob1() + j];
      for (std::size_t k = 0; k < In; ++k) z += w[j * In + k] * x[k];
      z1[j] = z, h1[j] = sp(z), s1[j] = sg(z) * w[j * In + D];
    }
    f = w[ob3()], g = 0;
    for (std::size_t j = 0; j < H; ++j) {
      double z = w[ob2() + j], u2 = 0;
      for (std::size_t k = 0; k < H; ++k) z += w[oW2() + j * H + k] * h1[k], u2 += w[oW2() + j * H + k] * s1[k];
      f += w[ow3() + j] * sp(z), g += w[ow3() + j] * sg(z) * u2;
    }
  }

  /// The loss's gradient for one sample, added to grad (scaled by wt); returns the sample's loss.
  double gradient(const double* x, double y, double dy, double wt, std::vector<double>& grad) const {
    std::vector<double> z1(H), h1(H), q1(H), c(H), s1(H), z2(H), q2(H), u2(H), h2(H);
    for (std::size_t j = 0; j < H; ++j) {
      double z = w[ob1() + j];
      for (std::size_t k = 0; k < In; ++k) z += w[j * In + k] * x[k];
      z1[j] = z, h1[j] = sp(z), q1[j] = sg(z), c[j] = w[j * In + D], s1[j] = q1[j] * c[j];
    }
    double f = w[ob3()], g = 0;
    for (std::size_t j = 0; j < H; ++j) {
      double z = w[ob2() + j], uu = 0;
      for (std::size_t k = 0; k < H; ++k) z += w[oW2() + j * H + k] * h1[k], uu += w[oW2() + j * H + k] * s1[k];
      z2[j] = z, q2[j] = sg(z), u2[j] = uu, h2[j] = sp(z);
      f += w[ow3() + j] * h2[j], g += w[ow3() + j] * q2[j] * u2[j];
    }
    const double ef = 2 * (f - y) * wt, eg = 2 * lambda * (g - dy) * wt;
    // The last layer.
    grad[ob3()] += ef;
    std::vector<double> d2(H), e2(H);
    for (std::size_t j = 0; j < H; ++j) {
      grad[ow3() + j] += ef * h2[j] + eg * q2[j] * u2[j];
      d2[j] = w[ow3() + j] * q2[j];                                // df/dz2 = dg/du2
      e2[j] = w[ow3() + j] * u2[j] * q2[j] * (1 - q2[j]);          // dg/dz2
      grad[ob2() + j] += ef * d2[j] + eg * e2[j];
      for (std::size_t k = 0; k < H; ++k) grad[oW2() + j * H + k] += ef * d2[j] * h1[k] + eg * (d2[j] * s1[k] + e2[j] * h1[k]);
    }
    // The first layer.
    for (std::size_t k = 0; k < H; ++k) {
      double a = 0, b = 0;  // (W2^T d2)_k, (W2^T e2)_k
      for (std::size_t j = 0; j < H; ++j) a += w[oW2() + j * H + k] * d2[j], b += w[oW2() + j * H + k] * e2[j];
      const double d1 = a * q1[k];                                          // df/dz1 = dg/dc
      const double e1 = a * c[k] * q1[k] * (1 - q1[k]) + b * q1[k];         // dg/dz1
      grad[ob1() + k] += ef * d1 + eg * e1;
      for (std::size_t i = 0; i < In; ++i) grad[k * In + i] += ef * d1 * x[i] + eg * e1 * x[i];
      grad[k * In + D] += eg * d1;  // g's own dependence on the column W1[:, D]
    }
    return (f - y) * (f - y) + lambda * (g - dy) * (g - dy);
  }

  /// One Adam step on a mini-batch (rows of x [n][In], labels y, dy).
  void step(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& dy, std::size_t n) {
    if (n == 0) return;
    std::vector<double> grad(size(), 0.0);
    for (std::size_t r = 0; r < n; ++r) gradient(&x[r * In], y[r], dy[r], 1.0 / double(n), grad);
    steps += 1;
    const double b1 = 0.9, b2 = 0.999, c1 = 1 - std::pow(b1, steps), c2 = 1 - std::pow(b2, steps);
    for (std::size_t k = 0; k < size(); ++k) {
      m[k] = b1 * m[k] + (1 - b1) * grad[k], v[k] = b2 * v[k] + (1 - b2) * grad[k] * grad[k];
      w[k] -= lr * (m[k] / c1) / (std::sqrt(v[k] / c2) + 1e-8);
    }
  }
};

}  // namespace dnet
