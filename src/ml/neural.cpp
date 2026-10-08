#include "sat/ml/neural.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "sat/core/random.hpp"

namespace sat {

namespace {

/// Adam optimiser state for one parameter vector.
struct Adam {
  std::vector<double> m, v;
  std::size_t t = 0;
  void init(std::size_t n) {
    m.assign(n, 0.0);
    v.assign(n, 0.0);
    t = 0;
  }
  // Applies the averaged gradient g (cleared afterwards) to w.
  void step(std::vector<double>& w, std::vector<double>& g, double lr, double decay, double count) {
    ++t;
    const double b1 = 0.9, b2 = 0.999;
    const double c1 = 1.0 - std::pow(b1, static_cast<double>(t)), c2 = 1.0 - std::pow(b2, static_cast<double>(t));
    for (std::size_t k = 0; k < w.size(); ++k) {
      const double gk = g[k] / count + decay * w[k];
      m[k] = b1 * m[k] + (1 - b1) * gk;
      v[k] = b2 * v[k] + (1 - b2) * gk * gk;
      w[k] -= lr * (m[k] / c1) / (std::sqrt(v[k] / c2) + 1e-8);
      g[k] = 0.0;
    }
  }
};

void initUniform(std::vector<double>& w, double bound, Rng& rng) {
  for (double& x : w) x = (2.0 * rng.uniform() - 1.0) * bound;
}

void checkData(const Matrix& X, const std::vector<double>& y, const char* who) {
  if (X.rows() == 0 || X.rows() != y.size()) throw std::invalid_argument(std::string(who) + ": empty or mismatched training data");
}

std::vector<std::size_t> epochOrder(std::size_t n, std::size_t maxSamples, Rng& rng) {
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  shuffle(order, rng);
  if (maxSamples > 0 && maxSamples < n) order.resize(maxSamples);
  return order;
}

}  // namespace

// ------------------------------------------------------------------ MLP

double Mlp::forward(const double* z, std::vector<double>& a) const {
  double m = b2_;
  for (std::size_t j = 0; j < hid_; ++j) {
    double s = b1_[j];
    const double* w = W1_.data() + j * in_;
    for (std::size_t f = 0; f < in_; ++f) s += w[f] * z[f];
    a[j] = std::tanh(s);
    m += W2_[j] * a[j];
  }
  return m;
}

void Mlp::fit(const Matrix& X, const std::vector<double>& y) {
  checkData(X, y, "MLP");
  std_.fit(X);
  const Matrix Z = std_.transform(X);
  in_ = Z.cols();
  hid_ = std::max<std::size_t>(1, spec_.hidden);
  Rng rng(spec_.seed);
  // Parameters in one vector: W1 | b1 | W2 | b2.
  std::vector<double> w(hid_ * in_ + 2 * hid_ + 1, 0.0), g(w.size(), 0.0);
  std::vector<double> head(hid_ * in_);
  initUniform(head, std::sqrt(6.0 / static_cast<double>(in_ + hid_)), rng);
  std::copy(head.begin(), head.end(), w.begin());
  std::vector<double> out(hid_);
  initUniform(out, std::sqrt(6.0 / static_cast<double>(hid_ + 1)), rng);
  std::copy(out.begin(), out.end(), w.begin() + static_cast<std::ptrdiff_t>(hid_ * in_ + hid_));
  Adam adam;
  adam.init(w.size());
  const std::size_t oB1 = hid_ * in_, oW2 = oB1 + hid_, oB2 = oW2 + hid_;
  auto unpack = [&] {
    W1_.assign(w.begin(), w.begin() + static_cast<std::ptrdiff_t>(oB1));
    b1_.assign(w.begin() + static_cast<std::ptrdiff_t>(oB1), w.begin() + static_cast<std::ptrdiff_t>(oW2));
    W2_.assign(w.begin() + static_cast<std::ptrdiff_t>(oW2), w.begin() + static_cast<std::ptrdiff_t>(oB2));
    b2_ = w[oB2];
  };
  unpack();
  std::vector<double> a(hid_);
  const std::size_t batch = std::max<std::size_t>(1, spec_.batch);
  for (std::size_t epoch = 0; epoch < std::max<std::size_t>(1, spec_.epochs); ++epoch) {
    const auto order = epochOrder(Z.rows(), spec_.maxSamples, rng);
    std::size_t inBatch = 0;
    for (std::size_t k = 0; k < order.size(); ++k) {
      const std::size_t r = order[k];
      const double* z = Z.row(r);
      const double e = sigmoid(forward(z, a)) - y[r];
      for (std::size_t j = 0; j < hid_; ++j) {
        g[oW2 + j] += e * a[j];
        const double d = e * W2_[j] * (1.0 - a[j] * a[j]);
        g[oB1 + j] += d;
        double* gw = g.data() + j * in_;
        for (std::size_t f = 0; f < in_; ++f) gw[f] += d * z[f];
      }
      g[oB2] += e;
      if (++inBatch == batch || k + 1 == order.size()) {
        adam.step(w, g, spec_.learningRate, spec_.weightDecay, static_cast<double>(inBatch));
        inBatch = 0;
        unpack();
      }
    }
  }
}

std::vector<double> Mlp::predictProba(const Matrix& X) const {
  if (W1_.empty()) throw std::logic_error("MLP is not fitted");
  std::vector<double> out(X.rows()), z(X.cols()), a(hid_);
  for (std::size_t r = 0; r < X.rows(); ++r) {
    std_.apply(X.row(r), z.data());
    out[r] = sigmoid(forward(z.data(), a));
  }
  return out;
}

// ------------------------------------------------------------------ LSTM

// Cache layout per time step s (stride 7H + K): u (K) | i | f | g | o | c | tanh(c) | c_prev.
double Lstm::forward(const double* z, std::vector<double>* cache) const {
  const std::size_t H = hid_, F = in_, K = F + H + 1, stride = K + 7 * H;
  std::vector<double> h(H, 0.0), c(H, 0.0), u(K), pre(4 * H);
  for (std::size_t s = 0; s < steps_; ++s) {
    std::copy(z + s * F, z + (s + 1) * F, u.begin());
    std::copy(h.begin(), h.end(), u.begin() + static_cast<std::ptrdiff_t>(F));
    u[K - 1] = 1.0;
    for (std::size_t r = 0; r < 4 * H; ++r) {
      const double* w = W_.data() + r * K;
      double acc = 0.0;
      for (std::size_t k = 0; k < K; ++k) acc += w[k] * u[k];
      pre[r] = acc;
    }
    double* cs = cache ? cache->data() + s * stride : nullptr;
    if (cs) std::copy(u.begin(), u.end(), cs);
    for (std::size_t j = 0; j < H; ++j) {
      const double ig = sigmoid(pre[j]), fg = sigmoid(pre[H + j]), gg = std::tanh(pre[2 * H + j]), og = sigmoid(pre[3 * H + j]);
      const double cPrev = c[j];
      c[j] = fg * cPrev + ig * gg;
      const double tc = std::tanh(c[j]);
      h[j] = og * tc;
      if (cs) {
        double* q = cs + K;
        q[j] = ig;
        q[H + j] = fg;
        q[2 * H + j] = gg;
        q[3 * H + j] = og;
        q[4 * H + j] = c[j];
        q[5 * H + j] = tc;
        q[6 * H + j] = cPrev;
      }
    }
  }
  double m = V_[H];
  for (std::size_t j = 0; j < H; ++j) m += V_[j] * h[j];
  return m;
}

void Lstm::fit(const Matrix& X, const std::vector<double>& y) {
  checkData(X, y, "LSTM");
  steps_ = std::max<std::size_t>(1, spec_.seqLen);
  if (X.cols() % steps_ != 0) throw std::invalid_argument("LSTM: columns are not a multiple of the sequence length");
  in_ = X.cols() / steps_;
  hid_ = std::max<std::size_t>(1, spec_.hidden);
  std_.fit(X);
  const Matrix Z = std_.transform(X);
  const std::size_t H = hid_, F = in_, K = F + H + 1, stride = K + 7 * H;
  Rng rng(spec_.seed);
  W_.assign(4 * H * K, 0.0);
  initUniform(W_, std::sqrt(6.0 / static_cast<double>(K + H)), rng);
  for (std::size_t j = 0; j < H; ++j) {
    // Gate biases 0, except the forget gate at 1: remember by default.
    for (std::size_t g = 0; g < 4; ++g) W_[(g * H + j) * K + K - 1] = g == 1 ? 1.0 : 0.0;
  }
  V_.assign(H + 1, 0.0);
  initUniform(V_, std::sqrt(6.0 / static_cast<double>(H + 1)), rng);
  V_[H] = 0.0;
  std::vector<double> gW(W_.size(), 0.0), gV(V_.size(), 0.0);
  Adam adamW, adamV;
  adamW.init(W_.size());
  adamV.init(V_.size());
  std::vector<double> cache(steps_ * stride);
  std::vector<double> dh(H), dc(H), dpre(4 * H);
  const std::size_t batch = std::max<std::size_t>(1, spec_.batch);
  for (std::size_t epoch = 0; epoch < std::max<std::size_t>(1, spec_.epochs); ++epoch) {
    const auto order = epochOrder(Z.rows(), spec_.maxSamples, rng);
    std::size_t inBatch = 0;
    for (std::size_t k = 0; k < order.size(); ++k) {
      const std::size_t r = order[k];
      const double e = sigmoid(forward(Z.row(r), &cache)) - y[r];
      // Output layer: h_T = o * tanh(c) of the last step.
      const double* last = cache.data() + (steps_ - 1) * stride + K;
      for (std::size_t j = 0; j < H; ++j) {
        const double hT = last[3 * H + j] * last[5 * H + j];
        gV[j] += e * hT;
        dh[j] = e * V_[j];
        dc[j] = 0.0;
      }
      gV[H] += e;
      for (std::size_t s = steps_; s-- > 0;) {
        const double* u = cache.data() + s * stride;
        const double* q = u + K;
        for (std::size_t j = 0; j < H; ++j) {
          const double ig = q[j], fg = q[H + j], gg = q[2 * H + j], og = q[3 * H + j], tc = q[5 * H + j], cPrev = q[6 * H + j];
          const double dO = dh[j] * tc;
          const double dC = dc[j] + dh[j] * og * (1.0 - tc * tc);
          dpre[j] = dC * gg * ig * (1.0 - ig);
          dpre[H + j] = dC * cPrev * fg * (1.0 - fg);
          dpre[2 * H + j] = dC * ig * (1.0 - gg * gg);
          dpre[3 * H + j] = dO * og * (1.0 - og);
          dc[j] = dC * fg;
        }
        std::fill(dh.begin(), dh.end(), 0.0);
        for (std::size_t row = 0; row < 4 * H; ++row) {
          const double d = dpre[row];
          if (d == 0.0) continue;
          double* gw = gW.data() + row * K;
          const double* w = W_.data() + row * K;
          for (std::size_t kk = 0; kk < K; ++kk) gw[kk] += d * u[kk];
          for (std::size_t j = 0; j < H; ++j) dh[j] += d * w[F + j];
        }
      }
      if (++inBatch == batch || k + 1 == order.size()) {
        // Clip the averaged gradient norm at 5 to keep BPTT stable.
        double norm = 0.0;
        for (double v : gW) norm += v * v;
        for (double v : gV) norm += v * v;
        norm = std::sqrt(norm) / static_cast<double>(inBatch);
        if (norm > 5.0) {
          const double sc = 5.0 / norm;
          for (double& v : gW) v *= sc;
          for (double& v : gV) v *= sc;
        }
        adamW.step(W_, gW, spec_.learningRate, spec_.weightDecay, static_cast<double>(inBatch));
        adamV.step(V_, gV, spec_.learningRate, spec_.weightDecay, static_cast<double>(inBatch));
        inBatch = 0;
      }
    }
  }
}

std::vector<double> Lstm::predictProba(const Matrix& X) const {
  if (W_.empty()) throw std::logic_error("LSTM is not fitted");
  if (X.cols() != steps_ * in_) throw std::invalid_argument("LSTM: input width does not match the fitted sequence");
  std::vector<double> out(X.rows()), z(X.cols());
  for (std::size_t r = 0; r < X.rows(); ++r) {
    std_.apply(X.row(r), z.data());
    out[r] = sigmoid(forward(z.data(), nullptr));
  }
  return out;
}

}  // namespace sat
