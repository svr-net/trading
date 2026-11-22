// The differential network proxy of the adaptive nested Monte Carlo (docs/adaptive-nested-mc.md): differential machine
// learning (Huge and Savine 2020) in place of the Hermite regression, for horizon states of many dimensions. Header-only,
// C++17, no dependencies.
//
//   labels    every training path gives one value label, its payoff Y, and one label per state number, its pathwise
//             derivative dY/dx_j (OuterScenarios::gradient): E[dY/dx | x] = dV/dx, so a path teaches the shape of V,
//             not only its level, and the network learns from far fewer paths than a regression on values alone
//   network   a feed-forward network with softplus activations (hidden layers x width); its derivative by its inputs
//             by backpropagation (the "twin network")
//   loss      alpha MSE(value) + (1 - alpha) sum_j lambda_j MSE(derivative j) / d, alpha = 1 / (1 + d) and
//             lambda_j = 1 / mean(g_j^2) on the standardized labels, as the paper
//   training  Adam on minibatches, the learning rate decayed by a cosine to a tenth; the weight gradient is exact,
//             second order included (it differentiates through the twin network's backpropagation;
//             DifferentialProxy::gradientCheck tests it against central differences)
//
// Reproducible: the initial weights and the minibatch order come from SplitMix64 (no std:: distribution, whose numbers
// differ between standard libraries), and each minibatch's gradient is the sum of 8 fixed chunks added in order, so a
// fit is the same on any number of threads and on every platform.
//
// Estimation variance v(x), as the Hermite proxy's: the output layer is a linear regression on the last hidden layer's
// W features (and 1), so its HC0 sandwich G^-1 (sum e^2 phi phi^T) G^-1 over the training rows gives v(x) = phi^T Cov phi.
// It leaves out the hidden layers' own uncertainty, as the Hermite proxy's leaves out the basis'; without it the network
// claims to be exact, the audit cannot see a bias below the payoff noise, and the loop stops early (docs).
//
// A fit is laid out as a DifferentialProblem (standardized rows, initial weights, the batch order and the learning rate
// of every step) and trained on it; the fused GPU kernels (include/strategy/diffnet_gpu.hpp) train on the same problem.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "strategy/adaptive_nested.hpp"

namespace anmc {

struct DifferentialOptions {
  std::size_t hiddenLayers = 3;
  std::size_t width = 32;
  std::size_t epochs = 60;      // passes over the training rows
  std::size_t batch = 256;      // rows a step
  double learningRate = 0.01;   // Adam's, decayed by a cosine to a tenth over the fit
  bool useDerivatives = true;   // false: an ordinary network on the values alone (the comparison)
  bool estimationVariance = true;  // v(x) from the output layer's HC0 sandwich; false: v = 0
  double valueWeight = -1;      // alpha, the loss's share on values; below 0: 1 / (1 + d)
  std::uint64_t seed = 7;

  void validate() const {
    if (hiddenLayers < 1) throw std::invalid_argument("DifferentialOptions: at least one hidden layer");
    if (width < 1) throw std::invalid_argument("DifferentialOptions: width must be positive");
    if (batch < 1) throw std::invalid_argument("DifferentialOptions: batch must be positive");
    if (epochs < 1) throw std::invalid_argument("DifferentialOptions: at least one epoch");
    if (!(learningRate > 0)) throw std::invalid_argument("DifferentialOptions: the learning rate must be positive");
    if (valueWeight > 1) throw std::invalid_argument("DifferentialOptions: valueWeight must be at most 1");
  }
};

// A fit laid out flat: what DifferentialProxy trains on, and what the fused kernels' plans are compiled from.
// Parameters: per hidden layer l, W_l (width x previous, row by row) then b_l; then the output weights and bias.
struct DifferentialProblem {
  std::size_t inputs = 0, rows = 0, hiddenLayers = 0, width = 0, epochs = 0, batch = 0;
  std::vector<std::size_t> wOffset, bOffset;  // per hidden layer (index 0 unused)
  std::size_t outOffset = 0;
  std::vector<double> x, g;                   // rows x inputs, standardized states and derivative labels
  std::vector<double> y, w;                   // per row: standardized value, weight (mean 1)
  std::vector<char> hasGradient;
  std::vector<double> lambda;                 // per input: the weight of its derivative errors
  double alpha = 1;                           // the loss's share on values
  std::vector<double> theta;                  // initial weights
  std::vector<std::uint32_t> order;           // epochs x rows: each epoch's rows in training order
  std::vector<double> learningRate;           // per step
  std::vector<double> xMean, xScale, lo, hi;  // standardization; the box of the training states
  double yMean = 0, yScale = 1;
  std::size_t parameters() const { return theta.size(); }
  std::size_t stepsPerEpoch() const { return batch ? (rows + batch - 1) / batch : 0; }
  std::size_t steps() const { return epochs * stepsPerEpoch(); }
};

namespace diffnet {

constexpr double kPi = 3.14159265358979323846;
constexpr double kAdamBeta1 = 0.9, kAdamBeta2 = 0.999, kAdamEpsilon = 1e-8;
constexpr std::size_t kChunks = 8;  // a minibatch's fixed split: the sums do not depend on the thread count

inline double softplus(double z) { return z > 30 ? z : std::log1p(std::exp(z)); }
inline double sigmoid(double z) { return 1 / (1 + std::exp(-z)); }

// A standard normal from SplitMix64 (Box-Muller, the cosine half).
inline double gauss(SplitMix64& g) {
  double u1 = g.uniform();
  if (u1 < 1e-300) u1 = 1e-300;
  return std::sqrt(-2 * std::log(u1)) * std::cos(2 * kPi * g.uniform());
}

// The network's arithmetic in double precision: forward, the twin network, and the exact gradient of the loss.
struct Net {
  std::size_t d = 0, L = 0, W = 0;
  std::vector<std::size_t> size, wo, bo;  // layer sizes (inputs, hidden...); weight and bias offsets per hidden layer
  std::size_t oo = 0;                     // output weights, then the output bias

  Net(std::size_t inputs, std::size_t layers, std::size_t width) : d(inputs), L(layers), W(width) {
    size.assign(L + 1, W);
    size[0] = d;
    wo.assign(L + 1, 0);
    bo.assign(L + 1, 0);
    std::size_t at = 0;
    for (std::size_t l = 1; l <= L; ++l) {
      wo[l] = at;
      at += size[l] * size[l - 1];
      bo[l] = at;
      at += size[l];
    }
    oo = at;
  }
  std::size_t parameters() const { return oo + W + 1; }

  // Scratch for one row: activations and the adjoints, per layer.
  struct Work {
    std::vector<std::vector<double>> z, a, gb, gz, u, vz, ab;
    std::vector<double> zb;
    explicit Work(const Net& n) : z(n.L + 1), a(n.L + 1), gb(n.L + 1), gz(n.L + 1), u(n.L + 1), vz(n.L + 1), ab(n.L + 1), zb(n.W) {
      for (std::size_t l = 0; l <= n.L; ++l)
        for (auto* v : {&z[l], &a[l], &gb[l], &gz[l], &u[l], &vz[l], &ab[l]}) v->assign(n.size[l], 0.0);
    }
  };

  double forward(const double* theta, const double* x, Work& k) const {
    std::copy(x, x + d, k.a[0].begin());
    for (std::size_t l = 1; l <= L; ++l) {
      const std::size_t cols = size[l - 1];
      for (std::size_t i = 0; i < W; ++i) {
        double s = theta[bo[l] + i];
        const double* Wi = theta + wo[l] + i * cols;
        for (std::size_t j = 0; j < cols; ++j) s += Wi[j] * k.a[l - 1][j];
        k.z[l][i] = s;
        k.a[l][i] = softplus(s);
      }
    }
    double y = theta[oo + W];
    for (std::size_t i = 0; i < W; ++i) y += theta[oo + i] * k.a[L][i];
    return y;
  }

  // The network's derivative by its inputs into k.gb[0] (after forward).
  void twin(const double* theta, Work& k) const {
    for (std::size_t i = 0; i < W; ++i) k.gb[L][i] = theta[oo + i];
    for (std::size_t l = L; l >= 1; --l) {
      const std::size_t cols = size[l - 1];
      for (std::size_t i = 0; i < W; ++i) k.gz[l][i] = k.gb[l][i] * sigmoid(k.z[l][i]);
      std::fill(k.gb[l - 1].begin(), k.gb[l - 1].end(), 0.0);
      for (std::size_t i = 0; i < W; ++i) {
        const double* Wi = theta + wo[l] + i * cols;
        for (std::size_t j = 0; j < cols; ++j) k.gb[l - 1][j] += Wi[j] * k.gz[l][i];
      }
    }
  }

  // The loss of rows order[begin, end) of problem p at weights theta, and its gradient added to grad.
  double lossAndGradient(const DifferentialProblem& p, const double* theta, const std::uint32_t* order, std::size_t begin, std::size_t end,
                         double* grad, Work& k) const {
    const double beta = 1 - p.alpha;
    double loss = 0;
    for (std::size_t q = begin; q < end; ++q) {
      const std::size_t r = order[q];
      const double yhat = forward(theta, p.x.data() + r * d, k);
      const double omega = p.w[r], ey = yhat - p.y[r];
      loss += p.alpha * omega * ey * ey;
      const double dy = 2 * p.alpha * omega * ey;
      if (p.hasGradient[r] && beta > 0) {
        twin(theta, k);
        const double* gt = p.g.data() + r * d;
        for (std::size_t j = 0; j < d; ++j) {
          const double e = k.gb[0][j] - gt[j];
          loss += beta * omega * p.lambda[j] * e * e / static_cast<double>(d);
          k.u[0][j] = 2 * beta * omega * p.lambda[j] * e / static_cast<double>(d);
        }
        // The adjoint of the twin network, layer 1 upwards: through gb[l-1] = W^T gz and gz = gb s(z).
        for (std::size_t l = 1; l <= L; ++l) {
          const std::size_t cols = size[l - 1];
          for (std::size_t i = 0; i < W; ++i) {
            const double* Wi = theta + wo[l] + i * cols;
            double* gWi = grad + wo[l] + i * cols;
            double s = 0;
            for (std::size_t j = 0; j < cols; ++j) {
              gWi[j] += k.gz[l][i] * k.u[l - 1][j];
              s += Wi[j] * k.u[l - 1][j];
            }
            const double sg = sigmoid(k.z[l][i]);
            k.u[l][i] = s * sg;
            k.vz[l][i] = s * k.gb[l][i] * sg * (1 - sg);
          }
        }
        for (std::size_t i = 0; i < W; ++i) grad[oo + i] += k.u[L][i];
      } else {
        for (std::size_t l = 1; l <= L; ++l) std::fill(k.vz[l].begin(), k.vz[l].end(), 0.0);
      }
      // The value's adjoint and the derivative loss's z-adjoints, back through the forward pass.
      for (std::size_t i = 0; i < W; ++i) {
        k.ab[L][i] = dy * theta[oo + i];
        grad[oo + i] += dy * k.a[L][i];
      }
      grad[oo + W] += dy;
      for (std::size_t l = L; l >= 1; --l) {
        const std::size_t cols = size[l - 1];
        for (std::size_t i = 0; i < W; ++i) k.zb[i] = k.ab[l][i] * sigmoid(k.z[l][i]) + k.vz[l][i];
        for (std::size_t i = 0; i < W; ++i) {
          double* gWi = grad + wo[l] + i * cols;
          for (std::size_t j = 0; j < cols; ++j) gWi[j] += k.zb[i] * k.a[l - 1][j];
          grad[bo[l] + i] += k.zb[i];
        }
        if (l > 1) {
          std::fill(k.ab[l - 1].begin(), k.ab[l - 1].end(), 0.0);
          for (std::size_t i = 0; i < W; ++i) {
            const double* Wi = theta + wo[l] + i * cols;
            for (std::size_t j = 0; j < cols; ++j) k.ab[l - 1][j] += Wi[j] * k.zb[i];
          }
        }
      }
    }
    return loss;
  }
};

// This thread's scratch for a network of n's shape (every entry is written before it is read).
inline Net::Work& scratch(const Net& n) {
  thread_local std::unique_ptr<Net::Work> w;
  thread_local std::size_t d = 0, L = 0, W = 0;
  if (!w || d != n.d || L != n.L || W != n.W) w = std::make_unique<Net::Work>(n), d = n.d, L = n.L, W = n.W;
  return *w;
}

}  // namespace diffnet

class DifferentialProxy : public Proxy {
 public:
  explicit DifferentialProxy(DifferentialOptions o = DifferentialOptions()) : o_(o) { o_.validate(); }

  // The problem a fit on these rows trains on (standardization from the rows, the initial weights, the batch order
  // and the learning rates), without training.
  DifferentialProblem problem(const OuterScenarios& o, const std::vector<std::size_t>& rows) const {
    const std::size_t n = rows.size(), d = o.d;
    if (n < 2) throw std::invalid_argument("DifferentialProxy: at least two training rows");
    const diffnet::Net net(d, o_.hiddenLayers, o_.width);
    DifferentialProblem p;
    p.inputs = d, p.rows = n, p.hiddenLayers = o_.hiddenLayers, p.width = o_.width, p.epochs = o_.epochs, p.batch = std::min(o_.batch, n);
    p.wOffset = net.wo, p.bOffset = net.bo, p.outOffset = net.oo;
    // standardization: population moments of the training rows
    p.xMean.assign(d, 0), p.xScale.assign(d, 1), p.lo.assign(d, kInf), p.hi.assign(d, -kInf);
    for (std::size_t r : rows)
      for (std::size_t j = 0; j < d; ++j) {
        const double x = o.at(r)[j];
        p.xMean[j] += x, p.lo[j] = std::min(p.lo[j], x), p.hi[j] = std::max(p.hi[j], x);
      }
    for (double& m : p.xMean) m /= static_cast<double>(n);
    for (std::size_t j = 0; j < d; ++j) {
      double ss = 0;
      for (std::size_t r : rows) ss += (o.at(r)[j] - p.xMean[j]) * (o.at(r)[j] - p.xMean[j]);
      const double sd = std::sqrt(ss / static_cast<double>(n));
      p.xScale[j] = sd > 1e-12 ? sd : 1;
    }
    double ym = 0, yv = 0;
    for (std::size_t r : rows) ym += o.payoff[r];
    ym /= static_cast<double>(n);
    for (std::size_t r : rows) yv += (o.payoff[r] - ym) * (o.payoff[r] - ym);
    p.yMean = ym;
    p.yScale = std::sqrt(yv / static_cast<double>(n));
    if (!(p.yScale > 1e-12)) p.yScale = 1;
    // the standardized rows
    const bool labels = o_.useDerivatives && !o.gradient.empty();
    p.x.resize(n * d), p.g.assign(n * d, 0), p.y.resize(n), p.w.assign(n, 1.0), p.hasGradient.assign(n, labels ? 1 : 0);
    for (std::size_t k = 0; k < n; ++k) {
      const std::size_t r = rows[k];
      for (std::size_t j = 0; j < d; ++j) p.x[k * d + j] = (o.at(r)[j] - p.xMean[j]) / p.xScale[j];
      p.y[k] = (o.payoff[r] - p.yMean) / p.yScale;
      if (labels)
        for (std::size_t j = 0; j < d; ++j) p.g[k * d + j] = o.gradientAt(r)[j] * p.xScale[j] / p.yScale;
    }
    p.lambda.assign(d, 1.0);
    p.alpha = 1;
    if (labels) {
      for (std::size_t j = 0; j < d; ++j) {
        double s = 0;
        for (std::size_t k = 0; k < n; ++k) s += p.g[k * d + j] * p.g[k * d + j];
        s /= static_cast<double>(n);
        p.lambda[j] = s > 1e-12 ? 1 / s : 1;
      }
      p.alpha = o_.valueWeight < 0 ? 1 / (1 + static_cast<double>(d)) : o_.valueWeight;
    }
    // initial weights: N(0, 1 / fan-in)
    p.theta.assign(net.parameters(), 0.0);
    SplitMix64 g(derive(o_.seed, -11, 0));
    for (std::size_t l = 1; l <= p.hiddenLayers; ++l) {
      const double sd = std::sqrt(1 / static_cast<double>(net.size[l - 1]));
      for (std::size_t q = 0; q < net.size[l] * net.size[l - 1]; ++q) p.theta[net.wo[l] + q] = sd * diffnet::gauss(g);
    }
    const double sdOut = std::sqrt(1 / static_cast<double>(o_.width));
    for (std::size_t i = 0; i < o_.width; ++i) p.theta[net.oo + i] = sdOut * diffnet::gauss(g);
    // the batch order (a Fisher-Yates shuffle of the running order every epoch) and the learning rates
    std::vector<std::uint32_t> ord(n);
    for (std::size_t k = 0; k < n; ++k) ord[k] = static_cast<std::uint32_t>(k);
    SplitMix64 sh(derive(o_.seed, -12, 0));
    for (std::size_t e = 0; e < p.epochs; ++e) {
      for (std::size_t i = n; i > 1; --i) std::swap(ord[i - 1], ord[sh.below(i)]);
      p.order.insert(p.order.end(), ord.begin(), ord.end());
    }
    const std::size_t total = p.steps();
    const double lrMin = 0.1 * o_.learningRate;
    for (std::size_t s = 0; s < total; ++s)
      p.learningRate.push_back(lrMin + 0.5 * (o_.learningRate - lrMin) * (1 + std::cos(diffnet::kPi * static_cast<double>(s) / static_cast<double>(total))));
    return p;
  }

  // Trains on the problem in double precision: Adam, each step's gradient the sum of kChunks chunks in order.
  static std::vector<double> train(const DifferentialProblem& p, std::vector<double>* stepLoss = nullptr) {
    const diffnet::Net net(p.inputs, p.hiddenLayers, p.width);
    std::vector<double> theta = p.theta, m(theta.size(), 0), v(theta.size(), 0), grad(theta.size());
    std::vector<std::vector<double>> part(diffnet::kChunks, std::vector<double>(theta.size()));
    std::vector<double> partLoss(diffnet::kChunks);
    if (stepLoss) stepLoss->clear();
    const std::size_t per = p.stepsPerEpoch();
    for (std::size_t s = 0; s < p.steps(); ++s) {
      const std::size_t start = (s / per) * p.rows + (s % per) * p.batch, count = std::min(p.batch, p.rows - (s % per) * p.batch);
      detail::parallelFor(diffnet::kChunks, -1, [&](std::size_t c) {
        std::fill(part[c].begin(), part[c].end(), 0.0);
        const std::size_t lo = start + count * c / diffnet::kChunks, hi = start + count * (c + 1) / diffnet::kChunks;
        partLoss[c] = lo < hi ? net.lossAndGradient(p, theta.data(), p.order.data(), lo, hi, part[c].data(), diffnet::scratch(net)) : 0.0;
      });
      std::fill(grad.begin(), grad.end(), 0.0);
      double loss = 0;
      for (std::size_t c = 0; c < diffnet::kChunks; ++c) {
        loss += partLoss[c];
        for (std::size_t q = 0; q < grad.size(); ++q) grad[q] += part[c][q];
      }
      if (stepLoss) stepLoss->push_back(loss);
      const double t = static_cast<double>(s + 1), bc1 = 1 - std::pow(diffnet::kAdamBeta1, t), bc2 = 1 - std::pow(diffnet::kAdamBeta2, t);
      for (std::size_t q = 0; q < theta.size(); ++q) {
        const double gq = grad[q] / static_cast<double>(count);
        m[q] = diffnet::kAdamBeta1 * m[q] + (1 - diffnet::kAdamBeta1) * gq;
        v[q] = diffnet::kAdamBeta2 * v[q] + (1 - diffnet::kAdamBeta2) * gq * gq;
        theta[q] -= p.learningRate[s] * (m[q] / bc1) / (std::sqrt(v[q] / bc2) + diffnet::kAdamEpsilon);
      }
    }
    return theta;
  }

  // Installs weights trained on `p` (here, or by the fused kernels).
  void adopt(const DifferentialProblem& p, std::vector<double> theta) {
    if (theta.size() != p.parameters()) throw std::invalid_argument("DifferentialProxy::adopt: wrong number of weights");
    net_ = std::make_shared<diffnet::Net>(p.inputs, p.hiddenLayers, p.width);
    theta_ = std::move(theta);
    xMean_ = p.xMean, xScale_ = p.xScale, lo_ = p.lo, hi_ = p.hi, yMean_ = p.yMean, yScale_ = p.yScale;
    derivatives_ = p.alpha < 1;
    cov_.clear();
    if (o_.estimationVariance) outputCovariance(p);
  }

  void fit(const OuterScenarios& o, const std::vector<std::size_t>& rows) override {
    const DifferentialProblem p = problem(o, rows);
    adopt(p, train(p));
  }
  void predict(const double* x, double& mu, double& v) const override { mu = value(x, nullptr, &v); }
  // The value and, if `gradient` is given, its gradient by the state (d numbers), in the original units; if `variance`
  // is given, its estimation variance (0 without estimationVariance).
  double value(const double* x, double* gradient, double* variance = nullptr) const {
    if (!net_) throw std::logic_error("DifferentialProxy: predict before fit");
    diffnet::Net::Work& work = diffnet::scratch(*net_);
    std::vector<double> z(net_->d);
    for (std::size_t j = 0; j < net_->d; ++j) z[j] = (x[j] - xMean_[j]) / xScale_[j];
    const double y = net_->forward(theta_.data(), z.data(), work);
    if (variance) {
      *variance = 0;
      if (!cov_.empty()) {
        const std::size_t q = net_->W + 1;
        const std::vector<double>& phi = work.a[net_->L];
        for (std::size_t a = 0; a < q; ++a) {
          double s = 0;
          for (std::size_t b = 0; b < q; ++b) s += cov_[a * q + b] * (b < net_->W ? phi[b] : 1.0);
          *variance += (a < net_->W ? phi[a] : 1.0) * s;
        }
        *variance = *variance > 0 ? *variance * yScale_ * yScale_ : 0;
      }
    }
    if (gradient) {
      net_->twin(theta_.data(), work);
      for (std::size_t j = 0; j < net_->d; ++j) gradient[j] = work.gb[0][j] * yScale_ / xScale_[j];
    }
    return yMean_ + yScale_ * y;
  }
  bool outsideTraining(const double* x) const override {
    for (std::size_t j = 0; j < lo_.size(); ++j)
      if (x[j] < lo_[j] || x[j] > hi_[j]) return true;
    return false;
  }
  std::string describe() const override {
    return std::string(derivatives_ ? "differential network " : "network on values ") + std::to_string(o_.hiddenLayers) + "x" + std::to_string(o_.width) +
           ", " + std::to_string(theta_.size()) + " weights";
  }

  const DifferentialOptions& options() const { return o_; }
  const std::vector<double>& weights() const { return theta_; }

  // The loss of `p` at weights theta over its rows `rows` (summed) and its exact gradient: the double-precision
  // reference for the kernels' per-sample gradients.
  static double lossAndGradient(const DifferentialProblem& p, const std::vector<double>& theta, const std::vector<std::uint32_t>& rows,
                                std::vector<double>& grad) {
    const diffnet::Net net(p.inputs, p.hiddenLayers, p.width);
    grad.assign(theta.size(), 0.0);
    return net.lossAndGradient(p, theta.data(), rows.data(), 0, rows.size(), grad.data(), diffnet::scratch(net));
  }

  // The largest relative difference between the loss's exact gradient (values and derivatives, second order included)
  // and central differences, on a small random network with some rows unweighted and some without derivatives.
  static double gradientCheck(std::uint64_t seed) {
    const std::size_t d = 3, n = 7;
    const diffnet::Net net(d, 2, 5);
    DifferentialProblem p;
    p.inputs = d, p.rows = n, p.hiddenLayers = 2, p.width = 5;
    SplitMix64 g(seed);
    p.theta.resize(net.parameters());
    for (double& t : p.theta) t = 0.7 * diffnet::gauss(g);
    p.lambda = {1.0, 0.5, 2.0};
    p.alpha = 0.3;
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t j = 0; j < d; ++j) p.x.push_back(diffnet::gauss(g)), p.g.push_back(diffnet::gauss(g));
      p.y.push_back(diffnet::gauss(g));
      p.w.push_back(0.5 + static_cast<double>(k % 3));
      p.hasGradient.push_back(k % 4 != 3);
    }
    std::vector<std::uint32_t> rows(n);
    for (std::size_t k = 0; k < n; ++k) rows[k] = static_cast<std::uint32_t>(k);
    std::vector<double> grad, scratch;
    lossAndGradient(p, p.theta, rows, grad);
    double worst = 0;
    const double h = 1e-6;
    for (std::size_t q = 0; q < p.theta.size(); ++q) {
      std::vector<double> t = p.theta;
      t[q] = p.theta[q] + h;
      const double up = lossAndGradient(p, t, rows, scratch);
      t[q] = p.theta[q] - h;
      const double down = lossAndGradient(p, t, rows, scratch);
      const double fd = (up - down) / (2 * h);
      worst = std::max(worst, std::fabs(fd - grad[q]) / std::max(1e-6, std::fabs(fd) + std::fabs(grad[q])));
    }
    return worst;
  }

 private:
  // HC0 of the output layer as a regression on phi = (last hidden layer, 1) over the problem's rows (weighted, a tiny
  // relative ridge), on the standardized scale.
  void outputCovariance(const DifferentialProblem& p) {
    const std::size_t q = net_->W + 1, n = p.rows;
    std::vector<double> g(q * q, 0), meat(q * q, 0), phi(q);
    diffnet::Net::Work& work = diffnet::scratch(*net_);
    for (std::size_t r = 0; r < n; ++r) {
      const double e = net_->forward(theta_.data(), p.x.data() + r * p.inputs, work) - p.y[r], w = p.w[r];
      for (std::size_t a = 0; a < q; ++a) phi[a] = a < net_->W ? work.a[net_->L][a] : 1.0;
      for (std::size_t a = 0; a < q; ++a)
        for (std::size_t b = 0; b <= a; ++b) g[a * q + b] += w * phi[a] * phi[b], meat[a * q + b] += w * w * e * e * phi[a] * phi[b];
    }
    double tr = 0;
    for (std::size_t a = 0; a < q; ++a) tr += g[a * q + a];
    for (std::size_t a = 0; a < q; ++a) {
      g[a * q + a] += 1e-10 * tr / static_cast<double>(q);
      for (std::size_t b = 0; b < a; ++b) g[b * q + a] = g[a * q + b], meat[b * q + a] = meat[a * q + b];
    }
    if (!detail::cholesky(g, q)) return;  // degenerate features: no variance
    std::vector<double> gm(q * q), col(q);
    for (std::size_t c = 0; c < q; ++c) {
      for (std::size_t a = 0; a < q; ++a) col[a] = meat[a * q + c];
      detail::cholSolve(g, q, col.data());
      for (std::size_t a = 0; a < q; ++a) gm[a * q + c] = col[a];
    }
    cov_.assign(q * q, 0);
    for (std::size_t c = 0; c < q; ++c) {
      for (std::size_t a = 0; a < q; ++a) col[a] = gm[c * q + a];
      detail::cholSolve(g, q, col.data());
      for (std::size_t a = 0; a < q; ++a) cov_[c * q + a] = col[a];
    }
    for (std::size_t a = 0; a < q; ++a)
      for (std::size_t b = 0; b < a; ++b) cov_[a * q + b] = cov_[b * q + a] = 0.5 * (cov_[a * q + b] + cov_[b * q + a]);
  }

  DifferentialOptions o_;
  std::shared_ptr<const diffnet::Net> net_;
  std::vector<double> theta_, xMean_, xScale_, lo_, hi_, cov_;
  double yMean_ = 0, yScale_ = 1;
  bool derivatives_ = false;
};

// For Options::proxy: a fresh differential proxy for each estimate.
inline ProxyFactory differentialProxy(DifferentialOptions o = DifferentialOptions()) {
  return [o](std::size_t) { return std::make_shared<DifferentialProxy>(o); };
}

}  // namespace anmc
