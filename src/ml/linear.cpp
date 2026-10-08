#include "sat/ml/linear.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "sat/core/random.hpp"

namespace sat {

void Standardizer::fit(const Matrix& X) {
  const std::size_t F = X.cols();
  mean.assign(F, 0.0);
  scale.assign(F, 1.0);
  if (X.rows() == 0) return;
  for (std::size_t r = 0; r < X.rows(); ++r)
    for (std::size_t f = 0; f < F; ++f) mean[f] += X(r, f);
  for (double& m : mean) m /= static_cast<double>(X.rows());
  std::vector<double> var(F, 0.0);
  for (std::size_t r = 0; r < X.rows(); ++r)
    for (std::size_t f = 0; f < F; ++f) var[f] += (X(r, f) - mean[f]) * (X(r, f) - mean[f]);
  for (std::size_t f = 0; f < F; ++f) {
    const double sd = std::sqrt(var[f] / static_cast<double>(X.rows()));
    scale[f] = sd > 1e-12 ? 1.0 / sd : 1.0;
  }
}

void Standardizer::apply(const double* in, double* out) const {
  for (std::size_t f = 0; f < mean.size(); ++f) out[f] = (in[f] - mean[f]) * scale[f];
}

Matrix Standardizer::transform(const Matrix& X) const {
  Matrix Z(X.rows(), X.cols());
  for (std::size_t r = 0; r < X.rows(); ++r) apply(X.row(r), Z.row(r));
  return Z;
}

namespace {

void checkData(const Matrix& X, const std::vector<double>& y, const char* who) {
  if (X.rows() == 0 || X.rows() != y.size()) throw std::invalid_argument(std::string(who) + ": empty or mismatched training data");
}

std::vector<double> absNormalised(const std::vector<double>& w, std::size_t n) {
  std::vector<double> imp(n);
  for (std::size_t f = 0; f < n; ++f) imp[f] = std::fabs(w[f]);
  const double s = std::accumulate(imp.begin(), imp.end(), 0.0);
  if (s > 0)
    for (double& v : imp) v /= s;
  return imp;
}

}  // namespace

void LogisticRegression::fit(const Matrix& X, const std::vector<double>& y) {
  checkData(X, y, "logistic regression");
  std_.fit(X);
  const Matrix Z = std_.transform(X);
  const std::size_t F = Z.cols(), D = F + 1, n = Z.rows();
  w_.assign(D, 0.0);
  const double l2 = std::max(spec_.l2, 1e-8);
  std::vector<double> p(n), grad(D);
  for (int iter = 0; iter < 30; ++iter) {
    Matrix H(D, D, 0.0);
    std::fill(grad.begin(), grad.end(), 0.0);
    for (std::size_t r = 0; r < n; ++r) {
      const double* z = Z.row(r);
      double m = w_[F];
      for (std::size_t f = 0; f < F; ++f) m += w_[f] * z[f];
      const double pr = sigmoid(m), e = pr - y[r], wt = std::max(pr * (1.0 - pr), 1e-9);
      for (std::size_t a = 0; a < D; ++a) {
        const double za = a < F ? z[a] : 1.0;
        grad[a] += e * za;
        for (std::size_t b = 0; b <= a; ++b) H(a, b) += wt * za * (b < F ? z[b] : 1.0);
      }
    }
    for (std::size_t a = 0; a < D; ++a) {
      for (std::size_t b = 0; b < a; ++b) H(b, a) = H(a, b);
      if (a < F) {
        H(a, a) += l2;
        grad[a] += l2 * w_[a];
      } else {
        H(a, a) += 1e-9;
      }
    }
    const auto step = solveSpd(H, grad);
    double norm = 0.0;
    for (std::size_t a = 0; a < D; ++a) {
      w_[a] -= step[a];
      norm = std::max(norm, std::fabs(step[a]));
    }
    if (norm < 1e-8) break;
  }
}

std::vector<double> LogisticRegression::predictProba(const Matrix& X) const {
  if (w_.empty()) throw std::logic_error("logistic regression is not fitted");
  const std::size_t F = X.cols();
  std::vector<double> out(X.rows()), z(F);
  for (std::size_t r = 0; r < X.rows(); ++r) {
    std_.apply(X.row(r), z.data());
    double m = w_[F];
    for (std::size_t f = 0; f < F; ++f) m += w_[f] * z[f];
    out[r] = sigmoid(m);
  }
  return out;
}

std::vector<double> LogisticRegression::featureImportance() const { return absNormalised(w_, w_.empty() ? 0 : w_.size() - 1); }

double LinearSvm::margin(const double* z) const {
  double m = b_;
  for (std::size_t f = 0; f < w_.size(); ++f) m += w_[f] * z[f];
  return m;
}

void LinearSvm::fit(const Matrix& X, const std::vector<double>& y) {
  checkData(X, y, "linear SVM");
  std_.fit(X);
  const Matrix Z = std_.transform(X);
  const std::size_t F = Z.cols(), n = Z.rows();
  w_.assign(F, 0.0);
  b_ = 0.0;
  const double lambda = std::max(spec_.l2, 1e-6) / static_cast<double>(n);
  Rng rng(spec_.seed);
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::size_t t = 0;
  for (std::size_t epoch = 0; epoch < std::max<std::size_t>(1, spec_.epochs); ++epoch) {
    shuffle(order, rng);
    for (std::size_t r : order) {
      ++t;
      const double eta = 1.0 / (lambda * static_cast<double>(t + 100));
      const double s = y[r] > 0.5 ? 1.0 : -1.0;
      const double m = margin(Z.row(r));
      for (double& w : w_) w *= 1.0 - eta * lambda;
      if (s * m < 1.0) {
        const double* z = Z.row(r);
        const double step = std::min(eta, 1.0);
        for (std::size_t f = 0; f < F; ++f) w_[f] += step * s * z[f];
        b_ += step * s * 0.1;
      }
    }
  }
  // Platt scaling: P(y = 1) = sigmoid(a m + b), fitted by Newton's method.
  std::vector<double> m(n);
  for (std::size_t r = 0; r < n; ++r) m[r] = margin(Z.row(r));
  plattA_ = 1.0;
  plattB_ = 0.0;
  for (int iter = 0; iter < 50; ++iter) {
    double g0 = 0, g1 = 0, h00 = 1e-9, h01 = 0, h11 = 1e-9;
    for (std::size_t r = 0; r < n; ++r) {
      const double p = sigmoid(plattA_ * m[r] + plattB_), e = p - y[r], w = p * (1 - p);
      g0 += e * m[r];
      g1 += e;
      h00 += w * m[r] * m[r];
      h01 += w * m[r];
      h11 += w;
    }
    const double det = h00 * h11 - h01 * h01;
    if (!(std::fabs(det) > 1e-300)) break;
    const double da = (h11 * g0 - h01 * g1) / det, db = (h00 * g1 - h01 * g0) / det;
    plattA_ -= da;
    plattB_ -= db;
    if (std::fabs(da) + std::fabs(db) < 1e-9) break;
  }
}

std::vector<double> LinearSvm::predictProba(const Matrix& X) const {
  if (w_.empty()) throw std::logic_error("linear SVM is not fitted");
  std::vector<double> out(X.rows()), z(X.cols());
  for (std::size_t r = 0; r < X.rows(); ++r) {
    std_.apply(X.row(r), z.data());
    out[r] = sigmoid(plattA_ * margin(z.data()) + plattB_);
  }
  return out;
}

std::vector<double> LinearSvm::featureImportance() const { return absNormalised(w_, w_.size()); }

}  // namespace sat
