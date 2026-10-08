#pragma once

#include <vector>

#include "sat/ml/classifier.hpp"

namespace sat {

/// L2-regularised logistic regression fitted by Newton's method (IRLS) on standardised inputs.
class LogisticRegression : public Classifier {
 public:
  explicit LogisticRegression(const ModelSpec& spec) : spec_(spec) {}
  void fit(const Matrix& X, const std::vector<double>& y) override;
  std::vector<double> predictProba(const Matrix& X) const override;
  std::vector<double> featureImportance() const override;
  const std::vector<double>& coefficients() const { return w_; }  ///< intercept last

 private:
  ModelSpec spec_;
  Standardizer std_;
  std::vector<double> w_;
};

/// Linear support vector machine trained with Pegasos (stochastic sub-gradient on the
/// hinge loss), with Platt scaling of the margin into a probability.
class LinearSvm : public Classifier {
 public:
  explicit LinearSvm(const ModelSpec& spec) : spec_(spec) {}
  void fit(const Matrix& X, const std::vector<double>& y) override;
  std::vector<double> predictProba(const Matrix& X) const override;
  std::vector<double> featureImportance() const override;

 private:
  double margin(const double* z) const;
  ModelSpec spec_;
  Standardizer std_;
  std::vector<double> w_;
  double b_ = 0.0, plattA_ = 1.0, plattB_ = 0.0;
};

}  // namespace sat
