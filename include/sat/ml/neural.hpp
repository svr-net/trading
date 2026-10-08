#pragma once

#include <vector>

#include "sat/ml/classifier.hpp"

namespace sat {

/// One-hidden-layer perceptron (tanh) with a sigmoid output, trained by Adam on the
/// cross-entropy with L2 weight decay.
class Mlp : public Classifier {
 public:
  explicit Mlp(const ModelSpec& spec) : spec_(spec) {}
  void fit(const Matrix& X, const std::vector<double>& y) override;
  std::vector<double> predictProba(const Matrix& X) const override;

 private:
  double forward(const double* z, std::vector<double>& hidden) const;
  ModelSpec spec_;
  Standardizer std_;
  std::size_t in_ = 0, hid_ = 0;
  std::vector<double> W1_, b1_, W2_;  // W1: hid x in, W2: hid
  double b2_ = 0.0;
};

/// Long short-term memory network (Hochreiter and Schmidhuber, 1997): one LSTM layer read
/// over the last `seqLen` daily feature vectors of a stock, then a sigmoid output on the
/// final hidden state. Trained by back-propagation through time with Adam.
/// Input rows hold the sequence oldest first (see assemble() with lags = seqLen).
class Lstm : public Classifier {
 public:
  explicit Lstm(const ModelSpec& spec) : spec_(spec) {}
  void fit(const Matrix& X, const std::vector<double>& y) override;
  std::vector<double> predictProba(const Matrix& X) const override;

 private:
  double forward(const double* z, std::vector<double>* cache) const;
  ModelSpec spec_;
  Standardizer std_;
  std::size_t steps_ = 0, in_ = 0, hid_ = 0;
  // Gate weights, 4 blocks (input, forget, cell, output) of hid rows over [x, h_prev, 1].
  std::vector<double> W_;
  std::vector<double> V_;  // output weights over [h, 1]
};

}  // namespace sat
