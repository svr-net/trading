#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sat/core/matrix.hpp"
#include "sat/core/random.hpp"
#include "sat/ml/classifier.hpp"

namespace sat {

/// Quantile binning of each feature into at most `maxBins` bins, the basis of histogram
/// split finding (as in XGBoost's hist method and LightGBM). Bin b holds the values
/// x <= thresholds[b] above the previous threshold; the last bin holds the rest.
class FeatureBinner {
 public:
  void fit(const Matrix& X, std::size_t maxBins);
  std::size_t numFeatures() const { return thresholds_.size(); }
  std::size_t numBins(std::size_t f) const { return thresholds_[f].size() + 1; }
  double threshold(std::size_t f, std::size_t bin) const { return thresholds_[f][bin]; }
  std::uint8_t code(std::size_t f, double x) const;
  /// Column-major codes: codes[f * rows + r].
  std::vector<std::uint8_t> transform(const Matrix& X) const;

 private:
  std::vector<std::vector<double>> thresholds_;
};

struct TreeNode {
  int feature = -1;  ///< -1 for a leaf
  double threshold = 0.0;
  int left = -1, right = -1;
  double value = 0.0;
};

struct RegressionTree {
  std::vector<TreeNode> nodes;
  double predict(const double* x) const;
};

struct TreeParams {
  std::size_t maxDepth = 3;
  std::size_t maxLeaves = 0;       ///< 0: no limit
  double lambda = 1.0;
  double gamma = 0.0;
  double minChildWeight = 1.0;
  bool leafWise = false;           ///< LightGBM: always split the leaf with the largest gain
  double featureFractionPerNode = 1.0;  ///< random forest: features tried at each split
};

/// Grows a second-order regression tree on gradients g and hessians h (XGBoost's
/// formulation): leaf value -G / (H + lambda), split gain
/// 1/2 [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)] - gamma.
/// Squared loss with g = -y, h = 1 and lambda = 0 gives an ordinary CART regression tree.
/// `rows` are the training rows used, `features` the columns allowed; split gains are
/// added to `importance`.
RegressionTree growTree(const FeatureBinner& binner, const std::vector<std::uint8_t>& codes, std::size_t numRows,
                        const std::vector<double>& g, const std::vector<double>& h, const std::vector<std::uint32_t>& rows,
                        const std::vector<std::size_t>& features, const TreeParams& params, Rng& rng,
                        std::vector<double>& importance);

/// Random forest of CART trees on bootstrap samples (Breiman, 2001).
class RandomForest : public Classifier {
 public:
  explicit RandomForest(const ModelSpec& spec, bool single = false) : spec_(spec), single_(single) {}
  void fit(const Matrix& X, const std::vector<double>& y) override;
  std::vector<double> predictProba(const Matrix& X) const override;
  std::vector<double> featureImportance() const override { return importance_; }

 private:
  ModelSpec spec_;
  bool single_;
  std::vector<RegressionTree> trees_;
  std::vector<double> importance_;
};

/// Gradient-boosted trees on the logistic loss. Depth-wise growth reproduces XGBoost
/// (Chen and Guestrin, 2016); leaf-wise growth with a leaf budget reproduces LightGBM
/// (Ke et al., 2017).
class GradientBoosting : public Classifier {
 public:
  GradientBoosting(const ModelSpec& spec, bool leafWise) : spec_(spec), leafWise_(leafWise) {}
  void fit(const Matrix& X, const std::vector<double>& y) override;
  std::vector<double> predictProba(const Matrix& X) const override;
  std::vector<double> featureImportance() const override { return importance_; }
  /// Raw margin (log-odds) of one row.
  double margin(const double* x) const;

 private:
  ModelSpec spec_;
  bool leafWise_;
  double base_ = 0.0;
  std::vector<RegressionTree> trees_;
  std::vector<double> importance_;
};

}  // namespace sat
