#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sat/core/matrix.hpp"

namespace sat {

/// A probabilistic binary classifier: fit on rows of X with labels in {0, 1}, then
/// predict P(y = 1 | x) for new rows.
class Classifier {
 public:
  virtual ~Classifier() = default;
  virtual void fit(const Matrix& X, const std::vector<double>& y) = 0;
  virtual std::vector<double> predictProba(const Matrix& X) const = 0;
  /// Relative importance of each input column (empty if the model has none); sums to 1.
  virtual std::vector<double> featureImportance() const { return {}; }
};

/// Settings of every model family; each family reads the fields it uses.
struct ModelSpec {
  /// logistic | svm | tree | forest | xgboost | lightgbm | mlp | lstm
  std::string type = "xgboost";
  std::string name;  ///< display name (defaults to the family name)

  // Linear models
  double l2 = 1.0;               ///< ridge penalty (logistic), lambda (SVM)
  // Trees and ensembles
  std::size_t trees = 60;
  std::size_t maxDepth = 3;
  std::size_t maxLeaves = 0;     ///< 0: no limit (LightGBM grows leaf-wise up to this count)
  std::size_t minLeaf = 20;      ///< minimum samples (forest) / hessian weight x 4 (boosting) per leaf
  double learningRate = 0.1;     ///< shrinkage (boosting) or Adam step size (networks)
  double lambda = 1.0;           ///< L2 penalty on leaf values (boosting)
  double gamma = 0.0;            ///< minimum split gain (boosting)
  double subsample = 0.8;        ///< rows per tree (bootstrap fraction for the forest)
  double colsample = 0.8;        ///< features per tree (boosting) or per split (forest)
  std::size_t bins = 32;         ///< histogram bins per feature
  // Neural networks
  std::size_t hidden = 16;
  std::size_t epochs = 5;
  std::size_t batch = 64;
  std::size_t seqLen = 5;        ///< LSTM input sequence length (days)
  std::size_t maxSamples = 0;    ///< networks: samples per epoch (0 = all)
  double weightDecay = 1e-4;     ///< networks: L2 penalty on the weights

  std::uint64_t seed = 1;

  /// Feature vectors per sample: the sequence length for an LSTM, 1 otherwise.
  std::size_t lags() const { return type == "lstm" ? (seqLen < 1 ? 1 : seqLen) : 1; }
  std::string displayName() const;
};

/// Builds an untrained classifier. Throws std::invalid_argument for an unknown type.
std::unique_ptr<Classifier> makeClassifier(const ModelSpec& spec);

/// Model families, in display order.
const std::vector<std::string>& modelTypes();

/// Per-column standardisation fitted on training data (zero-variance columns pass through).
struct Standardizer {
  std::vector<double> mean, scale;
  void fit(const Matrix& X);
  Matrix transform(const Matrix& X) const;
  void apply(const double* in, double* out) const;
};

inline double sigmoid(double z) { return z >= 0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z)); }

}  // namespace sat
