#pragma once

#include <cstdint>
#include <vector>

#include "sat/afml/sampling.hpp"
#include "sat/core/matrix.hpp"
#include "sat/ml/classifier.hpp"

namespace sat::afml {

/// Cross-validated score of a model over the given splits: mean out-of-fold accuracy, AUC
/// and log loss, and the per-fold accuracies.
struct CvScore {
  double accuracy = 0, auc = 0, logLoss = 0;
  std::vector<double> foldAccuracy;
};

CvScore crossValidate(const ModelSpec& model, const Matrix& X, const std::vector<double>& y, const std::vector<Split>& splits);

/// Feature importance with its spread across folds or trees.
struct Importance {
  std::vector<double> mean, sd;
};

/// Mean decrease impurity: the split-gain importance of a tree ensemble fitted in sample
/// (fast, but biased towards features with many split points, and blind to substitution).
Importance meanDecreaseImpurity(const ModelSpec& model, const Matrix& X, const std::vector<double>& y);

/// Mean decrease accuracy: per fold, the rise in out-of-fold log loss when one feature's
/// column is shuffled; works with any model and measures out-of-sample usefulness.
Importance meanDecreaseAccuracy(const ModelSpec& model, const Matrix& X, const std::vector<double>& y,
                                const std::vector<Split>& splits, std::uint64_t seed = 3);

/// Single feature importance: the cross-validated AUC of a model trained on each feature
/// alone (no substitution effects, but no interactions either).
Importance singleFeatureImportance(const ModelSpec& model, const Matrix& X, const std::vector<double>& y,
                                   const std::vector<Split>& splits);

}  // namespace sat::afml
