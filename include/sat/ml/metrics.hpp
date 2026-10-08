#pragma once

#include <cstddef>
#include <vector>

namespace sat {

/// Out-of-sample quality of probabilistic predictions of a binary label.
struct ClassificationMetrics {
  std::size_t count = 0;
  double accuracy = 0.0;   ///< at the 0.5 threshold
  double precision = 0.0;  ///< of the "up" predictions
  double recall = 0.0;
  double f1 = 0.0;
  double auc = 0.0;        ///< area under the ROC curve (0.5 = no skill)
  double logLoss = 0.0;
  double baseRate = 0.0;   ///< share of label 1
};

/// Metrics over the pairs where both label and prediction are finite.
ClassificationMetrics classificationMetrics(const std::vector<double>& y, const std::vector<double>& p);

/// ROC curve as (false positive rate, true positive rate) pairs at up to `points` thresholds.
std::vector<std::vector<double>> rocCurve(const std::vector<double>& y, const std::vector<double>& p, std::size_t points = 50);

}  // namespace sat
