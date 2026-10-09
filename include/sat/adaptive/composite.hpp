#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "sat/core/panel.hpp"
#include "sat/ml/walk_forward.hpp"

namespace sat {

/// How the composite model combines the out-of-sample probabilities of its member models.
enum class CompositeMethod : int {
  None = 0,
  /// Equal-weight mean of the members' probabilities. The "forecast combination puzzle"
  /// (Stock and Watson, 2004; Rapach, Strauss and Zhou, 2010 for stock returns): estimated
  /// combination weights rarely beat the simple average out of sample.
  Average = 1,
  /// Stacked generalisation (Wolpert, 1992; Breiman, 1996): a logistic meta-learner on the
  /// members' log-odds, re-fitted on a rolling window of resolved labels, with non-negative
  /// weights and a ridge penalty that shrinks them towards the equal-weight average.
  /// The members' predictions are themselves out of sample, so the meta-learner never sees
  /// in-sample fits.
  Stacked = 2,
  /// Online expert aggregation with Bernstein Online Aggregation (Wintenberger, 2017), as
  /// applied to stock-return forecasts by Remlinger, Alasseur, Briere and Mikael (2023):
  /// multiplicative weights on each member's linearised log loss with a second-order
  /// correction and per-member adaptive learning rates; updated daily as labels resolve.
  Online = 3,
};

CompositeMethod parseCompositeMethod(const std::string& name);
std::string compositeMethodName(CompositeMethod m);

/// examples/composite_study: the equal-weight average, traded alone (keepMembers = false), gave
/// the self-adaptive selector a higher Sharpe ratio than the separate models on synthetic
/// selection and validation markets and on LSE stocks; the stacked meta-learner was worst.
struct CompositeSpec {
  CompositeMethod method = CompositeMethod::None;
  std::size_t window = 252;     ///< Stacked: dates of resolved labels in each fit
  std::size_t refitEvery = 21;  ///< Stacked: dates between fits
  double ridge = 1.0;           ///< Stacked: penalty on (w - 1/M)^2 per 1,000 training rows
  std::size_t maxRows = 20000;  ///< Stacked: rows per fit (an even sub-sample of the window)
  bool keepMembers = true;      ///< ExperimentSpec: trade the members as well as the composite
};

/// The composite forecast and the weight it gave each member over time.
struct CompositePredictions {
  ModelPredictions model;                     ///< usable like any member's predictions
  std::vector<std::string> members;
  std::vector<std::vector<double>> weights;   ///< [date - model.start][member]
};

/// Combines members' predictions over their common out-of-sample range. Only labels whose
/// last date (`labelEnds`, from makeLabels) is before a prediction date are used to set the
/// weights of that date. Members must all have the same panel shape.
CompositePredictions compositePredictions(const std::vector<ModelPredictions>& members, const Panel& labels,
                                          const Panel& labelEnds, const CompositeSpec& spec);

}  // namespace sat
