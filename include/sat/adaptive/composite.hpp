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
  /// The self-adaptive forecast: the paper's self-adaptive rule applied to the forecasts
  /// themselves. Every `adaptEvery` dates each candidate (each member, and their equal-weight
  /// average) is scored by its mean daily cross-sectional rank correlation (information
  /// coefficient) with the labels over the last `lookback` resolved dates, and the best one
  /// is used until the next adaptation; with no candidate scoring above zero, the average is.
  Adaptive = 4,
};

CompositeMethod parseCompositeMethod(const std::string& name);
std::string compositeMethodName(CompositeMethod m);

/// examples/forecast_study: traded alone (keepMembers = false), the equal-weight average gave
/// the self-adaptive selector a higher Sharpe ratio than the separate models and than the
/// self-adaptive forecast, on synthetic selection and validation markets.
struct CompositeSpec {
  CompositeMethod method = CompositeMethod::None;
  bool keepMembers = true;      ///< ExperimentSpec: trade the members as well as the composite
  std::size_t lookback = 63;    ///< Adaptive: resolved dates scored (the selector's default)
  std::size_t adaptEvery = 21;  ///< Adaptive: dates between choices (the selector's default)
};

/// The composite forecast and the weight it gave each member over time.
struct CompositePredictions {
  ModelPredictions model;                     ///< usable like any member's predictions
  std::vector<std::string> members;
  std::vector<std::vector<double>> weights;   ///< [date - model.start][member]
  /// Adaptive: the candidate used each date (-1 = the average, otherwise the member).
  std::vector<int> choice;
};

/// Combines members' predictions over their common out-of-sample range. Only labels whose
/// last date (`labelEnds`, from makeLabels) is before a prediction date are used to set the
/// weights of that date. Members must all have the same panel shape.
CompositePredictions compositePredictions(const std::vector<ModelPredictions>& members, const Panel& labels,
                                          const Panel& labelEnds, const CompositeSpec& spec);

}  // namespace sat
