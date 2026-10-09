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

/// How the self-adaptive forecast remembers each candidate's record.
enum class ScoringWindow : int {
  /// The last `lookback` resolved dates, re-scored every `adaptEvery` dates (the selector's rule).
  Fixed = 0,
  /// Progressive and incremental: every resolved date counts, with weights halving every
  /// `halfLife` dates (0: no decay, an expanding window); updated daily as labels resolve.
  Exponential = 1,
  /// Progressive, incremental and adaptive: ADWIN (Bifet and Gavalda, 2007). The window grows
  /// by each newly resolved date and drops its older part whenever that part's mean differs
  /// from the recent part's by more than chance allows (confidence `adwinDelta`), so memory is
  /// long while a candidate's record is stable and short after it changes. Updated daily.
  Adwin = 2,
};

/// How the scores pick the forecast.
enum class ScoringDecision : int {
  /// The candidate (member or average) with the highest mean information coefficient; the
  /// average when none is positive.
  Best = 0,
  /// A member only while its lead in information coefficient over the average is significant
  /// (t-statistic of the mean daily difference above `minT`); otherwise the average.
  Evidence = 1,
};

ScoringWindow parseScoringWindow(const std::string& name);
std::string scoringWindowName(ScoringWindow w);

/// examples/forecast_study: traded alone (keepMembers = false), the equal-weight average gave
/// the self-adaptive selector a higher Sharpe ratio than the separate models and than the
/// self-adaptive forecast, on synthetic selection and validation markets.
struct CompositeSpec {
  CompositeMethod method = CompositeMethod::None;
  bool keepMembers = true;      ///< ExperimentSpec: trade the members as well as the composite
  std::size_t lookback = 63;    ///< Adaptive: resolved dates scored (the selector's default)
  std::size_t adaptEvery = 21;  ///< Adaptive: dates between choices (the selector's default)
  ScoringWindow window = ScoringWindow::Fixed;
  ScoringDecision decision = ScoringDecision::Best;
  double halfLife = 63.0;       ///< Exponential: dates for a weight to halve (0 = expanding)
  double adwinDelta = 0.002;    ///< Adwin: confidence of a cut
  double minT = 2.0;            ///< Evidence: t-statistic a member's lead must exceed
};

/// The composite forecast and the weight it gave each member over time.
struct CompositePredictions {
  ModelPredictions model;                     ///< usable like any member's predictions
  std::vector<std::string> members;
  std::vector<std::vector<double>> weights;   ///< [date - model.start][member]
  /// Adaptive: the candidate used each date (-1 = the average, otherwise the member).
  std::vector<int> choice;
  /// Adaptive: the scoring memory each date (resolved dates in the window, or the effective
  /// number for exponential weights), for the first scored series.
  std::vector<double> memory;
};

/// Combines members' predictions over their common out-of-sample range. Only labels whose
/// last date (`labelEnds`, from makeLabels) is before a prediction date are used to set the
/// weights of that date. Members must all have the same panel shape.
CompositePredictions compositePredictions(const std::vector<ModelPredictions>& members, const Panel& labels,
                                          const Panel& labelEnds, const CompositeSpec& spec);

}  // namespace sat
