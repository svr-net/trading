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
  /// Market-driven: as Adwin, and a second ADWIN watches the market itself (the size of the
  /// equal-weight market's daily moves, i.e. its volatility state). Memory grows while the
  /// market stays in one state and recedes to the start of the new state when it changes.
  MarketAdwin = 3,
  /// Market-driven: every recorded date counts in proportion to how closely its market state
  /// (21-day market volatility and 63-day market trend) resembles today's, with a Gaussian
  /// kernel of width `stateBandwidth` (in units of each feature's spread). Memory recedes to
  /// the dates that match the current state, so a forecast that does well in calm markets and
  /// another that does well in turbulent ones are each used in their own state.
  SimilarState = 4,
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

/// examples/forecast_study: over the two models and the four market-state specialists
/// (ExperimentSpec::stateSpecialists), the self-adaptive forecast with MarketAdwin memory and
/// the Evidence decision scored highest on the selection markets and lifted the selector's
/// Sharpe ratio on unseen markets from 1.57 (the average of the two models) to 2.00.
struct CompositeSpec {
  CompositeMethod method = CompositeMethod::None;
  bool keepMembers = true;      ///< ExperimentSpec: trade the members as well as the composite
  std::size_t lookback = 63;    ///< Adaptive: resolved dates scored (the selector's default)
  std::size_t adaptEvery = 21;  ///< Adaptive: dates between choices (the selector's default)
  ScoringWindow window = ScoringWindow::Fixed;
  ScoringDecision decision = ScoringDecision::Best;
  double halfLife = 63.0;       ///< Exponential: dates for a weight to halve (0 = expanding)
  double adwinDelta = 1e-4;     ///< Adwin, MarketAdwin: confidence of a cut
  double minT = 2.0;            ///< Evidence: t-statistic a member's lead must exceed
  double stateBandwidth = 1.0;  ///< SimilarState: kernel width in units of each state feature's spread
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

/// The market state of every date, from the equal-weight market's returns up to that date's
/// close: the log of its 21-day volatility and its 63-day trend (sum of returns over their
/// volatility times sqrt(63)). NaN for the first 63 dates.
struct MarketState {
  std::vector<double> volatility, trend;
};
MarketState marketState(const Panel& nextReturns);

/// Which side of a market-state feature a specialist model is trained on.
enum class StateSide : int { Calm = 0, Turbulent = 1, Rising = 2, Falling = 3 };
std::string stateSideName(StateSide s);

/// Training mask (dates x assets, 1 = train on it) of the dates in one market state: calm or
/// turbulent (21-day market volatility below or above its median so far), rising or falling
/// (63-day market trend above or below zero). Uses only information up to each date.
Panel marketStateMask(const Panel& nextReturns, StateSide side);

/// Combines members' predictions over their common out-of-sample range. Only labels whose
/// last date (`labelEnds`, from makeLabels) is before a prediction date are used to set the
/// weights of that date. Members must all have the same panel shape. `nextReturns` (the
/// close-to-close return of the following day, as in PredictionSet) is needed for the
/// market-driven scoring windows; the market state of a date uses returns up to its close.
CompositePredictions compositePredictions(const std::vector<ModelPredictions>& members, const Panel& labels,
                                          const Panel& labelEnds, const CompositeSpec& spec, const Panel* nextReturns = nullptr);

}  // namespace sat
