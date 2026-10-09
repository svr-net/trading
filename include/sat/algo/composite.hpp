#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "sat/adaptive/experiment.hpp"
#include "sat/algo/ensemble.hpp"
#include "sat/algo/regimes.hpp"
#include "sat/core/panel.hpp"
#include "sat/data/market_data.hpp"
#include "sat/strategy/performance.hpp"

namespace sat::algo {

/// How the composite strategy weighs its signals.
enum class SignalWeighting : int {
  /// The same weight for every signal (the forecast-combination default).
  Equal = 0,
  /// Exponential weights on each signal's trailing information coefficient: weight in
  /// proportion to exp(eta x t-statistic of the daily rank correlation between the signal and
  /// the next day's returns over the look-back), so a signal that stops working fades out.
  Adaptive = 1,
};

SignalWeighting parseSignalWeighting(const std::string& name);
std::string signalWeightingName(SignalWeighting w);

/// Multi-signal stock selection: a long-only book with risk overlays, combining what the
/// library found to work on its own:
///  - signals, each turned into a cross-sectional percentile rank every day:
///    the machine-learning forecast (P(up), e.g. the composite model's), cross-sectional
///    momentum (Jegadeesh and Titman, 1993: the return from t - momentumLookback to
///    t - momentumSkip) and the trailing Sharpe ratio of each stock;
///  - the blended rank picks `holdings` stocks every `rebalanceEvery` days, held with
///    inverse-volatility weights;
///  - a volatility target on the book's own realised volatility (Barroso and Santa-Clara,
///    2015, "Momentum has its moments"), capped at `maxLeverage`;
///  - an optional regime gate: the HMM regime switch on the equal-weight market sets the
///    exposure to `regimes.riskOffExposure` in the high-volatility state (in the spirit of
///    Daniel and Moskowitz, 2016, "Momentum crashes").
/// The defaults are the variant examples/composite_study ranked first of 56 on synthetic
/// markets (all three signals, adaptive weights, 10 holdings, both overlays).
struct MultiSignalSpec {
  bool useMl = true, useMomentum = true, useTrailingSharpe = true;
  std::size_t momentumLookback = 252, momentumSkip = 21, sharpeLookback = 126;
  SignalWeighting weighting = SignalWeighting::Adaptive;
  std::size_t icLookback = 126;
  double eta = 0.5;
  std::size_t holdings = 10, rebalanceEvery = 5, volLookback = 63;
  double targetVol = 0.15;   ///< annualised; 0 = no volatility target
  double maxLeverage = 1.5;
  bool regimeGate = true;
  RegimeSwitchSpec regimes;
  double costBps = 10.0;
};

struct MultiSignalResult {
  std::vector<std::string> signals;           ///< the signals used, in order
  std::vector<double> returns;                ///< net; returns[k] is earned from date start + k to start + k + 1
  std::vector<double> gross, turnover, exposure;
  std::vector<std::vector<double>> signalWeights;  ///< [k][signal]
  std::vector<double> holdings;               ///< weights per stock held after the last date (with exposure)
  std::vector<double> book;                   ///< the last selection's weights before exposure (sum 1)
  std::vector<double> lastScore;              ///< blended rank per stock on the last date
  std::size_t start = 0;
  PerformanceMetrics metrics;
};

/// Runs the strategy over dates [start, end) (end = 0: the last date with a next-day return).
/// `probability` (dates x stocks, NaN where unknown) is needed when useMl is set; dates where
/// it is missing for every stock are traded on the other signals.
MultiSignalResult multiSignalSelection(const MarketData& data, const Panel* probability, const MultiSignalSpec& spec,
                                       std::size_t start, std::size_t end = 0);

/// The composite strategy: exponential-weights allocation (the Hedge algorithm, as in the
/// strategy tournament) between low-correlated sleeves:
///  - the self-adaptive selector over the trading rules on the composite model's forecasts
///    (the experiment's composite model when it has one, otherwise the equal-weight average
///    of its models);
///  - one or more stock-selection books (multiSignalSelection) on the same composite
///    forecast, momentum and trailing Sharpe, with their volatility target and regime gate;
///  - optionally the equal-weight market itself.
/// The selector chases the short-horizon forecast; the books hold slower signals and step
/// aside in turbulent regimes, so the allocator can lean on whichever is working.
struct CompositeStrategySpec {
  std::vector<MultiSignalSpec> books = {MultiSignalSpec{}};
  bool marketSleeve = false;
  AllocationSpec allocation;
};

/// Name of a stock-selection book from its signals, e.g. "Momentum book" or
/// "Multi-signal book (ML + momentum + trailing Sharpe)".
std::string bookName(const MultiSignalSpec& spec);

struct CompositeStrategyResult {
  /// The sleeves (selector, books, market when allocated to), then the composite strategy
  /// and the equal-weight market benchmark.
  std::vector<std::string> names;
  std::vector<std::vector<double>> returns;    ///< per name, over the common evaluation days
  std::vector<PerformanceMetrics> metrics;
  std::size_t sleeves = 0;                     ///< the first `sleeves` names are allocated to
  std::vector<std::vector<double>> weights;    ///< allocation to each sleeve per day
  std::vector<double> cash;
  std::vector<MultiSignalResult> books;        ///< each book's full result (holdings, signal weights)
  AdaptiveResult selector;
  std::vector<std::string> selectorCandidates;
  std::size_t start = 0;                       ///< date index of the first evaluated return
};

/// The selector starts trading after its look-back on the composite model's out-of-sample
/// dates, the multi-signal sleeve on the same day; the allocation then needs its own
/// look-back, after which every series is evaluated over the same days.
CompositeStrategyResult runCompositeStrategy(const MarketData& data, const PredictionSet& predictions, const ExperimentSpec& experiment,
                                             const CompositeStrategySpec& spec);

/// The same on a given composite forecast (e.g. one of compositePredictions' methods).
CompositeStrategyResult runCompositeStrategy(const MarketData& data, const ModelPredictions& composite, const Panel& nextReturns,
                                             const ExperimentSpec& experiment, const CompositeStrategySpec& spec);

}  // namespace sat::algo
