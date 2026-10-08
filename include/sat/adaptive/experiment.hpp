#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "sat/adaptive/self_adaptive.hpp"
#include "sat/data/market_data.hpp"
#include "sat/features/dataset.hpp"
#include "sat/ml/walk_forward.hpp"
#include "sat/strategy/strategy.hpp"

namespace sat {

/// The full method: alpha factors -> labels -> walk-forward machine-learning forecasts ->
/// a pool of fixed trading strategies on every model's forecasts -> the self-adaptive
/// strategy that keeps switching to the candidate with the best recent record.
struct ExperimentSpec {
  std::vector<int> alphaIds;           ///< empty: the paper's 23 alphas
  Normalisation normalisation = Normalisation::Rank;
  /// Extra features from Advances in Financial Machine Learning (afml::extraFeatureNames()).
  std::vector<std::string> extraFeatures;
  double ffdOrder = 0.4;               ///< order of the "ffd" feature
  /// Train on CUSUM events only (threshold = multiple x median daily volatility; 0 = every day).
  double cusumMultiple = 0.0;
  LabelSpec label;
  std::vector<ModelSpec> models;       ///< empty: defaultModels()
  WalkForwardSpec walkForward;
  std::vector<StrategySpec> strategies;  ///< empty: defaultStrategies()
  double costBps = 10.0;               ///< per unit of turnover, one way (HK stamp duty is 10 bp a side)
  SelectorSpec selector;

  static std::vector<ModelSpec> defaultModels();
  static std::vector<StrategySpec> defaultStrategies();
};

/// Features, labels and every model's out-of-sample predictions: the expensive part.
struct PredictionSet {
  FeatureSet features;
  Panel labels;
  Panel labelEnds;    ///< last date each label depends on
  Panel trainMask;    ///< CUSUM events (empty when every day is used)
  Panel nextReturns;  ///< close-to-close return of the following day
  std::vector<ModelPredictions> models;
};

PredictionSet runPredictions(const MarketData& data, const ExperimentSpec& spec);

struct Experiment {
  CandidateBook book;
  std::size_t evalFrom = 0;                       ///< first evaluated day (offset into the book)
  std::vector<PerformanceMetrics> candidateMetrics;  ///< fixed strategies over the evaluation days
  std::size_t bestFixed = 0;                      ///< candidate with the highest Sharpe ratio in hindsight
  BacktestResult benchmark;                       ///< equal-weight market over the evaluation days
  AdaptiveResult adaptive;
};

/// Backtests every candidate on the predictions and runs the selector. All strategies are
/// compared over the same days, from `evalFrom` (default: the selector's look-back) on.
Experiment runStrategies(const PredictionSet& predictions, const ExperimentSpec& spec, std::size_t evalFrom = SIZE_MAX);

}  // namespace sat
