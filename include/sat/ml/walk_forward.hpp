#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sat/core/panel.hpp"
#include "sat/features/dataset.hpp"
#include "sat/ml/classifier.hpp"
#include "sat/ml/metrics.hpp"

namespace sat {

/// Rolling-window re-training schedule.
///
/// The model is re-fitted every `retrainEvery` dates on the labelled samples of the
/// previous `trainWindow` dates and predicts until the next re-fit, so every prediction is
/// out of sample. Training data stop `label.lookahead()` dates before the first prediction
/// date, so no training label overlaps the prediction period (purging).
/// How training samples are weighted (Lopez de Prado, 2018, ch. 4).
enum class SampleWeighting {
  None,
  /// By average uniqueness: labels whose spans overlap many others count less.
  Uniqueness,
  /// By uniqueness, decaying linearly with age along cumulative uniqueness.
  UniquenessDecay,
};

SampleWeighting parseSampleWeighting(const std::string& name);

struct WalkForwardSpec {
  std::size_t trainWindow = 504;
  std::size_t retrainEvery = 63;
  std::size_t testStart = 0;     ///< first prediction date (0: as early as the window allows)
  std::size_t maxTrainRows = 8000;  ///< random subsample of the training rows (0: all)
  std::uint64_t seed = 11;
  /// With weighting, the training rows are a weighted bootstrap (with replacement, in
  /// proportion to the weights) of min(maxTrainRows, rows) draws instead of a plain subsample.
  SampleWeighting weighting = SampleWeighting::None;
  double decayOldest = 0.5;      ///< UniquenessDecay: weight of the oldest sample relative to the newest
};

/// First prediction date for the given features and labels.
std::size_t firstTestDate(const FeatureSet& features, const LabelSpec& label, const WalkForwardSpec& spec);

struct RetrainRecord {
  std::size_t date = 0;        ///< first date predicted by this fit
  std::size_t trainRows = 0;
  std::size_t trainFrom = 0, trainTo = 0;  ///< training dates [from, to)
  double fitMs = 0.0;
  ClassificationMetrics test;  ///< on the dates this fit predicted
};

/// Out-of-sample predictions of one model.
struct ModelPredictions {
  ModelSpec spec;
  std::string name;
  Panel probability;            ///< P(label = 1); NaN outside [start, end)
  std::size_t start = 0, end = 0;
  ClassificationMetrics oos;    ///< over all predicted, labelled samples
  std::vector<double> importance;  ///< of the last fit, per feature (summed over lags)
  std::vector<RetrainRecord> retrains;
  double elapsedMs = 0.0;
};

/// Walk-forward training and prediction of one model for dates [start, end), where end
/// defaults to the last date that has a next-day return (dates - 1).
/// `labelEnds` (from makeLabels) is needed for uniqueness weighting; with `trainMask`, only
/// samples whose mask entry is positive (e.g. CUSUM events) are used for training, while
/// predictions are still made for every date and stock.
ModelPredictions walkForward(const ModelSpec& model, const FeatureSet& features, const Panel& labels, const LabelSpec& label,
                             const WalkForwardSpec& spec, std::size_t end = 0, const Panel* labelEnds = nullptr,
                             const Panel* trainMask = nullptr);

/// Per-row weights of a training set: average uniqueness of each label's span [t, end]
/// among the labels of the same stock in the set, with optional time decay.
std::vector<double> trainingWeights(const Dataset& train, const Panel& labelEnds, SampleWeighting weighting, double decayOldest);

}  // namespace sat
