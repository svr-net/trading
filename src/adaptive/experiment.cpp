#include "sat/adaptive/experiment.hpp"

#include <algorithm>
#include <stdexcept>

#include "sat/factors/alpha101.hpp"

namespace sat {

std::vector<ModelSpec> ExperimentSpec::defaultModels() {
  std::vector<ModelSpec> m(6);
  m[0].type = "logistic";
  m[1].type = "forest";
  m[1].trees = 40;
  m[1].maxDepth = 6;
  m[1].colsample = 0.4;
  m[1].subsample = 0.7;
  m[2].type = "xgboost";
  m[3].type = "lightgbm";
  m[3].maxDepth = 8;
  m[3].maxLeaves = 12;
  m[4].type = "mlp";
  m[4].learningRate = 0.005;
  m[4].maxSamples = 4000;
  m[5].type = "lstm";
  m[5].hidden = 8;
  m[5].learningRate = 0.01;
  m[5].epochs = 2;
  m[5].maxSamples = 3000;
  return m;
}

std::vector<StrategySpec> ExperimentSpec::defaultStrategies() {
  return {
      {StrategyKind::LongTopK, 3, 1},       {StrategyKind::LongTopK, 5, 1},  {StrategyKind::LongTopK, 10, 5},
      {StrategyKind::LongShort, 3, 1},      {StrategyKind::LongShort, 5, 1}, {StrategyKind::Threshold, 0.52, 1},
      {StrategyKind::Threshold, 0.55, 1},   {StrategyKind::ProbabilityWeighted, 0.5, 1},
  };
}

PredictionSet runPredictions(const MarketData& data, const ExperimentSpec& spec) {
  data.validate();
  PredictionSet p;
  p.features = buildFeatures(data, spec.alphaIds.empty() ? paperAlphaIds() : spec.alphaIds, spec.normalisation);
  p.labels = makeLabels(data, spec.label);
  p.nextReturns = data.forwardReturns(1);
  const auto models = spec.models.empty() ? ExperimentSpec::defaultModels() : spec.models;
  for (const auto& m : models) p.models.push_back(walkForward(m, p.features, p.labels, spec.label, spec.walkForward));
  return p;
}

Experiment runStrategies(const PredictionSet& predictions, const ExperimentSpec& spec, std::size_t evalFrom) {
  Experiment e;
  e.book = CandidateBook(predictions.models, spec.strategies.empty() ? ExperimentSpec::defaultStrategies() : spec.strategies,
                         predictions.nextReturns, spec.costBps);
  const std::size_t D = e.book.days();
  e.evalFrom = evalFrom == SIZE_MAX ? std::min(spec.selector.lookback, D - 1) : evalFrom;
  if (e.evalFrom >= D) throw std::invalid_argument("evaluation starts after the last out-of-sample date");
  double best = -1e300;
  for (std::size_t c = 0; c < e.book.size(); ++c) {
    e.candidateMetrics.push_back(evaluatePerformance(e.book.netSeries(c, e.evalFrom), e.book.turnoverSeries(c, e.evalFrom)));
    if (e.candidateMetrics.back().sharpe > best) {
      best = e.candidateMetrics.back().sharpe;
      e.bestFixed = c;
    }
  }
  e.benchmark = equalWeightBenchmark(predictions.nextReturns, e.book.start() + e.evalFrom, e.book.end(), spec.costBps);
  e.adaptive = runSelector(e.book, spec.selector, e.evalFrom);
  return e;
}

}  // namespace sat
