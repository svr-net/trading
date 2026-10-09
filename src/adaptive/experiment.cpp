#include "sat/adaptive/experiment.hpp"

#include <algorithm>
#include <stdexcept>

#include "sat/afml/features.hpp"
#include "sat/factors/alpha101.hpp"

namespace sat {

// The default pool is the one examples/pool_ablation kept: backward elimination of models and
// rules on the self-adaptive strategy's Sharpe ratio over synthetic markets, confirmed on
// unseen ones. The other families (XGBoost, LightGBM, MLP, LSTM, SVM, tree) and rules remain
// available; on their own they added candidates the selector chased without adding value.
std::vector<ModelSpec> ExperimentSpec::defaultModels() {
  std::vector<ModelSpec> m(2);
  m[0].type = "logistic";
  m[1].type = "forest";
  m[1].trees = 40;
  m[1].maxDepth = 6;
  m[1].colsample = 0.4;
  m[1].subsample = 0.7;
  return m;
}

std::vector<StrategySpec> ExperimentSpec::defaultStrategies() {
  return {
      {StrategyKind::LongTopK, 5, 1},  {StrategyKind::LongTopK, 10, 5}, {StrategyKind::LongShort, 3, 1},
      {StrategyKind::Threshold, 0.52, 1}, {StrategyKind::ProbabilityWeighted, 0.5, 1},
  };
}

PredictionSet runPredictions(const MarketData& data, const ExperimentSpec& spec) {
  data.validate();
  PredictionSet p;
  p.features = buildFeatures(data, spec.alphaIds.empty() ? paperAlphaIds() : spec.alphaIds, spec.normalisation);
  if (!spec.extraFeatures.empty()) afml::appendExtraFeatures(p.features, data, spec.extraFeatures, spec.normalisation, spec.ffdOrder);
  p.labels = makeLabels(data, spec.label, &p.labelEnds);
  p.nextReturns = data.forwardReturns(1);
  if (spec.cusumMultiple > 0) p.trainMask = afml::cusumEventMask(data, spec.cusumMultiple);
  const auto models = spec.models.empty() ? ExperimentSpec::defaultModels() : spec.models;
  for (const auto& m : models)
    p.models.push_back(walkForward(m, p.features, p.labels, spec.label, spec.walkForward, 0, &p.labelEnds,
                                   p.trainMask.empty() ? nullptr : &p.trainMask));
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
