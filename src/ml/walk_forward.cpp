#include "sat/ml/walk_forward.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "sat/core/random.hpp"

namespace sat {

namespace {
double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

std::size_t firstTestDate(const FeatureSet& features, const LabelSpec& label, const WalkForwardSpec& spec) {
  const std::size_t earliest = features.warmup + spec.trainWindow + label.lookahead();
  return std::max(spec.testStart, earliest);
}

ModelPredictions walkForward(const ModelSpec& model, const FeatureSet& features, const Panel& labels, const LabelSpec& label,
                             const WalkForwardSpec& spec, std::size_t end) {
  const std::size_t T = features.dates();
  if (spec.trainWindow < 2 || spec.retrainEvery < 1) throw std::invalid_argument("walk-forward window and step must be positive");
  if (end == 0 || end > T - 1) end = T - 1;
  const std::size_t start = firstTestDate(features, label, spec);
  if (start >= end)
    throw std::invalid_argument("not enough dates: the first out-of-sample date is " + std::to_string(start) + " of " +
                                std::to_string(T) + " (warm-up " + std::to_string(features.warmup) + " + training window " +
                                std::to_string(spec.trainWindow) + ")");
  const double t0 = nowMs();
  ModelPredictions out;
  out.spec = model;
  out.name = model.displayName();
  out.probability = labels.like();
  out.start = start;
  out.end = end;
  const std::size_t lags = model.lags();
  std::vector<double> allY, allP;
  for (std::size_t r = start; r < end; r += spec.retrainEvery) {
    RetrainRecord rec;
    rec.date = r;
    rec.trainTo = r - label.lookahead();
    rec.trainFrom = rec.trainTo > spec.trainWindow ? std::max(features.warmup, rec.trainTo - spec.trainWindow) : features.warmup;
    Dataset train = assemble(features, labels, rec.trainFrom, rec.trainTo, lags, true);
    if (spec.maxTrainRows > 0 && train.X.rows() > spec.maxTrainRows) {
      std::vector<std::size_t> rows(train.X.rows());
      for (std::size_t k = 0; k < rows.size(); ++k) rows[k] = k;
      Rng rng(spec.seed + r);
      shuffle(rows, rng);
      rows.resize(spec.maxTrainRows);
      std::sort(rows.begin(), rows.end());
      std::vector<double> y;
      for (std::size_t k : rows) y.push_back(train.y[k]);
      train.X = train.X.selectRows(rows);
      train.y = std::move(y);
    }
    if (train.X.rows() < 10) throw std::invalid_argument("walk-forward: fewer than 10 labelled training samples before date " + std::to_string(r));
    double pos = 0.0;
    for (double v : train.y) pos += v;
    rec.trainRows = train.X.rows();
    const double f0 = nowMs();
    auto clf = makeClassifier(model);
    if (pos == 0.0 || pos == static_cast<double>(train.y.size())) {
      // One class only: predict the base rate.
      Dataset test = assemble(features, labels, r, std::min(r + spec.retrainEvery, end), lags, false);
      for (std::size_t k = 0; k < test.X.rows(); ++k) out.probability(test.date[k], test.asset[k]) = pos > 0 ? 1.0 : 0.0;
      out.retrains.push_back(rec);
      continue;
    }
    clf->fit(train.X, train.y);
    rec.fitMs = nowMs() - f0;
    const Dataset test = assemble(features, labels, r, std::min(r + spec.retrainEvery, end), lags, false);
    const auto p = clf->predictProba(test.X);
    for (std::size_t k = 0; k < p.size(); ++k) out.probability(test.date[k], test.asset[k]) = p[k];
    rec.test = classificationMetrics(test.y, p);
    allY.insert(allY.end(), test.y.begin(), test.y.end());
    allP.insert(allP.end(), p.begin(), p.end());
    const auto imp = clf->featureImportance();
    if (!imp.empty()) {
      out.importance.assign(features.size(), 0.0);
      for (std::size_t k = 0; k < imp.size(); ++k) out.importance[k % features.size()] += imp[k];
    }
    out.retrains.push_back(rec);
  }
  out.oos = classificationMetrics(allY, allP);
  out.elapsedMs = nowMs() - t0;
  return out;
}

}  // namespace sat
