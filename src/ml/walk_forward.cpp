#include "sat/ml/walk_forward.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "sat/afml/sampling.hpp"
#include "sat/core/random.hpp"

namespace sat {

namespace {
double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

SampleWeighting parseSampleWeighting(const std::string& name) {
  if (name == "none") return SampleWeighting::None;
  if (name == "uniqueness") return SampleWeighting::Uniqueness;
  if (name == "decay") return SampleWeighting::UniquenessDecay;
  throw std::invalid_argument("unknown sample weighting '" + name + "' (none, uniqueness, decay)");
}

std::vector<double> trainingWeights(const Dataset& train, const Panel& labelEnds, SampleWeighting weighting, double decayOldest) {
  const std::size_t n = train.X.rows();
  std::vector<double> w(n, 1.0);
  if (weighting == SampleWeighting::None || n == 0) return w;
  // Uniqueness per stock: labels of different stocks do not share returns.
  std::size_t lastDate = 0;
  for (std::size_t r = 0; r < n; ++r) {
    const double e = labelEnds(train.date[r], train.asset[r]);
    lastDate = std::max(lastDate, std::isfinite(e) ? static_cast<std::size_t>(e) : train.date[r]);
  }
  std::vector<std::vector<std::size_t>> rowsOf(labelEnds.assets());
  for (std::size_t r = 0; r < n; ++r) rowsOf[train.asset[r]].push_back(r);
  for (const auto& rows : rowsOf) {
    if (rows.empty()) continue;
    std::vector<afml::Span> spans;
    for (std::size_t r : rows) {
      const double e = labelEnds(train.date[r], train.asset[r]);
      spans.push_back({train.date[r], std::isfinite(e) ? std::max(train.date[r], static_cast<std::size_t>(e)) : train.date[r]});
    }
    const auto u = afml::averageUniqueness(spans, afml::concurrency(spans, lastDate + 1));
    for (std::size_t k = 0; k < rows.size(); ++k) w[rows[k]] = u[k];
  }
  if (weighting == SampleWeighting::UniquenessDecay) {
    std::vector<std::size_t> order(n);
    for (std::size_t r = 0; r < n; ++r) order[r] = r;
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return train.date[a] < train.date[b]; });
    std::vector<double> u;
    for (std::size_t r : order) u.push_back(w[r]);
    const auto d = afml::timeDecay(u, decayOldest);
    for (std::size_t k = 0; k < n; ++k) w[order[k]] *= d[k];
  }
  return w;
}

std::size_t firstTestDate(const FeatureSet& features, const LabelSpec& label, const WalkForwardSpec& spec) {
  const std::size_t earliest = features.warmup + spec.trainWindow + label.lookahead();
  return std::max(spec.testStart, earliest);
}

ModelPredictions walkForward(const ModelSpec& model, const FeatureSet& features, const Panel& labels, const LabelSpec& label,
                             const WalkForwardSpec& spec, std::size_t end, const Panel* labelEnds, const Panel* trainMask) {
  if (spec.weighting != SampleWeighting::None && !labelEnds) throw std::invalid_argument("sample weighting needs the label ends");
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
    // A mask that leaves too few samples (e.g. a market state not yet seen) falls back to
    // the whole window rather than failing.
    bool masked = trainMask != nullptr;
    if (masked) {
      std::size_t inMask = 0;
      for (std::size_t k = 0; k < train.X.rows(); ++k) inMask += (*trainMask)(train.date[k], train.asset[k]) > 0 ? 1 : 0;
      masked = inMask >= 10;
    }
    if (masked) {
      std::vector<std::size_t> keep;
      for (std::size_t k = 0; k < train.X.rows(); ++k)
        if ((*trainMask)(train.date[k], train.asset[k]) > 0) keep.push_back(k);
      std::vector<double> y;
      std::vector<std::size_t> dates, assets;
      for (std::size_t k : keep) {
        y.push_back(train.y[k]);
        dates.push_back(train.date[k]);
        assets.push_back(train.asset[k]);
      }
      train.X = train.X.selectRows(keep);
      train.y = std::move(y);
      train.date = std::move(dates);
      train.asset = std::move(assets);
    }
    if (spec.weighting != SampleWeighting::None && train.X.rows() > 0) {
      // Weighted bootstrap in proportion to the sample weights.
      const auto w = trainingWeights(train, *labelEnds, spec.weighting, spec.decayOldest);
      std::vector<double> cum(w.size());
      double total = 0;
      for (std::size_t k = 0; k < w.size(); ++k) cum[k] = total += w[k];
      const std::size_t draws = spec.maxTrainRows > 0 ? std::min(spec.maxTrainRows, w.size()) : w.size();
      Rng rng(spec.seed + r);
      std::vector<std::size_t> rows(draws);
      for (auto& k : rows) k = std::min<std::size_t>(w.size() - 1, static_cast<std::size_t>(std::upper_bound(cum.begin(), cum.end(), rng.uniform() * total) - cum.begin()));
      std::vector<double> y;
      for (std::size_t k : rows) y.push_back(train.y[k]);
      train.X = train.X.selectRows(rows);
      train.y = std::move(y);
    } else if (spec.maxTrainRows > 0 && train.X.rows() > spec.maxTrainRows) {
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
