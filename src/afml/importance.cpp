#include "sat/afml/importance.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sat/core/random.hpp"
#include "sat/core/stats.hpp"
#include "sat/ml/metrics.hpp"

namespace sat::afml {

namespace {

std::vector<double> pick(const std::vector<double>& v, const std::vector<std::size_t>& rows) {
  std::vector<double> out;
  out.reserve(rows.size());
  for (std::size_t r : rows) out.push_back(v[r]);
  return out;
}

double logLoss(const std::vector<double>& y, const std::vector<double>& p) { return classificationMetrics(y, p).logLoss; }

}  // namespace

CvScore crossValidate(const ModelSpec& model, const Matrix& X, const std::vector<double>& y, const std::vector<Split>& splits) {
  CvScore s;
  std::vector<double> allY, allP;
  for (const auto& sp : splits) {
    if (sp.train.size() < 10 || sp.test.empty()) continue;
    auto m = makeClassifier(model);
    m->fit(X.selectRows(sp.train), pick(y, sp.train));
    const auto p = m->predictProba(X.selectRows(sp.test));
    const auto yt = pick(y, sp.test);
    s.foldAccuracy.push_back(classificationMetrics(yt, p).accuracy);
    allY.insert(allY.end(), yt.begin(), yt.end());
    allP.insert(allP.end(), p.begin(), p.end());
  }
  if (allY.empty()) throw std::invalid_argument("cross-validation: no usable fold");
  const auto m = classificationMetrics(allY, allP);
  s.accuracy = m.accuracy;
  s.auc = m.auc;
  s.logLoss = m.logLoss;
  return s;
}

Importance meanDecreaseImpurity(const ModelSpec& model, const Matrix& X, const std::vector<double>& y) {
  auto m = makeClassifier(model);
  m->fit(X, y);
  Importance imp;
  imp.mean = m->featureImportance();
  if (imp.mean.empty()) throw std::invalid_argument("MDI needs a model with split-gain importance (tree, forest, boosting)");
  imp.sd.assign(imp.mean.size(), 0.0);
  return imp;
}

Importance meanDecreaseAccuracy(const ModelSpec& model, const Matrix& X, const std::vector<double>& y,
                                const std::vector<Split>& splits, std::uint64_t seed) {
  const std::size_t F = X.cols();
  std::vector<std::vector<double>> perFold(F);
  Rng rng(seed);
  for (const auto& sp : splits) {
    if (sp.train.size() < 10 || sp.test.empty()) continue;
    auto m = makeClassifier(model);
    m->fit(X.selectRows(sp.train), pick(y, sp.train));
    Matrix Xt = X.selectRows(sp.test);
    const auto yt = pick(y, sp.test);
    const double base = logLoss(yt, m->predictProba(Xt));
    for (std::size_t f = 0; f < F; ++f) {
      std::vector<double> col = Xt.column(f);
      std::vector<double> shuffled = col;
      shuffle(shuffled, rng);
      for (std::size_t r = 0; r < Xt.rows(); ++r) Xt(r, f) = shuffled[r];
      perFold[f].push_back(logLoss(yt, m->predictProba(Xt)) - base);
      for (std::size_t r = 0; r < Xt.rows(); ++r) Xt(r, f) = col[r];
    }
  }
  Importance imp;
  for (const auto& v : perFold) {
    imp.mean.push_back(v.empty() ? 0.0 : mean(v));
    imp.sd.push_back(v.size() > 1 ? stdev(v) / std::sqrt(static_cast<double>(v.size())) : 0.0);
  }
  return imp;
}

Importance singleFeatureImportance(const ModelSpec& model, const Matrix& X, const std::vector<double>& y,
                                   const std::vector<Split>& splits) {
  Importance imp;
  for (std::size_t f = 0; f < X.cols(); ++f) {
    Matrix one(X.rows(), 1);
    for (std::size_t r = 0; r < X.rows(); ++r) one(r, 0) = X(r, f);
    const CvScore s = crossValidate(model, one, y, splits);
    imp.mean.push_back(s.auc);
    imp.sd.push_back(s.foldAccuracy.size() > 1 ? stdev(s.foldAccuracy) / std::sqrt(static_cast<double>(s.foldAccuracy.size())) : 0.0);
  }
  return imp;
}

}  // namespace sat::afml
