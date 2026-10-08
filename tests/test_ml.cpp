#include <cmath>

#include "sat/core/random.hpp"
#include "sat/data/synthetic_market.hpp"
#include "sat/ml/classifier.hpp"
#include "sat/ml/metrics.hpp"
#include "sat/ml/trees.hpp"
#include "sat/ml/walk_forward.hpp"
#include "test_framework.hpp"

using namespace sat;

namespace {

// y = 1 if 2 x0 - x1 + noise > 0; x2 is irrelevant.
void linearProblem(std::size_t n, std::uint64_t seed, Matrix& X, std::vector<double>& y) {
  Rng rng(seed);
  X = Matrix(n, 3);
  y.resize(n);
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t f = 0; f < 3; ++f) X(r, f) = rng.normal();
    y[r] = 2 * X(r, 0) - X(r, 1) + 0.5 * rng.normal() > 0 ? 1.0 : 0.0;
  }
}

// y = 1 if x0 and x1 have the same sign: an interaction no linear model can learn.
void xorProblem(std::size_t n, std::uint64_t seed, Matrix& X, std::vector<double>& y) {
  Rng rng(seed);
  X = Matrix(n, 2);
  y.resize(n);
  for (std::size_t r = 0; r < n; ++r) {
    X(r, 0) = rng.normal();
    X(r, 1) = rng.normal();
    y[r] = X(r, 0) * X(r, 1) > 0 ? 1.0 : 0.0;
  }
}

double auc(const ModelSpec& spec, void (*problem)(std::size_t, std::uint64_t, Matrix&, std::vector<double>&)) {
  Matrix X, Xt;
  std::vector<double> y, yt;
  problem(3000, 1, X, y);
  problem(2000, 2, Xt, yt);
  auto m = makeClassifier(spec);
  m->fit(X, y);
  const auto p = m->predictProba(Xt);
  for (double v : p) CHECK(v >= 0.0 && v <= 1.0);
  return classificationMetrics(yt, p).auc;
}

ModelSpec spec(const char* type) {
  ModelSpec s;
  s.type = type;
  if (s.type == "mlp") {
    s.learningRate = 0.01;
    s.epochs = 10;
  }
  return s;
}

}  // namespace

TEST(every_model_learns_a_linear_rule) {
  for (const char* type : {"logistic", "svm", "tree", "forest", "xgboost", "lightgbm", "mlp"}) {
    const double a = auc(spec(type), linearProblem);
    if (!(a > 0.85)) ::sattest::fail(std::string(type) + " AUC " + std::to_string(a), __FILE__, __LINE__);
  }
  CHECK_THROWS(makeClassifier(spec("nope")));
}

TEST(trees_learn_interactions_that_linear_models_cannot) {
  CHECK(auc(spec("logistic"), xorProblem) < 0.6);
  CHECK(auc(spec("xgboost"), xorProblem) > 0.9);
  CHECK(auc(spec("lightgbm"), xorProblem) > 0.9);
  CHECK(auc(spec("forest"), xorProblem) > 0.9);
  CHECK(auc(spec("mlp"), xorProblem) > 0.85);
}

TEST(feature_importance_finds_the_signal) {
  Matrix X;
  std::vector<double> y;
  linearProblem(3000, 5, X, y);
  for (const char* type : {"logistic", "forest", "xgboost"}) {
    auto m = makeClassifier(spec(type));
    m->fit(X, y);
    const auto imp = m->featureImportance();
    CHECK(imp.size() == 3);
    CHECK_NEAR(imp[0] + imp[1] + imp[2], 1.0, 1e-9);
    CHECK(imp[0] > imp[1] && imp[1] > imp[2]);
  }
}

TEST(lstm_reads_the_sequence) {
  // Two features over 4 steps; the label is the sign of feature 0 at the first step, so
  // the network has to carry information through time.
  Rng rng(9);
  const std::size_t steps = 4, n = 3000;
  auto make = [&](Matrix& X, std::vector<double>& y) {
    X = Matrix(n, steps * 2);
    y.resize(n);
    for (std::size_t r = 0; r < n; ++r) {
      for (std::size_t k = 0; k < steps * 2; ++k) X(r, k) = rng.normal();
      y[r] = X(r, 0) > 0 ? 1.0 : 0.0;
    }
  };
  Matrix X, Xt;
  std::vector<double> y, yt;
  make(X, y);
  make(Xt, yt);
  ModelSpec s = spec("lstm");
  s.seqLen = steps;
  s.hidden = 8;
  s.epochs = 8;
  s.learningRate = 0.02;
  auto m = makeClassifier(s);
  m->fit(X, y);
  CHECK(classificationMetrics(yt, m->predictProba(Xt)).auc > 0.9);
  CHECK(s.lags() == steps && spec("xgboost").lags() == 1);
  Matrix wrong(2, 3);
  CHECK_THROWS(m->predictProba(wrong));
}

TEST(classification_metrics) {
  const auto perfect = classificationMetrics({0, 0, 1, 1}, {0.1, 0.2, 0.8, 0.9});
  CHECK(perfect.accuracy == 1.0 && perfect.auc == 1.0 && perfect.f1 == 1.0 && perfect.count == 4);
  const auto inverted = classificationMetrics({0, 1}, {0.9, 0.1});
  CHECK(inverted.auc == 0.0 && inverted.accuracy == 0.0);
  const auto ties = classificationMetrics({0, 1, 0, 1}, {0.5, 0.5, 0.5, 0.5});
  CHECK_NEAR(ties.auc, 0.5, 1e-12);
  const auto roc = rocCurve({0, 0, 1, 1}, {0.1, 0.2, 0.8, 0.9}, 10);
  CHECK(roc.front()[0] == 0 && roc.back()[0] == 1 && roc.back()[1] == 1);
}

TEST(binner_and_tree_thresholds) {
  Matrix X(100, 1);
  for (std::size_t r = 0; r < 100; ++r) X(r, 0) = static_cast<double>(r % 4);
  FeatureBinner b;
  b.fit(X, 32);
  CHECK(b.numBins(0) == 4);
  CHECK(b.code(0, 0.0) == 0 && b.code(0, 3.0) == 3 && b.code(0, 1.4) == 1);
}

TEST(walk_forward_is_out_of_sample_and_purged) {
  SyntheticMarketSpec ms;
  ms.numAssets = 10;
  ms.numDates = 420;
  const MarketData d = generateSyntheticMarket(ms);
  const FeatureSet f = buildFeatures(d, {2, 12, 33, 41}, Normalisation::Rank);
  const LabelSpec label{LabelKind::Direction, 3, 10};
  const Panel y = makeLabels(d, label);
  WalkForwardSpec wf;
  wf.trainWindow = 150;
  wf.retrainEvery = 50;
  ModelSpec m = spec("logistic");
  const ModelPredictions p = walkForward(m, f, y, label, wf);
  CHECK(p.start == f.warmup + 150 + 3 && p.end == 419);
  CHECK(p.retrains.size() == (p.end - p.start + 49) / 50);
  for (const auto& r : p.retrains) {
    CHECK(r.trainTo + label.lookahead() <= r.date);  // labels of training dates end before prediction
    CHECK(r.trainTo - r.trainFrom <= 150);
    CHECK(r.trainRows > 0);
  }
  for (std::size_t t = 0; t < d.numDates(); ++t)
    for (std::size_t i = 0; i < 10; ++i) CHECK(std::isfinite(p.probability(t, i)) == (t >= p.start && t < p.end));
  CHECK(p.oos.count > 0 && p.oos.auc > 0.4);
  CHECK(p.importance.size() == 4);
  wf.trainWindow = 410;  // warm-up + window + purge runs past the last date
  CHECK_THROWS(walkForward(m, f, y, label, wf));
}
