#include <cmath>

#include "sat/adaptive/composite.hpp"
#include "sat/adaptive/experiment.hpp"
#include "sat/core/random.hpp"
#include "sat/data/synthetic_market.hpp"
#include "test_framework.hpp"

using namespace sat;

namespace {

// Labels on 300 days x 20 stocks; a "good" member knows the label (0.7 / 0.3), a "noise"
// member is random around 0.5.
struct Members {
  Panel labels, ends;
  std::vector<ModelPredictions> models;
  explicit Members(std::size_t T = 300, std::size_t N = 20) : labels(T, N), ends(T, N) {
    Rng rng(5);
    ModelPredictions good, noise;
    good.name = "good";
    noise.name = "noise";
    good.probability = Panel(T, N);
    noise.probability = Panel(T, N);
    for (std::size_t t = 0; t + 1 < T; ++t)
      for (std::size_t i = 0; i < N; ++i) {
        labels(t, i) = rng.uniform() < 0.5 ? 1.0 : 0.0;
        ends(t, i) = static_cast<double>(t + 1);
        good.probability(t, i) = (labels(t, i) > 0.5 ? 0.6 : 0.4) + 0.2 * (rng.uniform() - 0.5);
        noise.probability(t, i) = 0.5 + 0.3 * (rng.uniform() - 0.5);
      }
    good.start = noise.start = 20;
    good.end = noise.end = T - 1;
    models = {good, noise};
  }
};

}  // namespace

TEST(composite_average_is_the_mean_of_members) {
  Members w;
  CompositeSpec spec;
  spec.method = CompositeMethod::Average;
  const auto c = compositePredictions(w.models, w.labels, w.ends, spec);
  CHECK(c.model.start == 20 && c.model.end == 299);
  for (std::size_t t = 20; t < 299; t += 37)
    for (std::size_t i = 0; i < 20; i += 7)
      CHECK_NEAR(c.model.probability(t, i), 0.5 * (w.models[0].probability(t, i) + w.models[1].probability(t, i)), 1e-12);
  CHECK(std::isnan(c.model.probability(10, 0)));
  CHECK(c.weights.size() == 279);
}

TEST(self_adaptive_forecast_follows_the_informative_member) {
  Members w;
  CompositeSpec spec;
  spec.method = CompositeMethod::Adaptive;
  const auto c = compositePredictions(w.models, w.labels, w.ends, spec);
  CHECK(c.model.name == "Composite (self-adaptive)");
  CHECK(c.choice.size() == c.weights.size() && c.choice.size() == 279);
  // Until 5 resolved dates are scored, the average; then the informative member.
  CHECK(c.choice.front() == -1);
  CHECK(c.choice.back() == 0 && c.weights.back()[0] == 1.0);
  for (std::size_t t = 250; t < 299; t += 7)
    for (std::size_t i = 0; i < 20; i += 5) CHECK_NEAR(c.model.probability(t, i), w.models[0].probability(t, i), 1e-12);
  CHECK(c.model.oos.auc > w.models[1].oos.auc + 0.2);
  // The choice changes only on adaptation dates.
  for (std::size_t k = 1; k < c.choice.size(); ++k)
    if (k % spec.adaptEvery != 0) CHECK(c.choice[k] == c.choice[k - 1]);
}

TEST(self_adaptive_forecast_uses_only_resolved_labels) {
  Members a, b;
  // Scramble every label from day 150 on: forecasts up to day 150 must not change.
  for (std::size_t t = 150; t < 299; ++t)
    for (std::size_t i = 0; i < 20; ++i) b.labels(t, i) = 1.0 - b.labels(t, i);
  CompositeSpec spec;
  spec.method = CompositeMethod::Adaptive;
  spec.adaptEvery = 5;
  const auto ca = compositePredictions(a.models, a.labels, a.ends, spec);
  const auto cb = compositePredictions(b.models, b.labels, b.ends, spec);
  for (std::size_t t = 20; t <= 151; ++t)
    for (std::size_t i = 0; i < 20; ++i) CHECK_NEAR(ca.model.probability(t, i), cb.model.probability(t, i), 1e-12);
  bool differs = false;
  for (std::size_t k = 0; k < ca.choice.size() && !differs; ++k) differs = ca.choice[k] != cb.choice[k];
  CHECK(differs);
}

TEST(composite_rejects_bad_input) {
  Members w;
  CompositeSpec spec;
  CHECK_THROWS(compositePredictions(w.models, w.labels, w.ends, spec));  // method none
  spec.method = CompositeMethod::Average;
  CHECK_THROWS(compositePredictions({}, w.labels, w.ends, spec));
  CHECK_THROWS(compositePredictions(w.models, Panel(10, 20), w.ends, spec));
  CHECK(parseCompositeMethod("self-adaptive") == CompositeMethod::Adaptive);
  CHECK_THROWS(parseCompositeMethod("stacked"));
  CHECK_THROWS(parseCompositeMethod("median"));
}

TEST(experiment_appends_the_composite_model) {
  SyntheticMarketSpec ms;
  ms.numAssets = 12;
  ms.numDates = 420;
  ExperimentSpec exp;
  exp.walkForward.trainWindow = 150;
  exp.walkForward.retrainEvery = 60;
  exp.composite.method = CompositeMethod::Adaptive;
  const auto p = runPredictions(generateSyntheticMarket(ms), exp);
  CHECK(p.models.size() == 3);
  CHECK(p.models.back().name == "Composite (self-adaptive)");
  CHECK(p.compositeMembers.size() == 2);
  CHECK(p.compositeWeights.size() == p.models.back().end - p.models.back().start);
  exp.composite.keepMembers = false;
  const auto q = runPredictions(generateSyntheticMarket(ms), exp);
  CHECK(q.models.size() == 1);
  const auto e = runStrategies(q, exp);
  CHECK(e.book.size() == ExperimentSpec::defaultStrategies().size());
}

