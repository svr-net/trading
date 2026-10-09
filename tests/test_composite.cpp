#include <algorithm>
#include <cmath>
#include <cstdint>

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

TEST(scoring_windows_are_progressive_incremental_and_adaptive) {
  Members w;
  for (auto window : {ScoringWindow::Exponential, ScoringWindow::Adwin})
    for (auto decision : {ScoringDecision::Best, ScoringDecision::Evidence}) {
      CompositeSpec spec;
      spec.method = CompositeMethod::Adaptive;
      spec.window = window;
      spec.decision = decision;
      spec.halfLife = 0;  // expanding
      const auto c = compositePredictions(w.models, w.labels, w.ends, spec);
      // The informative member wins once its record is long enough, and stays chosen.
      CHECK(c.choice.back() == 0);
      CHECK(c.model.oos.auc > w.models[1].oos.auc + 0.2);
      // Incremental: the memory grows by about one date a day while nothing changes.
      CHECK(c.memory.size() == c.choice.size());
      CHECK(c.memory.back() > c.memory[c.memory.size() / 2] + 100);
    }
}

TEST(adwin_forgets_after_a_change) {
  // The informative member and the noise member swap roles at day 160.
  Members w;
  const std::size_t T = 300, N = 20;
  for (std::size_t t = 160; t + 1 < T; ++t)
    for (std::size_t i = 0; i < N; ++i) std::swap(w.models[0].probability(t, i), w.models[1].probability(t, i));
  CompositeSpec adwin;
  adwin.method = CompositeMethod::Adaptive;
  adwin.window = ScoringWindow::Adwin;
  const auto a = compositePredictions(w.models, w.labels, w.ends, adwin);
  CompositeSpec expanding = adwin;
  expanding.window = ScoringWindow::Exponential;
  expanding.halfLife = 0;
  const auto x = compositePredictions(w.models, w.labels, w.ends, expanding);
  // ADWIN cuts its window after the change and switches to the new informative member well
  // before the expanding record does.
  const std::size_t off = 160 - 20;
  std::size_t switchA = SIZE_MAX, switchX = SIZE_MAX;
  for (std::size_t k = off; k < a.choice.size(); ++k) {
    if (switchA == SIZE_MAX && a.choice[k] == 1) switchA = k;
    if (switchX == SIZE_MAX && x.choice[k] == 1) switchX = k;
  }
  CHECK(switchA < a.choice.size());
  CHECK(switchA + 10 < switchX);
  double minMemory = 1e9;
  for (std::size_t k = off; k < a.memory.size(); ++k) minMemory = std::min(minMemory, a.memory[k]);
  CHECK(minMemory < 80);  // the window was cut
  CHECK(a.choice.back() == 1);
}

TEST(evidence_decision_stays_with_the_average_without_a_significant_lead) {
  // Two equally uninformative members: the evidence rule should almost never leave the average.
  Members w;
  Rng rng(9);
  for (std::size_t t = 0; t + 1 < 300; ++t)
    for (std::size_t i = 0; i < 20; ++i) w.models[0].probability(t, i) = 0.5 + 0.3 * (rng.uniform() - 0.5);
  CompositeSpec spec;
  spec.method = CompositeMethod::Adaptive;
  spec.window = ScoringWindow::Exponential;
  spec.decision = ScoringDecision::Evidence;
  const auto c = compositePredictions(w.models, w.labels, w.ends, spec);
  std::size_t onAverage = 0;
  for (int k : c.choice) onAverage += k < 0 ? 1 : 0;
  CHECK(onAverage > 0.8 * static_cast<double>(c.choice.size()));
  CHECK(parseScoringWindow("adwin") == ScoringWindow::Adwin);
  CHECK_THROWS(parseScoringWindow("weekly"));
  spec.window = ScoringWindow::Adwin;
  spec.adwinDelta = 0;
  CHECK_THROWS(compositePredictions(w.models, w.labels, w.ends, spec));
}

namespace {

// A market that alternates between calm and turbulent blocks of `block` days (daily market
// moves of 0.4% vs 3%), and two members: one informative in calm blocks only, the other in
// turbulent blocks only.
struct RegimeWorld {
  Panel labels, ends, next;
  std::vector<ModelPredictions> models;
  std::vector<bool> turbulent;
  RegimeWorld(std::size_t T, std::size_t block, std::size_t firstSwitch = 0, std::size_t N = 20)
      : labels(T, N), ends(T, N), next(T, N), turbulent(T) {
    Rng rng(17);
    ModelPredictions calm, storm;
    calm.name = "calm";
    storm.name = "storm";
    calm.probability = Panel(T, N);
    storm.probability = Panel(T, N);
    for (std::size_t t = 0; t < T; ++t) {
      turbulent[t] = firstSwitch ? t >= firstSwitch : (t / block) % 2 == 1;
      const double m = (turbulent[t] ? 0.03 : 0.004) * rng.normal();
      for (std::size_t i = 0; i < N; ++i) next(t, i) = m + 0.01 * rng.normal();
    }
    for (std::size_t t = 0; t + 1 < T; ++t)
      for (std::size_t i = 0; i < N; ++i) {
        labels(t, i) = rng.uniform() < 0.5 ? 1.0 : 0.0;
        ends(t, i) = static_cast<double>(t + 1);
        const double informative = (labels(t, i) > 0.5 ? 0.6 : 0.4) + 0.2 * (rng.uniform() - 0.5);
        const double noise = 0.5 + 0.3 * (rng.uniform() - 0.5);
        // A forecast for date t is made at its close, when the market state of t is known.
        calm.probability(t, i) = turbulent[t] ? noise : informative;
        storm.probability(t, i) = turbulent[t] ? informative : noise;
      }
    calm.start = storm.start = 70;
    calm.end = storm.end = T - 1;
    models = {calm, storm};
  }
};

}  // namespace

TEST(market_state_adwin_recedes_when_the_market_changes_state) {
  // The market turns turbulent at day 300 while the forecasts keep the same quality: only the
  // market-driven memory has a reason to recede.
  RegimeWorld w(600, 0, 300);
  for (std::size_t t = 0; t < 600; ++t)
    for (std::size_t i = 0; i < 20; ++i) {
      const double informative = w.models[0].probability(t, i), other = w.models[1].probability(t, i);
      if (w.turbulent[t]) w.models[0].probability(t, i) = other, w.models[1].probability(t, i) = informative;
    }
  CompositeSpec spec;
  spec.method = CompositeMethod::Adaptive;
  spec.window = ScoringWindow::MarketAdwin;
  const auto m = compositePredictions(w.models, w.labels, w.ends, spec, &w.next);
  spec.window = ScoringWindow::Adwin;
  const auto a = compositePredictions(w.models, w.labels, w.ends, spec, &w.next);
  const std::size_t change = 300 - 70;
  CHECK(m.memory[change - 1] > 200 && a.memory[change - 1] > 200);
  double leastM = 1e9, leastA = 1e9;
  for (std::size_t k = change; k < change + 60; ++k) leastM = std::min(leastM, m.memory[k]), leastA = std::min(leastA, a.memory[k]);
  CHECK(leastM < 60);    // recedes to the new state within weeks
  CHECK(leastA > 200);   // the forecasts' own records saw no change
  CHECK(m.memory.back() > 150);  // and grows again while the new state lasts
  CHECK(m.choice.back() == 0 && a.choice.back() == 0);
  spec.window = ScoringWindow::MarketAdwin;
  CHECK_THROWS(compositePredictions(w.models, w.labels, w.ends, spec, nullptr));
}

TEST(similar_state_memory_uses_each_model_in_its_own_state) {
  // Calm and turbulent blocks of 60 days alternate: no single model is good all the time.
  RegimeWorld w(900, 60);
  CompositeSpec spec;
  spec.method = CompositeMethod::Adaptive;
  spec.window = ScoringWindow::SimilarState;
  const auto sim = compositePredictions(w.models, w.labels, w.ends, spec, &w.next);
  spec.window = ScoringWindow::Exponential;
  spec.halfLife = 0;
  const auto exp = compositePredictions(w.models, w.labels, w.ends, spec, &w.next);
  // After the first blocks, the similar-state memory never uses the model that is wrong for the
  // current state (it uses the right one, or the average around transitions); the expanding
  // record cannot tell the states apart.
  std::size_t wrong = 0, right = 0, n = 0;
  for (std::size_t k = 300; k < sim.choice.size(); ++k, ++n) {
    const int good = w.turbulent[70 + k] ? 1 : 0;
    wrong += sim.choice[k] == 1 - good ? 1 : 0;
    right += sim.choice[k] == good ? 1 : 0;
  }
  CHECK(wrong < 0.05 * static_cast<double>(n));
  CHECK(right > 0.25 * static_cast<double>(n));
  CHECK(sim.model.oos.auc > exp.model.oos.auc + 0.02);
  CHECK(parseScoringWindow("similar-state") == ScoringWindow::SimilarState);
}

TEST(market_state_masks_use_only_the_past) {
  RegimeWorld w(400, 0, 200);
  const auto st = marketState(w.next);
  CHECK(std::isnan(st.volatility[62]) && std::isfinite(st.volatility[63]));
  const Panel calm = marketStateMask(w.next, StateSide::Calm), storm = marketStateMask(w.next, StateSide::Turbulent);
  double calmBefore = 0, stormAfter = 0;
  for (std::size_t t = 100; t < 200; ++t) calmBefore += calm(t, 0) / 100.0;
  for (std::size_t t = 230; t < 400; ++t) stormAfter += storm(t, 0) / 170.0;
  CHECK(stormAfter > 0.9);   // the turbulent days are above the running median
  CHECK(calmBefore > 0.3);   // about half of a calm stretch is below its own running median
  for (std::size_t t = 63; t < 400; ++t) CHECK(calm(t, 3) + storm(t, 3) == 1.0);
  // The mask of a date does not change when later returns change.
  auto later = w.next;
  for (std::size_t t = 300; t < 400; ++t)
    for (std::size_t i = 0; i < 20; ++i) later(t, i) *= 3.0;
  const Panel again = marketStateMask(later, StateSide::Calm);
  for (std::size_t t = 0; t < 300; ++t) CHECK(again(t, 0) == calm(t, 0));
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
  exp.stateSpecialists = true;
  const auto r = runPredictions(generateSyntheticMarket(ms), exp);
  CHECK(r.models.size() == 2 + 4 + 1 && r.models[2].name == "Logistic (calm)" && r.models[5].name == "Logistic (falling)");
  CHECK(r.compositeMembers.size() == 6);
  exp.stateSpecialists = false;
  exp.composite.keepMembers = false;
  const auto q = runPredictions(generateSyntheticMarket(ms), exp);
  CHECK(q.models.size() == 1);
  const auto e = runStrategies(q, exp);
  CHECK(e.book.size() == ExperimentSpec::defaultStrategies().size());
}

