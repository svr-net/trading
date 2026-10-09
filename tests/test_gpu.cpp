#include <algorithm>
#include <cmath>

#include "sat/adaptive/experiment.hpp"
#include "sat/data/synthetic_market.hpp"
#include "sat/gpu/fused_backtest.hpp"
#include "test_framework.hpp"

using namespace sat;

namespace {

const PredictionSet& predictions() {
  static const PredictionSet p = [] {
    SyntheticMarketSpec ms;
    ms.numAssets = 20;
    ms.numDates = 700;
    const MarketData d = generateSyntheticMarket(ms);
    ExperimentSpec spec;
    spec.alphaIds = {2, 6, 12, 33, 35, 41};
    spec.walkForward.trainWindow = 300;
    spec.models.resize(3);
    spec.models[0].type = "logistic";
    spec.models[1].type = "xgboost";
    spec.models[1].trees = 25;
    spec.models[2].type = "forest";
    spec.models[2].trees = 15;
    return runPredictions(d, spec);
  }();
  return p;
}

std::vector<SelectorSpec> selectorGrid() {
  std::vector<SelectorSpec> s;
  for (std::size_t lb : {21, 42, 63})
    for (std::size_t step : {5, 21})
      for (ScoreMetric m : {ScoreMetric::Return, ScoreMetric::Sharpe, ScoreMetric::Sortino}) s.push_back({lb, step, m, 1, lb != 42, 0.0});
  return s;
}

}  // namespace

TEST(gpu_kernels_reproduce_the_cpu_grid) {
  const auto& p = predictions();
  const auto strategies = ExperimentSpec::defaultStrategies();
  const auto selectors = selectorGrid();
  const double cost = 10.0;
  const CandidateBook book(p.models, strategies, p.nextReturns, cost);
  const std::size_t evalFrom = 63;
  const GridResult cpu = evaluateGrid(book, selectors, evalFrom);
  const gpu::FusedPlan plan = gpu::compile(p.models, strategies, p.nextReturns, cost, selectors, evalFrom);
  CHECK(plan.numCandidates == book.size() && plan.numDays == book.days() && plan.header.size() == gpu::kHeaderWords);
  const GridResult g = gpu::summarise(plan, gpu::runFusedReference(plan));
  CHECK(g.candidates.size() == cpu.candidates.size() && g.selectors.size() == cpu.selectors.size());
  // Fixed candidates: the same arithmetic in single precision.
  for (std::size_t c = 0; c < cpu.candidates.size(); ++c) {
    CHECK_NEAR(g.candidates[c].totalReturn, cpu.candidates[c].totalReturn, 1e-3 * (1 + std::fabs(cpu.candidates[c].totalReturn)));
    CHECK_NEAR(g.candidates[c].sharpe, cpu.candidates[c].sharpe, 1e-3);
    CHECK_NEAR(g.candidates[c].maxDrawdown, cpu.candidates[c].maxDrawdown, 1e-4);
    CHECK_NEAR(g.candidates[c].averageTurnover, cpu.candidates[c].averageTurnover, 1e-5);
  }
  // Selectors: near-ties of the scores may resolve differently in f32, so most days agree
  // and the performance is close.
  std::size_t agree = 0, total = 0;
  for (std::size_t s = 0; s < selectors.size(); ++s) {
    for (std::size_t d = 0; d < cpu.selection[s].size(); ++d) {
      agree += cpu.selection[s][d] == g.selection[s][d];
      ++total;
    }
    CHECK_NEAR(g.selectors[s].sharpe, cpu.selectors[s].sharpe, 0.1);
  }
  CHECK(agree >= total * 97 / 100);
}

TEST(gpu_selection_and_turnover_are_exact_where_selection_agrees) {
  const auto& p = predictions();
  const std::vector<StrategySpec> strategies = {{StrategyKind::LongTopK, 3, 1}, {StrategyKind::LongShort, 3, 5}};
  const std::vector<SelectorSpec> selectors = {{21, 10, ScoreMetric::Return, 1, false, 0.0}};
  const CandidateBook book(p.models, strategies, p.nextReturns, 15.0);
  const GridResult cpu = evaluateGrid(book, selectors, 21);
  const gpu::FusedPlan plan = gpu::compile(p.models, strategies, p.nextReturns, 15.0, selectors, 21);
  const GridResult g = gpu::summarise(plan, gpu::runFusedReference(plan));
  for (std::size_t d = 0; d < cpu.selection[0].size(); ++d)
    if (cpu.selection[0][d] == g.selection[0][d] && (d == 0 || cpu.selection[0][d - 1] == g.selection[0][d - 1])) {
      CHECK_NEAR(g.adaptiveTurnover[0][d], cpu.adaptiveTurnover[0][d], 1e-5);
      CHECK_NEAR(g.adaptiveNet[0][d], cpu.adaptiveNet[0][d], 1e-5);
    }
}

TEST(gpu_limitations_and_validation) {
  const auto& p = predictions();
  const auto strategies = ExperimentSpec::defaultStrategies();
  CHECK(gpu::limitation(p.models, strategies, {{63, 21, ScoreMetric::Sharpe, 1, true, 0.0}}).empty());
  CHECK(!gpu::limitation(p.models, strategies, {{63, 21, ScoreMetric::Sharpe, 2, true, 0.0}}).empty());
  std::vector<ModelPredictions> many(9, p.models[0]);
  CHECK(!gpu::limitation(many, strategies, {}).empty());
  CHECK_THROWS(gpu::compile(p.models, strategies, p.nextReturns, 10, {{63, 21, ScoreMetric::Sharpe, 3, true, 0.0}}, 63));
  CHECK_THROWS(gpu::compile(p.models, strategies, p.nextReturns, 10, {}, 100000));
  const gpu::FusedPlan plan = gpu::compile(p.models, strategies, p.nextReturns, 10, {}, 10);
  gpu::FusedOutput bad;
  bad.stats.assign(3, 0.0f);
  CHECK_THROWS(gpu::summarise(plan, bad));
  CHECK(plan.adaptBytes() > 0 && plan.statsBytes() == plan.numSeries() * 32);
  // The CPU reference (the browser's emulated GPU) rejects plans whose header points outside the tables.
  CHECK(gpu::runFusedReference(plan).stats.size() == plan.numSeries() * gpu::kStats);
  gpu::FusedPlan truncated = plan;
  truncated.tables.resize(truncated.tables.size() - 1);
  CHECK_THROWS(gpu::runFusedReference(truncated));
  gpu::FusedPlan tooWide = plan;
  tooWide.header[1] = static_cast<std::uint32_t>(gpu::kMaxAssets + 1);
  CHECK_THROWS(gpu::runFusedReference(tooWide));
  gpu::FusedPlan shortHeader = plan;
  shortHeader.header.pop_back();
  CHECK_THROWS(gpu::runFusedReference(shortHeader));
}

TEST(gpu_kernel_sources) {
  for (const std::string* k : {&gpu::candidateBacktestKernel(), &gpu::adaptiveSelectKernel(), &gpu::seriesSummaryKernel()}) {
    CHECK(k->find("fn main") != std::string::npos);
    CHECK(k->find("@workgroup_size(64)") != std::string::npos);
    // Reserved WGSL words must not be used as identifiers.
    for (const char* reserved : {" active ", "let from", "var from", " filter ", " target "}) CHECK(k->find(reserved) == std::string::npos);
  }
  CHECK(gpu::candidateBacktestKernel().find("var<workgroup> sProb : array<f32, 1024>") != std::string::npos);
}

TEST(gpu_candidate_book_from_the_kernels) {
  // The candidate-backtest kernel's book (gross return and turnover per candidate and day)
  // rebuilds the CPU candidate book to f32 precision, and the selector runs on it unchanged.
  const auto& p = predictions();
  const auto strategies = ExperimentSpec::defaultStrategies();
  const double cost = 10.0;
  const CandidateBook cpu(p.models, strategies, p.nextReturns, cost);
  const gpu::FusedPlan plan = gpu::compile(p.models, strategies, p.nextReturns, cost, {}, 0);  // candidates only
  const gpu::FusedOutput out = gpu::runFusedReference(plan);
  CHECK(out.book.size() == cpu.size() * cpu.days() * 2);
  const CandidateBook kernels(p.models, strategies, p.nextReturns, cost, out.book);
  CHECK(kernels.size() == cpu.size() && kernels.days() == cpu.days() && kernels.start() == cpu.start());
  double worst = 0;
  for (std::size_t c = 0; c < cpu.size(); ++c)
    for (std::size_t d = 0; d < cpu.days(); ++d) worst = std::max(worst, std::fabs(kernels.net(c, d) - cpu.net(c, d)));
  CHECK(worst < 1e-5);
  const SelectorSpec sel;
  const auto a = runSelector(cpu, sel), b = runSelector(kernels, sel);
  CHECK_NEAR(b.metrics.sharpe, a.metrics.sharpe, 0.05);
  CHECK_THROWS(CandidateBook(p.models, strategies, p.nextReturns, cost, std::vector<float>(out.book.begin(), out.book.end() - 2)));
}
