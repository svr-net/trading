// End-to-end demonstration of the self-adaptive trading strategy on a synthetic
// regime-switching stock market: alpha factors, walk-forward machine-learning forecasts,
// a pool of fixed strategies, and the selector that adapts to the market.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "sat/sat.hpp"

using namespace sat;

namespace {

void row(const std::string& name, const PerformanceMetrics& m) {
  std::printf("  %-44s %8.1f%% %7.1f%% %6.2f %6.2f %7.1f%% %6.2f\n", name.c_str(), 100 * m.annualReturn, 100 * m.annualVolatility,
              m.sharpe, m.sortino, 100 * m.maxDrawdown, m.averageTurnover);
}

void header(const char* title) {
  std::printf("\n%s\n  %-44s %9s %8s %6s %6s %8s %6s\n", title, "strategy", "ann.ret", "ann.vol", "Sharpe", "Sortino", "max DD",
              "turn.");
}

}  // namespace

int main() {
  SyntheticMarketSpec ms;  // 30 stocks, 5 years of daily bars, bull / bear / range-bound regimes
  const MarketData data = generateSyntheticMarket(ms);
  std::printf("Synthetic market: %zu stocks x %zu days (%s .. %s)\n", data.numAssets(), data.numDates(), data.dates.front().c_str(),
              data.dates.back().c_str());

  ExperimentSpec spec;  // the paper's 23 alphas, next-day direction labels, six model families
  const PredictionSet p = runPredictions(data, spec);
  std::printf("\nWalk-forward forecasts (train %zu days, re-fit every %zu days, out of sample from %s)\n", spec.walkForward.trainWindow,
              spec.walkForward.retrainEvery, data.dates[p.models[0].start].c_str());
  std::printf("  %-22s %8s %8s %8s %8s\n", "model", "accuracy", "AUC", "F1", "time ms");
  for (const auto& m : p.models)
    std::printf("  %-22s %8.3f %8.3f %8.3f %8.0f\n", m.name.c_str(), m.oos.accuracy, m.oos.auc, m.oos.f1, m.elapsedMs);

  const Experiment e = runStrategies(p, spec);
  header("Fixed strategies (top 8 by Sharpe ratio, in hindsight)");
  std::vector<std::size_t> order(e.book.size());
  for (std::size_t c = 0; c < order.size(); ++c) order[c] = c;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return e.candidateMetrics[a].sharpe > e.candidateMetrics[b].sharpe; });
  for (std::size_t k = 0; k < std::min<std::size_t>(8, order.size()); ++k) row(e.book.candidates()[order[k]].label, e.candidateMetrics[order[k]]);

  header("Self-adaptive strategy against the alternatives");
  row(e.adaptive.spec.label(), e.adaptive.metrics);
  row("Best fixed strategy (chosen in hindsight)", e.candidateMetrics[e.bestFixed]);
  double avgSharpe = 0;
  for (const auto& m : e.candidateMetrics) avgSharpe += m.sharpe / static_cast<double>(e.candidateMetrics.size());
  row(e.benchmark.label, e.benchmark.metrics);
  std::printf("  average Sharpe ratio of the %zu fixed strategies: %.2f; switches: %zu; days in cash: %.0f%%\n", e.book.size(), avgSharpe,
              e.adaptive.switches, 100 * e.adaptive.share.back());

  // Robustness: the selector's look-back and adaptation period, evaluated over the same days.
  std::vector<SelectorSpec> grid;
  for (std::size_t lb : {21, 42, 63, 126})
    for (std::size_t step : {5, 21, 63}) grid.push_back({lb, step, ScoreMetric::Sharpe, 1, true, 0.0});
  const GridResult g = evaluateGrid(e.book, grid, 126);
  std::printf("\nSelector grid (Sharpe ratio from day 126):\n  look-back \\ every   5d     21d    63d\n");
  for (std::size_t a = 0; a < 4; ++a)
    std::printf("  %9zu        %6.2f %6.2f %6.2f\n", grid[a * 3].lookback, g.selectors[a * 3].sharpe, g.selectors[a * 3 + 1].sharpe,
                g.selectors[a * 3 + 2].sharpe);

  // The same grid on the GPU kernels (executed here by their CPU reference).
  const auto plan = gpu::compile(p.models, ExperimentSpec::defaultStrategies(), p.nextReturns, spec.costBps, grid, 126);
  const GridResult gg = gpu::summarise(plan, gpu::runFusedReference(plan));
  double worst = 0;
  for (std::size_t s = 0; s < grid.size(); ++s) worst = std::max(worst, std::fabs(gg.selectors[s].sharpe - g.selectors[s].sharpe));
  std::printf("GPU kernels (f32 reference) vs CPU: largest Sharpe difference %.4f over %zu selectors x %zu candidates\n", worst,
              grid.size(), plan.numCandidates);

  // Advances in Financial Machine Learning: how much of this survives the search?
  const auto rep = afml::assessOverfitting(e.book, e.adaptive.net, e.evalFrom);
  const double ann = std::sqrt(kTradingDaysPerYear);
  std::printf("\nBacktest overfitting (%zu candidates = trials, %zu days)\n", rep.trials, rep.days);
  std::printf("  expected maximum Sharpe ratio of %zu unskilled trials: %.2f (annualised)\n", rep.trials, rep.expectedMaxSharpe * ann);
  std::printf("  best fixed:    Sharpe %.2f, PSR %.1f%%, deflated Sharpe %.1f%%\n", rep.best.annualSharpe, 100 * rep.best.psr, 100 * rep.best.dsr);
  std::printf("  self-adaptive: Sharpe %.2f, PSR %.1f%%, deflated Sharpe %.1f%%\n", rep.adaptive.annualSharpe, 100 * rep.adaptive.psr,
              100 * rep.adaptive.dsr);
  std::printf("  probability of backtest overfitting (CSCV, %zu combinations): %.1f%%\n", rep.pbo.combinations, 100 * rep.pbo.pbo);
  return 0;
}
