import { barChart, fmt, heatmap, lineChart } from '../charts.js';
import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'robustness',
  title: 'Robustness',
  context: 'sat/adaptive: evaluateGrid · GridResult',
  description: 'Does adaptation depend on lucky settings? The selector is run on a grid of look-backs, adaptation periods and score metrics, all over the same evaluation days (after the longest look-back). ' +
    'Separate runs at several transaction costs show how much turnover the strategy can afford. The year-by-year table compares it with the best fixed rule and the market. ' +
    'Every setting is a separate run over the same candidate record, which is the workload the WebGPU kernels parallelise.',
});
specEditor(page, ['market', 'csv', 'models', 'strategies', 'costs', 'selector', 'robustness']);

const button = runButton(page, 'Run the robustness study', async (spec) => {
  const t0 = performance.now();
  const runResult = await runAnalysis('robustness', spec);
  const r = runResult.result;
  window.__satEngineRun = { page: 'robustness', engine: runResult.engine, shareBeatingMedianFixed: r.shareBeatingMedianFixed, sharpe: r.adaptive.metrics.sharpe };
  tiles(page.content, [
    { label: 'Settings tested', value: fmt.num(r.gridSize), hint: `${r.lookbacks.length} look-backs × ${r.steps.length} steps × ${r.grids.length} scores` },
    { label: 'Beat the median fixed rule', value: fmt.pct(r.shareBeatingMedianFixed, 0), hint: `Sharpe above ${fmt.ratio(r.medianFixedSharpe)}` },
    { label: 'Beat the market', value: fmt.pct(r.shareBeatingBenchmark, 0), hint: `Sharpe above ${fmt.ratio(r.benchmark.metrics.sharpe)}` },
    { label: 'Your setting', value: fmt.ratio(r.adaptive.metrics.sharpe), hint: `Sharpe, ${fmt.num(r.evalDays)} days` },
  ]);
  const g = grid(page.content);
  for (const gr of r.grids)
    heatmap(card(g, `Sharpe ratio, scored by ${gr.metric}`, 'Rows: look-back (days). Columns: adaptation period (days).'), {
      rows: r.lookbacks, cols: r.steps, values: gr.sharpe.map((row) => Array.from(row)), format: (v) => v.toFixed(2),
      xLabel: 'adapt every', yLabel: 'look-back', center: r.medianFixedSharpe, rowWidth: 56,
    });
  const costs = r.costs;
  lineChart(card(g, 'Transaction costs', 'Sharpe ratio against the cost per unit of turnover.'), {
    x: costs.map((c) => c.costBps), xLabel: 'cost (bp)', yFormat: (v) => v.toFixed(2),
    series: [
      { name: 'self-adaptive', y: costs.map((c) => c.adaptive.sharpe), markers: true },
      { name: 'best fixed at that cost', y: costs.map((c) => c.bestFixed.sharpe), colorIndex: 1, markers: true },
      { name: 'average fixed', y: costs.map((c) => c.averageFixedSharpe), colorIndex: 4, markers: true },
      { name: 'market', y: costs.map((c) => c.benchmark.sharpe), colorIndex: 2, dash: true },
    ],
  });
  barChart(card(g, 'Year by year', 'Return of each calendar year (part years at the ends).'), {
    labels: r.years.map((y) => y.year),
    series: [{ name: 'self-adaptive', values: r.years.map((y) => y.adaptive) }, { name: 'best fixed', values: r.years.map((y) => y.bestFixed) }, { name: 'market', values: r.years.map((y) => y.benchmark) }],
    yFormat: (v) => fmt.pct(v, 0), tooltipFormat: (v) => fmt.pct(v, 1),
  });
  const box = card(page.content, 'Costs', 'The best fixed rule is re-chosen in hindsight at each cost level.');
  table(box, ['cost (bp)', 'self-adaptive return', 'Sharpe', 'turnover/day', 'best fixed', 'its Sharpe', 'market Sharpe'],
    costs.map((c) => [String(c.costBps), fmt.pct(c.adaptive.annualReturn, 1), fmt.ratio(c.adaptive.sharpe), fmt.ratio(c.adaptive.averageTurnover), c.bestFixedLabel, fmt.ratio(c.bestFixed.sharpe), fmt.ratio(c.benchmark.sharpe)]));
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
