import { dateAxis, fmt, heatmap, lineChart } from '../charts.js';
import { engineNote, engineSelector, gpuScopeNote, runAnalysis } from '../gpu/backend.js';
import { PERF_HEADERS, card, grid, initPage, perfRow, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'strategies',
  title: 'Fixed strategies',
  context: 'sat/strategy: StrategySpec · strategyWeights · backtest · CandidateBook',
  description: 'Earlier studies turn forecasts into a single fixed trading rule. Here every model drives every rule in the pool: ' +
    'long the top <i>k</i>, dollar-neutral long-short, long whatever beats a probability threshold, or probability-weighted. Each rule rebalances at the close on its schedule and pays costs on turnover. ' +
    'The table ranks all candidates over the evaluation period. The rolling-winner view shows that the best rule changes from quarter to quarter, which is why the paper adapts the choice.',
});
specEditor(page, ['market', 'csv', 'factors', 'labels', 'walkForward', 'models', 'strategies', 'costs']);

const button = runButton(page, 'Backtest every candidate', async (spec) => {
  const t0 = performance.now();
  const runResult = await runAnalysis('strategies', spec);
  const r = runResult.result;
  const onGpu = runResult.engine === 'gpu';
  window.__satEngineRun = { page: 'strategies', engine: runResult.engine, bestSharpe: r.bestFixed.metrics.sharpe, medianSharpe: r.medianFixedSharpe };
  if (onGpu) gpuScopeNote(page.content, 'the equity curve of every candidate and the rolling winners');
  const sorted = r.candidates.map((c, i) => ({ ...c, i })).sort((a, b) => b.metrics.sharpe - a.metrics.sharpe);
  const winners = r.blocks ? new Set(r.blocks.map((b) => b.winner)).size : null;
  tiles(page.content, [
    { label: 'Candidates', value: fmt.num(r.candidates.length), hint: `${r.models.length} models × ${r.strategyLabels.length} rules` },
    { label: 'Best (hindsight)', value: fmt.ratio(r.bestFixed.metrics.sharpe), hint: `Sharpe: ${r.bestFixed.label}` },
    { label: 'Median Sharpe', value: fmt.ratio(r.medianFixedSharpe), hint: `median annual return ${fmt.pct(r.medianFixedAnnualReturn, 1)}` },
    { label: 'Market Sharpe', value: fmt.ratio(r.benchmark.metrics.sharpe), hint: 'equal-weight benchmark' },
    winners === null ? { label: 'Distinct quarterly winners', value: '–', hint: 'WebAssembly engine only' }
      : { label: 'Distinct quarterly winners', value: `${winners} of ${r.blocks.length}`, hint: 'quarters won by different candidates' },
  ]);
  const g = grid(page.content);
  heatmap(card(g, 'Sharpe ratio by model and rule', `Over ${fmt.num(r.evalDays)} out-of-sample days, net of ${r.costBps} bp costs.`), {
    rows: r.models, cols: r.strategyLabels, values: r.models.map((_, m) => r.strategyLabels.map((_, s) => r.candidates[m * r.strategyLabels.length + s].metrics.sharpe)),
    format: (v) => v.toFixed(2), rowWidth: 130,
  });
  if (!onGpu) {
    const x = r.dates.map((_, i) => i);
    const bestOfModel = r.models.map((_, m) => sorted.find((c) => c.modelIndex === m));
    lineChart(card(g, 'Best rule of each model', 'Wealth of the highest-Sharpe rule per model, against the market.'), {
      x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
      series: [...bestOfModel.map((c, k) => ({ name: `${c.model} · ${c.strategy}`, y: Array.from(r.equityCurves[c.i]), colorIndex: k })),
        { name: 'market', y: Array.from(r.benchmark.equity), colorIndex: 7, dash: true }],
    });
    const box = card(page.content, 'Winner of each quarter', 'The candidate with the highest return in each 63-day block, and where the overall best rule ranked. Hindsight is no strategy.');
    table(box, ['from', 'to', 'regime', 'winner', 'return', 'rank of overall best', 'its return', 'self-adaptive'],
      r.blocks.map((b) => [b.from, b.to, r.regimeNames[b.regime] ?? '–', `${r.candidates[b.winner].model} · ${r.candidates[b.winner].strategy}`, fmt.pct(b.winnerReturn, 1),
        `${b.bestFixedRank} / ${r.candidates.length}`, fmt.pct(b.bestFixedReturn, 1), fmt.pct(b.adaptiveReturn, 1)]));
  }
  const all = card(page.content, 'All candidates', 'Sorted by Sharpe ratio over the evaluation period.');
  table(all, ['model', ...PERF_HEADERS], sorted.map((c) => [c.model, ...perfRow(c.strategy, c.metrics)]));
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
