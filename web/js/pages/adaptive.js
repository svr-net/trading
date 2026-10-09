import { dateAxis, fmt, lineChart } from '../charts.js';
import { engineNote, engineSelector, gpuScopeNote, runAnalysis } from '../gpu/backend.js';
import { PERF_HEADERS, card, grid, initPage, perfRow, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'adaptive',
  title: 'Self-adaptive strategy',
  context: 'sat/adaptive: SelectorSpec · runSelector · windowScore',
  description: 'The paper\'s contribution: instead of committing to one rule, re-score every candidate (model × rule) at regular intervals on its recent out-of-sample record, and trade the best. ' +
    'The score is the Sharpe ratio, Sortino ratio or mean return over the look-back. When nothing scores above the minimum, the strategy stays in cash, which controls the risk in falling markets. ' +
    'A switch pays the cost of the turnover between the two portfolios. All strategies are compared over the same days, after the first look-back.',
});
specEditor(page, ['market', 'csv', 'factors', 'labels', 'walkForward', 'models', 'strategies', 'costs', 'selector']);

const button = runButton(page, 'Run the self-adaptive strategy', async (spec) => {
  const t0 = performance.now();
  const runResult = await runAnalysis('adaptive', spec);
  const r = runResult.result;
  const onGpu = runResult.engine === 'gpu' || runResult.engine === 'emulator';  // kernel results, real or emulated GPU
  const a = r.adaptive;
  window.__satEngineRun = { page: 'adaptive', engine: runResult.engine, sharpe: a.metrics.sharpe, annualReturn: a.metrics.annualReturn, switches: a.switches };
  if (onGpu) gpuScopeNote(page.content, 'the average of all fixed rules', runResult.engine);
  const cash = a.share[a.share.length - 1];
  tiles(page.content, [
    { label: 'Annual return', value: fmt.pct(a.metrics.annualReturn, 1), hint: `best fixed ${fmt.pct(r.bestFixed.metrics.annualReturn, 1)}, market ${fmt.pct(r.benchmark.metrics.annualReturn, 1)}` },
    { label: 'Sharpe ratio', value: fmt.ratio(a.metrics.sharpe), hint: `median fixed ${fmt.ratio(r.medianFixedSharpe)}` },
    { label: 'Max drawdown', value: fmt.pct(a.metrics.maxDrawdown, 1), hint: `best fixed ${fmt.pct(r.bestFixed.metrics.maxDrawdown, 1)}` },
    { label: 'Switches', value: fmt.num(a.switches), hint: `${fmt.pct(cash, 0)} of days in cash` },
    { label: 'Models trained in', value: `${fmt.num(r.trainMs)} ms`, hint: r.cachedPredictions ? 'cached' : 'WebAssembly' },
  ]);
  const g = grid(page.content);
  const x = r.dates.map((_, i) => i), xf = dateAxis(r.dates);
  lineChart(card(g, 'Wealth', `${a.label}. Net of ${r.costBps} bp per unit of turnover.`), {
    x, xFormat: xf, xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [
      { name: 'self-adaptive', y: Array.from(a.equity) },
      { name: `best fixed: ${r.bestFixed.label}`, y: Array.from(r.bestFixed.equity), colorIndex: 1 },
      ...(r.averageFixed ? [{ name: 'average of all fixed rules', y: Array.from(r.averageFixed.equity), colorIndex: 4 }] : []),
      { name: 'equal-weight market', y: Array.from(r.benchmark.equity), colorIndex: 2, dash: true },
    ],
  });
  lineChart(card(g, 'Drawdown', 'Loss from the running peak of wealth.'), {
    x, xFormat: xf, xLabel: 'date', yFormat: (v) => fmt.pct(-v, 0),
    series: [
      { name: 'self-adaptive', y: Array.from(a.drawdown, (v) => -v) },
      { name: 'best fixed', y: Array.from(r.bestFixed.drawdown, (v) => -v), colorIndex: 1 },
      { name: 'market', y: Array.from(r.benchmark.drawdown, (v) => -v), colorIndex: 2, dash: true },
    ],
  });
  const S = r.strategyLabels.length;
  const sel = Array.from(a.selection);
  const days = sel.map((_, i) => i + 1);
  lineChart(card(g, 'What the strategy holds', 'Model and rule held each day (−1 = cash), with the hidden market regime.'), {
    x: days, xFormat: xf, xLabel: 'date', height: 240,
    yFormat: (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : ''),
    tooltipFormat: (v) => String(v),
    series: [
      { name: 'model (index)', y: sel.map((c) => (c < 0 ? -1 : Math.floor(c / S))), step: true },
      { name: 'rule (index)', y: sel.map((c) => (c < 0 ? -1 : c % S)), step: true, colorIndex: 1 },
      ...(r.regimeNames.length ? [{ name: 'regime', y: Array.from(r.regime), step: true, colorIndex: 3, dash: true }] : []),
    ],
  });
  const shares = Array.from(a.share).slice(0, -1).map((s, i) => ({ s, i })).filter((v) => v.s > 0).sort((p, q) => q.s - p.s).slice(0, 10);
  const box = card(g, 'Most held candidates', `Share of days held. Models: ${r.models.map((m, i) => `${i} ${m}`).join(', ')}.`);
  table(box, ['candidate', 'share of days', 'Sharpe as a fixed rule'], [
    ...shares.map(({ s, i }) => [`${r.candidates[i].model} · ${r.candidates[i].strategy}`, fmt.pct(s, 1), fmt.ratio(r.candidates[i].metrics.sharpe)]),
    ['cash', fmt.pct(cash, 1), '–'],
  ]);
  const perf = card(page.content, 'Performance over the evaluation period', '');
  table(perf, PERF_HEADERS, [
    perfRow('Self-adaptive', a.metrics),
    perfRow(`Best fixed (hindsight): ${r.bestFixed.label}`, r.bestFixed.metrics),
    ...(r.averageFixed ? [perfRow('Average of all fixed rules', r.averageFixed.metrics)] : []),
    perfRow('Equal-weight market', r.benchmark.metrics),
  ]);
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
