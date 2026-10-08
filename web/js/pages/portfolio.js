import { run } from '../sat-client.js';
import { barChart, dateAxis, fmt, heatmap, lineChart } from '../charts.js';
import { PERF_HEADERS, card, grid, initPage, perfRow, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'portfolio',
  title: 'Hierarchical risk parity',
  context: 'sat/afml/portfolio: clusterAssets · hierarchicalRiskParity · minimumVarianceWeights',
  description: 'Mean-variance optimisers invert the covariance matrix, which amplifies estimation error, and they tend to concentrate in a few assets. ' +
    '<b>Hierarchical risk parity</b> (chapter 16) needs no inversion. It clusters the assets by correlation, reorders the covariance matrix so that similar assets sit together (quasi-diagonalisation), and splits the capital top-down between clusters in inverse proportion to their variance. ' +
    'Here it is compared with inverse-variance and minimum-variance weights: on the stock universe, in a walk-forward backtest with periodic re-estimation, and in a Monte Carlo study of out-of-sample variance.',
});
specEditor(page, ['market', 'csv', 'portfolio'], { open: false });

runButton(page, 'Allocate', async (spec) => {
  const r = await run('afmlPortfolio', spec);
  const mc = r.monteCarlo;
  const avg = (v) => Array.from(v).reduce((a, b) => a + b, 0) / v.length;
  const maxW = (w) => Math.max(...Array.from(w));
  tiles(page.content, [
    { label: 'Largest weight: HRP', value: fmt.pct(maxW(r.hrp), 1), hint: 'last estimation window' },
    { label: 'Largest weight: min-variance', value: fmt.pct(maxW(r.minVar), 1), hint: `gross ${fmt.pct(Array.from(r.minVar).reduce((a, b) => a + Math.abs(b), 0), 0)}` },
    { label: 'MC out-of-sample vol: HRP', value: fmt.pct(Math.sqrt(avg(mc.hrp)), 2), hint: `${mc.trials} trials, ${mc.assets} assets` },
    { label: 'MC out-of-sample vol: IVP', value: fmt.pct(Math.sqrt(avg(mc.ivp)), 2) },
    { label: 'MC out-of-sample vol: min-variance', value: fmt.pct(Math.sqrt(avg(mc.minVar)), 2), hint: `unconstrained, gross ${fmt.pct(mc.minVarGross, 0)}` },
    { label: 'MC out-of-sample vol: long-only min-var', value: fmt.pct(Math.sqrt(avg(mc.longOnly)), 2), hint: `largest weight ${fmt.pct(mc.longOnlyMaxWeight, 0)} vs HRP ${fmt.pct(mc.hrpMaxWeight, 0)}` },
  ]);
  const g = grid(page.content);
  heatmap(card(g, 'Correlation, original order', `${r.tickers.length} stocks over the last ${r.window} days.`), {
    rows: r.tickers, cols: r.tickers, values: r.correlation.map((x) => Array.from(x)), format: (v) => v.toFixed(1), rowWidth: 64,
  }).canvas.style.height = '420px';
  heatmap(card(g, 'Correlation, quasi-diagonalised', 'Rows and columns in the order of the clustering tree: correlated stocks form blocks along the diagonal.'), {
    rows: r.orderedTickers, cols: r.orderedTickers, values: r.orderedCorrelation.map((x) => Array.from(x)), format: (v) => v.toFixed(1), rowWidth: 64,
  }).canvas.style.height = '420px';
  barChart(card(g, 'Weights on the last window', 'Minimum-variance weights may be negative (short) and concentrated.'), {
    labels: r.tickers,
    series: [{ name: 'HRP', values: Array.from(r.hrp) }, { name: 'inverse variance', values: Array.from(r.ivp) }, { name: 'minimum variance', values: Array.from(r.minVar) }],
    yFormat: (v) => fmt.pct(v, 0), tooltipFormat: (v) => fmt.pct(v, 2),
  });
  const bt = r.backtest;
  lineChart(card(g, 'Walk-forward backtest', `Weights re-estimated every ${r.rebalance} days on the previous ${r.window} days.`), {
    x: Array.from(bt[0].equity, (_, i) => i), xFormat: dateAxis(r.backtestDates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: bt.map((b, i) => ({ name: b.name, y: Array.from(b.equity), colorIndex: i })),
  });
  const box = card(page.content, 'Backtest', 'Volatility is what these allocations target; HRP aims at stability and diversification rather than the lowest in-sample variance.');
  table(box, PERF_HEADERS, bt.map((b) => perfRow(b.name, b.metrics)));
  const mcBox = card(page.content, 'Monte Carlo: out-of-sample variance', 'Clusters of correlated assets with random volatilities and occasional common shocks. Weights are estimated on 260 days and held for the next 22. ' +
    'In this setting the minimum-variance portfolios, long-only or not, reach a lower out-of-sample variance than HRP, at the price of much larger single positions (and shorts, unconstrained). HRP stays close to inverse-variance risk while spreading capital across clusters.');
  table(mcBox, ['method', 'mean OOS variance (annual)', 'OOS volatility', 'average largest weight'], [
    ['HRP', avg(mc.hrp).toExponential(3), fmt.pct(Math.sqrt(avg(mc.hrp)), 2), fmt.pct(mc.hrpMaxWeight, 1)],
    ['inverse variance', avg(mc.ivp).toExponential(3), fmt.pct(Math.sqrt(avg(mc.ivp)), 2), fmt.pct(mc.ivpMaxWeight, 1)],
    ['minimum variance (long-only)', avg(mc.longOnly).toExponential(3), fmt.pct(Math.sqrt(avg(mc.longOnly)), 2), fmt.pct(mc.longOnlyMaxWeight, 1)],
    ['minimum variance (unconstrained)', avg(mc.minVar).toExponential(3), fmt.pct(Math.sqrt(avg(mc.minVar)), 2), fmt.pct(mc.minVarMaxWeight, 1)],
  ]);
});
