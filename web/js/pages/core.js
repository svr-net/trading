import { run } from '../sat-client.js';
import { barChart, dateAxis, fmt, histogram, lineChart } from '../charts.js';
import { card, grid, initPage, passFail, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'core',
  title: 'Core numerics',
  context: 'sat/core · sat/factors/operators: Panel · Rng · ts_* and cross-sectional operators',
  description: 'Every factor is a formula in a small operator algebra over date × stock panels. ' +
    'Time-series operators look back over a stock\'s past <i>d</i> days, including today. Cross-sectional operators compare all stocks on one date. ' +
    'The generator is a seeded xoshiro256** with explicit Box–Muller normals, so a seed gives the same market on every platform. ' +
    'The last check recomputes all 23 alphas without the final 50 days. No value may change, which proves that no factor looks ahead.',
});
specEditor(page, ['market']);

runButton(page, 'Run', async (spec) => {
  const r = await run('coreDemo', spec);
  const x = Array.from(r.close, (_, i) => i), xf = dateAxis(r.dates);
  const passed = r.lookahead.filter((a) => a.maxDiff < 1e-9 && a.nanMismatches === 0).length;
  tiles(page.content, [
    { label: 'Normal mean', value: r.normals.mean.toFixed(4), hint: '20,000 draws' },
    { label: 'Normal sd', value: r.normals.sd.toFixed(4) },
    { label: 'Skew / kurtosis', value: `${r.normals.skew.toFixed(3)} / ${r.normals.kurtosis.toFixed(2)}`, hint: 'normal: 0 / 3' },
    { label: 'No look-ahead', value: `${passed} / ${r.lookahead.length}`, hint: 'alphas unchanged without the last 50 days' },
  ]);
  const g = grid(page.content);
  lineChart(card(g, `ts_mean and ts_stddev: ${r.ticker}`, 'Close with its 20-day mean and a band of two 20-day standard deviations.'), {
    x, xFormat: xf, xLabel: 'date',
    series: [
      { name: 'close', y: Array.from(r.close) },
      { name: 'ts_mean(close, 20)', y: Array.from(r.mean20), colorIndex: 1, band: { lower: Array.from(r.lower), upper: Array.from(r.upper) } },
    ],
  });
  lineChart(card(g, 'ts_rank and correlation', 'Rank of today\'s close among the last 10 days, and the 10-day correlation of close and volume.'), {
    x, xFormat: xf, xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [{ name: 'ts_rank(close, 10)', y: Array.from(r.tsRank10) }, { name: 'corr(close, volume, 10)', y: Array.from(r.corr10), colorIndex: 2 }],
  });
  barChart(card(g, 'Cross-sectional rank on the last date', 'rank() maps each stock\'s close to its percentile among all stocks.'), {
    labels: r.tickers, series: [{ name: 'rank(close)', values: Array.from(r.lastRank) }], yFormat: (v) => v.toFixed(2),
  });
  histogram(card(g, 'Standard normals from the generator', 'Box–Muller transform of xoshiro256** uniforms.'), Array.from(r.normals.samples), { bins: 50, name: 'draws', xFormat: (v) => v.toFixed(1) });
  const box = card(page.content, 'Look-ahead check of every alpha', `Each alpha computed on all dates and on the first ${fmt.num(r.lookaheadCut)} dates only; values on the common dates must agree.`);
  table(box, ['alpha', 'max |Δ|', 'NaN mismatches', 'result'], r.lookahead.map((a) => [`#${a.id}`, a.maxDiff.toExponential(1), fmt.num(a.nanMismatches), passFail(a.maxDiff < 1e-9 && a.nanMismatches === 0)]));
  window.__satLookahead = passed === r.lookahead.length;
});
