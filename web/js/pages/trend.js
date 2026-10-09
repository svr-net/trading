import { run } from '../sat-client.js';
import { dateAxis, fmt, lineChart } from '../charts.js';
import { PERF_HEADERS as ALL_HEADERS, card, grid, initPage, perfRow as fullRow, runButton, specEditor, table, tiles } from '../ui.js';

// These series carry no turnover, so the turnover column is left out.
const PERF_HEADERS = ALL_HEADERS.slice(0, -1);
const perfRow = (name, m) => fullRow(name, m).slice(0, -1);

const page = initPage({
  id: 'trend',
  gpu: false,
  title: 'Trend following',
  context: 'sat/algo/trend: trendFollowing (EWMAC rules, forecast scaling, volatility targeting, adaptive rule weights)',
  description: 'A systematic futures-style trend system after Carver\'s <i>Systematic Trading</i>, run on every stock. Six <b>EWMAC</b> rules compare a fast and a slow exponential moving average, from 2/8 to 64/256 days; each raw forecast is divided by price volatility, <b>scaled</b> so its average absolute value is 10 and <b>capped</b> at 20. ' +
    'Forecasts are combined with weights and diversification multipliers, and positions are sized so every stock carries an equal share of the portfolio\'s <b>volatility target</b>; a buffer skips small trades. ' +
    '<b>Self-adaptation</b>: every few months the rule weights are reset in proportion to each rule\'s positive Sharpe ratio over the past year, so the system leans to the speeds that are working. ' +
    'On the synthetic market the regimes are short and individual stocks show only short-horizon momentum and reversal, so trend following has little to work with here — the tournament page lets the meta-allocator discover that.',
});
specEditor(page, ['market', 'csv', 'trend'], { open: false });

runButton(page, 'Run', async (spec) => {
  const r = await run('algoTrend', spec);
  const [ad, st, bench] = r.series;
  const avg = (v) => Array.from(v).reduce((a, b) => a + b, 0) / v.length;
  tiles(page.content, [
    { label: 'Adaptive weights: Sharpe', value: fmt.ratio(ad.metrics.sharpe), hint: `equal weights ${fmt.ratio(st.metrics.sharpe)}` },
    { label: 'Realised volatility', value: fmt.pct(ad.metrics.annualVolatility, 1), hint: `target ${fmt.pct(spec.trend.targetVol, 0)}` },
    { label: 'Average gross leverage', value: fmt.ratio(avg(r.grossLeverage)), hint: `diversification multiplier ${fmt.ratio(r.idm[r.idm.length - 1])}` },
    { label: 'Average |forecast|', value: fmt.ratio(avg(r.forecast), 1), hint: 'scaled towards 10' },
  ]);
  const g = grid(page.content);
  const x = Array.from(ad.equity, (_, i) => i);
  lineChart(card(g, 'Equity', 'Net of costs.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: r.series.map((s, i) => ({ name: s.name, y: Array.from(s.equity), colorIndex: i, dash: i === 2 })),
  });
  const w = r.rules[0].weights;
  lineChart(card(g, 'Adaptive rule weights', 'Reset from each rule\'s trailing Sharpe ratio; equal when none is positive.'), {
    x: Array.from(w, (_, i) => i), xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => fmt.pct(v, 0), step: true,
    series: r.rules.map((rule, i) => ({ name: rule.name, y: Array.from(rule.weights), colorIndex: i, step: true })),
  });
  lineChart(card(g, 'Gross leverage', 'Sum of absolute stock weights, set by the volatility target and the forecasts.'), {
    x: Array.from(r.grossLeverage, (_, i) => i), xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(1),
    series: [{ name: 'gross leverage', y: Array.from(r.grossLeverage) }],
  });
  table(card(page.content, 'Systems', ''), PERF_HEADERS, r.series.map((s) => perfRow(s.name, s.metrics)));
  table(card(page.content, 'Rules traded alone', 'Same sizing, one rule at a time, before costs. Forecast scalar: what makes the average |forecast| 10.'),
    ['rule', 'ann. return', 'ann. vol', 'Sharpe', 'max DD', 'forecast scalar', 'final weight'],
    r.rules.map((rule) => [rule.name, fmt.pct(rule.metrics.annualReturn, 1), fmt.pct(rule.metrics.annualVolatility, 1), fmt.ratio(rule.metrics.sharpe),
      fmt.pct(rule.metrics.maxDrawdown, 1), fmt.ratio(rule.scalar, 1), fmt.pct(rule.weights[rule.weights.length - 1], 0)]));
});
