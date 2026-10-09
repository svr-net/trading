import { run } from '../sat-client.js';
import { dateAxis, fmt, lineChart } from '../charts.js';
import { PERF_HEADERS as ALL_HEADERS, card, grid, initPage, perfRow as fullRow, runButton, specEditor, table, tiles } from '../ui.js';

// These series carry no turnover, so the turnover column is left out.
const PERF_HEADERS = ALL_HEADERS.slice(0, -1);
const perfRow = (name, m) => fullRow(name, m).slice(0, -1);

const page = initPage({
  id: 'hedging',
  title: 'Beta hedging & position sizing',
  context: 'sat/hedge/hedging: rollingBeta · kalmanRegression · applyHedge · volatilityTarget · kellyScale',
  description: 'The self-adaptive strategy is long stocks most of the time, so part of its return is plain market exposure. ' +
    'A <b>beta hedge</b> shorts the market in proportion to the strategy\'s sensitivity to it, the minimum-variance hedge ratio of the futures-hedging literature (Hull). ' +
    'The ratio is estimated on a rolling window, or tracked day by day by a <b>Kalman filter</b> that treats beta as a random walk (as Chan does for pairs). ' +
    'Separately, the size of the bet adapts: <b>volatility targeting</b> scales the position to a constant expected risk (Carver), and <b>fractional Kelly</b> to the trailing mean over variance. ' +
    'Every estimate uses only data up to the day before. All overlays are compared over the same days, after the longest warm-up.',
});
specEditor(page, ['market', 'csv', 'models', 'strategies', 'costs', 'selector', 'hedging'], { open: false });

runButton(page, 'Hedge', async (spec) => {
  const r = await run('hedgeOverlays', spec);
  const s = r.series;
  const by = (n) => s.find((x) => x.name === n);
  const un = by('self-adaptive (unhedged)'), kal = by('Kalman-beta hedge'), vt = by('volatility target'), kel = by('fractional Kelly');
  tiles(page.content, [
    { label: 'Market correlation: unhedged', value: fmt.ratio(r.marketCorrelation[0]), hint: `beta in hindsight ${fmt.ratio(r.hindsightBeta)}` },
    { label: 'Market correlation: Kalman hedge', value: fmt.ratio(r.marketCorrelation[2]), hint: `Sharpe ${fmt.ratio(kal.metrics.sharpe)} vs ${fmt.ratio(un.metrics.sharpe)}` },
    { label: 'Volatility target', value: fmt.pct(vt.metrics.annualVolatility, 1), hint: `max DD ${fmt.pct(vt.metrics.maxDrawdown, 1)} vs ${fmt.pct(un.metrics.maxDrawdown, 1)}` },
    { label: 'Fractional Kelly', value: fmt.pct(kel.metrics.annualReturn, 1), hint: `volatility ${fmt.pct(kel.metrics.annualVolatility, 1)}, max DD ${fmt.pct(kel.metrics.maxDrawdown, 1)}` },
  ]);
  const g = grid(page.content);
  const x = Array.from(s[0].equity, (_, i) => i);
  lineChart(card(g, 'Equity of the overlays', `${r.selector}. Log-scale growth differs a lot between sizing rules; compare the Sharpe ratios in the table.`), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: s.map((e, i) => ({ name: e.name, y: Array.from(e.equity), colorIndex: i, dash: e.name === 'equal-weight market' })),
  });
  lineChart(card(g, 'Estimated beta to the market', 'Rolling OLS against the Kalman filter; the dashed line is the full-sample (hindsight) beta.'), {
    x: Array.from(r.kalmanBeta, (_, i) => i), xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [{ name: 'rolling', y: Array.from(r.rollingBeta) }, { name: 'Kalman', y: Array.from(r.kalmanBeta) },
      { name: 'hindsight', y: Array.from(r.kalmanBeta, () => r.hindsightBeta), dash: true }],
  });
  lineChart(card(g, 'Leverage', 'Volatility target and fractional Kelly, both capped. Kelly leans on a noisy estimate of the mean, so it swings far more.'), {
    x: Array.from(r.volLeverage, (_, i) => i), xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(1),
    series: [{ name: 'volatility target', y: Array.from(r.volLeverage), colorIndex: 3 }, { name: 'fractional Kelly', y: Array.from(r.kellyLeverage), colorIndex: 4 }],
  });
  lineChart(card(g, 'Drawdown', 'Peak-to-trough loss of wealth.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => fmt.pct(v, 0),
    series: [un, kal, vt, by('Kalman hedge + vol target')].map((e, i) => ({ name: e.name, y: Array.from(e.drawdown, (v) => -v), colorIndex: [0, 2, 3, 5][i] })),
  });
  const box = card(page.content, 'Overlays', `Over ${fmt.num(s[0].equity.length - 1)} days after a ${fmt.num(r.warmup)}-day warm-up. Hedge trades cost ${spec.hedging.hedgeCostBps} bp per unit of beta changed.`);
  table(box, [...PERF_HEADERS, 'corr. to market'], s.map((e, i) => [...perfRow(e.name, e.metrics), fmt.ratio(r.marketCorrelation[i])]));
  return `Done · models ${r.cachedPredictions ? 'cached' : `trained in ${fmt.num(r.trainMs)} ms`} · WebAssembly`;
});
