import { run } from '../sat-client.js';
import { dateAxis, fmt, histogram, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'pairs',
  title: 'Cointegrated pairs',
  context: 'sat/algo/pairs: engleGranger · halfLife · kalmanPairs',
  description: 'Mean reversion between two prices, following Chan\'s <i>Algorithmic Trading</i>. Two prices are <b>cointegrated</b> when a combination y − βx is stationary even though each wanders; the <b>Engle–Granger</b> test regresses y on x and applies an augmented Dickey–Fuller test to the residual (5% critical value about −3.34). ' +
    'The <b>half-life</b> of the spread, from an AR(1) fit, says how long a deviation takes to halve. The strategy tracks β with a <b>Kalman filter</b>, so the hedge ratio adapts, and trades the spread\'s z-score: long below −entry, short above +entry, out when it crosses the exit level. ' +
    'Top: a generated pair with a known hedge ratio and half-life. Bottom: every pair of the stock universe is tested on the first half of the history and the most cointegrated are traded on the second half — on a universe without planted relationships, about 5% of pairs pass by chance and do not persist.',
});
specEditor(page, ['market', 'csv', 'pairs'], { open: false });

runButton(page, 'Trade', async (spec) => {
  const r = await run('algoPairs', spec);
  const sy = r.synthetic, c = sy.cointegration, st = sy.strategy;
  tiles(page.content, [
    { label: 'ADF of the spread', value: fmt.ratio(c.adf), hint: c.cointegrated ? 'cointegrated at 5%' : 'not cointegrated at 5%' },
    { label: 'Hedge ratio', value: fmt.ratio(c.hedgeRatio, 3), hint: `true ${spec.pairs.beta}` },
    { label: 'Half-life', value: `${fmt.ratio(c.halfLife, 1)} days`, hint: `true ${spec.pairs.halfLife}` },
    { label: 'Kalman pairs Sharpe', value: fmt.ratio(st.metrics.sharpe), hint: `${fmt.num(st.trades)} trades, max DD ${fmt.pct(st.metrics.maxDrawdown, 1)}` },
    { label: 'Universe pairs passing the test', value: `${fmt.num(r.pairsCointegrated)} / ${fmt.num(r.pairsTested)}`, hint: `${fmt.pct(r.pairsCointegrated / r.pairsTested, 1)}; 5% expected by chance` },
  ]);
  const g = grid(page.content);
  const x = Array.from(sy.y, (_, i) => i);
  lineChart(card(g, 'Generated pair', 'y and β·x share a stochastic trend; the gap between them mean-reverts.'), {
    x, xLabel: 'day', yFormat: (v) => v.toFixed(0),
    series: [{ name: 'y', y: Array.from(sy.y) }, { name: `${fmt.ratio(c.hedgeRatio)}·x + ${fmt.ratio(c.intercept)}`, y: Array.from(sy.x, (v) => c.hedgeRatio * v + c.intercept) }],
  });
  lineChart(card(g, 'Z-score and position', `Kalman forecast error over its standard deviation. Entry at ±${spec.pairs.entryZ}, exit at ${spec.pairs.exitZ}.`), {
    x, xLabel: 'day', yFormat: (v) => v.toFixed(1),
    series: [{ name: 'z-score', y: Array.from(st.zscore) }, { name: 'position', y: Array.from(st.position), step: true, colorIndex: 2 }],
  });
  lineChart(card(g, 'Kalman hedge ratio', 'Adapts to the data; the dashed line is the true β.'), {
    x, xLabel: 'day', yFormat: (v) => v.toFixed(2),
    series: [{ name: 'Kalman β', y: Array.from(st.beta) }, { name: 'true β', y: x.map(() => spec.pairs.beta), dash: true }],
  });
  lineChart(card(g, 'Equity of the pairs strategy', 'Per unit of gross capital, net of costs.'), {
    x: Array.from(st.equity, (_, i) => i), xLabel: 'day', yFormat: (v) => v.toFixed(2), series: [{ name: 'Kalman pairs', y: Array.from(st.equity) }],
  });
  histogram(card(g, 'ADF statistics of every pair in the universe', `First ${fmt.num(r.scanDays)} days. Left of −3.34 counts as cointegrated.`), Array.from(r.scanAdf), { bins: 30, name: 'pairs', xFormat: (v) => v.toFixed(1) });
  if (r.scan.length) lineChart(card(g, 'Most cointegrated pairs, traded out of sample', 'Selected on the first half, traded on the second.'), {
    x: Array.from(r.scan[0].equity, (_, i) => i), xFormat: dateAxis(r.tradeDates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: r.scan.map((s, i) => ({ name: s.pair, y: Array.from(s.equity), colorIndex: i })),
  });
  table(card(page.content, 'Universe scan: in sample vs out of sample', 'A relationship found in one period rarely survives into the next unless there is a reason for it.'),
    ['pair', 'ADF in sample', 'ADF out of sample', 'half-life (in)', 'trades', 'ann. return', 'Sharpe', 'max DD'],
    r.scan.map((s) => [s.pair, fmt.ratio(s.inSample.adf), fmt.ratio(s.outOfSample.adf), fmt.ratio(s.inSample.halfLife, 1), fmt.num(s.trades),
      fmt.pct(s.metrics.annualReturn, 1), fmt.ratio(s.metrics.sharpe), fmt.pct(s.metrics.maxDrawdown, 1)]));
});
