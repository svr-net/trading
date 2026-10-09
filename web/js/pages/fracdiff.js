import { run } from '../sat-client.js';
import { dateAxis, fmt, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'fracdiff',
  gpu: false,
  title: 'Fractional differentiation',
  context: 'sat/afml/fracdiff: fracDiffWeights · fracDiff · adfTest · scanFracDiff',
  description: 'Prices are not stationary, so models are usually fed returns, the first difference of log prices. That throws away the memory of the price level. ' +
    'Differentiating by a fraction <i>d</i> between 0 and 1 removes just enough of the trend to pass a unit-root test while keeping as much memory as possible. ' +
    'The scan below finds the smallest <i>d</i> whose fixed-window fractional difference passes the augmented Dickey-Fuller test at 5%. The correlation with the original series shows what that order keeps (chapter 5). ' +
    'The <code>ffd</code> extra feature of the models uses this transform of each stock\'s log price.',
});
specEditor(page, ['market', 'csv']);

runButton(page, 'Scan d', async (spec) => {
  const r = await run('afmlFracDiff', { ...spec, fracdiffSeries: 'index' });
  const k = r.d.findIndex((d) => Math.abs(d - r.minimumD) < 1e-9);
  tiles(page.content, [
    { label: 'Series', value: r.series, hint: 'log level' },
    { label: 'Minimum d (ADF 5%)', value: fmt.ratio(r.minimumD), hint: `ADF ${fmt.ratio(r.adf[k])} < ${r.critical5}` },
    { label: 'Memory kept at that d', value: fmt.pct(r.correlation[k], 1), hint: 'correlation with the log level' },
    { label: 'First difference keeps', value: fmt.pct(r.correlation[r.correlation.length - 1], 1), hint: 'd = 1 (returns)' },
    { label: 'Window width', value: fmt.num(r.width[k]), hint: `weights above ${r.threshold}` },
  ]);
  const g = grid(page.content);
  const dd = Array.from(r.d);
  lineChart(card(g, 'Stationarity against memory', 'ADF statistic (stationary below the dashed 5% critical value) and correlation with the original series.'), {
    x: dd, xLabel: 'd', yFormat: (v) => v.toFixed(1),
    series: [
      { name: 'ADF statistic', y: Array.from(r.adf), markers: true },
      { name: '5% critical value', y: dd.map(() => r.critical5), dash: true, colorIndex: 7 },
      { name: 'correlation × 10', y: Array.from(r.correlation, (v) => 10 * v), colorIndex: 2, markers: true },
    ],
  });
  const x = r.dates.map((_, i) => i), xf = dateAxis(r.dates);
  lineChart(card(g, 'The series and its fractional difference', `Log level, its fractional difference at d = ${fmt.ratio(r.minimumD)}, and the first difference (returns, scaled ×10).`), {
    x, xFormat: xf, xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [
      { name: 'log level', y: Array.from(r.logPrice) },
      { name: `d = ${fmt.ratio(r.minimumD)}`, y: Array.from(r.ffdMinimum), colorIndex: 1 },
      { name: 'd = 1 (×10)', y: Array.from(r.firstDifference, (v) => 10 * v), colorIndex: 2, width: 1 },
    ],
  });
  lineChart(card(g, 'Weights of (1 − B)^d', 'The weight on the value k days back. Small d keeps long memory.'), {
    x: Array.from({ length: 30 }, (_, i) => i), xLabel: 'lag k', yFormat: (v) => v.toFixed(2),
    series: r.weights.map((w, i) => ({ name: `d = ${w.d}`, y: Array.from(w.weights), colorIndex: i, markers: true })),
  });
  const box = card(page.content, 'Scan', '');
  table(box, ['d', 'ADF', 'stationary (5%)', 'correlation', 'window'],
    dd.map((d, i) => [fmt.ratio(d), fmt.ratio(r.adf[i]), r.adf[i] < r.critical5 ? 'yes' : 'no', fmt.ratio(r.correlation[i], 3), fmt.num(r.width[i])]));
});
