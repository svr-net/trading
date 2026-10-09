import { run } from '../sat-client.js';
import { barChart, fmt, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'bars',
  gpu: false,
  title: 'Information-driven bars',
  context: 'sat/afml/bars: generateTrades · timeBars · tickBars · volumeBars · dollarBars · tickImbalanceBars',
  description: 'Markets do not process information at a constant rate, so sampling prices every fixed slice of clock time mixes quiet and busy periods. ' +
    'Bars that close after a fixed number of trades, shares or dollars instead sample more often when more happens, and their returns come closer to independent, identically distributed normal draws. ' +
    'Imbalance bars go further and close when the order flow becomes more one-sided than expected. ' +
    'The trade stream here is synthetic. Each day draws its own activity level, and every trade moves the price by a draw of the same size, the situation that motivates these bars (López de Prado, 2018, chapter 2).',
});
specEditor(page, ['bars'], { open: false });

runButton(page, 'Build bars', async (spec) => {
  const r = await run('afmlBars', spec);
  const byName = Object.fromEntries(r.kinds.map((k) => [k.name, k]));
  tiles(page.content, [
    { label: 'Trades', value: fmt.compact(r.trades), hint: `${fmt.num(r.days)} days` },
    { label: 'Time bars: Jarque-Bera', value: fmt.num(byName.time.stats.jarqueBera, 1), hint: 'non-normality of returns' },
    { label: 'Dollar bars: Jarque-Bera', value: fmt.num(byName.dollar.stats.jarqueBera, 1), hint: 'lower is closer to normal' },
    { label: 'Time bars: kurtosis', value: fmt.ratio(byName.time.stats.kurtosis), hint: 'normal: 3' },
    { label: 'Dollar bars: kurtosis', value: fmt.ratio(byName.dollar.stats.kurtosis) },
  ]);
  const g = grid(page.content);
  const days = Array.from(r.tradesPerDay, (_, i) => i + 1);
  lineChart(card(g, 'Activity and bars per day', 'Time bars ignore activity; trade-based bars follow it.'), {
    x: days, xLabel: 'day', yFormat: (v) => fmt.compact(v),
    series: [
      { name: 'trades / 100', y: Array.from(r.tradesPerDay, (v) => v / 100), colorIndex: 7, dash: true },
      ...r.kinds.map((k, i) => ({ name: `${k.name} bars`, y: Array.from(k.barsPerDay), colorIndex: i })),
    ],
  });
  // Distribution of standardised returns: time vs dollar bars against the normal density.
  const edges = Array.from({ length: 41 }, (_, i) => -5 + i * 0.25);
  const density = (v) => {
    const c = new Array(edges.length - 1).fill(0);
    for (const x of v) { const k = Math.floor((x + 5) / 0.25); if (k >= 0 && k < c.length) c[k]++; }
    return c.map((n) => n / v.length / 0.25);
  };
  const mids = edges.slice(0, -1).map((e) => e + 0.125);
  lineChart(card(g, 'Distribution of standardised returns', 'Log scale: fat tails show as the slow decay away from the normal curve.'), {
    x: mids, xLabel: 'standard deviations', yFormat: (v) => (v > 0 ? v.toExponential(0) : '0'),
    series: [
      { name: 'normal', y: mids.map((x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)), colorIndex: 7, dash: true },
      { name: 'time bars', y: density(byName.time.standardised), colorIndex: 0 },
      { name: 'dollar bars', y: density(byName.dollar.standardised), colorIndex: 3 },
    ].map((s) => ({ ...s, y: s.y.map((v) => (v > 0 ? Math.log10(v) : NaN)) })),
    yLabel: 'log10 density',
  });
  barChart(card(g, 'Jarque-Bera statistic by bar type', 'Distance from normality of bar returns (skewness and excess kurtosis).'), {
    labels: r.kinds.map((k) => k.name), series: [{ name: 'Jarque-Bera', values: r.kinds.map((k) => k.stats.jarqueBera) }], yFormat: (v) => fmt.compact(v),
  });
  const box = card(page.content, 'Statistical properties', 'Variance of variance: dispersion of the return variance across ten consecutive subsamples, relative to its mean (0 = stable).');
  table(box, ['bars', 'count', 'per day (mean ± sd)', 'return sd', 'skew', 'kurtosis', 'Jarque-Bera', 'serial corr.', 'variance of variance'],
    r.kinds.map((k) => [k.name, fmt.num(k.stats.count), `${fmt.num(k.stats.barsPerDayMean, 1)} ± ${fmt.num(k.stats.barsPerDaySd, 1)}`, fmt.pct(k.stats.returnSd, 3),
      fmt.ratio(k.stats.skewness), fmt.ratio(k.stats.kurtosis), fmt.num(k.stats.jarqueBera, 1), fmt.ratio(k.stats.serialCorrelation, 3), fmt.ratio(k.stats.varianceOfVariance)]));
  return `Done in ${fmt.num(r.elapsedMs)} ms · WebAssembly`;
});
