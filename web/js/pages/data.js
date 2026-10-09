import { run } from '../sat-client.js';
import { dateAxis, fmt, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'data',
  gpu: false,
  title: 'Market data',
  context: 'sat/data: MarketData · parseCsv · generateSyntheticMarket',
  description: 'The paper\'s experiments use daily bars of Hong Kong stocks. Those data are not published, so by default the library simulates a market with the features that matter here. ' +
    'A hidden Markov chain switches between bull, bear and range-bound regimes. The regime sets the market\'s drift and volatility, and the sign of the stocks\' short-term autocorrelation: trends in bull markets, reversals in range-bound ones. ' +
    'Moves made on abnormal volume partly revert the next day. The predictable part of returns therefore changes with the regime, which is the situation a single fixed strategy cannot keep up with. ' +
    'Upload your own daily bars (CSV) to run every page on real data instead.',
});
specEditor(page, ['market', 'csv'], { open: false });

runButton(page, 'Load market', async (spec) => {
  const r = await run('marketData', spec);
  const x = r.dates.map((_, i) => i), xf = dateAxis(r.dates);
  tiles(page.content, [
    { label: 'Source', value: r.source === 'csv' ? 'CSV upload' : 'synthetic', hint: `${r.dates[0]} – ${r.dates[r.dates.length - 1]}` },
    { label: 'Stocks', value: fmt.num(r.numAssets) },
    { label: 'Trading days', value: fmt.num(r.numDates) },
    ...r.regimes.map((g) => ({ label: `Regime: ${g.name}`, value: fmt.pct(g.days / (r.numDates - 1), 0), hint: `market ${fmt.pct(g.annualReturn, 0)} a year, vol ${fmt.pct(g.annualVolatility, 0)}` })),
  ]);
  const g = grid(page.content);
  lineChart(card(g, 'Equal-weight market index', 'Daily rebalanced average of all stocks.'), {
    x, xFormat: xf, xLabel: 'date', yFormat: (v) => v.toFixed(2), series: [{ name: 'index', y: Array.from(r.index) }],
  });
  if (r.regimeNames.length)
    lineChart(card(g, 'Hidden regime', 'Known only because the market is simulated; the models never see it.'), {
      x, xFormat: xf, xLabel: 'date', yFormat: (v) => r.regimeNames[Math.round(v)] ?? '', height: 160,
      series: [{ name: 'regime', y: Array.from(r.regime), step: true, colorIndex: 3 }],
    });
  lineChart(card(g, 'Prices of the first stocks (rebased)', ''), {
    x, xFormat: xf, xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: r.paths.map((p, i) => ({ name: r.tickers[i], y: Array.from(p), colorIndex: i % 8, width: 1.2 })),
  });
  const bx = r.bars.dates.map((_, i) => i);
  lineChart(card(g, `Daily bars: ${r.tickers[0]}, last 120 days`, 'High–low range as a band, close and VWAP.'), {
    x: bx, xFormat: dateAxis(r.bars.dates), xLabel: 'date',
    series: [
      { name: 'close', y: Array.from(r.bars.close), band: { lower: Array.from(r.bars.low), upper: Array.from(r.bars.high) } },
      { name: 'VWAP', y: Array.from(r.bars.vwap), colorIndex: 1, dash: true },
    ],
  });
  const box = card(page.content, 'Stocks', 'Over the whole sample. β is measured against the equal-weight market.');
  table(box, ['ticker', 'ann. return', 'ann. vol', 'Sharpe', 'max DD', 'β', 'avg volume', 'last close'],
    r.stocks.map((s) => [s.ticker, fmt.pct(s.annualReturn, 1), fmt.pct(s.annualVolatility, 1), fmt.ratio(s.sharpe), fmt.pct(s.maxDrawdown, 1), fmt.ratio(s.beta), fmt.compact(s.averageVolume), fmt.num(s.lastClose, 2)]));
});
