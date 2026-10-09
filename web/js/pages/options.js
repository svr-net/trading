import { run } from '../sat-client.js';
import { barChart, dateAxis, fmt, histogram, lineChart } from '../charts.js';
import { PERF_HEADERS as ALL_HEADERS, card, grid, initPage, perfRow as fullRow, runButton, specEditor, table, tiles } from '../ui.js';

// These series carry no turnover, so the turnover column is left out.
const PERF_HEADERS = ALL_HEADERS.slice(0, -1);
const perfRow = (name, m) => fullRow(name, m).slice(0, -1);

const page = initPage({
  id: 'options',
  gpu: false,
  title: 'Option hedges',
  context: 'sat/hedge/options: blackScholes · simulateDeltaHedge · optionOverlay',
  description: 'Two uses of options from Hull\'s <i>Options, Futures, and Other Derivatives</i>. ' +
    'First, <b>delta hedging</b>: a dealer sells a call at the Black–Scholes price and holds delta shares, rebalancing at discrete times. With continuous rebalancing the hedge would be perfect; the simulation shows how the hedging error shrinks roughly with the square root of the time between rebalances, and what happens when the option was priced with the wrong volatility. ' +
    'Second, <b>protective puts and collars</b> on the equal-weight index: a rolling out-of-the-money put caps the loss over each period, and selling an out-of-the-money call (a collar) pays for it by giving up the upside. Options are priced with Black–Scholes at the trailing realised volatility plus a premium, as implied volatility usually is.',
});
specEditor(page, ['market', 'csv', 'options'], { open: false });

runButton(page, 'Simulate', async (spec) => {
  const r = await run('hedgeOptions', spec);
  const f = r.frequencies;
  const daily = f.find((x) => x.perDay === 1) || f[0];
  const [idx, put, collar] = r.overlays;
  tiles(page.content, [
    { label: 'Call premium', value: r.premium.toFixed(2), hint: `spot 100, ${spec.options.years} years, implied vol ${fmt.pct(spec.options.impliedVol, 0)}` },
    { label: 'Hedging error, daily rebalancing', value: fmt.pct(daily.sd, 1), hint: 'standard deviation of P&L, % of premium' },
    { label: 'Mean hedged P&L', value: fmt.pct(daily.mean, 1), hint: 'of premium; non-zero when implied ≠ true volatility' },
    { label: 'Put cost per roll', value: fmt.pct(r.averagePutCost, 2), hint: `call income ${fmt.pct(r.averageCallIncome, 2)}, ${fmt.num(r.rolls)} rolls` },
    { label: 'Max drawdown: index → put → collar', value: `${fmt.pct(idx.metrics.maxDrawdown, 0)} → ${fmt.pct(put.metrics.maxDrawdown, 0)} → ${fmt.pct(collar.metrics.maxDrawdown, 0)}` },
  ]);
  const g = grid(page.content);
  barChart(card(g, 'Hedging error against rebalancing frequency', 'Standard deviation and 5% quantile of the hedged P&L, in units of the premium.'), {
    labels: f.map((x) => (x.perDay >= 1 ? `${x.perDay}/day` : `every ${Math.round(1 / x.perDay)} days`)),
    series: [{ name: 'sd', values: f.map((x) => x.sd) }, { name: '5% quantile', values: f.map((x) => x.q05) }],
    yFormat: (v) => fmt.pct(v, 0), tooltipFormat: (v) => fmt.pct(v, 1),
  });
  histogram(card(g, 'Hedged P&L, daily rebalancing', `${fmt.num(r.histograms.daily.length)} paths, in units of the premium.`), Array.from(r.histograms.daily), { bins: 40, xFormat: (v) => fmt.pct(v, 0) });
  if (r.histograms.weekly) histogram(card(g, 'Hedged P&L, weekly rebalancing', 'The same paths: the distribution widens.'), Array.from(r.histograms.weekly), { bins: 40, xFormat: (v) => fmt.pct(v, 0) });
  lineChart(card(g, 'Payoff at expiry', 'Return of the index plus the option positions, net of the average premia.'), {
    x: Array.from(r.payoffMove), xLabel: 'index move', xFormat: (v) => fmt.pct(v, 0), yFormat: (v) => fmt.pct(v, 0),
    series: [{ name: 'index', y: Array.from(r.payoffMove), dash: true }, { name: 'protective put', y: Array.from(r.payoffPut) }, { name: 'collar', y: Array.from(r.payoffCollar) }],
  });
  const x = Array.from(idx.equity, (_, i) => i);
  lineChart(card(g, 'Overlays on the equal-weight index', 'Rolled every tenor at the close.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: r.overlays.map((o, i) => ({ name: o.name, y: Array.from(o.equity), colorIndex: i })),
  });
  lineChart(card(g, 'Drawdown', 'The put floors each period\'s loss; drawdowns build only from consecutive losing periods and premia.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => fmt.pct(v, 0),
    series: r.overlays.map((o, i) => ({ name: o.name, y: Array.from(o.drawdown, (v) => -v), colorIndex: i })),
  });
  table(card(page.content, 'Overlays', 'Insurance costs return: expect a lower mean, a lower volatility and smaller drawdowns.'), PERF_HEADERS, r.overlays.map((o) => perfRow(o.name, o.metrics)));
  table(card(page.content, 'Delta hedging', 'Mean and dispersion of the hedged P&L of a short call, in units of the premium.'), ['rebalancing', 'rebalances', 'mean', 'sd', '5% quantile'],
    f.map((x) => [x.perDay >= 1 ? `${x.perDay} per day` : `every ${Math.round(1 / x.perDay)} days`, fmt.num(x.rebalances), fmt.pct(x.mean, 2), fmt.pct(x.sd, 2), fmt.pct(x.q05, 2)]));
});
