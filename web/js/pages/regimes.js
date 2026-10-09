import { run } from '../sat-client.js';
import { dateAxis, fmt, lineChart } from '../charts.js';
import { PERF_HEADERS as ALL_HEADERS, card, grid, initPage, perfRow as fullRow, runButton, specEditor, table, tiles } from '../ui.js';

// These series carry no turnover, so the turnover column is left out.
const PERF_HEADERS = ALL_HEADERS.slice(0, -1);
const perfRow = (name, m) => fullRow(name, m).slice(0, -1);

const page = initPage({
  id: 'regimes',
  gpu: false,
  title: 'Regime switching',
  context: 'sat/algo/regimes: fitHmm (Baum–Welch) · filterHmm · regimeSwitch',
  description: 'Hamilton\'s <b>regime-switching</b> model: daily market returns are drawn from one of a few Gaussian states — calm and volatile, say — and the state follows a Markov chain. The model is fitted by <b>Baum–Welch</b> (expectation maximisation), and the <b>filtered</b> probability of each state uses only returns up to that day, which is what a trader could know. ' +
    'The trading rule adapts its exposure to the regime: the model is re-fitted on a rolling window and the market is held only while the probability of the most volatile state at the previous close is below a threshold. ' +
    'On the synthetic market the true regimes are known, so the page also measures how well the filter recovers them. Volatility targeting is shown as the continuous alternative to switching.',
});
specEditor(page, ['market', 'csv', 'regimes'], { open: false });

runButton(page, 'Fit', async (spec) => {
  const r = await run('algoRegimes', spec);
  const K = r.model.sd.length;
  const [mkt, sw, vt] = r.series;
  tiles(page.content, [
    { label: 'State volatilities (annual)', value: Array.from(r.model.sd, (v) => fmt.pct(v * Math.sqrt(252), 0)).join(' · '), hint: `${fmt.num(r.model.iterations)} EM iterations` },
    { label: 'Persistence of each state', value: Array.from(r.model.transition, (row, k) => fmt.pct(row[k], 1)).join(' · '), hint: `expected duration ${Array.from(r.model.transition, (row, k) => fmt.num(1 / (1 - row[k]))).join(' · ')} days` },
    { label: 'Volatile regime recovered', value: r.accuracy >= 0 ? fmt.pct(r.accuracy, 1) : '–', hint: 'filtered P > ½ vs the generator\'s most volatile regime' },
    { label: 'Regime switch: max DD', value: fmt.pct(sw.metrics.maxDrawdown, 1), hint: `market ${fmt.pct(mkt.metrics.maxDrawdown, 1)}, Sharpe ${fmt.ratio(sw.metrics.sharpe)} vs ${fmt.ratio(mkt.metrics.sharpe)}` },
  ]);
  const g = grid(page.content);
  const x = Array.from(r.returns, (_, i) => i);
  const truth = Array.from(r.regime);
  const series = [{ name: `P(${K === 2 ? 'volatile' : 'most volatile'} state)`, y: Array.from(r.probabilities[K - 1]) }];
  if (truth[0] >= 0) {
    const names = r.regimeNames;
    const vol = names.indexOf('bear') >= 0 ? names.indexOf('bear') : 1;
    series.push({ name: `true regime = ${names[vol]}`, y: truth.map((v) => (v === vol ? 1 : 0)), step: true, colorIndex: 3, width: 1 });
  }
  lineChart(card(g, 'Filtered probability of the volatile state', 'Full-sample model, filtered forward in time.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(1), series,
  });
  lineChart(card(g, 'Market returns', 'Daily returns of the equal-weight market.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => fmt.pct(v, 0), series: [{ name: 'return', y: Array.from(r.returns), width: 1 }],
  });
  const e = Array.from(mkt.equity, (_, i) => i);
  const sd = r.dates.slice(r.switchStart);
  lineChart(card(g, 'Regime-switched exposure', 'Out of sample: the model is re-fitted on a rolling window.'), {
    x: e, xFormat: dateAxis(sd), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: r.series.map((s, i) => ({ name: s.name, y: Array.from(s.equity), colorIndex: i })),
  });
  lineChart(card(g, 'Exposure', 'Held while P(volatile) at the previous close is below the threshold.'), {
    x: Array.from(r.exposure, (_, i) => i), xFormat: dateAxis(sd), xLabel: 'date', yFormat: (v) => v.toFixed(1),
    series: [{ name: 'exposure', y: Array.from(r.exposure), step: true }, { name: 'P(volatile)', y: Array.from(r.switchProbability), colorIndex: 1, width: 1 }],
  });
  table(card(page.content, 'Fitted states', 'Daily mean and volatility of each state and its transition probabilities.'), ['state', 'mean (annual)', 'volatility (annual)', ...Array.from({ length: K }, (_, k) => `→ ${k}`)],
    Array.from({ length: K }, (_, k) => [String(k), fmt.pct(r.model.mean[k] * 252, 1), fmt.pct(r.model.sd[k] * Math.sqrt(252), 1), ...Array.from(r.model.transition[k], (p) => fmt.pct(p, 2))]));
  table(card(page.content, 'Strategies', `From day ${fmt.num(r.switchStart)} (after the first estimation window).`), PERF_HEADERS, [mkt, sw, vt].map((s) => perfRow(s.name, s.metrics)));
});
