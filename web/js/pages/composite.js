import { run } from '../sat-client.js';
import { dateAxis, fmt, lineChart } from '../charts.js';
import { COMPOSITE_NAMES, PERF_HEADERS as ALL_HEADERS, card, grid, initPage, perfRow as fullRow, runButton, specEditor, table, tiles } from '../ui.js';

// These series carry no turnover, so the turnover column is left out.
const PERF_HEADERS = ALL_HEADERS.slice(0, -1);
const perfRow = (name, m) => fullRow(name, m).slice(0, -1);

const page = initPage({
  id: 'composite',
  gpu: false,
  title: 'Composite model & strategy',
  context: 'sat/adaptive/composite: compositePredictions · sat/algo/composite: multiSignalSelection, runCompositeStrategy',
  description: 'The <b>composite model</b> combines the models\' out-of-sample forecasts into one, three ways from the forecast-combination literature: the <b>equal-weight average</b> (the "forecast combination puzzle" — estimated weights rarely beat it out of sample), a <b>stacked</b> logistic meta-learner on the models\' log-odds (non-negative weights shrunk towards equal), and <b>Bernstein Online Aggregation</b>, the online expert-weighting method applied to stock-return forecasts by Remlinger et al. (2023). Each composite only learns from labels that have resolved. ' +
    'The <b>composite strategy</b> lets exponential weights (the tournament\'s meta-allocator) move capital between the paper\'s self-adaptive selector trading the composite forecast and slower <b>stock-selection books</b>: the forecast, 12-1 month <b>momentum</b> and the trailing Sharpe ratio, ranked across stocks and blended with weights that follow each signal\'s recent information coefficient, held with inverse-volatility weights, a <b>volatility target</b> (Barroso and Santa-Clara) and an HMM <b>regime gate</b>. ' +
    'Settings were chosen by <code>examples/composite_study</code> on synthetic markets and re-tested on unseen ones and on 120 LSE stocks; see the README for what held up.',
});
specEditor(page, ['market', 'csv', 'models', 'composite', 'strategies', 'costs', 'selector', 'multiSignal', 'tournament'], { open: false });

runButton(page, 'Run', async (spec) => {
  const r = await run('compositeStrategy', spec);
  const s = r.series;
  const named = (n) => s.find((x) => x.name === n);
  const comp = named('Composite strategy');
  const members = named('Self-adaptive selector (separate members)');
  const market = named('Equal-weight market (benchmark)');
  const chosen = r.composites.find((c) => c.method === r.chosen);
  const bestMember = r.members.reduce((b, m) => (m.auc > b.auc ? m : b), r.members[0]);
  tiles(page.content, [
    { label: 'Composite strategy', value: fmt.ratio(comp.metrics.sharpe), hint: `Sharpe · max DD ${fmt.pct(comp.metrics.maxDrawdown, 1)}` },
    { label: 'Selector on the separate models', value: members ? fmt.ratio(members.metrics.sharpe) : '–', hint: members ? `Sharpe · max DD ${fmt.pct(members.metrics.maxDrawdown, 1)}` : 'one model only' },
    { label: 'Equal-weight market', value: fmt.ratio(market.metrics.sharpe), hint: `Sharpe · max DD ${fmt.pct(market.metrics.maxDrawdown, 1)}` },
    { label: `Composite forecast (${COMPOSITE_NAMES[r.chosen]})`, value: chosen ? fmt.ratio(chosen.auc, 4) : fmt.ratio(bestMember.auc, 4), hint: `AUC · best single model ${fmt.ratio(bestMember.auc, 4)}` },
  ]);
  const g = grid(page.content);
  const x = Array.from(comp.equity, (_, i) => i);
  lineChart(card(g, 'Equity', 'Every series over the same days, net of costs. Dashed: the market.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: s.map((q, i) => ({ name: q.name, y: Array.from(q.equity), colorIndex: i, width: q === comp ? 3 : 1.5, dash: q === market })),
  });
  const sleeves = s.slice(0, r.sleeves);
  lineChart(card(g, 'Capital per sleeve', 'Exponential weights on each sleeve\'s trailing Sharpe ratio; the rest is cash.'), {
    x: Array.from(r.cash, (_, i) => i), xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => fmt.pct(v, 0),
    series: [...sleeves.map((q, k) => ({ name: q.name, y: Array.from(r.weights[k]), colorIndex: k, step: true })), { name: 'cash', y: Array.from(r.cash), dash: true, colorIndex: 7, step: true }],
  });
  if (r.composites.length) {
    lineChart(card(g, `Weight of each model in the ${COMPOSITE_NAMES.online}`, 'Moves towards the models whose recent forecasts had the lower log loss.'), {
      x: Array.from(r.composites[2].weights[0], (_, i) => 5 * i), xLabel: 'out-of-sample day', yFormat: (v) => fmt.pct(v, 0),
      series: r.memberNames.map((n, k) => ({ name: n, y: Array.from(r.composites[2].weights[k]), colorIndex: k })),
    });
  }
  const blend = r.books.find((b) => b.signals.length > 1) || r.books[0];
  if (blend) {
    lineChart(card(g, `${blend.name}: signal weights`, 'Each signal\'s share of the blended rank, from its trailing information coefficient.'), {
      x: Array.from(blend.signalWeights[0], (_, i) => i), xFormat: dateAxis(r.bookDates), xLabel: 'date', yFormat: (v) => fmt.pct(v, 0),
      series: [...blend.signals.map((n, k) => ({ name: n, y: Array.from(blend.signalWeights[k]), colorIndex: k, step: true })),
        { name: 'exposure (vol target × regime gate)', y: Array.from(blend.exposure), colorIndex: 6, dash: true }],
    });
  }
  table(card(page.content, 'Forecast quality', 'Out of sample, over each model\'s predicted days.'), ['model', 'AUC', 'log loss', 'accuracy'],
    [...r.members, ...r.composites].map((m) => [m.name + (m.method === r.chosen ? ' (traded)' : ''), fmt.ratio(m.auc, 4), fmt.ratio(m.logLoss, 4), fmt.pct(m.accuracy, 1)]));
  table(card(page.content, 'Strategies', `${fmt.num(x.length - 1)} common days, after the selector\'s and the allocator\'s look-backs.`), PERF_HEADERS,
    s.map((q) => perfRow(q.name, q.metrics)));
  if (blend) {
    table(card(page.content, `${blend.name}: latest selection`, 'Target weights of the last rebalance before exposure; "held" includes the volatility target and regime gate.'),
      ['stock', 'weight', 'held', 'blended rank'],
      blend.holdings.map((h) => [h.ticker, fmt.pct(h.weight, 1), fmt.pct(h.held, 1), fmt.ratio(h.score, 2)]));
  }
  return `Done · models ${r.cachedPredictions ? 'cached' : `trained in ${fmt.num(r.trainMs)} ms`} · traded forecast: ${r.forecast}`;
});
