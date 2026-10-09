import { engineSelector, kernelNote, runWithKernels } from '../gpu/backend.js';
import { dateAxis, fmt, heatmap, lineChart } from '../charts.js';
import { ALLOCATION_NAMES, PERF_HEADERS as ALL_HEADERS, card, grid, initPage, perfRow as fullRow, runButton, specEditor, table, tiles } from '../ui.js';

// These series carry no turnover, so the turnover column is left out.
const PERF_HEADERS = ALL_HEADERS.slice(0, -1);
const perfRow = (name, m) => fullRow(name, m).slice(0, -1);

const page = initPage({
  id: 'tournament',
  title: 'Strategy tournament & meta-allocation',
  context: 'sat/algo/tournament: runTournament · sat/algo/ensemble: allocate · pairsBook',
  description: 'Every approach in the library, backtested on the <b>same days</b>: each machine-learning model (with equal capital in every trading rule, so no rule is chosen in hindsight), the paper\'s self-adaptive selector and its volatility-targeted and beta-hedged versions, trend following, an adaptive book of cointegrated pairs that re-scans the universe every quarter, and the HMM regime switch. ' +
    'On top sits a second level of self-adaptation: a <b>meta-allocator</b> that moves capital between the approaches using only their past returns — equal weight, follow the leader, Sharpe-weighted, inverse volatility, Sharpe over volatility, and <b>exponential weights</b> (the multiplicative-weights or Hedge algorithm of online learning). ' +
    'The default — exponential weights on a 63-day look-back, re-weighted weekly — was chosen by <code>examples/strategy_tournament</code>: 90 settings ranked on 6 ten-year synthetic markets, then re-tested on 6 unseen ones. There it reached a validation Sharpe ratio of about 1.3 against 1.2 for the paper\'s selector alone, with a drawdown of about 32% instead of 42%. ' +
    'The deflated Sharpe ratio below accounts for having tried all six methods on this market.',
});
specEditor(page, ['market', 'csv', 'models', 'strategies', 'costs', 'selector', 'tournament'], { open: false });

const button = runButton(page, 'Run tournament', async (spec) => {
  const k = await runWithKernels('strategyTournament', spec);
  const r = k.result;
  window.__satEngineRun = { page: 'tournament', engine: k.engine, result: r };
  const chosen = r.allocators.find((a) => a.name === r.chosen) || r.allocators[0];
  const sleeves = r.sleeves;
  const paper = sleeves.find((s) => s.family === 'ML self-adaptive');
  const bestSleeve = sleeves.slice(0, -1).reduce((b, s) => (s.metrics.sharpe > b.metrics.sharpe ? s : b));
  const bestAlloc = r.allocators.reduce((b, s) => (s.metrics.sharpe > b.metrics.sharpe ? s : b));
  tiles(page.content, [
    { label: `Chosen allocator: ${chosen.name}`, value: fmt.ratio(chosen.metrics.sharpe), hint: `Sharpe · max DD ${fmt.pct(chosen.metrics.maxDrawdown, 1)} · DSR ${fmt.pct(chosen.dsr, 0)}` },
    { label: 'Paper\'s self-adaptive selector', value: fmt.ratio(paper.metrics.sharpe), hint: `Sharpe · max DD ${fmt.pct(paper.metrics.maxDrawdown, 1)}` },
    { label: 'Best single approach (hindsight)', value: fmt.ratio(bestSleeve.metrics.sharpe), hint: bestSleeve.name },
    { label: 'Best allocator here (hindsight)', value: fmt.ratio(bestAlloc.metrics.sharpe), hint: bestAlloc.name },
  ]);
  const g = grid(page.content);
  const x = Array.from(chosen.equity, (_, i) => i);
  lineChart(card(g, 'Every approach', `${fmt.num(x.length - 1)} common days. Dashed: the benchmark.`), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: sleeves.map((s, i) => ({ name: s.name, y: Array.from(s.equity), colorIndex: i, width: 1.5, dash: s.family === 'benchmark' })),
  });
  lineChart(card(g, 'Meta-allocators', 'Each re-weights the approaches from their trailing returns.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [...r.allocators.map((a, i) => ({ name: a.name, y: Array.from(a.equity), colorIndex: i, width: a === chosen ? 3 : 1.5 })),
      { name: paper.name, y: Array.from(paper.equity), colorIndex: 7, dash: true }],
  });
  const w = Array.from(chosen.weights, (v) => Array.from(v));
  const top = w.map((v, k) => [k, v.reduce((a, b) => a + b, 0)]).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k]) => k);
  lineChart(card(g, `Capital allocated by ${chosen.name}`, 'The six approaches it used most; the rest and cash make up the remainder.'), {
    x: Array.from(chosen.cash, (_, i) => i), xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => fmt.pct(v, 0),
    series: [...top.map((k) => ({ name: r.names[k], y: w[k], colorIndex: k, step: true })), { name: 'cash', y: Array.from(chosen.cash), dash: true, colorIndex: 7, step: true }],
  });
  heatmap(card(g, 'Correlation of the approaches', 'Low correlations are what make allocation across them worthwhile.'), {
    rows: r.names, cols: r.names.map((_, i) => String(i + 1)), values: r.correlation.map((v) => Array.from(v)), format: (v) => v.toFixed(1), rowWidth: 200,
  });
  table(card(page.content, 'Approaches', 'Each over the same days, net of its own costs.'), [...PERF_HEADERS, 'family'],
    sleeves.map((s) => [...perfRow(s.name, s.metrics), s.family]));
  table(card(page.content, 'Meta-allocators', `Look-back ${r.lookback} days, rebalanced every ${r.rebalanceEvery}. DSR: probability the Sharpe ratio beats the best of six unskilled trials.`),
    [...PERF_HEADERS, 'DSR', 'avg. cash', 'turnover / rebalance'],
    r.allocators.map((a) => [...perfRow(a.name + (a === chosen ? ' (chosen)' : ''), a.metrics), fmt.pct(a.dsr, 0),
      fmt.pct(Array.from(a.cash).reduce((p, q) => p + q, 0) / a.cash.length, 0), fmt.pct(a.averageTurnover, 0)]));
  return kernelNote(k, ` · models ${r.cachedPredictions ? 'cached' : `trained in ${fmt.num(r.trainMs)} ms`} · method ${ALLOCATION_NAMES[spec.tournament.method] || spec.tournament.method}`);
});

engineSelector(page, () => button.click());
