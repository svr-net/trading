import { dateAxis, fmt, lineChart } from '../charts.js';
import { engineNote, engineSelector, runAnalysis } from '../gpu/backend.js';
import { PAGES, card, el, grid, initPage, runButton, tiles } from '../ui.js';

const page = initPage({
  id: 'index',
  title: 'Self-adaptive trading with machine learning, in the browser',
  context: 'trading · libsat (C++17) → WebAssembly + WebGPU',
  description:
    'Interactive front end for the library modelling <i>Predicting Stock Prices Based on Machine Learning to Build Self-adaptive Trading Strategy</i> ' +
    '(Wang, Huang and Luo, <i>Computational Economics</i>, 2025). Formulaic alpha factors feed machine-learning models that forecast each stock\'s next move. ' +
    'Every model drives a pool of fixed trading rules, and the self-adaptive strategy keeps switching to the rule with the best recent record. ' +
    'The C++ library is compiled to WebAssembly and runs in a worker. The strategy search also runs as WebGPU compute kernels. Each page below exercises one part of the library on a shared, editable specification. ' +
    'By default the models\' forecasts are averaged into one composite forecast before the rules trade it, which the forecast study found works better than trading the models separately.',
});

const DESCRIPTIONS = {
  core: 'The alpha operator algebra on real series, the seeded generator, and a check that no factor looks ahead.',
  data: 'A regime-switching synthetic stock market (bull, bear, range-bound), or your own daily bars from a CSV file.',
  factors: 'The 23 formulaic alphas used as features: information coefficients, coverage and correlations.',
  labels: 'Next-day direction, excess over the median, and N-period min-max labels with their look-ahead.',
  models: 'Logistic regression, SVM, trees, random forest, XGBoost, LightGBM, MLP and LSTM, trained walk-forward, and their composite forecast.',
  strategies: 'Fixed rules (top-k, long-short, thresholds) on every model. The best one changes from period to period.',
  adaptive: 'The self-adaptive strategy: re-scores every candidate and switches to the best, or to cash.',
  robustness: 'Look-back and adaptation-period grid, score metrics, transaction costs and year-by-year results.',
  gpu: 'Candidate backtests and the selector grid as WebGPU kernels, validated against WASM and benchmarked.',
};

const map = card(page.main, 'Contexts', 'Each page maps onto one part of the library and one step of the method.');
const g = el('div', { class: 'grid' });
for (const group of PAGES.slice(1))
  for (const p of group.items)
    g.append(el('a', { href: p.href, class: 'tile', style: 'text-decoration:none;color:inherit' },
      el('div', { class: 'label', text: group.group }), el('div', { style: 'font-weight:650;font-size:15px', text: p.title }),
      el('div', { class: 'hint', text: DESCRIPTIONS[p.id] })));
map.append(g);
page.main.insertBefore(map.parentNode, page.toolbar);

const button = runButton(page, 'Run the self-adaptive strategy', async (spec) => {
  const t0 = performance.now();
  const runResult = await runAnalysis('adaptive', spec);
  const r = runResult.result;
  const a = r.adaptive.metrics, b = r.bestFixed.metrics, m = r.benchmark.metrics;
  window.__satEngineRun = { page: 'index', engine: runResult.engine, sharpe: a.sharpe, annualReturn: a.annualReturn };
  tiles(page.content, [
    { label: 'Self-adaptive: annual return', value: fmt.pct(a.annualReturn, 1), hint: `Sharpe ${fmt.ratio(a.sharpe)}, max drawdown ${fmt.pct(a.maxDrawdown, 1)}` },
    { label: 'Best fixed rule (hindsight)', value: fmt.pct(b.annualReturn, 1), hint: `${r.bestFixed.label}, Sharpe ${fmt.ratio(b.sharpe)}` },
    { label: 'Median fixed rule', value: fmt.pct(r.medianFixedAnnualReturn, 1), hint: `Sharpe ${fmt.ratio(r.medianFixedSharpe)} over ${r.numCandidates} candidates` },
    { label: 'Equal-weight market', value: fmt.pct(m.annualReturn, 1), hint: `Sharpe ${fmt.ratio(m.sharpe)}` },
    { label: 'Switches', value: fmt.num(r.adaptive.switches), hint: `${fmt.pct(r.adaptive.share[r.adaptive.share.length - 1], 0)} of days in cash` },
  ]);
  const gr = grid(page.content);
  const x = r.dates.map((_, i) => i);
  lineChart(card(gr, 'Wealth over the evaluation period', `Out-of-sample, net of ${r.costBps} bp costs per unit of turnover. Full analysis on the <a href="adaptive.html">self-adaptive page</a>.`), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [
      { name: 'self-adaptive', y: Array.from(r.adaptive.equity) },
      { name: 'best fixed (hindsight)', y: Array.from(r.bestFixed.equity), colorIndex: 1 },
      { name: 'equal-weight market', y: Array.from(r.benchmark.equity), colorIndex: 2, dash: true },
    ],
  });
  const arch = card(gr, 'How it runs', '');
  arch.append(el('pre', { class: 'code', text:
`C++17 library (include/sat, src/)
   │  emcmake cmake -S . -B build-wasm   (Embind wrapper: wasm/bindings.cpp)
   ▼
web/wasm/sat.{js,wasm}  ── run in a Worker (js/sat-worker.js)
   │   standalone build: worker + .wasm embedded, started from a blob URL
   │   marketData · factors · labels · models (walk-forward training)
   │   strategies · adaptive · robustness
   ▼
sat::gpu (C++): compile() → one table (candidates, selectors, next-day
   │            returns, every model's predictions and ranks)
   │            + the WGSL kernels; summarise() + analytics on the read-back
   ▼
WebGPU host (js/gpu/engine.js): upload ▸ candidate-backtest ▸ adaptive-select
                                ▸ series-summary ▸ read back → gpuAnalyse` }));
  return engineNote(runResult, performance.now() - t0);
});
engineSelector(page, () => button.click());
