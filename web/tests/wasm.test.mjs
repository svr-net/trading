// Node smoke test of the WASM wrapper: every entry point on a reduced default spec,
// with sanity checks against the library's own invariants.
//   node web/tests/wasm.test.mjs
import createSatModule from '../wasm/sat.js';
import { defaultSpec } from '../js/spec.js';

const sat = await createSatModule();
let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.log('  FAIL', msg); } };
const near = (a, b, tol, msg) => check(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

// A smaller universe and lighter models keep the run short.
const spec = defaultSpec();
spec.market = { ...spec.market, numAssets: 16, numDates: 760 };
spec.walkForward = { ...spec.walkForward, trainWindow: 300, retrainEvery: 63, maxTrainRows: 4000 };
spec.models = spec.models.filter((m) => ['logistic', 'xgboost', 'lstm'].includes(m.type)).map((m) => ({ ...m, trees: 25, epochs: 1, maxSamples: 1500 }));
spec.robustness = { lookbacks: [21, 63], steps: [5, 21], metrics: ['return', 'sharpe'], costs: [0, 20] };

function run(name, fn) {
  const t0 = performance.now();
  const r = fn();
  if (r && r.error) { failures++; console.log(`FAIL ${name}: ${r.error}`); return null; }
  console.log(`ok   ${name} (${(performance.now() - t0).toFixed(0)} ms)`);
  return r;
}

const ver = run('version', () => sat.version());
const cmakeVersion = (await import('node:fs')).readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8').match(/^\s*VERSION ([0-9.]+)$/m)[1];
if (ver) {
  check(ver.library === `trading sat ${cmakeVersion}`, `version() reports the CMake project version: ${ver.library}`);
  check(ver.alphas.length === 23, 'the paper uses 23 alphas');
}
const core = run('coreDemo', () => sat.coreDemo(spec));
if (core) {
  near(core.normals.mean, 0, 0.03, 'normal mean');
  near(core.normals.sd, 1, 0.03, 'normal sd');
  check(core.lookahead.every((a) => a.maxDiff < 1e-9 && a.nanMismatches === 0), 'no alpha looks ahead');
}
const market = run('marketData', () => sat.marketData(spec));
if (market) {
  check(market.numAssets === 16 && market.dates.length === 760 && market.regimeNames.length === 3, 'market shape');
  check(market.regimes.reduce((s, r) => s + r.days, 0) === 759, 'regime days add up');
}
const factors = run('factors', () => sat.factors(spec));
if (factors) {
  check(factors.factors.length === 23 && factors.correlation.length === 23, 'one entry per alpha');
  near(factors.correlation[4][4], 1, 1e-9, 'unit diagonal');
  check(factors.factors.some((f) => Math.abs(f.icMean) > 0.01), 'some factor predicts next-day returns');
}
const labels = run('labels', () => sat.labels(spec));
if (labels) check(labels.kinds[3].labelled < 0.3 && labels.kinds[0].labelled > 0.9, 'min-max labels only the turning points');
const models = run('models', () => sat.models(spec));
if (models) {
  check(models.models.length === 3, 'three models');
  for (const m of models.models) check(m.oos.auc > 0.5 && m.oos.count > 1000, `${m.name} beats chance out of sample (AUC ${m.oos.auc.toFixed(3)})`);
  check(models.models[1].importance.length === 23, 'importance per feature');
}
const again = run('models (cached)', () => sat.models(spec));
if (again) check(again.cached, 'predictions are reused');
const strat = run('strategies', () => sat.strategies(spec));
if (strat) {
  check(strat.candidates.length === 3 * spec.strategies.length, 'one candidate per model and strategy');
  check(strat.equityCurves.length === strat.candidates.length && strat.blocks.length > 2, 'curves and rolling winners');
  const winners = new Set(strat.blocks.map((b) => b.winner));
  check(winners.size > 1, 'different strategies win in different periods');
}
const adapt = run('adaptive', () => sat.adaptive(spec));
if (adapt) {
  check(adapt.adaptive.equity.length === adapt.dates.length, 'equity aligned with dates');
  check(adapt.adaptive.metrics.days === adapt.evalDays, 'adaptive evaluated over the evaluation days');
  near(adapt.adaptive.share.reduce((s, v) => s + v, 0), 1, 1e-9, 'shares add up');
}
const rob = run('robustness', () => sat.robustness(spec));
if (rob) {
  check(rob.grids.length === 2 && rob.grids[0].sharpe.length === 2 && rob.costs.length === 2, 'grid and cost jobs');
  check(rob.costs[0].benchmark.annualReturn >= rob.costs[1].benchmark.annualReturn, 'costs reduce returns');
}

// WebGPU path: plans and the combine step. No GPU in Node, so the kernels run through the
// C++ CPU reference (gpuEmulate); results must match the CPU library closely.
const kernels = run('gpuKernels', () => sat.gpuKernels({}));
if (kernels) check(kernels.candidateBacktest.includes('fn main') && kernels.adaptiveSelect.includes('fn main') && kernels.seriesSummary.includes('fn main'), 'WGSL sources');
const jobs = run('gpuJobs', () => sat.gpuJobs({ ...spec, analysis: 'robustness' }));
if (jobs) {
  check(jobs.plans.length === 1 + 2, 'robustness plans: the grid plus one per cost');
  const p = jobs.plans[0];
  check(p.header instanceof Uint32Array && p.tables instanceof Float32Array && p.header[3] === p.numCandidates, 'plan buffers');
  check(p.numSelectors === 2 * 2 * 2 + 1, 'grid selectors plus the spec selector');
}
const mix = sat.gpuJobs({ ...spec, analysis: 'adaptive', selector: { ...spec.selector, topM: 2 } });
check(typeof mix.unsupported === 'string', 'a top-2 mix is reported as unsupported on the GPU');
const emuAdapt = run('gpuEmulate adaptive', () => sat.gpuEmulate({ ...spec, analysis: 'adaptive' }));
if (emuAdapt && adapt) {
  near(emuAdapt.adaptive.metrics.sharpe, adapt.adaptive.metrics.sharpe, 0.05, 'emulated kernels: adaptive Sharpe vs CPU');
  near(emuAdapt.bestFixed.metrics.sharpe, adapt.bestFixed.metrics.sharpe, 1e-3, 'emulated kernels: best fixed Sharpe vs CPU');
  check(emuAdapt.equityCurves === undefined, 'path-level views stay on WebAssembly');
}
const valid = run('gpuEmulate validation', () => sat.gpuEmulate({ ...spec, analysis: 'validation' }));
if (valid) {
  const c = valid.checks;
  check(c.candidateSharpeDiff < 1e-3 && c.candidateReturnDiff < 1e-3 && c.selectionAgreement > 0.95 && c.selectorSharpeDiff < 0.1,
    `validation checks ${JSON.stringify(c)}`);
}
// gpuAnalyse with read-backs of the wrong size is rejected.
const badOut = sat.gpuAnalyse({ ...spec, analysis: 'adaptive', gpuOutputs: [{ stats: new Float32Array(3), adapt: new Float32Array(1) }] });
check(typeof badOut.error === 'string', 'gpuAnalyse rejects read-backs that do not match the plan');
const bad = sat.models({ ...spec, models: [{ type: 'nonsense' }] });
check(typeof bad.error === 'string', 'errors are reported, not thrown');
// Advances in Financial Machine Learning entry points.
const bars = run('afmlBars', () => sat.afmlBars({ ...spec, bars: { days: 40, tradesPerDay: 1500, barsPerDay: 20 } }));
if (bars) {
  const k = Object.fromEntries(bars.kinds.map((x) => [x.name, x.stats]));
  check(k.time.jarqueBera > 2 * k.dollar.jarqueBera, `dollar bars closer to normal than time bars (JB ${k.time.jarqueBera} vs ${k.dollar.jarqueBera})`);
  check(Math.abs(k.dollar.count - 800) < 100 && k['tick imbalance'].count > 50, 'bar counts');
}
const fd = run('afmlFracDiff', () => sat.afmlFracDiff(spec));
if (fd) {
  check(fd.minimumD > 0 && fd.minimumD <= 1 && fd.adf[fd.adf.length - 1] < fd.critical5, 'a d in (0, 1] makes the index stationary');
  check(fd.correlation[0] > 0.999, 'd = 0 is the series itself');
}
const lab = run('afmlLabeling', () => sat.afmlLabeling({ ...spec, labeling: { cusumMultiple: 2, maxHolding: 10, metaHolding: 2, metaCusumMultiple: 1 } }));
if (lab) {
  check(lab.events.length > 20 && lab.events.every((e) => e.t1 > e.t0 && e.t1 <= e.t0 + 10), 'events within the vertical barrier');
  check(lab.counts.upper + lab.counts.lower + lab.counts.vertical === lab.events.length, 'every event hits one barrier');
  check(lab.uniqueness.every((u) => u > 0 && u <= 1), 'uniqueness in (0, 1]');
  check(lab.meta.testEvents > 100 && lab.meta.filtered.bets <= lab.meta.primary.bets, 'meta-labels filter the primary bets');
}
const val = run('afmlValidation', () => sat.afmlValidation({ ...spec, validation: { horizon: 5, folds: 4, embargo: 5, trees: 20, maxRows: 2500, groups: 5, testGroups: 2 } }));
if (val) {
  check(val.cv.length === 3 && val.cv[0].trainRows > val.cv[2].trainRows, 'purging removes training rows');
  check(val.cpcv.paths.length === 4 && val.cpcv.splits === 10, 'CPCV: C(5, 2) = 10 splits, k C(N, k) / N = 4 paths');
  check(val.mdi.length === 23 && val.mda.length === 23 && val.sfi.length === 23, 'importance per feature');
}
const pf = run('afmlPortfolio', () => sat.afmlPortfolio({ ...spec, portfolio: { window: 200, rebalance: 21, trials: 20 } }));
if (pf) {
  near(Array.from(pf.hrp).reduce((a, b) => a + b, 0), 1, 1e-9, 'HRP weights sum to 1');
  check(Array.from(pf.hrp).every((w) => w > 0), 'HRP is long-only');
  check(pf.orderedTickers.length === 16 && new Set(pf.orderedTickers).size === 16, 'quasi-diagonal order is a permutation');
  check(pf.backtest.length === 4 && pf.monteCarlo.hrp.length === 20, 'backtest and Monte Carlo');
}
const of = run('afmlOverfitting', () => sat.afmlOverfitting({ ...spec, overfitting: { blocks: 8 } }));
if (of) {
  check(of.trials === 3 * spec.strategies.length && of.pbo.combinations === 70, 'trials and CSCV combinations');
  check(of.best.dsr <= of.best.psr + 1e-12 && of.pbo.pbo >= 0 && of.pbo.pbo <= 1, 'DSR <= PSR, PBO a probability');
}
const afmlModels = run('models (triple barrier, extra features, CUSUM, uniqueness)', () => sat.models({ ...spec,
  label: { kind: 'triple', horizon: 5, barrierWidth: 1, volSpan: 50 }, extraFeatures: ['ffd', 'vol', 'amihud'], cusumMultiple: 1,
  walkForward: { ...spec.walkForward, weighting: 'decay' } }));
if (afmlModels) check(afmlModels.features.length === 26 && afmlModels.features[23] === 'ffd' && !afmlModels.cached, 'extra features retrain the models');
const emuBet = run('gpuEmulate (bet-sized rule)', () => sat.gpuEmulate({ ...spec, analysis: 'strategies', strategies: [{ kind: 'betsize', param: 0.1, holding: 1 }] }));
const cpuBet = sat.strategies({ ...spec, strategies: [{ kind: 'betsize', param: 0.1, holding: 1 }] });
if (emuBet && !cpuBet.error) near(emuBet.candidates[0].metrics.sharpe, cpuBet.candidates[0].metrics.sharpe, 1e-3, 'bet-sized rule: kernels vs CPU');

// Hedging and algorithmic trading.
const hd = run('hedgeOverlays', () => sat.hedgeOverlays(spec));
if (hd) {
  check(hd.series.length === 7 && hd.dates.length === hd.series[0].equity.length, 'one series per overlay, dates align with equity');
  const vt = hd.series.find((s) => s.name === 'volatility target');
  check(Math.abs(vt.metrics.annualVolatility - 0.10) < 0.05, `volatility target near 10%: ${vt.metrics.annualVolatility}`);
  check(Math.abs(hd.marketCorrelation[2]) < Math.abs(hd.marketCorrelation[0]) + 0.05, 'the Kalman hedge does not add market exposure');
}
const op = run('hedgeOptions', () => sat.hedgeOptions({ ...spec, options: { paths: 400 } }));
if (op) {
  const f = op.frequencies;
  check(f.length === 7 && f[0].sd < f[f.length - 1].sd, 'hedging error grows as rebalancing thins out');
  check(op.overlays[1].metrics.maxDrawdown < op.overlays[0].metrics.maxDrawdown, 'the protective put cuts the drawdown');
  check(op.histograms.daily.length === 400 && op.histograms.weekly.length === 400, 'P&L histograms');
}
const pr = run('algoPairs', () => sat.algoPairs(spec));
if (pr) {
  check(pr.synthetic.cointegration.cointegrated && pr.synthetic.strategy.metrics.sharpe > 0.5, 'the generated pair is cointegrated and tradable');
  check(pr.pairsTested === 16 * 15 / 2 && pr.scan.length === 5, 'every pair of the universe is scanned');
}
const tr = run('algoTrend', () => sat.algoTrend(spec));
if (tr) {
  check(tr.series.length === 3 && tr.rules.length === 6, 'three series, six rules');
  const last = tr.rules.reduce((s, r) => s + r.weights[r.weights.length - 1], 0);
  near(last, 1, 1e-9, 'rule weights sum to 1');
  check(tr.dates.length === tr.series[0].equity.length, 'dates align with equity');
}
const rg = run('algoRegimes', () => sat.algoRegimes({ ...spec, regimes: { window: 300, refitEvery: 63 } }));
if (rg) {
  check(rg.model.sd[1] > rg.model.sd[0] && rg.accuracy > 0.6, `HMM separates the volatile regime (accuracy ${rg.accuracy})`);
  check(rg.probabilities.length === 2 && rg.probabilities[0].length === 759, 'filtered probabilities per state and day');
}
const ex = run('algoExecution', () => sat.algoExecution({ execution: { paths: 2000 } }));
if (ex) {
  const [twap, ac] = ex.plans;
  check(ac.expectedCost > twap.expectedCost && ac.sd < twap.sd, 'risk aversion trades cost for risk');
  near(ac.simMean, ac.expectedCost, 0.1 * ac.expectedCost, 'Monte Carlo mean matches the closed form');
  check(ex.frontierCost.every((c, i) => i === 0 || c >= ex.frontierCost[i - 1] - 1e-6), 'frontier cost rises with risk aversion');
}
const tn = run('strategyTournament', () => sat.strategyTournament(spec));
if (tn) {
  check(tn.sleeves.length === 3 + 3 + 4 && tn.sleeves[tn.sleeves.length - 1].family === 'benchmark', 'three models, three self-adaptive variants, trend, pairs, regimes, benchmark');
  check(tn.allocators.length === 6 && tn.chosen === 'exponential weights', 'six allocators, exponential weights by default');
  const n = tn.sleeves[0].equity.length;
  check(tn.sleeves.every((s) => s.equity.length === n) && tn.allocators.every((a) => a.equity.length === n) && tn.dates.length === n, 'every series on the same days');
  const eq = tn.allocators.find((a) => a.name === 'equal weight');
  const w = eq.weights.map((v) => v[10]);
  near(w.reduce((a, b) => a + b, 0), 1, 1e-9, 'equal-weight allocator is fully invested');
}

const csv = 'date,ticker,open,high,low,close,volume\n2024-01-02,A,1,1,1,1,1\n';
check(typeof sat.marketData({ ...spec, csv }).error === 'string', 'a one-stock CSV is rejected');

console.log(failures ? `\n${failures} failure(s)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
