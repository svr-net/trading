// Node smoke test of the WASM wrapper: every entry point on a reduced default spec,
// with sanity checks against the library's own invariants.
//   node web/tests/wasm.test.mjs
import createSatModule from '../wasm/sat.js';
import { MODEL_DEFAULTS, defaultSpec } from '../js/spec.js';

const sat = await createSatModule();
let failures = 0;
const check = (cond, msg) => { if (!cond) { failures++; console.log('  FAIL', msg); } };
const near = (a, b, tol, msg) => check(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

// A smaller universe and lighter models keep the run short.
const spec = defaultSpec();
spec.market = { ...spec.market, numAssets: 16, numDates: 760 };
spec.walkForward = { ...spec.walkForward, trainWindow: 300, retrainEvery: 63, maxTrainRows: 4000 };
// Three families of different kinds (linear, boosted trees, recurrent network), not only the default pool.
spec.models = ['logistic', 'xgboost', 'lstm'].map((t) => ({ ...MODEL_DEFAULTS[t], trees: 25, epochs: 1, maxSamples: 1500 }));
spec.robustness = { lookbacks: [21, 63], steps: [5, 21], metrics: ['return', 'sharpe'], costs: [0, 20] };
// The models on their own (the default trades their composite alone; checked below).
spec.composite = { ...spec.composite, method: 'none' };

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
  const withComposite = sat.models({ ...spec, composite: defaultSpec().composite });
  check(withComposite.models.length === 4 && withComposite.models[3].name === 'Composite (average)', 'the Models page shows the models and their composite');
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
// Pipeline options drawn from Advances in Financial Machine Learning.
const afmlModels = run('models (triple barrier, extra features, CUSUM, uniqueness)', () => sat.models({ ...spec,
  label: { kind: 'triple', horizon: 5, barrierWidth: 1, volSpan: 50 }, extraFeatures: ['ffd', 'vol', 'amihud'], cusumMultiple: 1,
  walkForward: { ...spec.walkForward, weighting: 'decay' } }));
if (afmlModels) check(afmlModels.features.length === 26 && afmlModels.features[23] === 'ffd' && !afmlModels.cached, 'extra features retrain the models');
const emuBet = run('gpuEmulate (bet-sized rule)', () => sat.gpuEmulate({ ...spec, analysis: 'strategies', strategies: [{ kind: 'betsize', param: 0.1, holding: 1 }] }));
const cpuBet = sat.strategies({ ...spec, strategies: [{ kind: 'betsize', param: 0.1, holding: 1 }] });
if (emuBet && !cpuBet.error) near(emuBet.candidates[0].metrics.sharpe, cpuBet.candidates[0].metrics.sharpe, 1e-3, 'bet-sized rule: kernels vs CPU');

// The composite forecast: the web default trades the equal-weight average in place of the
// models; the self-adaptive forecast picks the model or average with the best recent record.
const withComposite = run('models (with the default composite)', () => sat.models({ ...spec, composite: defaultSpec().composite }));
if (withComposite) check(withComposite.models.length === 4 && withComposite.models[3].name === 'Composite (average)', 'the Models page shows the models and their composite');
const def = run('strategies (default composite pool)', () => sat.strategies({ ...spec, composite: defaultSpec().composite }));
if (def) check(def.candidates.length === spec.strategies.length && def.candidates.every((c) => JSON.stringify(c).includes('Composite (average)')), 'the default pool trades the composite alone');
const sa = run('adaptive (self-adaptive forecast, cached models)', () => sat.adaptive({ ...spec, composite: { method: 'adaptive', keepMembers: false } }));
if (sa) check(sa.cachedPredictions !== false, 'changing only the composite never re-trains the models');
const both = run('strategies (models + self-adaptive forecast)', () => sat.strategies({ ...spec, composite: { method: 'adaptive', keepMembers: true } }));
if (both) check(both.candidates.length === 4 * spec.strategies.length, 'three models and the self-adaptive forecast');
check(typeof sat.strategies({ ...spec, composite: { method: 'stacked' } }).error === 'string', 'removed combinations are rejected');

// The emulated GPU device: each plan of gpuJobs run through gpuRunPlan and handed to
// gpuAnalyse must give exactly what gpuEmulate (compile + reference + analyse in one call) gives.
const emuJobs = run('gpuJobs (for the emulated GPU)', () => sat.gpuJobs({ ...spec, analysis: 'adaptive' }));
if (emuJobs && !emuJobs.unsupported) {
  const outputs = emuJobs.plans.map((plan) => {
    if (!plan) return null;
    const out = sat.gpuRunPlan({ header: plan.header, tables: plan.tables });
    check(!out.error && out.stats.length === plan.bytes.stats / 4 && out.adapt.length === plan.bytes.adapt / 4, `gpuRunPlan read-back sizes ${out.error || ''}`);
    return { stats: out.stats, adapt: out.adapt };
  });
  const viaDevice = run('gpuAnalyse (emulated GPU read-back)', () => sat.gpuAnalyse({ ...spec, analysis: 'adaptive', gpuOutputs: outputs }));
  const direct = sat.gpuEmulate({ ...spec, analysis: 'adaptive' });
  if (viaDevice && !direct.error) near(viaDevice.adaptive.metrics.sharpe, direct.adaptive.metrics.sharpe, 1e-12, 'emulated device = gpuEmulate');
  const plan = emuJobs.plans.find(Boolean);
  check(typeof sat.gpuRunPlan({ header: plan.header, tables: plan.tables.subarray(0, plan.tables.length - 1) }).error === 'string', 'a truncated plan is rejected');
  check(typeof sat.gpuRunPlan({ header: plan.header.subarray(0, 4), tables: plan.tables }).error === 'string', 'a short header is rejected');
}

// Candidate backtests on the kernels: the 'book' plan read back through the emulated GPU.
const bookJobs = run('gpuJobs (candidate book)', () => sat.gpuJobs({ ...spec, analysis: 'book' }));
if (bookJobs && !bookJobs.unsupported) {
  const plan = bookJobs.plans[0];
  const out = sat.gpuRunPlan({ header: plan.header, tables: plan.tables, readBook: true });
  check(!out.error && out.book.length === plan.numCandidates * plan.numDays * 2, `kernel book size ${out.error || out.book?.length}`);
  check(plan.numSelectors === 0, 'the book plan has no selectors');
}

const csv = 'date,ticker,open,high,low,close,volume\n2024-01-02,A,1,1,1,1,1\n';
check(typeof sat.marketData({ ...spec, csv }).error === 'string', 'a one-stock CSV is rejected');

console.log(failures ? `\n${failures} failure(s)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
