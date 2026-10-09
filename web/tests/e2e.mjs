// End-to-end test of the web front end in headless Chromium (Playwright).
//
//   node web/tests/e2e.mjs [page-filter] [--root DIR] [--file]
//
// Serves web/ (or --root DIR, e.g. dist/standalone) on a local port, or with --file opens the
// pages straight from disk as file:// URLs (the copy-and-open deployment of the standalone
// build, which must then make no network request at all). Opens every page, waits for its automatic run and
// checks that it finished without errors and rendered charts. On the WebGPU page
// it also checks that the strategy-search kernels agree with the WASM library.
// WebGPU runs on SwiftShader in headless mode, so the GPU page works without a GPU.
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const argv = process.argv.slice(2);
const option = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
const fileMode = argv.includes('--file');
const root = path.resolve(option('--root') || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'));
const filter = argv.find((a, i) => !a.startsWith('--') && argv[i - 1] !== '--root');
const { chromium, devices } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright').catch(() =>
  import('/opt/node22/lib/node_modules/playwright/index.mjs'));

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm', '.png': 'image/png' };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = path.join(root, url === '/' ? 'index.html' : url);
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || (fs.existsSync('/opt/pw-browsers/chromium-1194/chrome-linux/chrome') ? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' : undefined),
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'],
});

// A small universe and light models keep the run fast; the pages pick the spec up from localStorage.
const testSpec = {
  market: { numAssets: 16, numDates: 760, seed: 7, persistence: 0.985, idiosyncraticVol: 0.016, betaDispersion: 0.3,
    regimes: [
      { name: 'bull', drift: 0.0008, volatility: 0.009, momentum: 0.12, volumeReversal: 0.15 },
      { name: 'bear', drift: -0.001, volatility: 0.018, momentum: -0.06, volumeReversal: 0.22 },
      { name: 'range', drift: 0, volatility: 0.008, momentum: -0.16, volumeReversal: 0.3 },
    ] },
  models: [
    { type: 'logistic', l2: 1 },
    { type: 'xgboost', trees: 25, maxDepth: 3, learningRate: 0.1, subsample: 0.8, colsample: 0.8, seed: 1 },
    { type: 'lstm', hidden: 8, seqLen: 5, epochs: 1, learningRate: 0.01, maxSamples: 1500, seed: 1 },
  ],
  walkForward: { trainWindow: 300, retrainEvery: 63, maxTrainRows: 4000, seed: 11 },
  robustness: { lookbacks: [21, 42, 63], steps: [5, 21], metrics: ['return', 'sharpe'], costs: [0, 10, 30] },
};
// The AFML pages on the same small universe, with lighter workloads.
const afmlSpec = {
  ...testSpec,
  bars: { days: 30, tradesPerDay: 1000, activityDispersion: 0.6, persistence: 0.6, barsPerDay: 20, seed: 5 },
  validation: { model: 'xgboost', trees: 20, horizon: 5, folds: 4, embargo: 5, groups: 5, testGroups: 2, maxRows: 2500 },
  portfolio: { window: 200, rebalance: 21, trials: 30, simAssets: 10 },
  overfitting: { blocks: 8 },
};
// Hedging and algorithmic trading pages, lighter simulations.
const algoSpec = {
  ...testSpec,
  options: { vol: 0.2, impliedVol: 0.2, years: 0.25, strike: 1, costBps: 0, paths: 400, seed: 13, tenorDays: 21, putMoneyness: 0.95, callMoneyness: 1.05, volPremium: 0.02 },
  regimes: { states: 2, window: 300, refitEvery: 63, threshold: 0.5, riskOffExposure: 0, costBps: 5, targetVol: 0.12 },
  execution: { shares: 1e6, price: 50, horizonDays: 5, periods: 25, sigma: 0.95, epsilon: 0.0625, eta: 2.5e-6, gamma: 2.5e-7, riskAversion: 1e-6, paths: 1000 },
};
// Pages with the engine selector. With no stored choice they run on Auto, which must pick the
// WebGPU kernels on desktop too; some also run once forced onto WebAssembly.
const enginePages = ['index', 'strategies', 'adaptive', 'robustness'];
const cases = [
  ...['index', 'core', 'data', 'factors', 'labels', 'models', 'strategies', 'adaptive', 'robustness']
    .map((name) => ({ name, file: name, spec: testSpec, ...(enginePages.includes(name) ? { engine: 'auto', expectEngine: 'WebGPU' } : {}) })),
  // Advances in Financial Machine Learning pages (WebAssembly only).
  ...['bars', 'fracdiff', 'labeling', 'validation', 'portfolio', 'overfitting'].map((name) => ({ name, file: name, spec: afmlSpec })),
  // Hedging and algorithmic trading pages (WebAssembly only).
  ...['hedging', 'options', 'pairs', 'trend', 'regimes', 'execution', 'tournament'].map((name) => ({ name, file: name, spec: algoSpec })),
  ...['index', 'strategies'].map((name) => ({ name: `${name}-wasm`, file: name, spec: testSpec, engine: 'wasm', expectEngine: 'WebAssembly' })),
  // A first visit (nothing stored): the default is Auto, which must dispatch the kernels.
  { name: 'adaptive-first-visit', file: 'adaptive', spec: testSpec, expectEngine: 'WebGPU' },
  // A mix of the top two candidates needs the CPU selector: Auto must fall back and say why.
  { name: 'adaptive-top2', file: 'adaptive', spec: { ...testSpec, selector: { lookback: 63, adaptEvery: 21, metric: 'sharpe', topM: 2, allowCash: true, minScore: 0 } }, engine: 'auto', expectEngine: 'WebAssembly', expectReason: 'top 2' },
  { name: 'gpu', file: 'gpu', spec: testSpec },
  // WASM/GPU pairs on the same spec whose headline numbers must agree.
  ...['adaptive', 'robustness'].flatMap((name) => [
    { name: `${name}-wasm-pair`, file: name, spec: testSpec, engine: 'wasm', expectEngine: 'WebAssembly', pair: name },
    { name: `${name}-gpu-pair`, file: name, spec: testSpec, engine: 'gpu', expectEngine: 'WebGPU', pair: name },
  ]),
  // Android drivers: no adapter for a high-performance request but one for a plain request
  // (Auto must still reach WebGPU), and no adapter at all (WebAssembly, with the reason).
  { name: 'adaptive-android-quirk', file: 'adaptive', spec: testSpec, engine: 'auto', expectEngine: 'WebGPU', gpuStub: 'no-high-performance', device: 'Pixel 7' },
  // Chrome offers only a WebGPU compatibility-mode adapter (OpenGL ES): Auto must use it.
  { name: 'adaptive-compat-only', file: 'adaptive', spec: testSpec, engine: 'auto', expectEngine: 'WebGPU', gpuStub: 'compat-only', device: 'Pixel 7', expectReason: 'compatibility mode' },
  { name: 'adaptive-no-adapter', file: 'adaptive', spec: testSpec, engine: 'auto', expectEngine: 'WebAssembly', gpuStub: 'no-adapter', expectReason: 'WebGPU unavailable' },
  // A phone: Auto must pick WebGPU, and the collapsed menu must leave the page content in view.
  { name: 'adaptive-mobile', file: 'adaptive', spec: testSpec, engine: 'auto', expectEngine: 'WebGPU', device: 'iPhone 14' },
];
const pairs = {};
let failures = 0;
console.log(`testing ${root} ${fileMode ? 'from file:// (no server)' : `over ${base}`}`);

for (const { name, file, spec, engine, expectEngine, pair, device, gpuStub, expectReason } of cases.filter((c) => !filter || c.name.includes(filter))) {
  const context = await browser.newContext(device ? { ...devices[device] } : { viewport: { width: 1400, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const network = [];
  page.on('request', (r) => { if (!/^(file|data|blob):/.test(r.url())) network.push(r.url()); });
  if (gpuStub) await page.addInitScript((stub) => {
    const gpu = navigator.gpu;
    if (!gpu) return;
    const original = gpu.requestAdapter.bind(gpu);
    gpu.requestAdapter = (options = {}) =>
      stub === 'no-adapter' || (stub === 'compat-only' ? !('featureLevel' in options) : options.powerPreference === 'high-performance')
        ? Promise.resolve(null) : original(options);
  }, gpuStub);
  await page.addInitScript(({ spec, engine }) => {
    if (!sessionStorage.getItem('seeded')) {
      localStorage.setItem('sat-spec-v1', JSON.stringify(spec));
      if (engine) localStorage.setItem('sat-engine', engine); else localStorage.removeItem('sat-engine');
      sessionStorage.setItem('seeded', '1');
    }
  }, { spec, engine });
  const t0 = Date.now();
  await page.goto(fileMode ? pathToFileURL(path.join(root, `${file}.html`)).href : `${base}/${file}.html`);
  let status = '';
  try {
    await page.waitForFunction(() => {
      const s = document.querySelector('#run-status');
      return s && /Done|GPU .* ms|Error|ready/.test(s.textContent);
    }, null, { timeout: 240000 });
    status = await page.textContent('#run-status');
  } catch (e) {
    status = 'timeout';
  }
  const charts = await page.$$eval('.chart canvas', (cs) => cs.filter((c) => c.width > 0).length);
  const problems = [...errors];
  if (/Error|timeout/.test(status)) problems.push('status: ' + status);
  if (charts === 0) problems.push('no charts rendered');
  if (fileMode && network.length) problems.push(`network requests from a file:// page: ${network.slice(0, 3).join(', ')}`);
  if (expectEngine && !status.includes(`· ${expectEngine}`)) problems.push(`expected the ${expectEngine} engine: ${status}`);
  if (expectReason && !status.includes(expectReason)) problems.push(`expected the reason "${expectReason}": ${status}`);
  // The status line is not enough: the kernels must really have been dispatched on WebGPU
  // (and never when WebAssembly ran).
  const gpuRuns = await page.evaluate(() => globalThis.__satGpuRuns || 0);
  if (expectEngine === 'WebGPU' && gpuRuns === 0) problems.push('no kernel pipeline was dispatched on WebGPU');
  if (expectEngine === 'WebAssembly' && gpuRuns > 0) problems.push(`${gpuRuns} WebGPU dispatches while WebAssembly ran`);
  if (pair) pairs[pair] = { ...(pairs[pair] || {}), [engine]: await page.evaluate(() => window.__satEngineRun) };
  if (device) {
    const layout = await page.evaluate(() => ({
      menuHidden: document.getElementById('nav-links').offsetParent === null,
      titleTop: document.querySelector('main h1').getBoundingClientRect().top,
      scrollW: document.documentElement.scrollWidth, width: document.documentElement.clientWidth,
    }));
    if (!layout.menuHidden) problems.push('mobile menu is not collapsed');
    if (layout.titleTop > 200) problems.push(`page title starts ${Math.round(layout.titleTop)}px down: content hidden below the menu`);
    if (layout.scrollW > layout.width + 1) problems.push(`horizontal scroll on mobile (${layout.scrollW} > ${layout.width})`);
    await page.click('.menu-toggle');
    if (await page.evaluate(() => document.getElementById('nav-links').offsetParent === null)) problems.push('menu button does not open the menu');
  }
  if (file === 'gpu') {
    const probe = await page.waitForFunction(() => window.__satWebGpuProbe, null, { timeout: 30000 }).then((h) => h.jsonValue()).catch(() => null);
    if (!probe || !probe.adapters.some((a) => a.adapter)) problems.push(`WebGPU diagnostics found no adapter: ${JSON.stringify(probe)}`);
  }
  if (file === 'core' && !problems.length && !(await page.evaluate(() => window.__satLookahead))) problems.push('an alpha looks ahead');
  if (file === 'gpu' && !problems.length) {
    const r = await page.evaluate(() => window.__satLastRun);
    if (!r) problems.push('no GPU result');
    else {
      const c = r.checks;
      if (!(c.candidateSharpeDiff < 1e-3)) problems.push(`candidate Sharpe mismatch ${c.candidateSharpeDiff}`);
      if (!(c.candidateReturnDiff < 1e-3)) problems.push(`candidate return mismatch ${c.candidateReturnDiff}`);
      if (!(c.selectionAgreement >= 0.95)) problems.push(`selection agreement ${c.selectionAgreement}`);
      if (!(c.selectorSharpeDiff < 0.1)) problems.push(`selector Sharpe mismatch ${c.selectorSharpeDiff}`);
      console.log(`       GPU checks: candidates ΔSharpe ${c.candidateSharpeDiff.toExponential(1)}, Δreturn ${c.candidateReturnDiff.toExponential(1)}; selectors same choice ${(100 * c.selectionAgreement).toFixed(2)}%, ΔSharpe ${c.selectorSharpeDiff.toFixed(4)}`);
    }
  }
  if (process.env.SCREENSHOTS) await page.screenshot({ path: path.join(process.env.SCREENSHOTS, `${name}.png`), fullPage: true });
  console.log(`${problems.length ? 'FAIL' : 'ok  '} ${name.padEnd(12)} ${String(Date.now() - t0).padStart(6)} ms  charts=${charts}  gpu=${gpuRuns}  ${status.trim().slice(0, 80)}`);
  for (const p of problems) console.log('       ' + p);
  failures += problems.length ? 1 : 0;
  await context.close();
}

// The same spec on both engines: the GPU kernels (f32) and WASM (f64) run the same
// deterministic backtests, so the headline numbers must agree closely.
const metrics = {
  adaptive: [['adaptive Sharpe', (r) => r.sharpe, 0.05], ['adaptive annual return', (r) => r.annualReturn, 0.02]],
  robustness: [['share of settings beating the median', (r) => r.shareBeatingMedianFixed, 0.1], ['selector Sharpe', (r) => r.sharpe, 0.05]],
};
for (const [name, r] of Object.entries(pairs)) {
  if (!r.wasm || !r.gpu) continue;
  for (const [label, get, tol] of metrics[name]) {
    const w = get(r.wasm), g = get(r.gpu);
    const ok = Math.abs(g - w) <= tol;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${`${name} engines`.padEnd(18)} ${label}: WASM ${w.toFixed(4)} vs GPU ${g.toFixed(4)} (tol ${tol})`);
    if (!ok) failures++;
  }
}

await browser.close();
server.close();
console.log(failures ? `\n${failures} page(s) failed` : '\nall pages passed');
process.exit(failures ? 1 : 0);
