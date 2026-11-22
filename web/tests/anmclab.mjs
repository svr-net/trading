// The adaptive nested Monte Carlo in the browser: its WebAssembly builds (docker/wasm/Dockerfile's anmc stage) run the
// differential network's basket check (apps/ofm_anmc.cpp --basket) and give the native build's answers.
//   CPU build   ofm_anmc.js in Node: the fused kernel's f32 reference on the CPU, as the native build runs it
//   GPU build   ofm_anmc_gpu.js in headless Chromium (its software GPU, SwiftShader): the same fits on WebGPU through
//               backends/webgpu.cpp (diffnetOffload: the fused kernel diffnetFused), every one of them on the GPU
// Each run's adaptive VaR and ES must be within 0.5% of the exact value from the native build's.
//   node web/tests/anmclab.mjs WASM_DIR NATIVE_OFM_ANMC
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [wasmDir, native] = process.argv.slice(2);
if (!wasmDir || !native) { console.error('usage: node web/tests/anmclab.mjs WASM_DIR NATIVE_OFM_ANMC'); process.exit(2); }
// 10 assets, 2,000 training rows (480 Adam steps a fit), two statistics: two network fits a run
const args = ['--basket', '10', '--outer', '4000', '--budget', '200000', '--methods', 'fused', '--stats', 'VaR,ES', '--estimates'];
const estimates = (lines) => Object.fromEntries(lines.filter((l) => l.startsWith('estimate,')).map((l) => {
  const [, method, stat, , estimate, baseline, exact, paths] = l.split(',');
  return [stat, { method, estimate: +estimate, baseline: +baseline, exact: +exact, paths: +paths }];
}));

let ok = true;
const fail = (m) => { console.log(`FAIL: ${m}`); ok = false; };
const compare = (name, got, ref) => {
  for (const stat of ['VaR', 'ES']) {
    const g = got[stat], r = ref[stat];
    if (!g) { fail(`${name}: no ${stat} estimate`); continue; }
    const rel = Math.abs(g.estimate - r.estimate) / Math.abs(r.exact);
    console.log(`  ${stat}: ${g.estimate.toPrecision(8)} (native ${r.estimate.toPrecision(8)}, exact ${r.exact.toPrecision(8)}; ${rel === 0 ? 'identical' : `${(100 * rel).toFixed(4)}% of exact apart`}), ${g.paths} inner paths (native ${r.paths})`);
    if (!(rel <= 0.005)) fail(`${name}: ${stat} ${(100 * rel).toFixed(3)}% of the exact value from the native build's`);
  }
};

// native: the reference
let t0 = Date.now();
const nativeLines = execFileSync(path.resolve(native), args, { encoding: 'utf8' }).split('\n');
const ref = estimates(nativeLines);
console.log(`native: ${Date.now() - t0} ms; ${nativeLines.find((l) => l.includes('fused-kernel')).trim()}`);

// the CPU build in Node
t0 = Date.now();
{
  const lines = [];
  const create = (await import(pathToFileURL(path.resolve(wasmDir, 'ofm_anmc.js')).href)).default;
  const m = await create({ print: (s) => lines.push(s), printErr: (s) => lines.push(s) });
  try { m.callMain([...args]); } catch (e) { if (!(e && e.name === 'ExitStatus')) throw e; }  // a copy: callMain prepends argv[0] in place
  console.log(`WebAssembly, CPU build: ${Date.now() - t0} ms; ${(lines.find((l) => l.includes('fused-kernel')) || '?').trim()}`);
  compare('CPU build', estimates(lines), ref);
}

// the GPU build in Chromium
const types = { '.js': 'text/javascript', '.wasm': 'application/wasm' };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (url === '/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end('<!doctype html><title>anmclab</title>'); return; }
  const file = path.join(path.resolve(wasmDir), path.basename(url));
  if (!/^\/lab\/ofm_anmc_gpu\.(js|wasm)$/.test(url) || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(file)] });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader'],
});
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/`);
const LIMIT = Number(process.env.ANMCLAB_LIMIT_S || 900);
const gpu = await page.evaluate(async ({ args, limit }) => {
  const lines = [], t0 = performance.now();
  try {
    const create = (await import('/lab/ofm_anmc_gpu.js')).default;
    let ended;
    const end = new Promise((r) => { ended = r; });
    const m = await create({ print: (s) => lines.push(s), printErr: (s) => lines.push(s), onExit: ended });
    try { m.callMain([...args]); } catch (e) { if (!(e && e.name === 'ExitStatus')) throw e; }
    // Asyncify: callMain returns at the first wait for the GPU; the run ends at exit
    const code = await Promise.race([end, new Promise((r) => setTimeout(() => r('timeout'), limit * 1000))]);
    return { code, lines, ms: performance.now() - t0 };
  } catch (e) {
    return { error: String(e && e.message || e), lines, ms: performance.now() - t0 };
  }
}, { args, limit: LIMIT });
await browser.close();
server.close();
if (gpu.error || gpu.code) fail(`GPU build: ${gpu.error || (gpu.code === 'timeout' ? `no end after ${LIMIT} s` : `exit ${gpu.code}`)}\n${gpu.lines.slice(-8).join('\n')}`);
else {
  const ran = (gpu.lines.find((l) => l.startsWith('anmc: webgpu: ')) || '').replace('anmc: webgpu: ', '');
  console.log(`WebAssembly, GPU build: ${(gpu.ms / 1000).toFixed(1)} s, ${ran || 'no GPU report'}; ${(gpu.lines.find((l) => l.includes('fused-kernel')) || '?').trim()}`);
  const m = /^(\d+) network fits on (.+?)(, (\d+) on the CPU)?( \(|$)/.exec(ran);
  if (!m || /no GPU/.test(m[2])) fail(`the GPU build did not train on the GPU (${ran || 'no report'})`);
  else if (+m[1] !== 2) fail(`the GPU build ran ${m[1]} network fits on the GPU, not 2`);
  else if (m[4]) fail(`${m[4]} network fits fell back to the CPU`);
  if (!gpu.lines.some((l) => l.includes('fused-kernel') && l.includes('(GPU)'))) fail('the estimator did not report a GPU-trained network');
  compare('GPU build', estimates(gpu.lines), ref);
}
console.log(ok ? 'anmclab: the adaptive nested Monte Carlo runs in the browser, its network trained on WebGPU, with the native answers; OK' : 'anmclab: FAILED');
process.exit(ok ? 0 : 1);
