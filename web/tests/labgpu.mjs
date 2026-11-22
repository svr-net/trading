// The Lab with the live settings (the nested Monte Carlo) on WebGPU, end to end in headless Chromium (its software GPU,
// SwiftShader): the Lab's worker (web/lab/worker.js) runs the GPU build (--backend webgpu, backends/webgpu.cpp) and the CPU build on a
// generated market, and the native engine runs the same: the CPU build gives the native engine's results, the GPU
// build the native engine's in the GPU's single precision (X_MCGPU_EMULATE), and both make the same orders.
//   node web/tests/labgpu.mjs WASM_DIR NATIVE_ENGINE
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const [wasmDir, native] = process.argv.slice(2);
if (!wasmDir || !native) { console.error('usage: node web/tests/labgpu.mjs WASM_DIR NATIVE_ENGINE'); process.exit(2); }
const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'labgpu-'));
execFileSync(path.resolve(native), ['--synthetic', 'syn.csv'], { cwd: dir, stdio: 'ignore' });
const live = JSON.parse(fs.readFileSync(path.join(web, 'data', 'strategies.json'))).markets.uk.find((s) => s.role === 'live');
const params = Object.entries(live.params).flatMap(([k, v]) => ['--param', `${k}=${v}`]);
const args = ['--csv', '/bars.csv', '--costs', 'US', '--strategy', live.strategy, '--min-value', '0', ...params];
const nativeRun = (env, tag) => {
  const t0 = Date.now();
  execFileSync(path.resolve(native), [...args.map((a) => (a === '/bars.csv' ? 'syn.csv' : a)), '--backend', 'serial', '--json', `${tag}.json`, '--orders', `${tag}-o.json`],
    { cwd: dir, stdio: 'ignore', env: { ...process.env, ...env } });
  return { ms: Date.now() - t0, report: JSON.parse(fs.readFileSync(path.join(dir, `${tag}.json`))), orders: JSON.parse(fs.readFileSync(path.join(dir, `${tag}-o.json`))) };
};
const n64 = nativeRun({}, 'n64'), n32 = nativeRun({ X_MCGPU_EMULATE: '1' }, 'n32');

const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.wgsl': 'text/plain', '.json': 'application/json', '.csv': 'text/csv' };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (url === '/') { res.writeHead(200, { 'content-type': 'text/html' }); res.end('<!doctype html><title>labgpu</title>'); return; }
  if (url === '/syn.csv') { res.writeHead(200, { 'content-type': 'text/csv' }); fs.createReadStream(path.join(dir, 'syn.csv')).pipe(res); return; }
  const name = url.replace(/^\/lab\//, '');
  const file = url.startsWith('/lab/') && /^ofm_strategy/.test(name) ? path.join(path.resolve(wasmDir), name) : path.join(web, url);
  if (!fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader'],
});
const page = await browser.newPage();
await page.goto(`${base}/`);
page.on('console', (m) => console.log(`  [page] ${m.text()}`));
const LIMIT = Number(process.env.LABGPU_LIMIT_S || 900);
const runLab = (gpu) => page.evaluate(async ({ args, gpu, limit }) => {
  const csv = new Uint8Array(await (await fetch('/syn.csv')).arrayBuffer());
  const w = new Worker('/lab/worker.js', { type: 'module' });
  const t0 = performance.now();
  return await new Promise((resolve) => {
    let last = '';
    const timer = setTimeout(() => { w.terminate(); resolve({ error: `no end after ${limit} s; last progress: ${last}` }); }, limit * 1000);
    w.onmessage = ({ data }) => {
      if (data.progress) { last = data.progress; console.log(`${gpu ? 'GPU' : 'CPU'} ${((performance.now() - t0) / 1000).toFixed(0)} s: ${data.progress.slice(0, 110)}`); return; }
      clearTimeout(timer); w.terminate(); resolve(data);
    };
    w.onerror = (e) => { clearTimeout(timer); resolve({ error: e.message }); };
    w.postMessage({ csv, args, env: {}, files: {}, gpu }, [csv.buffer]);
  });
}, { args, gpu, limit: LIMIT });
const gpu = await runLab(true), cpu = await runLab(false);
await browser.close();
server.close();

let ok = true;
const fail = (m) => { console.log(`FAIL: ${m}`); ok = false; };
for (const [name, r] of [['GPU build', gpu], ['CPU build', cpu]]) if (r.error) fail(`${name}: ${r.error}\n${(r.log || []).slice(-8).join('\n')}`);
if (ok) {
  const bt = (r) => JSON.stringify(r.report.backtest), od = (r) => JSON.stringify(r.orders.orders.map((o) => o.ticker));
  console.log(`native (double): ${n64.ms} ms; native in the GPU's precision: ${n32.ms} ms`);
  console.log(`Lab CPU build: ${(cpu.ms / 1000).toFixed(1)} s ${cpu.ran}; Lab GPU build: ${(gpu.ms / 1000).toFixed(1)} s, ${gpu.ran}`);
  const m = /^(\d+) Monte Carlo batches on (.+?)(, (\d+) on the CPU)?( \(|$)/.exec(gpu.ran);
  if (!m || !(+m[1] > 0) || /no GPU/.test(m[2])) fail(`the GPU build did not run on the GPU (${gpu.ran})`);
  else if (m[4]) fail(`${m[4]} batches fell back to the CPU (${gpu.ran})`);
  if (bt(cpu) !== bt(n64)) fail(`the CPU build's backtest differs from the native engine's\n${bt(cpu)}\n${bt(n64)}`);
  if (bt(gpu) !== bt(n32)) fail(`the GPU build's backtest differs from the native engine's in single precision\n${bt(gpu)}\n${bt(n32)}`);
  if (od(gpu) !== od(cpu)) fail(`the GPU and CPU builds make different orders: ${od(gpu)} / ${od(cpu)}`);
  const b = gpu.report.backtest, c = cpu.report.backtest;
  console.log(`backtest: GPU ${b.trades} trades, Sharpe ${b.sharpe}; CPU ${c.trades} trades, Sharpe ${c.sharpe}; orders ${od(gpu)}`);
}
console.log(ok ? 'labgpu: the Lab runs the live settings on WebGPU with the engine\'s results; OK' : 'labgpu: FAILED');
process.exit(ok ? 0 : 1);
