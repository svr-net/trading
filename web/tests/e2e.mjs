// Browser test of the page in headless Chromium (Playwright): node web/tests/e2e.mjs
// Serves web/, runs the sample market on WebGPU (SwiftShader in headless mode), on the emulated
// GPU and on the reference engine, and checks that each run renders the trades for the next open and
// the backtest, without page errors, and that the WebGPU kernels agree with their emulation.
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright').catch(() => import('/opt/node22/lib/node_modules/playwright/index.mjs'));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.wasm': 'application/wasm' };
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
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'],
});

const results = {};
let failed = 0;
// One engine by default (WebGPU on SwiftShader, the page's own path); E2E_ENGINES=gpu,gpu-cpu,emulated,reference for all.
const ENGINES = { gpu: 'WebGPU', 'gpu-cpu': 'CPU-simulated GPU', emulated: 'Emulated GPU', reference: 'Reference' };
for (const [engine, expect] of (process.env.E2E_ENGINES || 'gpu').split(',').map((e) => [e, ENGINES[e]])) {
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  // A server without a published UK scan answers data/uk.json with 404; the page says so.
  page.on('console', (m) => { if (m.type() === 'error' && !/404/.test(m.text())) errors.push(m.text()); });
  try {
    await page.goto(`${base}/index.html`);
    await page.waitForFunction(() => /ready/.test(document.getElementById('status').textContent), null, { timeout: 60000 });
    await page.selectOption('#engine', engine);
    await page.click('#run');
    await page.waitForFunction(() => /^Done|^Could not/.test(document.getElementById('status').textContent), null, { timeout: 300000 });
    const statusText = await page.textContent('#status');
    if (!statusText.startsWith('Done')) throw new Error(statusText);
    if (!statusText.includes(expect)) throw new Error(`ran on the wrong engine: ${statusText}`);
    const annual = await page.$eval('#verdict .tile .v', (e) => parseFloat(e.textContent) / 100);
    const drawn = await page.$eval('#chart', (c) => c.width > 0);
    if (!drawn || !Number.isFinite(annual)) throw new Error('backtest not rendered');
    if (errors.length) throw new Error(errors.join(' | '));
    // The trades for the next open: at most 10, each expected to make money after costs.
    if (!/Sample market/.test(await page.textContent('#asof'))) throw new Error('trades not rendered');
    const trades = await page.$$eval('#plan tbody tr:not(.gap)', (rows) => rows.map((r) => [r.cells[0].textContent, parseFloat(r.cells[1].textContent)]));
    if (trades.length > 10 || trades.some(([, e]) => !(e > 0))) throw new Error(`bad trades: ${JSON.stringify(trades)}`);
    const recent = await page.$$eval('#recent tbody tr', (r) => r.length);
    if (recent < 1) throw new Error('last 20 days not rendered');
    results[engine] = { annual, top: trades.map(([t]) => t) };
    console.log(`ok   ${engine.padEnd(9)} daily trades ${(100 * annual).toFixed(1)}% a year, ${trades.length} trades  ${statusText}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${engine}: ${e.message}`);
  }
  await page.close();
}
if (results.gpu && results.emulated) {
  const overlap = results.gpu.top.filter((t) => results.emulated.top.includes(t)).length;
  const n = Math.max(results.gpu.top.length, results.emulated.top.length);
  const ok = Math.abs(results.gpu.annual - results.emulated.annual) < 0.05 && overlap >= n - 2;
  console.log(`${ok ? 'ok  ' : 'FAIL'} WebGPU vs emulated: ${(100 * results.gpu.annual).toFixed(1)}% vs ${(100 * results.emulated.annual).toFixed(1)}% a year, trades in common ${overlap}/${n}`);
  if (!ok) failed++;
}
await browser.close();
server.close();
process.exit(failed ? 1 : 0);
