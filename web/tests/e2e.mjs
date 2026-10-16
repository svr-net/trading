// Browser test of the page in headless Chromium (Playwright): node web/tests/e2e.mjs
// Serves web/, runs the sample market on WebGPU (SwiftShader in headless mode) and checks that the
// run renders the trades for the next open and the backtest, without page errors.
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

let failed = 0;
{
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  // A server without published scans answers data/uk.json with 404; the page says so.
  page.on('console', (m) => { if (m.type() === 'error' && !/404/.test(m.text())) errors.push(m.text()); });
  try {
    await page.goto(`${base}/index.html`);
    await page.waitForFunction(() => /ready/.test(document.getElementById('status').textContent), null, { timeout: 60000 });
    await page.click('#run');
    await page.waitForFunction(() => /^Done|^Could not/.test(document.getElementById('status').textContent), null, { timeout: 300000 });
    const statusText = await page.textContent('#status');
    if (!statusText.startsWith('Done')) throw new Error(statusText);
    if (!statusText.includes('WebGPU')) throw new Error(`did not run on WebGPU: ${statusText}`);
    const annual = await page.$eval('#verdict .tile .v', (e) => parseFloat(e.textContent) / 100);
    const drawn = await page.$eval('#chart', (c) => c.width > 0);
    if (!drawn || !Number.isFinite(annual)) throw new Error('backtest not rendered');
    if (errors.length) throw new Error(errors.join(' | '));
    // The trades for the next open: at most 10, each expected to make money after costs.
    if (!/Sample market/.test(await page.textContent('#asof'))) throw new Error('trades not rendered');
    const trades = await page.$$eval('#plan tbody tr:not(.gap)', (rows) => rows.map((r) => [r.cells[0].textContent, parseFloat(r.cells[2].textContent)]));
    if (trades.length > 10 || trades.some(([, e]) => !(e > 0))) throw new Error(`bad trades: ${JSON.stringify(trades)}`);
    const recent = await page.$$eval('#recent tbody tr', (r) => r.length);
    if (recent < 1) throw new Error('last 20 days not rendered');
    console.log(`ok   daily trades ${(100 * annual).toFixed(1)}% a year, ${trades.length} trades  ${statusText}`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${e.message}`);
  }
  await page.close();
}
await browser.close();
server.close();
process.exit(failed ? 1 : 0);
