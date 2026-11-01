// Browser test of the page in headless Chromium (Playwright): node web/tests/e2e.mjs
// Serves web/ and checks that each market's published scan renders: the summary, the next open's
// trades (at most 10), and, opened, the backtest tile, the chart and the last days, without page errors.
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = path.join(root, url === '/' ? 'index.html' : url);
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });

let failed = 0;
for (const m of ['uk', 'us']) {
  if (!fs.existsSync(path.join(root, 'data', `${m}.json`))) { console.log(`skip ${m}: no published scan`); continue; }
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  try {
    await page.goto(`${base}/index.html?market=${m}`);
    await page.waitForFunction(() => /close of/.test(document.getElementById('asof').textContent), null, { timeout: 30000 });
    const summary = await page.$eval('#summary', (e) => e.textContent.trim());
    if (!summary) throw new Error('no summary');
    await page.click('.views button[data-view="backtest"]');  // the backtest is its own section; open it to draw the chart
    await page.waitForTimeout(200);
    const trades = await page.$$eval('#plan tbody tr:not(.gap)', (rows) => rows.length);
    const annual = await page.$eval('#verdict .tile .v', (e) => parseFloat(e.textContent) / 100);
    const drawn = await page.$eval('#chart', (c) => c.width > 0);
    const recent = await page.$$eval('#recent tbody tr', (r) => r.length);
    if (trades > 10 || !Number.isFinite(annual) || !drawn || recent < 1) throw new Error(`not rendered: ${trades} trades, ${annual}, chart ${drawn}, ${recent} days`);
    if (errors.length) throw new Error(errors.join(' | '));
    console.log(`ok   ${m}: ${trades} trades, backtest ${(100 * annual).toFixed(1)}% a year`);
  } catch (e) {
    failed++;
    console.log(`FAIL ${m}: ${e.message}`);
  }
  await page.close();
}
await browser.close();
server.close();
process.exit(failed ? 1 : 0);
