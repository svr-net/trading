// The WebAssembly model (web/lab/ofm_uk.js, docker/wasm/Dockerfile) gives the native model's results: the same
// backtest and the same orders on a made-up market (two runs: the UK's costs, and Europe's with its venues).
//   node tools/check/wasm_same.mjs build/ofm_uk web/lab/ofm_uk.js
import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { pathToFileURL } from 'url';

const [native, wasmJs] = process.argv.slice(2);
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wasm-'));
let seed = 11;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const rows = ['date,ticker,open,high,low,close,volume'];
const days = [];
for (let t = Date.UTC(2019, 0, 1); days.length < 900; t += 86400000) { const d = new Date(t); if (d.getUTCDay() % 6) days.push(d.toISOString().slice(0, 10)); }
for (let k = 0; k < 80; k++) {
  let c = 100, prev = 0;
  for (const d of days) {
    const r = gauss() * 0.015 - 0.3 * prev, o = c * (1 + gauss() * 0.004), c2 = o * (1 + r);
    prev = r;
    rows.push(`${d},${k % 2 ? 'PAR' : 'AMS'}:S${k},${o.toFixed(4)},${(Math.max(o, c2) * 1.006).toFixed(4)},${(Math.min(o, c2) * 0.994).toFixed(4)},${c2.toFixed(4)},${30000 + Math.floor(rnd() * 60000)}`);
    c = c2;
  }
}
fs.writeFileSync(path.join(dir, 'p.csv'), rows.join('\n') + '\n');
fs.writeFileSync(path.join(dir, 'v.txt'), 'PAR 5000 0.0005 3 0 0 0 0.004\nAMS 5000 0.0005 3 0 0 0\n');
const create = (await import(pathToFileURL(path.resolve(wasmJs)).href)).default;
for (const [name, extra] of [['EU', ['--venues', 'v.txt']], ['UK', []]]) {
  const args = ['--csv', 'p.csv', '--costs', name, '--top', '60', '--hold', '5', '--min-value', '10000', ...extra];
  execFileSync(path.resolve(native), [...args, '--json', 'n.json', '--orders', 'n-o.json'], { cwd: dir, stdio: 'ignore' });
  const m = await create({ print: () => {}, printErr: () => {} });
  m.FS.writeFile('/p.csv', fs.readFileSync(path.join(dir, 'p.csv')));
  m.FS.writeFile('/v.txt', fs.readFileSync(path.join(dir, 'v.txt')));
  try { m.callMain(args.map((a) => (a.endsWith('.csv') || a.endsWith('.txt') ? `/${a}` : a)).concat(['--json', '/w.json', '--orders', '/w-o.json'])); } catch (e) { if (!(e && e.name === 'ExitStatus')) throw e; }
  const nb = JSON.parse(fs.readFileSync(path.join(dir, 'n.json'))).backtest, wb = JSON.parse(m.FS.readFile('/w.json', { encoding: 'utf8' })).backtest;
  const no = JSON.parse(fs.readFileSync(path.join(dir, 'n-o.json'))).orders.map((o) => `${o.ticker} ${o.limit}`), wo = JSON.parse(m.FS.readFile('/w-o.json', { encoding: 'utf8' })).orders.map((o) => `${o.ticker} ${o.limit}`);
  if (JSON.stringify(nb) !== JSON.stringify(wb)) throw new Error(`${name}: the backtests differ\n${JSON.stringify(nb)}\n${JSON.stringify(wb)}`);
  if (JSON.stringify(no) !== JSON.stringify(wo)) throw new Error(`${name}: the orders differ\n${no}\n${wo}`);
  console.log(`${name}: the same backtest (${nb.trades} trades, Sharpe ${nb.sharpe}) and the same ${no.length} orders`);
}
console.log('wasm: the WebAssembly model gives the native model\'s results; OK');
