// The WebAssembly engine (web/lab/ofm_strategy.js, docker/wasm/Dockerfile) gives the native engine's results:
// for every strategy built in, the same backtest and the same orders on a generated market (with Europe's costs).
//   node tools/check/wasm_same.mjs build/ofm_strategy web/lab/ofm_strategy.js
import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { pathToFileURL } from 'url';

const [native, wasmJs] = process.argv.slice(2);
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wasm-'));
execFileSync(path.resolve(native), ['--synthetic', 'syn.csv'], { cwd: dir, stdio: 'ignore' });
// The generated shares as Paris and Amsterdam listings, each exchange with its costs.
const csv = fs.readFileSync(path.join(dir, 'syn.csv'), 'utf8').split('\n')
  .map((l, k) => (k === 0 || !l ? l : l.replace(/,S(\d+),/, (_, n) => `,${+n % 2 ? 'PAR' : 'AMS'}:S${n},`))).join('\n');
fs.writeFileSync(path.join(dir, 'p.csv'), csv);
fs.writeFileSync(path.join(dir, 'v.txt'), 'PAR 5000 0.0005 3 0 0 0 0.004\nAMS 5000 0.0005 3 0 0 0\n');
const names = execFileSync(path.resolve(native), ['--list'], { encoding: 'utf8' }).split('back ends:')[0].split('\n').slice(1).map((l) => l.trim().split(/\s+/)[0]).filter(Boolean);
const create = (await import(pathToFileURL(path.resolve(wasmJs)).href)).default;
for (const s of names) {
  const args = ['--csv', 'p.csv', '--costs', 'EU', '--venues', 'v.txt', '--strategy', s, '--min-value', '0'];
  execFileSync(path.resolve(native), [...args, '--json', 'n.json', '--orders', 'n-o.json'], { cwd: dir, stdio: 'ignore' });
  const m = await create({ print: () => {}, printErr: () => {} });
  m.FS.writeFile('/p.csv', fs.readFileSync(path.join(dir, 'p.csv')));
  m.FS.writeFile('/v.txt', fs.readFileSync(path.join(dir, 'v.txt')));
  try { m.callMain(args.map((a) => (a.endsWith('.csv') || a.endsWith('.txt') ? `/${a}` : a)).concat(['--json', '/w.json', '--orders', '/w-o.json'])); } catch (e) { if (!(e && e.name === 'ExitStatus')) throw e; }
  const nb = JSON.parse(fs.readFileSync(path.join(dir, 'n.json'))).backtest, wb = JSON.parse(m.FS.readFile('/w.json', { encoding: 'utf8' })).backtest;
  const no = fs.readFileSync(path.join(dir, 'n-o.json'), 'utf8'), wo = new TextDecoder().decode(m.FS.readFile('/w-o.json'));
  if (JSON.stringify(nb) !== JSON.stringify(wb)) throw new Error(`${s}: the backtests differ\n${JSON.stringify(nb)}\n${JSON.stringify(wb)}`);
  if (no !== wo) throw new Error(`${s}: the orders differ\n${no}\n${wo}`);
  console.log(`${s}: the same backtest (${nb.trades} trades, Sharpe ${nb.sharpe}) and the same orders`);
}
console.log('wasm: the WebAssembly engine gives the native engine\'s results; OK');
