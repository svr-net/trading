// The stored 5-minute bars (<store root>/intraday/<EXCHANGE>/<TICKER>/<YYYY-MM>.json, kept in the
// private svr-net/eod, never committed or published) of the given exchanges, as one CSV for research:
// date,ticker,time,high,low,close,volume. Reads only what the store holds; fetches nothing.
//
//   node tools/eoddata/intraday_export.mjs --exchanges NYSE,NASDAQ --out data/intraday.csv
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './store.mjs';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const out = fs.createWriteStream(opt('out', 'data/intraday.csv'));
out.write('date,ticker,time,high,low,close,volume\n');
let days = 0, rows = 0, tickers = 0;
for (const ex of opt('exchanges', 'NYSE,NASDAQ').split(',')) {
  const dir = path.join(ROOT, 'intraday', ex);
  if (!fs.existsSync(dir)) continue;
  for (const t of fs.readdirSync(dir)) {
    ++tickers;
    for (const f of fs.readdirSync(path.join(dir, t)).filter((x) => x.endsWith('.json'))) {
      const m = JSON.parse(fs.readFileSync(path.join(dir, t, f), 'utf8'));
      for (const [d, bars] of Object.entries(m.days || {})) {
        ++days;
        for (const [tm, h, l, c, v] of bars) { out.write(`${d},${t},${tm},${h},${l},${c},${v ?? ''}\n`); ++rows; }
      }
    }
  }
}
out.end(() => console.log(`intraday export: ${tickers} symbols, ${days} share-days, ${rows} bars`));
