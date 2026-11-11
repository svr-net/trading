// The stored 5-minute bars (<store root>/intraday/<EXCHANGE>/<TICKER>/<YYYY-MM>.json, kept in the
// private svr-net/eod, never committed or published) of the given exchanges, as one CSV for research:
// date,ticker,time,high,low,close,volume. Reads only what the store holds; fetches nothing.
// With --paths MARKET, the bridge's recordings of the day's candidates (paths/<MARKET>/<date>.csv:
// time,ticker,last,bid,ask,type every 30 seconds, submit.py --paths) are added as 5-minute bars of their
// last prices (else bids), for the share-days the store has no bars of: a delayed row (type 3 or 4)
// shifted 15 minutes earlier, and the delayed rows of the first 15 minutes dropped (yesterday's prices).
// The prices are IBKR's (an LSE share in pence or pounds): apps/ofm_uk rescales a day to its daily bar.
//
//   node tools/eoddata/intraday_export.mjs --exchanges NYSE,NASDAQ --out data/intraday.csv
//   node tools/eoddata/intraday_export.mjs --exchanges LSE --paths UK --out data/intraday-uk.csv
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './store.mjs';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const out = fs.createWriteStream(opt('out', 'data/intraday.csv'));
out.write('date,ticker,time,high,low,close,volume\n');
let days = 0, rows = 0, tickers = 0;
const seen = new Set();  // the store's share-days (date,ticker)
for (const ex of opt('exchanges', 'NYSE,NASDAQ').split(',')) {
  const dir = path.join(ROOT, 'intraday', ex);
  if (!fs.existsSync(dir)) continue;
  for (const t of fs.readdirSync(dir)) {
    ++tickers;
    for (const f of fs.readdirSync(path.join(dir, t)).filter((x) => x.endsWith('.json'))) {
      const m = JSON.parse(fs.readFileSync(path.join(dir, t, f), 'utf8'));
      for (const [d, bars] of Object.entries(m.days || {})) {
        ++days;
        seen.add(`${d},${t}`);
        for (const [tm, h, l, c, v] of bars) { out.write(`${d},${t},${tm},${h},${l},${c},${v ?? ''}\n`); ++rows; }
      }
    }
  }
}
const OPENS = { UK: '08:00', EU: '09:00', US: '09:30', APAC: '10:00' };
const market = opt('paths', '');
let pathDays = 0;
if (market) {
  const dir = path.join(ROOT, 'paths', market);
  const open = OPENS[market] || '08:00';
  const minutes = (hm) => +hm.slice(0, 2) * 60 + +hm.slice(3, 5);
  for (const f of fs.existsSync(dir) ? fs.readdirSync(dir).filter((x) => /^\d{4}-\d{2}-\d{2}\.csv$/.test(x)).sort() : []) {
    const day = f.slice(0, 10), bars = new Map();  // ticker -> bucket -> [high, low, close]
    const lines = fs.readFileSync(path.join(dir, f), 'utf8').split('\n');
    const head = lines[0].split(','), col = (n) => head.indexOf(n);
    const [cT, cK, cL, cB, cY] = ['time', 'ticker', 'last', 'bid', 'type'].map(col);
    if (cT < 0 || cK < 0) continue;
    for (const l of lines.slice(1)) {
      const c = l.split(',');
      if (c.length <= cK || !c[cT]) continue;
      const delayed = ['3', '4'].includes(c[cY]);
      let m = minutes(c[cT].slice(11, 16)) - (delayed ? 15 : 0);
      if (m < minutes(open)) continue;  // before the open (a delayed row of the first 15 minutes: yesterday's prices)
      const px = +c[cL] > 0 ? +c[cL] : +c[cB];
      if (!(px > 0)) continue;
      const b = m - (m % 5), t = c[cK];
      if (seen.has(`${day},${t}`)) continue;  // the store's own bars of the day come first
      if (!bars.has(t)) bars.set(t, new Map());
      const k = bars.get(t), x = k.get(b);
      if (!x) k.set(b, [px, px, px]); else { x[0] = Math.max(x[0], px); x[1] = Math.min(x[1], px); x[2] = px; }
    }
    for (const [t, k] of bars) {
      ++pathDays;
      for (const [b, [h, l, c]] of [...k].sort((x, y) => x[0] - y[0]))
        out.write(`${day},${t},${String(Math.floor(b / 60)).padStart(2, '0')}:${String(b % 60).padStart(2, '0')},${h},${l},${c},\n`), ++rows;
    }
  }
}
out.end(() => console.log(`intraday export: ${tickers} symbols, ${days} share-days, ${rows} bars` + (market ? `; ${pathDays} share-days from the bridge's recordings (${market})` : '')));
