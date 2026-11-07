// The 5-minute bars of the backtest's filled trades (for apps/ofm_fills; research, never committed or
// published: EODData's licence). Reads the trade log (X_TRADELOG), fetches the month of 5-minute bars
// for each share-month with a fill (a share-day: <EX>:<CODE>, or a bare code on --exchange) and writes
// date,ticker,stamp,open,high,low,close for the filled days only. Prints counts only.
//
//   EODDATA_API_KEY=... node tools/eoddata/fills_bars.mjs --log data/tl.csv --exchange LSE --out data/bars5.csv
import fs from 'node:fs';
import { field } from './store.mjs';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const log = opt('log', 'data/tl.csv'), exchange = opt('exchange', 'LSE'), out = opt('out', 'data/bars5.csv');
const key = process.env.EODDATA_API_KEY;
if (!key) { console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).'); process.exit(2); }
const BASE = process.env.EODDATA_BASE || 'https://api.eoddata.com';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const lastDay = (mo) => { const t = new Date(`${mo}-01T12:00:00Z`); t.setUTCMonth(t.getUTCMonth() + 1); t.setUTCDate(0); return t.toISOString().slice(0, 10); };

const want = new Map();  // "<ticker>|<YYYY-MM>" -> the filled days
for (const line of fs.readFileSync(log, 'utf8').split('\n')) {
  const f = line.split(',');
  if (f.length < 18 || !(Number(f[17]) > 0)) continue;
  const k = `${f[1]}|${f[0].slice(0, 7)}`;
  (want.get(k) || want.set(k, new Set()).get(k)).add(f[0]);
}
async function month(ex, code, mo) {
  const url = new URL(`${BASE}/Quote/List/${ex}/${encodeURIComponent(code)}`);
  for (const [k, v] of Object.entries({ Interval: '5', FromDateStamp: `${mo}-01`, ToDateStamp: lastDay(mo), ApiKey: key })) url.searchParams.set(k, v);
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) { const b = await r.json(); return Array.isArray(b) ? b : field(b, 'data', 'items', 'quotes') || []; }
    if (r.status === 404) return [];
    if ((r.status === 429 || r.status >= 500) && attempt < 5) { await sleep(3000 * 2 ** attempt); continue; }
    return null;  // the URL holds the key: never print it
  }
}
const w = fs.createWriteStream(out);
w.write('date,ticker,stamp,open,high,low,close\n');
let months = 0, empty = 0, failed = 0, days = 0, rows = 0, stamp = null;
for (const [k, need] of want) {
  const [ticker, mo] = k.split('|');
  const [ex, code] = ticker.includes(':') ? ticker.split(':') : [exchange, ticker];
  const list = await month(ex, code, mo);
  ++months;
  if (list === null) { ++failed; continue; }
  if (!list.length) { ++empty; continue; }
  const seen = new Set();
  for (const q of list) {
    const s = String(field(q, 'dateStamp', 'date') ?? '');
    if (!need.has(s.slice(0, 10))) continue;
    const c = +field(q, 'close');
    if (!(c > 0)) continue;
    w.write(`${s.slice(0, 10)},${ticker},${s},${+field(q, 'open') || ''},${+field(q, 'high')},${+field(q, 'low')},${c}\n`);
    seen.add(s.slice(0, 10)), ++rows;
    stamp ||= s;
  }
  days += seen.size;
}
w.end(() => console.log(`fills bars: ${want.size} share-months with fills, ${months} asked, ${empty} empty, ${failed} failed; ${days} filled days with bars, ${rows} bars; a stamp: ${stamp ? stamp.replace(/^\d{4}-\d{2}-\d{2}/, 'YYYY-MM-DD') : '-'}`));
