// Intraday paths of the strongest candidates: 5-minute bars of the regular session (09:30-16:00,
// exchange time) for each (date, ticker) the model lists (ofm_report --candidates-out), kept in a
// store a month per symbol (data/eoddata/intraday/, kept between runs by actions/cache, never
// committed or published: EODData's licence). A month is fetched once (the current month again
// only for the days it is missing), so each run makes requests only for new candidates' months.
// Writes date,ticker,time,high,low,close for every stored day of the listed symbols' months.
//
//   EODDATA_API_KEY=... node tools/eoddata/intraday.mjs --candidates data/candidates.csv \
//       --exchanges NYSE,NASDAQ --out data/intraday.csv [--cached]
import fs from 'node:fs';
import path from 'node:path';
import { field, day } from './store.mjs';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const cached = argv.includes('--cached');
const exchanges = opt('exchanges', 'NYSE,NASDAQ').split(',');
const out = opt('out', 'data/intraday.csv');
const key = process.env.EODDATA_API_KEY;
if (!key && !cached) { console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).'); process.exit(2); }
const BASE = process.env.EODDATA_BASE || 'https://api.eoddata.com';
const ROOT = path.join('data', 'eoddata', 'intraday');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Which exchange each ticker is on (the share lists written by fetch.mjs).
const exchangeOf = new Map();
for (const ex of exchanges) {
  const f = `data/${ex}.symbols.json`;
  if (fs.existsSync(f)) for (const s of JSON.parse(fs.readFileSync(f, 'utf8')).symbols) exchangeOf.set(s.code, ex);
}
// The months needed: (ticker, YYYY-MM) of every candidate.
const lines = fs.readFileSync(opt('candidates', 'data/candidates.csv'), 'utf8').trim().split('\n').slice(1);
const months = new Map();  // ticker|month -> days wanted
for (const l of lines) {
  const [d, t] = l.split(',');
  if (!exchangeOf.has(t)) continue;
  const k = `${t}|${d.slice(0, 7)}`;
  (months.get(k) || months.set(k, new Set()).get(k)).add(d);
}
const fileOf = (ex, t, mo) => path.join(ROOT, ex, t.replace(/[^A-Za-z0-9.-]/g, '_'), `${mo}.json`);
const lastDay = (mo) => { const t = new Date(`${mo}-01T12:00:00Z`); t.setUTCMonth(t.getUTCMonth() + 1); t.setUTCDate(0); return t.toISOString().slice(0, 10); };
const today = new Date().toISOString().slice(0, 10);

async function fetchMonth(ex, t, mo) {
  const url = new URL(`${BASE}/Quote/List/${ex}/${encodeURIComponent(t)}`);
  for (const [k, v] of Object.entries({ Interval: '5', FromDateStamp: `${mo}-01`, ToDateStamp: lastDay(mo), ApiKey: key })) url.searchParams.set(k, v);
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) return await r.json();
    if (r.status === 404) return [];
    if ((r.status === 429 || r.status >= 500) && attempt < 4) { await sleep(2000 * 2 ** attempt); continue; }
    throw new Error(`${ex}:${t} ${mo}: HTTP ${r.status}`);  // the URL holds the key: never print it
  }
}

const counts = { stored: 0, fetched: 0, failed: 0 };
const jobs = [...months.entries()];
async function worker() {
  while (jobs.length) {
    const [k, want] = jobs.shift();
    const [t, mo] = k.split('|'), ex = exchangeOf.get(t), f = fileOf(ex, t, mo);
    const have = fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : null;
    const missing = [...want].some((d) => !have?.days[d]);
    if (have && (have.complete || !missing)) { ++counts.stored; continue; }
    if (cached) continue;
    try {
      const list = await fetchMonth(ex, t, mo);
      const rows = Array.isArray(list) ? list : field(list, 'data', 'items', 'quotes') || [];
      const days = {};
      for (const q of rows) {
        const s = String(field(q, 'dateStamp', 'date') ?? ''), tm = s.slice(11, 16);
        if (tm < '09:30' || tm >= '16:00') continue;  // the regular session
        const h = +field(q, 'high'), l = +field(q, 'low'), c = +field(q, 'close');
        if (!(h > 0 && l > 0 && c > 0)) continue;
        (days[day(s)] ||= []).push([tm, h, l, c]);
      }
      fs.mkdirSync(path.dirname(f), { recursive: true });
      fs.writeFileSync(f, JSON.stringify({ complete: lastDay(mo) < today, days }));
      ++counts.fetched;
    } catch (e) { ++counts.failed; console.warn(`  ${e.message}`); }
  }
}
await Promise.all(Array.from({ length: 8 }, worker));

// The paths of every stored day of the listed months.
const outRows = ['date,ticker,time,high,low,close'];
let stockDays = 0;
for (const k of months.keys()) {
  const [t, mo] = k.split('|'), f = fileOf(exchangeOf.get(t), t, mo);
  if (!fs.existsSync(f)) continue;
  for (const [d, bars] of Object.entries(JSON.parse(fs.readFileSync(f, 'utf8')).days)) {
    ++stockDays;
    for (const [tm, h, l, c] of bars) outRows.push(`${d},${t},${tm},${h},${l},${c}`);
  }
}
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, outRows.join('\n') + '\n');
console.log(`intraday: ${lines.length} candidate stock-days, ${months.size} symbol-months (${counts.stored} stored, ${counts.fetched} fetched, ${counts.failed} failed); wrote ${out}: ${stockDays} stock-days, ${outRows.length - 1} bars`);
