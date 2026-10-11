// Downloads daily bars of named series (futures, indices, currencies) from the EODData API and
// writes them in the long CSV format, with the ticker written as EXCHANGE:CODE:
//   date,ticker,open,high,low,close,volume
//
//   EODDATA_API_KEY=... node tools/eoddata/fetch_series.mjs --symbols LIFFE:XY00,CFE:VIY00 [--years 20] [--out data/futures.csv]
//
// The key is read from the environment only; it is never written to disk or printed. Bars are
// kept in the persistent store (store.mjs, git-ignored: EODData's licence does not allow
// redistributing its data): a series' history is downloaded once, year by year (so a plan limit
// on how far back one request may reach does not empty it); after that a run asks only for the
// days since its last bar. --cached makes no request; --refresh downloads every history again.
// Prints each series' first and last date and bar count (no prices).
import fs from 'node:fs';
import path from 'node:path';
import * as S from './store.mjs';

const argv = process.argv.slice(2);
const opt = (name, fallback) => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : fallback; };
const symbols = String(opt('symbols', '')).split(',').filter(Boolean);
const years = Number(opt('years', 20));
const out = opt('out', 'data/futures.csv');
const cached = argv.includes('--cached');
const refresh = argv.includes('--refresh');
const key = process.env.EODDATA_API_KEY;
if (!key && !cached) {
  console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).');
  process.exit(2);
}
const oldDir = path.join('data', 'eoddata', 'series');  // old-style cache files, migrated into the store

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function get(route, params) {
  const url = new URL((process.env.EODDATA_BASE || 'https://api.eoddata.com') + route);  // tests point this at a mock
  for (const [k, v] of Object.entries({ ...params, ApiKey: key })) url.searchParams.set(k, v);
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) return r.json();
    if ((r.status === 429 || r.status >= 500) && attempt < 6) { await sleep(5000 * 2 ** attempt); continue; }
    throw new Error(`${route}: HTTP ${r.status}`);  // the URL holds the key: never print it
  }
}
const asList = (b) => (Array.isArray(b) ? b : S.field(b, 'data', 'items', 'quotes') || []);

const stamp = new Date().toISOString().slice(0, 10);
const rows = ['date,ticker,open,high,low,close,volume'];
let requests = 0;
for (const s of symbols) {
  const [exchange, code] = s.split(':');
  const route = `/Quote/List/${encodeURIComponent(exchange)}/${encodeURIComponent(code)}`;
  const bars = refresh ? new Map() : S.load('series', s);
  if (!bars.size && fs.existsSync(oldDir)) {
    const stem = `${exchange}-${code.replace(/[^A-Za-z0-9.-]/g, '_')}-`;
    for (const f of fs.readdirSync(oldDir).filter((x) => x.startsWith(stem)).sort())
      S.merge(bars, asList(JSON.parse(fs.readFileSync(path.join(oldDir, f), 'utf8'))));
  }
  const L = S.lastDate(bars);
  try {
    if (!cached && !L) {
      for (let y = 0; y < years; y++) {
        const to = new Date(Date.now() - y * 365.25 * 864e5), from = new Date(to.getTime() - 365.25 * 864e5);
        const list = asList(await get(route, { Interval: 'd', FromDateStamp: from.toISOString().slice(0, 10), ToDateStamp: to.toISOString().slice(0, 10) }));
        ++requests;
        if (!list.length && y > 0) break;  // nothing further back
        S.merge(bars, list);
      }
    } else if (!cached && L < S.previousWeekday(stamp)) {
      S.merge(bars, asList(await get(route, { Interval: 'd', FromDateStamp: L, ToDateStamp: stamp })));
      ++requests;
    }
  } catch (e) {
    console.warn(`  ${s}: ${e.message}`);
  }
  if (bars.size) S.save('series', s, bars);
  const dates = [...bars.keys()].sort();
  console.log(`${s.padEnd(14)} ${String(dates.length).padStart(6)} bars  ${dates[0] ?? '-'} .. ${dates.at(-1) ?? '-'}`);
  for (const d of dates) {
    const b = bars.get(d);
    rows.push(`${d},${s},${b.o},${b.h},${b.l},${b.c},${b.v}`);
  }
}
if (!cached && fs.existsSync(oldDir)) fs.rmSync(oldDir, { recursive: true });  // now in the store
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, rows.join('\n') + '\n');
console.log(`wrote ${out}: ${rows.length - 1} bars; ${requests} requests`);
