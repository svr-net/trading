// Downloads daily bars of named series (futures, indices, currencies) from the EODData API and
// writes them in the long CSV format, with the ticker written as EXCHANGE:CODE:
//   date,ticker,open,high,low,close,volume
//
//   EODDATA_API_KEY=... node tools/eoddata/fetch_series.mjs --symbols LIFFE:XY00,CFE:VIY00 [--years 20] [--out data/futures.csv]
//
// The key is read from the environment only; it is never written to disk or printed. Responses
// are cached under data/eoddata/ (git-ignored: EODData's licence does not allow redistributing
// its data). Prints each series' first and last date and bar count (no prices).
import fs from 'node:fs';
import path from 'node:path';

const argv = process.argv.slice(2);
const opt = (name, fallback) => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : fallback; };
const symbols = String(opt('symbols', '')).split(',').filter(Boolean);
const years = Number(opt('years', 20));
const out = opt('out', 'data/futures.csv');
const key = process.env.EODDATA_API_KEY;
if (!key) {
  console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).');
  process.exit(2);
}
const cacheDir = path.join('data', 'eoddata', 'series');
fs.mkdirSync(cacheDir, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function get(route, params, cacheName) {
  const file = path.join(cacheDir, cacheName);
  if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  const url = new URL('https://api.eoddata.com' + route);
  for (const [k, v] of Object.entries({ ...params, ApiKey: key })) url.searchParams.set(k, v);
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) {
      const body = await r.json();
      fs.writeFileSync(file, JSON.stringify(body));
      return body;
    }
    if ((r.status === 429 || r.status >= 500) && attempt < 6) { await sleep(5000 * 2 ** attempt); continue; }
    throw new Error(`${route}: HTTP ${r.status}`);  // the URL holds the key: never print it
  }
}
const field = (o, ...names) => {
  for (const n of names) for (const k of Object.keys(o)) if (k.toLowerCase() === n.toLowerCase()) return o[k];
  return undefined;
};
const asList = (b) => (Array.isArray(b) ? b : field(b, 'data', 'items', 'quotes') || []);
const day = (v) => String(v).slice(0, 10);

const today = new Date();
const stamp = today.toISOString().slice(0, 10);
const rows = ['date,ticker,open,high,low,close,volume'];
for (const s of symbols) {
  const [exchange, code] = s.split(':');
  // Ask year by year, so a plan limit on how far back one request may reach does not empty the
  // whole series; years the plan does not serve simply come back empty.
  const bars = new Map();
  for (let y = 0; y < years; y++) {
    const to = new Date(today.getTime() - y * 365.25 * 864e5);
    const from = new Date(to.getTime() - 365.25 * 864e5);
    const f = from.toISOString().slice(0, 10), t = to.toISOString().slice(0, 10);
    let body = [];
    try {
      body = await get(`/Quote/List/${encodeURIComponent(exchange)}/${encodeURIComponent(code)}`, { Interval: 'd', FromDateStamp: f, ToDateStamp: t },
        `${exchange}-${code.replace(/[^A-Za-z0-9.-]/g, '_')}-${f}-${y === 0 ? stamp : t}.json`);
    } catch (e) {
      console.warn(`  ${s} ${f}..${t}: ${e.message}`);
    }
    const list = asList(body);
    if (!list.length && y > 0) break;  // nothing further back
    for (const b of list) {
      const d = day(field(b, 'dateStamp', 'date'));
      const c = +field(b, 'close');
      if (d && c > 0) bars.set(d, { o: +field(b, 'open') || c, h: +field(b, 'high') || c, l: +field(b, 'low') || c, c, v: +field(b, 'volume') || 0 });
    }
  }
  const dates = [...bars.keys()].sort();
  console.log(`${s.padEnd(14)} ${String(dates.length).padStart(6)} bars  ${dates[0] ?? '-'} .. ${dates.at(-1) ?? '-'}`);
  for (const d of dates) {
    const b = bars.get(d);
    rows.push(`${d},${s},${b.o},${b.h},${b.l},${b.c},${b.v}`);
  }
}
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, rows.join('\n') + '\n');
console.log(`wrote ${out}: ${rows.length - 1} bars`);
