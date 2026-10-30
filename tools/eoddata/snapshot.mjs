// The 5-minute bars of futures and index series over their whole session (not only New York's
// 09:30-16:00), a month per symbol in the store (<store root>/intraday/<EXCHANGE>/<CODE>/<YYYY-MM>.json,
// in the private svr-net/eod: EODData's licence), for snapshots at another market's open (such as
// the Euro Stoxx 50 future at New York's open). A month is fetched once (the current month again
// for the days it is missing). Each bar: [time stamp as EODData gives it, open, high, low, close,
// volume]. Prints each series' months, days and the first stamps (no prices).
//
//   EODDATA_API_KEY=... node tools/eoddata/snapshot.mjs --symbols EUREX:FX.C [--from 2019-01]
import fs from 'node:fs';
import path from 'node:path';
import { field, ROOT as STORE_ROOT } from './store.mjs';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const symbols = String(opt('symbols', 'EUREX:FX.C')).split(',').filter(Boolean);
const from = opt('from', '2019-01');
const key = process.env.EODDATA_API_KEY;
if (!key) { console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).'); process.exit(2); }
const BASE = process.env.EODDATA_BASE || 'https://api.eoddata.com';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const lastDay = (mo) => { const t = new Date(`${mo}-01T12:00:00Z`); t.setUTCMonth(t.getUTCMonth() + 1); t.setUTCDate(0); return t.toISOString().slice(0, 10); };
const today = new Date().toISOString().slice(0, 10);
const months = [];
for (let t = new Date(`${from}-01T12:00:00Z`); t.toISOString().slice(0, 7) <= today.slice(0, 7); t.setUTCMonth(t.getUTCMonth() + 1)) months.push(t.toISOString().slice(0, 7));

async function fetchMonth(ex, code, mo) {
  const url = new URL(`${BASE}/Quote/List/${ex}/${encodeURIComponent(code)}`);
  for (const [k, v] of Object.entries({ Interval: '5', FromDateStamp: `${mo}-01`, ToDateStamp: lastDay(mo), ApiKey: key })) url.searchParams.set(k, v);
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) return r.json();
    if (r.status === 404) return [];
    if ((r.status === 429 || r.status >= 500) && attempt < 5) { await sleep(3000 * 2 ** attempt); continue; }
    throw new Error(`${ex}:${code} ${mo}: HTTP ${r.status}`);  // the URL holds the key: never print it
  }
}

for (const sym of symbols) {
  const [ex, code] = sym.split(':');
  const dir = path.join(STORE_ROOT, 'intraday', ex, code.replace(/[^A-Za-z0-9.-]/g, '_'));
  fs.mkdirSync(dir, { recursive: true });
  let fetched = 0, kept = 0, empty = 0, first = null;
  for (const mo of months) {
    const f = path.join(dir, `${mo}.json`);
    if (fs.existsSync(f) && JSON.parse(fs.readFileSync(f, 'utf8')).complete) { ++kept; continue; }
    const body = await fetchMonth(ex, code, mo).catch((e) => { console.warn(`  ${e.message}`); return null; });
    if (body === null) continue;
    const rows = Array.isArray(body) ? body : field(body, 'data', 'items', 'quotes') || [];
    const days = {};
    for (const q of rows) {
      const s = String(field(q, 'dateStamp', 'date') ?? '');
      const o = +field(q, 'open'), h = +field(q, 'high'), l = +field(q, 'low'), c = +field(q, 'close'), v = +field(q, 'volume') || 0;
      if (!(c > 0)) continue;
      (days[s.slice(0, 10)] ||= []).push([s, o, h, l, c, v]);
      first ||= s;
    }
    if (!rows.length) ++empty;
    fs.writeFileSync(f, JSON.stringify({ complete: lastDay(mo) < today, days }));
    ++fetched;
  }
  const have = fs.readdirSync(dir).filter((x) => x.endsWith('.json')).sort();
  let nd = 0;
  for (const x of have) nd += Object.keys(JSON.parse(fs.readFileSync(path.join(dir, x), 'utf8')).days).length;
  console.log(`${sym}: ${have.length} months stored (${fetched} fetched, ${kept} already complete, ${empty} empty), ${nd} days, first ${have[0] || '-'}, a stamp: ${first || '-'}`);
}
