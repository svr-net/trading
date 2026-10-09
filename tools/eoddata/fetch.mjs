// Downloads daily bars of the most liquid stocks of an exchange from the EODData API
// (https://api.eoddata.com) and writes them as the library's long CSV format:
//   date,ticker,open,high,low,close,volume
//
//   EODDATA_API_KEY=... node tools/eoddata/fetch.mjs [--exchange LSE] [--universe 120] [--years 6] [--out data/LSE.csv]
//
// The key is read from the environment only; it is never written to disk or printed.
// Responses are cached under data/eoddata/ (git-ignored: EODData's licence does not allow
// redistributing its data), so a re-run only fetches what is missing. Behind an HTTP proxy,
// run Node with NODE_USE_ENV_PROXY=1 (Node >= 22.21).
//
// Universe: the latest quote list of the exchange, ordinary shares only where the API says so,
// ranked by traded value (close x volume); the top --universe symbols are downloaded, and the
// ones with fewer than --min-days of history are dropped.
import fs from 'node:fs';
import path from 'node:path';

const argv = process.argv.slice(2);
const opt = (name, fallback) => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : fallback; };
const exchange = opt('exchange', 'LSE');
const universe = Number(opt('universe', 120));
const years = Number(opt('years', 6));
const minDays = Number(opt('min-days', 750));
const out = opt('out', `data/${exchange}.csv`);
const cacheDir = path.join('data', 'eoddata', exchange);
const key = process.env.EODDATA_API_KEY;
if (!key) {
  console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).');
  process.exit(2);
}
fs.mkdirSync(cacheDir, { recursive: true });

const BASE = 'https://api.eoddata.com';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function get(route, params = {}, cacheName) {
  const file = cacheName && path.join(cacheDir, cacheName);
  if (file && fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  const url = new URL(BASE + route);
  for (const [k, v] of Object.entries({ ...params, ApiKey: key })) if (v !== undefined) url.searchParams.set(k, v);
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) {
      const body = await r.json();
      if (file) fs.writeFileSync(file, JSON.stringify(body));
      return body;
    }
    const text = (await r.text()).slice(0, 300);
    if ((r.status === 429 || r.status >= 500) && attempt < 4) { await sleep(2000 * 2 ** attempt); continue; }
    throw new Error(`${route}: HTTP ${r.status} ${text}`);  // the URL holds the key: never print it
  }
}

// Field names are matched case-insensitively and by alias, so small API changes don't break the parse.
const field = (o, ...names) => {
  for (const n of names) for (const k of Object.keys(o)) if (k.toLowerCase() === n.toLowerCase()) return o[k];
  return undefined;
};
const asList = (body) => (Array.isArray(body) ? body : field(body, 'data', 'items', 'quotes', 'symbols') || []);
const day = (v) => String(v).slice(0, 10);

const today = new Date();
const from = new Date(today.getTime() - years * 365.25 * 864e5).toISOString().slice(0, 10);
const stamp = today.toISOString().slice(0, 10);

console.log(`exchange ${exchange}: listing symbols and the latest quotes`);
const exchangeInfo = await get(`/Exchange/Get/${exchange}`, {}, 'exchange.json').catch(() => ({}));
const symbols = asList(await get(`/Symbol/List/${exchange}`, {}, `symbols-${stamp}.json`));
const typeOf = new Map(symbols.map((s) => [field(s, 'code', 'symbolCode', 'symbol'), String(field(s, 'type', 'symbolType', 'typeCode') ?? '')]));
const nameOf = new Map(symbols.map((s) => [field(s, 'code', 'symbolCode', 'symbol'), String(field(s, 'name', 'description') ?? '')]));
const quotes = asList(await get(`/Quote/List/${exchange}`, {}, `quotes-${stamp}.json`));
// The shapes (field names only) help diagnose a parse that finds nothing.
console.log(`  fields: exchange [${Object.keys(exchangeInfo)}]; symbol [${Object.keys(symbols[0] || {})}]; quote [${Object.keys(quotes[0] || {})}]`);
console.log(`  symbol types: ${[...new Set([...typeOf.values()])].slice(0, 20).join(', ')}`);
const isShare = (code) => {
  const t = (typeOf.get(code) || '').toLowerCase();
  if (t && !/(share|stock|equity|ord|common)/.test(t)) return false;  // funds, ETFs, warrants, bonds
  const n = (nameOf.get(code) || '').toLowerCase();
  return !/\b(etf|etc|fund|trust plc|investment trust|ucits|tracker|warrant|bond|gilt|pref)\b/.test(n);
};
const ranked = quotes
  .map((q) => ({ code: field(q, 'symbolCode', 'code', 'symbol'), value: Number(field(q, 'close')) * Number(field(q, 'volume')) }))
  .filter((q) => q.code && Number.isFinite(q.value) && q.value > 0 && isShare(q.code))
  .sort((a, b) => b.value - a.value)
  .slice(0, universe);
console.log(`  ${symbols.length} symbols, ${quotes.length} quotes; downloading ${ranked.length} most traded shares since ${from}`);

const rows = ['date,ticker,open,high,low,close,volume'];
const kept = [];
for (const [i, { code }] of ranked.entries()) {
  const body = await get(`/Quote/List/${exchange}/${encodeURIComponent(code)}`, { Interval: 'd', FromDateStamp: from, ToDateStamp: stamp },
    `history-${code.replace(/[^A-Za-z0-9.-]/g, '_')}-${from}-${stamp}.json`).catch((e) => { console.warn(`  skip ${code}: ${e.message}`); return []; });
  const bars = asList(body)
    .map((b) => ({ d: day(field(b, 'dateStamp', 'date')), o: +field(b, 'open'), h: +field(b, 'high'), l: +field(b, 'low'), c: +field(b, 'close'), v: +field(b, 'volume') }))
    .filter((b) => b.d && b.c > 0 && b.o > 0 && b.h > 0 && b.l > 0)
    .sort((a, b) => (a.d < b.d ? -1 : 1));
  if (i === 0) console.log(`  history fields [${Object.keys(asList(body)[0] || {})}], ${bars.length} bars`);
  if (bars.length < minDays) { console.log(`  ${code}: ${bars.length} days, dropped`); continue; }
  for (const b of bars) rows.push(`${b.d},${code},${b.o},${b.h},${b.l},${b.c},${Number.isFinite(b.v) ? b.v : 0}`);
  kept.push({ code, name: nameOf.get(code) || '', days: bars.length });
  if ((i + 1) % 20 === 0) console.log(`  ${i + 1}/${ranked.length}`);
}
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, rows.join('\n') + '\n');
fs.writeFileSync(out.replace(/\.csv$/, '.symbols.json'), JSON.stringify({ exchange, currency: field(exchangeInfo, 'currency', 'currencyCode') ?? null, from, to: stamp, symbols: kept }, null, 1));
console.log(`wrote ${out}: ${kept.length} stocks, ${rows.length - 1} bars`);
