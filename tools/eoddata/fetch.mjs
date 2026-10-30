// Downloads daily bars of the most liquid stocks of an exchange from the EODData API
// (https://api.eoddata.com) and writes them as the library's long CSV format:
//   date,ticker,open,high,low,close,volume
//
//   EODDATA_API_KEY=... node tools/eoddata/fetch.mjs [--exchange LSE] [--universe 120] [--years 6] [--out data/LSE.csv]
//
// The key is read from the environment only; it is never written to disk or printed.
// Bars are kept in a persistent store (store.mjs: data/eoddata/store/, kept between workflow runs
// by actions/cache, git-ignored: EODData's licence does not allow redistributing its data). A
// stock's history is downloaded once; after that a run adds only the days it is missing: the
// latest day from the exchange's quote list (already fetched for the ranking, so no request per
// stock), or one small request for a longer gap. --cached makes no request at all (the store and
// the newest cached quote list); --refresh downloads every history again. Behind an HTTP proxy,
// run Node with NODE_USE_ENV_PROXY=1 (Node >= 22.21).
//
// Universe: the latest quote list of the exchange, ordinary shares only where the API says so,
// ranked by traded value (close x volume); the top --universe symbols are downloaded, and the
// ones with fewer than --min-days of history are dropped.
import fs from 'node:fs';
import path from 'node:path';
import * as S from './store.mjs';

const argv = process.argv.slice(2);
const opt = (name, fallback) => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : fallback; };
const exchange = opt('exchange', 'LSE');
const universe = Number(opt('universe', 120));
const years = Number(opt('years', 6));
const minDays = Number(opt('min-days', 750));
const out = opt('out', `data/${exchange}.csv`);
const cacheDir = path.join(S.ROOT, exchange);
const cached = argv.includes('--cached');
const refresh = argv.includes('--refresh');
const key = process.env.EODDATA_API_KEY;
if (!key && !cached) {
  console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).');
  process.exit(2);
}
fs.mkdirSync(cacheDir, { recursive: true });

const BASE = process.env.EODDATA_BASE || 'https://api.eoddata.com';  // tests point this at a mock
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// Cached mode: the newest cached file with the same name up to its date stamps.
function newestCached(cacheName) {
  const stem = cacheName.replace(/(-\d{4}-\d{2}-\d{2})+\.json$/, '');
  const hits = fs.readdirSync(cacheDir).filter((f) => f.startsWith(stem + '-') && /(-\d{4}-\d{2}-\d{2})+\.json$/.test(f) &&
    f.slice(stem.length + 1).replace(/\.json$/, '').split('-').length % 3 === 0);
  if (!hits.length) return null;
  hits.sort();
  return path.join(cacheDir, hits[hits.length - 1]);
}

async function get(route, params = {}, cacheName) {
  const file = cacheName && path.join(cacheDir, cacheName);
  if (file && fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  if (cached) {
    const hit = cacheName && newestCached(cacheName);
    if (hit) return JSON.parse(fs.readFileSync(hit, 'utf8'));
    throw new Error(`${route}: not in the cache`);
  }
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

const { field, day } = S;
const asList = (body) => (Array.isArray(body) ? body : field(body, 'data', 'items', 'quotes', 'symbols') || []);

const today = new Date();
const from = new Date(today.getTime() - years * 365.25 * 864e5).toISOString().slice(0, 10);
const stamp = today.toISOString().slice(0, 10);

console.log(`exchange ${exchange}: listing symbols and the latest quotes`);
const exchangeInfo = await get(`/Exchange/Get/${exchange}`, {}, 'exchange.json').catch(() => ({}));
const symbols = asList(await get(`/Symbol/List/${exchange}`, {}, `symbols-${stamp}.json`));
const typeOf = new Map(symbols.map((s) => [field(s, 'code', 'symbolCode', 'symbol'), String(field(s, 'type', 'symbolType', 'typeCode') ?? '')]));
const nameOf = new Map(symbols.map((s) => [field(s, 'code', 'symbolCode', 'symbol'), String(field(s, 'name', 'description') ?? '')]));
const currencyOf = new Map(symbols.map((s) => [field(s, 'code', 'symbolCode', 'symbol'), String(field(s, 'currency', 'currencyCode') ?? '')]));
const currency = opt('currency', exchange === 'LSE' ? 'GBP' : '');  // LSE: sterling lines only (UK shares, quoted in pence; not the international order book)
const quotes = asList(await get(`/Quote/List/${exchange}`, {}, `quotes-${stamp}.json`));
// The shapes (field names only) help diagnose a parse that finds nothing.
console.log(`  fields: exchange [${Object.keys(exchangeInfo)}]; symbol [${Object.keys(symbols[0] || {})}]; quote [${Object.keys(quotes[0] || {})}]`);
console.log(`  symbol types: ${[...new Set([...typeOf.values()])].slice(0, 20).join(', ')}`);
const isShare = (code) => {
  if (!/^[A-Z]{1,5}(\.[A-Z])?$/.test(code)) return false;  // letters only: no warrants (YX58), alternative lines (RR-) or IOB codes (0O87)
  if (currency && currencyOf.get(code) && currencyOf.get(code).toUpperCase() !== currency.toUpperCase()) return false;
  const t = (typeOf.get(code) || '').toLowerCase();
  if (t && !/(share|stock|equity|ord|common)/.test(t)) return false;  // funds, ETFs, warrants, bonds
  const n = (nameOf.get(code) || '').toLowerCase();
  if (/\b(etf|etc|etn|fund|trust plc|investment trust|ucits|tracker|swap|s&p|msci|ftse|warrant|bond|gilt|pref)\b/.test(n)) return false;
  const company = /\b(plc|ltd|limited|s\.a\.|sa|ag|n\.v\.|nv|se|inc|corp|group|holdings|ord)\b/.test(n);  // a company's name, not a product's
  if (exchange === 'LSE') return company;
  // US: many companies' names carry no such word (JP Morgan Chase & Company, Alphabet Cl A); a fund's
  // does not either, but it names a fund family or a product.
  return company || !FUND_FAMILY.test(n);
};
const FUND_FAMILY = /\b(ishares|spdr|proshares|direxion|vanguard|invesco|wisdomtree|global x|vaneck|first trust|graniteshares|tidal|defiance|roundhill|yieldmax|grayscale|ark|amplify|pacer|schwab|fidelity|jpmorgan(?! chase)|franklin|janus|dimensional|avantis|t-rowe|xtrackers|simplify|rex|tuttle|leverage shares|microsectors|kraneshares|pgim|x-links|eaton vance|nuveen|blackrock|gabelli|templeton|voya|pimco|john hancock|legg mason|columbia|lazard|abrdn|aberdeen|cohen & steers|calamos|western asset|doubleline|guggenheim|mfs|putnam|neuberger|tortoise|kayne|clearbridge|flaherty|rivernorth|duff & phelps|babson|northern lights|virtus|allianz|liberty all-star|tri-continental|saba|trust|index|bitcoin|ether|ultra|2x|3x|bull|bear|daily|leveraged|inverse|portfolio|treasury|strategy shares|municipal|muni|tax-advantaged|tax-advantage|tax advantaged|tax-managed|buffer|laddered|definedterm|term|income fund|income fun|income f|income opportunity|income t|dividend opportunity|global dividend|dividend achievers|enhanced|managed duration|total return|stocksplus|precious metals|commodity strategy|source capital|income shares)\b/;
// US exchanges: ordinary shares only. Preferred shares, notes and other fixed-income lines are
// typed "Equity" too; they show in the name (a coupon, "Notes Due", "Pfd", "Depositary", "Series A",
// structured trusts) or, on NASDAQ, in a fifth ticker letter (L-P preferred and other classes, Q
// bankruptcy, R rights, T with warrants, U units, V when issued, W warrants, Z miscellaneous).
const NOT_ORDINARY = /(\d+(\.\d+)?\s*%|\b\d+\.\d{2,3}\b|\bnotes?\b|\bdue\b|\bpfd\b|\bpref(erred)?\b|\bdepositary\b|\bdebentures?\b|\bsenior\b|\bjunior\b|\bsubordinated?\b|\bfixed[- ]to[- ]floating\b|\bfloating rate\b|\bperpetual\b|\b(non-)?cumula|\bcum\b|\bseries [a-z]\b|\bunits?\b|\brights?\b|\bwarrants?\b|\bwt\b|\babs\b|\bstrats\b|\bsecurit)/i;
const isOrdinary = (code) => {
  if (exchange === 'LSE') return true;
  if (NOT_ORDINARY.test(nameOf.get(code) || '')) return false;
  return !(exchange === 'NASDAQ' && code.length === 5 && 'LMNOPQRTUVWZ'.includes(code[4]));
};
const ranked = quotes
  .map((q) => ({ code: field(q, 'symbolCode', 'code', 'symbol'), value: Number(field(q, 'close')) * Number(field(q, 'volume')) }))
  .filter((q) => q.code && Number.isFinite(q.value) && q.value > 0 && isShare(q.code))
  .sort((a, b) => b.value - a.value);
const currencies = {};
for (const c of currencyOf.values()) currencies[c] = (currencies[c] || 0) + 1;
console.log(`  currencies: ${JSON.stringify(currencies)}`);
console.log(`  ${symbols.length} symbols, ${quotes.length} quotes, ${ranked.length} candidate shares; keeping the ${universe} most traded with ${minDays}+ days since ${from}`);

// Splits: prices before a split are divided by its ratio, so returns are not distorted.
const splits = new Map();
try {
  const list = asList(await get(`/Splits/List/${exchange}`, {}, `splits-${stamp}.json`));
  console.log(`  splits: ${list.length}, fields [${Object.keys(list[0] || {})}], first ${JSON.stringify(list[0] || {})}`);
  for (const sp of list) {
    const code = field(sp, 'symbolCode', 'code', 'symbol');
    const date = day(field(sp, 'dateStamp', 'date', 'exDate') ?? '');
    let ratio = field(sp, 'ratio', 'splitRatio');
    if (typeof ratio === 'string' && /[-:/]/.test(ratio)) { const [a, b] = ratio.split(/[-:/]/).map(Number); ratio = a / b; }  // "2-1": two new for one old
    else if (ratio === undefined) { const n = +field(sp, 'numerator', 'to', 'newShares'), dn = +field(sp, 'denominator', 'from', 'oldShares'); ratio = n / dn; }
    ratio = Number(ratio);
    if (code && date && Number.isFinite(ratio) && ratio > 0 && ratio !== 1) (splits.get(code) || splits.set(code, []).get(code)).push({ date, ratio });
  }
} catch (e) { console.warn(`  splits unavailable: ${e.message}`); }

// Each stock's bars: from the store, completed with the days it is missing.
const quoteBar = new Map(quotes.map((q) => [field(q, 'symbolCode', 'code', 'symbol'), S.bar(q)]));
const latest = [...quoteBar.values()].filter(Boolean).map((b) => b[0]).sort().at(-1) ?? stamp;
const counts = { full: 0, gap: 0, quote: 0, current: 0 };
async function barsOf(code) {
  const bars = S.load(exchange, code);
  if (cached) {
    if (!S.has(exchange, code)) throw new Error(`${code}: not in the store`);  // the caller downloads
    return bars;
  }
  const L = S.lastDate(bars), qb = quoteBar.get(code);
  if (refresh || !L || L < from) {
    S.merge(bars, asList(await get(`/Quote/List/${exchange}/${encodeURIComponent(code)}`, { Interval: 'd', FromDateStamp: from, ToDateStamp: stamp })));
    ++counts.full;
  } else if (qb && qb[0] >= L && S.previousWeekday(qb[0]) <= L) {
    bars.set(qb[0], qb[1]);  // the latest day, from the quote list (overwrites a same-day bar)
    ++counts.quote;
  } else if (L < S.previousWeekday(latest) || (qb && qb[0] > L)) {
    S.merge(bars, asList(await get(`/Quote/List/${exchange}/${encodeURIComponent(code)}`, { Interval: 'd', FromDateStamp: S.addDays(L, 1), ToDateStamp: stamp })));
    ++counts.gap;
  } else {
    ++counts.current;
  }
  S.save(exchange, code, bars);
  return bars;
}

const rows = ['date,ticker,open,high,low,close,volume'];
const US = ['NYSE', 'NASDAQ', 'AMEX'];
let inferredSplits = 0, movedExchange = 0;
const kept = [];
let notOrdinary = 0;  // in the pool (the most traded) but not ordinary shares: counted, not kept
for (const [i, { code }] of ranked.entries()) {
  if (kept.length + notOrdinary >= universe) break;
  const stored = await barsOf(code).catch((e) => {
    if (cached) { console.error(`cached: ${e.message}`); process.exit(3); }
    console.warn(`  skip ${code}: ${e.message}`);
    return new Map();
  });
  const bars = [...stored.entries()]
    .filter(([d]) => d >= from)
    .map(([d, b]) => ({ d, o: b.o, h: b.h, l: b.l, c: b.c, v: b.v }))
    .filter((b) => b.c > 0 && b.o > 0 && b.h > 0 && b.l > 0)
    .sort((a, b) => (a.d < b.d ? -1 : 1));
  if (i === 0) console.log(`  ${code}: ${bars.length} bars`);
  // A share that moved between US exchanges (Walmart: NYSE to NASDAQ): its earlier days from the
  // other exchange, stored or (not --cached) downloaded once.
  if (bars.length < minDays && US.includes(exchange)) {
    for (const other of US.filter((x) => x !== exchange)) {
      let older = S.load(other, code);
      if (!older.size && !cached && !S.has(other, code)) {
        older = new Map();
        S.merge(older, asList(await get(`/Quote/List/${other}/${encodeURIComponent(code)}`, { Interval: 'd', FromDateStamp: from, ToDateStamp: stamp }).catch(() => [])));
        S.save(other, code, older);
      }
      const first = bars.length ? bars[0].d : '9999';
      const add = [...older.entries()].filter(([d]) => d >= from && d < first).map(([d, b]) => ({ d, o: b.o, h: b.h, l: b.l, c: b.c, v: b.v }))
        .filter((b) => b.c > 0 && b.o > 0 && b.h > 0 && b.l > 0);
      if (add.length) { bars.unshift(...add.sort((p, q) => (p.d < q.d ? -1 : 1))); ++movedExchange; }
    }
  }
  if (bars.length < minDays) { console.log(`  ${code}: ${bars.length} days, dropped`); continue; }
  for (const { date, ratio } of splits.get(code) || [])
    for (const b of bars) if (b.d < date) { b.o /= ratio; b.h /= ratio; b.l /= ratio; b.c /= ratio; b.v *= ratio; }
  // Splits the list lacks, read from the prices: a day whose close-to-close and close-to-open ratios
  // are both a split's (2, 3, 4, 5, 8, 10, ... for one, the inverse for a reverse split; within 3% and
  // 10%, or 10% and 15% from 5 up, where no day's move could be mistaken for one) and whose volume moves the other way (more shares after a split, fewer after a reverse
  // one: the 20 days' median after against before, at least half the split in logs).
  const med = (xs) => { const z = xs.filter((x) => x > 0).sort((p, q) => p - q); return z.length ? z[z.length >> 1] : 0; };
  for (let k = 1; k < bars.length; k++) {
    const r = bars[k - 1].c / bars[k].c, ro = bars[k - 1].c / bars[k].o;
    for (const n of [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100])
      for (const q of [n, 1 / n]) {
        if (!(Math.abs(Math.log(r / q)) < (n >= 5 ? 0.1 : 0.03) && Math.abs(Math.log(ro / q)) < (n >= 5 ? 0.15 : 0.1))) continue;
        const vb = med(bars.slice(Math.max(0, k - 20), k).map((b) => b.v)), va = med(bars.slice(k, k + 20).map((b) => b.v));
        if (!(vb > 0 && va > 0 && Math.log(va / vb) * Math.sign(Math.log(q)) > 0.5 * Math.abs(Math.log(q)))) continue;
        for (let j = 0; j < k; ++j) { bars[j].o /= q; bars[j].h /= q; bars[j].l /= q; bars[j].c /= q; bars[j].v *= q; }
        ++inferredSplits;
        if (argv.includes('--verbose')) console.log(`  ${code}: split ${q >= 1 ? q : '1/' + Math.round(1 / q)} read on ${bars[k].d}`);
      }
  }
  const jump = bars.findIndex((b, k) => k > 0 && Math.abs(Math.log(b.c / bars[k - 1].c)) > Math.log(1.6));
  if (jump > 0) { console.log(`  ${code}: unexplained move on ${bars[jump].d} (${bars[jump - 1].c} -> ${bars[jump].c}), dropped`); continue; }
  if (!isOrdinary(code)) { ++notOrdinary; continue; }
  for (const b of bars) rows.push(`${b.d},${code},${b.o},${b.h},${b.l},${b.c},${Number.isFinite(b.v) ? b.v : 0}`);
  kept.push({ code, name: nameOf.get(code) || '', days: bars.length });
  if (kept.length % 20 === 0) console.log(`  kept ${kept.length}`);
}
console.log(`  store: ${counts.full} full histories downloaded, ${counts.gap} gaps filled, ${counts.quote} days from the quote list, ${counts.current} up to date`);
// Dated lists keep only their newest copy.
if (!cached) {
  for (const p of ['symbols-', 'quotes-', 'splits-']) S.prune(cacheDir, p, 1);
}
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, rows.join('\n') + '\n');
fs.writeFileSync(out.replace(/\.csv$/, '.symbols.json'), JSON.stringify({ exchange, currency: field(exchangeInfo, 'currency', 'currencyCode') ?? null, from, to: stamp, symbols: kept }, null, 1));
console.log(`wrote ${out}: ${kept.length} stocks (${notOrdinary} of the most traded left out: not ordinary shares; ${inferredSplits} splits read from the prices; ${movedExchange} histories joined across exchanges), ${rows.length - 1} bars`);
