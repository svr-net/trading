// Survivorship probe: does EODData give an exchange's whole quote list for a past day (with the shares
// delisted since), and the history of a share no longer listed? Counts only: no prices, no tickers.
//
//   EODDATA_API_KEY=... node tools/check/survivors.mjs --exchange LSE [--top 500] [--years 1,3,6,9]
//
// For each past day (the first weekday on or after the date that many years ago) it asks for the
// exchange's quote list with a date parameter, under the names the API may use, and reports:
//   - whether the list is that day's (the quotes' own dates) or today's (the parameter ignored);
//   - its shares, how many of them are not in today's symbol list (delisted or renamed since);
//   - of that day's --top most traded (close x volume), how many are not listed today;
//   - for up to 10 of those, whether their daily history can still be downloaded.
import { field } from '../eoddata/store.mjs';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const exchange = opt('exchange', 'LSE');
const top = Number(opt('top', 500));
const yearsBack = String(opt('years', '1,3,6,9')).split(',').map(Number);
const key = process.env.EODDATA_API_KEY;
if (!key) { console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).'); process.exit(2); }
const BASE = process.env.EODDATA_BASE || 'https://api.eoddata.com';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const asList = (b) => (Array.isArray(b) ? b : field(b, 'data', 'items', 'quotes', 'symbols') || []);

async function get(route, params = {}) {
  const url = new URL(BASE + route);
  for (const [k, v] of Object.entries({ ...params, ApiKey: key })) url.searchParams.set(k, v);
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) return r.json();
    if (r.status === 404) return null;
    if ((r.status === 429 || r.status >= 500) && attempt < 4) { await sleep(2000 * 2 ** attempt); continue; }
    return { error: r.status };  // the URL holds the key: never print it
  }
}
const codeOf = (q) => field(q, 'symbolCode', 'code', 'symbol');
const dateOf = (q) => String(field(q, 'dateStamp', 'date', 'quoteDate', 'timestamp') ?? '').slice(0, 10);
const weekday = (d) => { while ([0, 6].includes(d.getUTCDay())) d.setUTCDate(d.getUTCDate() + 1); return d.toISOString().slice(0, 10); };

const listed = new Set(asList(await get(`/Symbol/List/${exchange}`)).map(codeOf));
console.log(`survivors ${exchange}: listed today ${listed.size}`);
for (const y of yearsBack) {
  const d = new Date(); d.setUTCFullYear(d.getUTCFullYear() - y);
  const day = weekday(d);
  let found = null;
  for (const name of ['DateStamp', 'Date', 'QuoteDate', 'FromDateStamp']) {
    const params = name === 'FromDateStamp' ? { FromDateStamp: day, ToDateStamp: day } : { [name]: day };
    const body = await get(`/Quote/List/${exchange}`, params);
    const list = body && !body.error ? asList(body) : [];
    const dates = new Set(list.map(dateOf));
    const onDay = dates.size === 1 && dates.has(day);
    console.log(`  ${day} (${y}y) param ${name}: ${body?.error ? `HTTP ${body.error}` : `${list.length} quotes, dates ${[...dates].slice(0, 3).join(' ')}${dates.size > 3 ? ' ...' : ''}`}${onDay ? ' <- that day' : ''}`);
    if (onDay && list.length) { found = list; break; }
  }
  if (!found) { console.log(`  ${day}: no past-day list`); continue; }
  const gone = found.filter((q) => !listed.has(codeOf(q)));
  const ranked = found.map((q) => ({ code: codeOf(q), v: Number(field(q, 'close')) * Number(field(q, 'volume')) }))
    .filter((x) => Number.isFinite(x.v) && x.v > 0).sort((a, b) => b.v - a.v).slice(0, top);
  const topGone = ranked.filter((x) => !listed.has(x.code));
  let histOk = 0, tried = 0;
  for (const x of topGone.slice(0, 10)) {
    tried++;
    const h = await get(`/Quote/List/${exchange}/${encodeURIComponent(x.code)}`, { Interval: 'd', FromDateStamp: day, ToDateStamp: weekday(new Date(Date.parse(day) + 30 * 864e5)) });
    if (h && !h.error && asList(h).length) histOk++;
  }
  console.log(`  ${day}: shares ${found.length}, not listed today ${gone.length} (${(100 * gone.length / found.length).toFixed(1)}%); top ${ranked.length} by traded value: ${topGone.length} not listed today (${(100 * topGone.length / Math.max(1, ranked.length)).toFixed(1)}%); their history downloadable: ${histOk} of ${tried}`);
}
