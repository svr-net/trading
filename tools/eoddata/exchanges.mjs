// Lists the exchanges the EODData key can see (code, name, country, currency, symbol count) and,
// for each one whose code or name looks like a futures market, a sample of its symbols:
//   EODDATA_API_KEY=... node tools/eoddata/exchanges.mjs [--out data/exchanges.json]
// Metadata only: no prices are fetched or written.
import fs from 'node:fs';
import path from 'node:path';

const key = process.env.EODDATA_API_KEY;
if (!key) {
  console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).');
  process.exit(2);
}
const argv = process.argv.slice(2);
const out = argv.includes('--out') ? argv[argv.indexOf('--out') + 1] : 'data/exchanges.json';

async function get(route) {
  const url = new URL('https://api.eoddata.com' + route);
  url.searchParams.set('ApiKey', key);
  for (let attempt = 0; ; attempt++) {
    const r = await fetch(url);
    if (r.ok) return r.json();
    if ((r.status === 429 || r.status >= 500) && attempt < 6) {  // rate limit: wait 5 s, 10 s, ... 160 s
      await new Promise((ok) => setTimeout(ok, 5000 * 2 ** attempt));
      continue;
    }
    throw new Error(`${route}: HTTP ${r.status}`);  // the URL holds the key: never print it
  }
}
const field = (o, ...names) => {
  for (const n of names) for (const k of Object.keys(o)) if (k.toLowerCase() === n.toLowerCase()) return o[k];
  return undefined;
};
const asList = (b) => (Array.isArray(b) ? b : field(b, 'data', 'items') || []);

const exchanges = asList(await get('/Exchange/List'));
console.log(`${exchanges.length} exchanges; fields [${Object.keys(exchanges[0] || {})}]`);
const rows = exchanges.map((e) => ({
  code: field(e, 'code', 'exchangeCode'), name: field(e, 'name', 'description'), country: field(e, 'country', 'countryCode'),
  currency: field(e, 'currency', 'currencyCode'), symbols: field(e, 'symbolCount', 'symbols'), type: field(e, 'type', 'exchangeType'),
}));
for (const r of rows) console.log(`  ${String(r.code).padEnd(10)} ${String(r.type ?? '').padEnd(10)} ${String(r.country ?? '').padEnd(4)} ${String(r.currency ?? '').padEnd(4)} ${String(r.symbols ?? '').padStart(6)}  ${r.name}`);
const futuresLike = rows.filter((r) => /fut|ice|liffe|cme|cbot|nymex|comex|eurex|cboe|lme|options|commod/i.test(`${r.code} ${r.name} ${r.type}`));
console.log(`\nfutures-like exchanges: ${futuresLike.map((r) => r.code).join(', ') || 'none'}`);
// Named symbols only (contract months carry no name), and a keyword search for the series a
// UK equity model can use: index futures, gilts, volatility, oil, sterling.
const wanted = /ftse|footsie|gilt|sterling|sonia|vix|volatility|brent|crude|dax|stoxx|bund|gbp|pound|msci/i;
const samples = {};
for (const code of [...futuresLike.map((r) => r.code), 'FOREX']) {
  try {
    const syms = asList(await get(`/Symbol/List/${encodeURIComponent(code)}`));
    const named = syms
      .map((s) => ({ code: field(s, 'code', 'symbolCode'), name: field(s, 'name', 'description') }))
      .filter((s) => s.name && s.name !== 'undefined');
    const hits = named.filter((s) => wanted.test(s.name) || (code === 'FOREX' && /GBP/.test(s.code)));
    samples[code] = hits;
    console.log(`\n${code}: ${syms.length} symbols, ${named.length} named, ${hits.length} relevant`);
    for (const s of hits.slice(0, 60)) console.log(`  ${s.code}: ${s.name}`);
  } catch (e) {
    console.log(`\n${code}: not available (${e.message})`);
  }
}
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify({ exchanges: rows, samples }, null, 1));
