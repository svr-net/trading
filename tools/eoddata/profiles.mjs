// Each share's sector and industry from EODData's profiles (/Profile/List/<EXCHANGE>), for the
// model's industry groups, into the private store (they are EODData's data); and, with --probe,
// whether the key reaches the profiles and the fundamentals: counts and how many shares have each
// field, never the values.
//   EODDATA_API_KEY=... node tools/eoddata/profiles.mjs --exchanges LSE,NYSE,NASDAQ --out data/eoddata/profiles [--probe]
// With --fundamentals DIR (the daily job): also the day's fundamentals of the model's shares (the
// tickers in --universe, a CSV with a ticker column) as DIR/<EXCHANGE>/<date>.csv.gz, so that a
// history builds up (EODData serves only today's), for the model to learn from once long enough.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const key = process.env.EODDATA_API_KEY;
if (!key) { console.error('Set EODDATA_API_KEY in the environment.'); process.exit(2); }
const BASE = process.env.EODDATA_BASE || 'https://api.eoddata.com';
const field = (o, ...names) => { for (const n of names) for (const k of Object.keys(o || {})) if (k.toLowerCase() === n.toLowerCase()) return o[k]; return undefined; };
const list = async (route) => {
  const url = new URL(BASE + route);
  url.searchParams.set('ApiKey', key);
  const r = await fetch(url);
  if (!r.ok) return { status: r.status, items: [] };  // the URL holds the key: never print it
  const body = await r.json();
  return { status: r.status, items: Array.isArray(body) ? body : field(body, 'data', 'items') || [] };
};
const filled = (items, name) => items.filter((x) => { const v = field(x, name); return v !== undefined && v !== null && v !== '' && v !== 0; }).length;

const out = opt('out');
const fundDir = opt('fundamentals');
const universe = opt('universe') ? new Set(fs.readFileSync(opt('universe'), 'utf8').split('\n').slice(1).map((l) => l.split(',')[1]).filter(Boolean)) : null;
const FUND = ['marketCapitalization', 'ebitda', 'peg', 'bookValue', 'dividendPerShare', 'dividendYield', 'earningsPerShare', 'revenuePerShare', 'priceToSales',
  'priceToBook', 'sharesOutstanding', 'grossMargin', 'profitMargin', 'operatingMargin', 'returnOnAssets', 'returnOnEquity', 'revenue', 'grossProfit', 'trailingPE',
  'forwardPE', 'totalCash', 'totalCashPerShare', 'totalDebt', 'bookValuePerShare'];
const stamp = new Date().toISOString().slice(0, 10);
for (const ex of String(opt('exchanges', 'LSE')).split(',').filter(Boolean)) {
  const p = await list(`/Profile/List/${ex}`);
  const sectors = new Set(p.items.map((x) => field(x, 'sector')).filter(Boolean));
  const industries = new Set(p.items.map((x) => field(x, 'industry')).filter(Boolean));
  console.log(`${ex} profiles: HTTP ${p.status}, ${p.items.length} symbols, sector on ${filled(p.items, 'sector')}, industry on ${filled(p.items, 'industry')}; ${sectors.size} sectors, ${industries.size} industries`);
  if (out && p.items.length) {
    fs.mkdirSync(out, { recursive: true });
    const rows = p.items.map((x) => ({ code: field(x, 'symbolCode', 'code'), sector: field(x, 'sector') || '', industry: field(x, 'industry') || '' })).filter((r) => r.code);
    fs.writeFileSync(path.join(out, `${ex}.json`), JSON.stringify(rows));
  }
  if (fundDir) {
    const f = await list(`/Fundamental/List/${ex}`);
    const rows = f.items.filter((x) => !universe || universe.has(field(x, 'symbolCode', 'code')));
    const csv = ['code,' + FUND.join(',')].concat(rows.map((x) => [field(x, 'symbolCode', 'code'), ...FUND.map((n) => { const v = field(x, n); return Number.isFinite(v) ? v : ''; })].join(',')));
    fs.mkdirSync(path.join(fundDir, ex), { recursive: true });
    fs.writeFileSync(path.join(fundDir, ex, `${stamp}.csv.gz`), zlib.gzipSync(csv.join('\n') + '\n', { level: 9 }));
    console.log(`${ex} fundamentals kept: HTTP ${f.status}, ${rows.length} of the model's shares`);
  }
  if (argv.includes('--probe')) {
    const f = await list(`/Fundamental/List/${ex}`);
    const names = ['marketCapitalization', 'priceToBook', 'trailingPE', 'returnOnEquity', 'totalDebtToEquity', 'beta', 'sharesOutstanding'];
    console.log(`${ex} fundamentals: HTTP ${f.status}, ${f.items.length} symbols; ` + names.map((n) => `${n} on ${filled(f.items, n)}`).join(', '));
  }
}
