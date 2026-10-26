// Lists an exchange's symbol codes and names that match given prefixes (no prices), to find the
// codes of series such as continuous futures:
//   EODDATA_API_KEY=... node tools/eoddata/symbols.mjs --find CME:ES,NQ,RTY;CBOT:ZN,YM;CFE:VX
const argv = process.argv.slice(2);
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : d; };
const key = process.env.EODDATA_API_KEY;
if (!key) { console.error('Set EODDATA_API_KEY in the environment.'); process.exit(2); }
const BASE = process.env.EODDATA_BASE || 'https://api.eoddata.com';
const field = (o, ...names) => { for (const n of names) for (const k of Object.keys(o || {})) if (k.toLowerCase() === n.toLowerCase()) return o[k]; return undefined; };
for (const part of String(opt('find', '')).split(';').filter(Boolean)) {
  const [ex, pre] = part.split(':');
  const prefixes = (pre || '').split(',').filter(Boolean);
  const url = new URL(`${BASE}/Symbol/List/${ex}`);
  url.searchParams.set('ApiKey', key);
  const r = await fetch(url);
  if (!r.ok) { console.log(`${ex}: HTTP ${r.status}`); continue; }
  const body = await r.json();
  const list = Array.isArray(body) ? body : field(body, 'data', 'items', 'symbols') || [];
  console.log(`${ex}: ${list.length} symbols`);
  for (const s of list) {
    const code = String(field(s, 'code', 'symbol') ?? ''), name = String(field(s, 'name', 'description') ?? '');
    if (prefixes.some((p) => code.startsWith(p))) console.log(`  ${code}\t${name}`);
  }
}
