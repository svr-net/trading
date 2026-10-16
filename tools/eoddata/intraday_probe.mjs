// Which intraday intervals the EODData key can fetch, and over how many recent days: asks for one
// liquid LSE share at each interval and prints the number of bars and the dates they cover.
//   EODDATA_API_KEY=... node tools/eoddata/intraday_probe.mjs [--symbol VOD] [--days 30]
// Counts and dates only: no prices are printed or written (EODData's licence).
const key = process.env.EODDATA_API_KEY;
if (!key) {
  console.error('Set EODDATA_API_KEY in the environment (never pass it on the command line).');
  process.exit(2);
}
const argv = process.argv.slice(2);
const opt = (n, d) => (argv.includes(`--${n}`) ? argv[argv.indexOf(`--${n}`) + 1] : d);
const symbol = opt('symbol', 'VOD'), days = +opt('days', '30');
const day = (d) => d.toISOString().slice(0, 10);
const to = new Date(), from = new Date(Date.now() - days * 864e5);
const field = (o, ...names) => {
  for (const n of names) for (const k of Object.keys(o)) if (k.toLowerCase() === n.toLowerCase()) return o[k];
  return undefined;
};
const asList = (b) => (Array.isArray(b) ? b : field(b, 'data', 'items', 'quotes') || []);
for (const interval of ['1', '5', '10', '15', '30', '60', 'h', '1h', 'm1', 'm5', 'm15', 'h1']) {
  const url = new URL(`https://api.eoddata.com/Quote/List/LSE/${encodeURIComponent(symbol)}`);
  for (const [k, v] of Object.entries({ Interval: interval, FromDateStamp: day(from), ToDateStamp: day(to), ApiKey: key })) url.searchParams.set(k, v);
  try {
    const r = await fetch(url);
    if (!r.ok) { console.log(`interval ${interval.padEnd(4)}: HTTP ${r.status}`); continue; }
    const list = asList(await r.json());
    const stamps = list.map((q) => String(field(q, 'dateStamp', 'date') ?? '')).filter(Boolean).sort();
    const perDay = new Map();
    for (const s of stamps) perDay.set(s.slice(0, 10), (perDay.get(s.slice(0, 10)) || 0) + 1);
    console.log(`interval ${interval.padEnd(4)}: ${list.length} bars, ${perDay.size} days, ${stamps[0] ?? '-'} .. ${stamps.at(-1) ?? '-'}, up to ${Math.max(0, ...perDay.values())} bars a day; fields [${Object.keys(list[0] || {})}]`);
  } catch (e) {
    console.log(`interval ${interval.padEnd(4)}: ${e.message}`);  // the URL holds the key: never print it
  }
}
