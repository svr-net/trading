// How far back the US intraday bars go, and which clock window matches the daily bar (the regular
// session). Counts, dates and match rates only: no prices are printed or written (EODData's licence).
//   EODDATA_API_KEY=... node tools/eoddata/intraday_depth.mjs [--targets NASDAQ:AAPL,NYSE:JPM]
const key = process.env.EODDATA_API_KEY;
if (!key) { console.error('Set EODDATA_API_KEY in the environment.'); process.exit(2); }
const argv = process.argv.slice(2);
const opt = (n, d) => (argv.includes(`--${n}`) ? argv[argv.indexOf(`--${n}`) + 1] : d);
const targets = opt('targets', 'NASDAQ:AAPL,NYSE:JPM').split(',');
const day = (d) => d.toISOString().slice(0, 10);
const field = (o, ...names) => { for (const n of names) for (const k of Object.keys(o)) if (k.toLowerCase() === n.toLowerCase()) return o[k]; };
const asList = (b) => (Array.isArray(b) ? b : field(b, 'data', 'items', 'quotes') || []);
async function get(ex, sym, interval, from, to) {
  const url = new URL(`https://api.eoddata.com/Quote/List/${ex}/${encodeURIComponent(sym)}`);
  for (const [k, v] of Object.entries({ Interval: interval, FromDateStamp: from, ToDateStamp: to, ApiKey: key })) url.searchParams.set(k, v);
  const r = await fetch(url);
  if (!r.ok) return { status: r.status, list: [] };
  return { status: 200, list: asList(await r.json()) };
}
const ago = (d) => day(new Date(Date.now() - d * 864e5));
for (const t of targets) {
  const [ex, sym] = t.split(':');
  for (const interval of ['1', '5', 'h'])
    for (const back of [90, 180, 365, 730, 1500, 3000]) {
      const { status, list } = await get(ex, sym, interval, ago(back), ago(back - 10));
      const days = new Set(list.map((q) => String(field(q, 'dateStamp')).slice(0, 10)));
      console.log(`${t} interval ${interval} ${back} days ago: HTTP ${status}, ${list.length} bars over ${days.size} days`);
    }
  // Session window: which start/end minutes make the bars' high, low and open/close equal the daily bar's.
  const from = ago(30), to = ago(0);
  const m1 = (await get(ex, sym, '1', from, to)).list, d1 = (await get(ex, sym, 'd', from, to)).list;
  const daily = new Map(d1.map((q) => [String(field(q, 'dateStamp')).slice(0, 10), q]));
  const byDay = new Map();
  for (const q of m1) { const s = String(field(q, 'dateStamp')); (byDay.get(s.slice(0, 10)) || byDay.set(s.slice(0, 10), []).get(s.slice(0, 10))).push({ t: s.slice(11, 16), o: +q.open, h: +q.high, l: +q.low, c: +q.close, v: +q.volume }); }
  const near = (a, b) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));
  const hist = {};
  for (const [d, bars] of byDay) {
    const D = daily.get(d); if (!D) continue;
    bars.sort((a, b) => (a.t < b.t ? -1 : 1));
    const openAt = bars.find((b) => near(b.o, +D.open))?.t ?? 'none', closeAt = [...bars].reverse().find((b) => near(b.c, +D.close))?.t ?? 'none';
    for (const [lo, hi] of [['09:00', '17:00'], ['09:30', '16:00'], ['04:00', '20:00'], ['08:30', '15:00']]) {
      const w = bars.filter((b) => b.t >= lo && b.t < hi); if (!w.length) continue;
      const ok = near(Math.max(...w.map((b) => b.h)), +D.high) && near(Math.min(...w.map((b) => b.l)), +D.low);
      hist[`${lo}-${hi}`] = (hist[`${lo}-${hi}`] || 0) + (ok ? 1 : 0);
    }
    hist[`open at ${openAt}`] = (hist[`open at ${openAt}`] || 0) + 1;
    hist[`close at ${closeAt}`] = (hist[`close at ${closeAt}`] || 0) + 1;
  }
  console.log(`${t} session over ${byDay.size} days: ${JSON.stringify(hist)}`);
}
