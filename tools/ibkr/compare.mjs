// The backtest against the broker, trade by trade (results only: no prices, keys or account numbers;
// this repository's logs are public). For each market and day the bridge recorded both the model's
// own result for the day's orders (the orders file's check: filled or not, entry, exit, net return)
// and IBKR's fills (the report), it compares them: how many orders filled in each, which agree, and
// the gross return (exit over entry) of the trades both filled, plus the model's net.
//   BRIDGE_URL=https://name:8765 BRIDGE_KEY=... node tools/ibkr/compare.mjs
const base = (process.env.BRIDGE_URL || '').replace(/\/+$/, '');
const key = process.env.BRIDGE_KEY || '';
if (!base || !key) { console.log('BRIDGE_URL and BRIDGE_KEY needed'); process.exit(1); }
const r = await fetch(`${base}/history`, { headers: { Origin: 'https://svr-net.github.io', 'X-Bridge-Key': key }, signal: AbortSignal.timeout(60000) });
const days = (await r.json()).days || [];
const sym = (t) => String(t || '').split(':').pop().replace('.', ' ').toUpperCase();
const pct = (x) => (x == null || !Number.isFinite(x) ? '-' : `${x >= 0 ? '+' : ''}${(100 * x).toFixed(2)}%`);
const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);
const tot = {};
for (const d of days.slice().reverse()) {
  if (!d.model || !d.trades) continue;
  const live = new Map(d.trades.map((t) => [sym(t.ticker), t]));
  const T = (tot[d.market] ||= { days: 0, orders: 0, mFill: 0, lFill: 0, both: 0, mOnly: 0, lOnly: 0, mGross: [], lGross: [], mNet: [], lAll: [] });
  let both = 0, mOnly = 0, lOnly = 0;
  const mg = [], lg = [];
  for (const m of d.model) {
    const l = live.get(sym(m.ticker));
    const lf = !!(l && l.bought > 0), mf = m.filled > 0;
    T.orders += 1, T.mFill += mf, T.lFill += lf;
    if (mf && lf) {
      both += 1;
      const g = m.entry > 0 && m.exitPrice > 0 ? m.exitPrice / m.entry - 1 : null;
      if (g != null) mg.push(g), T.mGross.push(g);
      if (Number.isFinite(l.ret)) lg.push(l.ret), T.lGross.push(l.ret);
      T.mNet.push(m.ret);
    } else if (mf) mOnly += 1;
    else if (lf) lOnly += 1;
    if (lf && Number.isFinite(l.ret)) T.lAll.push(l.ret);
  }
  T.days += 1, T.both += both, T.mOnly += mOnly, T.lOnly += lOnly;
  console.log(`${d.date} ${d.market}: ${d.model.length} orders; filled in both ${both}, only in the backtest ${mOnly}, only at IBKR ${lOnly}; `
    + `both-filled gross: backtest ${pct(mean(mg))}, IBKR ${pct(mean(lg))}`);
}
for (const [m, T] of Object.entries(tot)) {
  console.log(`\n${m}: ${T.days} days, ${T.orders} orders; filled: backtest ${T.mFill} (${pct(T.mFill / T.orders)}), IBKR ${T.lFill} (${pct(T.lFill / T.orders)}); `
    + `both ${T.both}, only backtest ${T.mOnly}, only IBKR ${T.lOnly}`);
  console.log(`${m}: trades both filled: gross backtest ${pct(mean(T.mGross))}, IBKR ${pct(mean(T.lGross))} (difference ${pct(mean(T.mGross) - mean(T.lGross))}); `
    + `backtest net ${pct(mean(T.mNet))}; every IBKR fill gross ${pct(mean(T.lAll))}`);
}
if (!Object.keys(tot).length) console.log('no day with both the model\'s check and IBKR\'s report yet');
