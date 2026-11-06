// The bridge's state, read from outside (GitHub's runners) so the bridge can be watched without the
// page: GET /status (no key: the gateway, the schedule and the latest automatic actions) and, with
// BRIDGE_KEY in the environment, GET /diag (the checks) and the day's order count. Prints no prices,
// no keys and no account numbers: this repository's logs are public.
//   BRIDGE_URL=https://name:8765 [BRIDGE_KEY=...] node tools/ibkr/monitor.mjs
const base = (process.env.BRIDGE_URL || '').replace(/\/+$/, '');
if (!base) { console.log('BRIDGE_URL not set'); process.exit(1); }
const key = process.env.BRIDGE_KEY || '';
const get = async (path, keyed, ms = 60000) => {
  const r = await fetch(base + path, { headers: { Origin: 'https://svr-net.github.io', ...(keyed ? { 'X-Bridge-Key': key } : {}) }, signal: AbortSignal.timeout(ms) });
  return { code: r.status, body: await r.json().catch(() => ({})) };
};
import { execFileSync } from 'node:child_process';
const AUTO_REPORT = (process.env.TRADES_MARKETS || 'UK,EU').split(',').map((x) => x.trim()).filter(Boolean);

let failed = false;
const problems = [];
// A failed action counts as new for 35 minutes (a little more than the runs' spacing), so one failure
// turns one run red (one email from GitHub), not every run for the rest of the day.
const OFFSET = { UTC: 0, GMT: 0, BST: 1, CET: 1, CEST: 2, EST: -5, EDT: -4, AEST: 10, AEDT: 11, SGT: 8 };
const age = (when) => {
  const m = /^(\d{4}-\d\d-\d\d) (\d\d):(\d\d) (\w+)$/.exec(when || '');
  if (!m || !(m[4] in OFFSET)) return 0;
  const t = Date.parse(`${m[1]}T${m[2]}:${m[3]}:00Z`) - OFFSET[m[4]] * 3600e3;
  return (Date.now() - t) / 60e3;
};
try {
  const s = await get('/status', false);
  const b = s.body;
  console.log(`bridge: HTTP ${s.code}; IB Gateway API ${b.gateway ? 'up' : 'DOWN'} (port ${b.port}); key set: ${b.keyed}`);
  // Is the NAS running the latest bridge? LATEST: the last commit that changed it (the workflow's).
  const v = String(b.version || 'unknown'), latest = process.env.LATEST || '';
  let current = null;
  if (/^[0-9a-f]{7,40}$/.test(v) && /^[0-9a-f]{7,40}$/.test(latest)) {
    try { execFileSync('git', ['merge-base', '--is-ancestor', latest, v], { stdio: 'ignore' }); current = true; } catch { current = false; }
  }
  console.log(`bridge version ${v}${b.started ? `, running since ${b.started}` : ''}: ${current === true ? 'the latest' : current === false ? `OUT OF DATE (the latest is ${latest}): update the NAS with sudo sh /volume1/DEV/trading/docker/ibkr/deploy-nas.sh` : 'not known'}`);
  if (current === false) problems.push('the NAS runs an old bridge');
  for (const [m, a] of Object.entries(b.auto || {}))
  {
    console.log(`schedule ${m}: place ${a.place}, auction ${a.auction}, close ${a.close}, report ${a.report} (${a.zone})`);
    if (a.today) console.log(`  today ${m}: ${Object.entries(a.today).map(([k, v]) => `${k} ${v ? 'done' : '-'}`).join(', ')}; running: ${(a.running || []).join(', ') || 'nothing'}`);
  }
  const g = b.gatewayApi;
  if (g) {
    console.log(`IB Gateway API watchdog: ${g.ok === null ? 'not checked yet' : g.ok ? 'answers' : 'DOES NOT ANSWER'} (checked ${g.checked}); restarts ${g.restarts}${g.restarted ? `, last ${g.restarted}: ${g.note}` : ''}`);
    if (g.ok === false) problems.push('IB Gateway API does not answer');
  }
  console.log('latest automatic actions:');
  for (const x of b.last || []) {
    console.log(`  ${x.when} ${x.market} ${x.action}: ${x.ok ? 'ok' : 'FAILED'} ${String(x.summary || '').replace(/\d+\.\d{2,}/g, '#')}`);
    if (!x.ok && age(x.when) < 35) problems.push(`${x.market} ${x.action} failed at ${x.when}`);
  }
  if (!b.gateway) problems.push('IB Gateway port closed');
} catch (e) {
  console.log(`bridge not reachable: ${e.message}`);
  process.exit(1);
}
if (key) {
  try {
    const d = await get('/diag', true, 150000);
    console.log(`diagnostics (HTTP ${d.code}):`);
    for (const c of d.body.checks || d.body || []) {
      if (!c || !c.check) continue;
      const detail = String(c.detail || '').replace(/\d+\.\d{2,}/g, '#').replace(/\b[DU]\d{5,}\b/g, 'ACCOUNT');
      console.log(`  ${c.ok === true ? 'ok  ' : c.ok === false ? 'FAIL' : 'warn'} ${c.check}: ${detail}`);
    }
    for (const m of ['UK', 'EU']) {
      const o = await get(`/orders?market=${m}`, true);
      console.log(`orders ${m}: ${(o.body.orders || []).length} in today's file (as of ${o.body.asOf || '?'})`);
    }
  } catch (e) {
    console.log(`diagnostics not read: ${e.message}`);
  }
  try {
      // Today's trades from IBKR (GET /trades), as returns only: each share's state and its move from
      // the purchase (to the sale, or to the last price while held); no prices, no amounts.
      const pct = (a, b) => (a > 0 && b > 0 ? 100 * (a / b - 1) : null);
      const f = (x) => (x == null ? '   ?  ' : `${x >= 0 ? '+' : ''}${x.toFixed(2)}%`);
      for (const m of AUTO_REPORT) {
        const t = await get(`/trades?market=${m}`, true, 120000);
        if (t.code !== 200) { console.log(`trades ${m}: not read (HTTP ${t.code}: ${t.body.error || ''})`); continue; }
        const rows = t.body.shares || [];
        const bought = rows.filter((r) => r.bought > 0 || r.held > 0);
        const moves = bought.map((r) => (r.sold >= r.bought && r.bought > 0 ? pct(r.sell, r.buy) : pct(r.price, r.buy))).filter((x) => x != null);
        const mean = moves.length ? moves.reduce((a, b) => a + b, 0) / moves.length : null;
        console.log(`trades ${m} (${t.body.at || ''}): ${rows.length} shares, ${bought.length} bought or held, ${moves.filter((x) => x > 0).length} up, ${moves.filter((x) => x < 0).length} down, mean move ${f(mean)} before costs`);
        for (const r of bought) {
          const done = r.sold >= r.bought && r.bought > 0;
          console.log(`  ${String(r.ticker).padEnd(8)} ${f(done ? pct(r.sell, r.buy) : pct(r.price, r.buy))}  ${r.state}${r.soldBy ? ` (${r.soldBy})` : ''}`);
        }
      }
  } catch (e) {
    console.log(`trades not read: ${e.message}`);
  }
} else {
  console.log('(BRIDGE_KEY secret not set: diagnostics skipped)');
}
if (problems.length) console.log(`\nNEEDS ATTENTION: ${problems.join('; ')}`);
process.exit(failed || problems.length ? 2 : 0);
