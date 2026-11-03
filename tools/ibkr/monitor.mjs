// The bridge's state, read from outside (GitHub's runners) so the bridge can be watched without the
// page: GET /status (no key: the gateway, the schedule and the latest automatic actions) and, with
// BRIDGE_KEY in the environment, GET /diag (the checks) and the day's order count. Prints no prices,
// no keys and no account numbers: this repository's logs are public.
//   BRIDGE_URL=https://name:8765 [BRIDGE_KEY=...] node tools/ibkr/monitor.mjs
const base = (process.env.BRIDGE_URL || '').replace(/\/+$/, '');
if (!base) { console.log('BRIDGE_URL not set'); process.exit(1); }
const key = process.env.BRIDGE_KEY || '';
const get = async (path, keyed) => {
  const r = await fetch(base + path, { headers: { Origin: 'https://svr-net.github.io', ...(keyed ? { 'X-Bridge-Key': key } : {}) }, signal: AbortSignal.timeout(60000) });
  return { code: r.status, body: await r.json().catch(() => ({})) };
};
let failed = false;
try {
  const s = await get('/status', false);
  const b = s.body;
  console.log(`bridge: HTTP ${s.code}; IB Gateway API ${b.gateway ? 'up' : 'DOWN'} (port ${b.port}); key set: ${b.keyed}`);
  for (const [m, a] of Object.entries(b.auto || {}))
  {
    console.log(`schedule ${m}: place ${a.place}, auction ${a.auction}, close ${a.close}, report ${a.report} (${a.zone})`);
    if (a.today) console.log(`  today ${m}: ${Object.entries(a.today).map(([k, v]) => `${k} ${v ? 'done' : '-'}`).join(', ')}; running: ${(a.running || []).join(', ') || 'nothing'}`);
  }
  const g = b.gatewayApi;
  if (g) {
    console.log(`IB Gateway API watchdog: ${g.ok === null ? 'not checked yet' : g.ok ? 'answers' : 'DOES NOT ANSWER'} (checked ${g.checked}); restarts ${g.restarts}${g.restarted ? `, last ${g.restarted}: ${g.note}` : ''}`);
    if (g.ok === false) failed = true;
  }
  console.log('latest automatic actions:');
  for (const x of b.last || []) {
    console.log(`  ${x.when} ${x.market} ${x.action}: ${x.ok ? 'ok' : 'FAILED'} ${String(x.summary || '').replace(/\d+\.\d{2,}/g, '#')}`);
    if (!x.ok) failed = true;
  }
  if (!b.gateway) failed = true;
} catch (e) {
  console.log(`bridge not reachable: ${e.message}`);
  process.exit(1);
}
if (key) {
  try {
    const d = await get('/diag', true);
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
} else {
  console.log('(BRIDGE_KEY secret not set: diagnostics skipped)');
}
process.exit(failed ? 2 : 0);
