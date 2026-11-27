// The Lab (lab.html): the strategy engine compiled to WebAssembly (lab/worker.js) run in this browser on a market's
// daily bars from the bridge (GET /bars.csv, keyed: the prices never leave this device), laid out as the trading desk.
const $ = (id) => document.getElementById(id);
const MARKETS = ['UK', 'EU', 'US', 'APAC', 'INDEX'];
const NAMES = { UK: 'London (AIM)', EU: 'Amsterdam · Xetra', US: 'New York', APAC: 'Sydney · Singapore', INDEX: 'Xetra ETFs' };
const STRATEGIES = [
  ['rsi2', 'RSI(2) pullback', 'RSI(2) under 10 in an uptrend'],
  ['ibs', 'Internal bar strength', 'closes near the day’s low'],
  ['revday', 'Daily residual reversal', 'the day’s fall against the market'],
  ['volrev', 'Volume reversal', 'heavy-volume 3-day falls'],
  ['hvp', 'High-volume return premium', 'days of extreme volume'],
  ['tom', 'Turn of the month', 'high betas, around the month end'],
  ['breakout', '20-day breakout', 'new highs in an uptrend'],
];
const stored = (k, d = '') => { try { return localStorage.getItem(k) || d; } catch { return d; } };
const store = (k, v) => { try { localStorage.setItem(k, v); } catch { /* not kept */ } };
const bridgeUrl = () => stored('bridgeUrl', 'http://127.0.0.1:8765').replace(/\/+$/, '');
const bridgeKey = () => stored('bridgeKey');
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
const pct = (x, d = 1) => (x == null || Number.isNaN(+x) ? '–' : `${x >= 0 ? '+' : ''}${(100 * x).toFixed(d)}%`);
const tone = (x) => (x > 0 ? 'pos' : x < 0 ? 'neg' : '');

const fromUrl = (new URLSearchParams(location.search).get('market') || '').toUpperCase();
let market = MARKETS.includes(fromUrl) ? fromUrl : (MARKETS.includes(stored('deskMarket')) ? stored('deskMarket') : 'UK');
let index = null;  // data/strategies.json: each market's live strategy and its parameters
let worker = null;

// ---- the markets
function live(m) { return (index?.markets?.[m.toLowerCase()] || []).find((s) => s.role === 'live') || null; }
function renderMarkets() {
  const box = $('markets');
  box.replaceChildren(...MARKETS.map((m) => {
    const b = el('button', null, 'mkt'), l = live(m), s = STRATEGIES.find((x) => x[0] === l?.strategy);
    b.type = 'button';
    b.setAttribute('aria-pressed', String(m === market));
    const row = el('div', null, 'row');
    row.append(el('span', m, 'name'));
    b.append(row, el('div', NAMES[m], 'clock'), el('div', s ? `live: ${s[1]}` : 'live: –', 'day'));
    b.addEventListener('click', () => { market = m; store('deskMarket', m); history.replaceState(null, '', `?market=${m.toLowerCase()}`); renderMarkets(); renderChoices(); clearResult(); });
    return b;
  }));
}

// ---- the strategy and its options
function renderChoices() {
  const l = live(market), chosen = stored(`labStrategy-${market}`, l?.strategy || 'rsi2');
  $('st-note').textContent = l ? `${market}'s live strategy is marked` : '';
  $('strategies').replaceChildren(...STRATEGIES.map(([k, name, what]) => {
    const lab = el('label', null, 'choice'), c = el('input');
    c.type = 'radio', c.name = 'strategy', c.value = k, c.checked = k === chosen;
    c.addEventListener('change', () => store(`labStrategy-${market}`, k));
    const t = el('span');
    t.append(el('b', name), el('small', `${what}${k === l?.strategy ? ' · live' : ''}`));
    lab.append(c, t);
    return lab;
  }));
  const opt = (id, name, what, on, disabled) => {
    const lab = el('label', null, 'choice'), c = el('input');
    c.type = 'checkbox', c.id = id, c.checked = on, c.disabled = !!disabled;
    const t = el('span');
    t.append(el('b', name), el('small', what));
    lab.append(c, t);
    return lab;
  };
  $('options').replaceChildren(
    opt('opt-live', 'The live settings in full', `every parameter ${market} trades with (the per-trade brackets, the buy-back, the nested Monte Carlo)${'gpu' in navigator ? ', on this device’s GPU' : ': no WebGPU here, so minutes on the CPU'}`, false, !l?.params),
    opt('opt-psar', 'A PSAR stop', 'bought back when its SAR turns up (ignored with the live settings)', false, false),
  );
}

// ---- the result
function clearResult() {
  $('result').replaceChildren();
  $('orders').replaceChildren();
  $('ord-note').textContent = '';
  $('status').textContent = 'Choose a market and a strategy, then run it.';
}
function tile(k, v, s, cls) { const t = el('div', null, 'tile'); t.append(el('span', k, 'k'), el('span', v, `v ${cls || ''}`), el('span', s || '', 's')); return t; }
function showResult(data) {
  const b = data.report?.backtest;
  $('result').replaceChildren(...(b && b.trades ? [
    tile('A year', pct(b.annualReturn), `${b.from} – ${b.to}`, tone(b.annualReturn)),
    tile('Sharpe', b.sharpe.toFixed(2), 'before costs'),
    tile('Worst fall', pct(-b.maxDrawdown, 0), 'from a peak', 'neg'),
    tile('Trades', String(b.trades), `${b.orders ? Math.round(100 * b.trades / b.orders) : 0}% of orders bought`),
    tile('Won', b.winRate != null ? `${Math.round(100 * b.winRate)}%` : '–', 'of the trades'),
    tile('Per trade', pct(b.avgTrade, 2), 'average, from the buy', tone(b.avgTrade)),
  ] : [tile('Result', '–', 'too little history in the bars for the strategy (it needs a year)')]));
  const orders = data.orders?.orders || [];
  const t = $('orders');
  t.replaceChildren();
  const head = t.createTHead().insertRow();
  for (const [h, c] of [['Share'], ['Buy', 'num'], ['Take-profit', 'num'], ['Stop-loss', 'num'], ['Buy-back', 'num'], ['Expected', 'num']]) head.append(el('th', h, c || ''));
  const body = t.createTBody();
  if (!orders.length) { const r = body.insertRow(), c = r.insertCell(); c.colSpan = 6; c.className = 'muted'; c.textContent = 'No orders: none expected to gain after its costs.'; }
  for (const o of orders) {
    const last = o.close, rel = (x) => (x == null || !last ? null : x / last - 1);
    const r = body.insertRow();
    const cells = [[o.ticker.split(':').pop()], [`${pct(rel(o.limit))} ${o.entryDay ? 'all day' : 'auction'}`, 'num'], [o.bracketTake == null ? 'none' : pct(rel(o.bracketTake)), 'num pos'],
      [o.bracketStop == null ? 'none' : pct(rel(o.bracketStop)), 'num neg'], [o.bracketRebuy == null ? 'none' : pct(rel(o.bracketRebuy)), 'num'], [pct(o.expected, 2), `num ${tone(o.expected)}`]];
    for (const [v, c] of cells) { const td = r.insertCell(); td.textContent = v; if (c) td.className = c; }
  }
  $('ord-note').textContent = orders.length ? `for the session after ${data.orders?.asOf || 'the last close'}, from the last close, after their costs` : '';
  $('log').textContent = data.log.filter((l) => /^strategy: /.test(l)).join('\n');
}

// ---- the run
$('run').addEventListener('click', async () => {
  const M = market, status = $('status'), btn = $('run');
  clearResult();
  if (!bridgeKey()) { status.textContent = 'Set your bridge on the Trading desk first (☰ → Today, then Bridge): the bars come from your IBKR account.'; return; }
  btn.disabled = true;
  status.textContent = 'Fetching the bars from your bridge…';
  let csv;
  try {
    const r = await fetch(`${bridgeUrl()}/bars.csv?market=${encodeURIComponent(M)}`, { headers: { 'X-Bridge-Key': bridgeKey() } });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
    csv = new Uint8Array(await r.arrayBuffer());
  } catch (e) {
    status.textContent = `No bars: ${e.message}.`;
    btn.disabled = false;
    return;
  }
  const chosen = document.querySelector('input[name="strategy"]:checked')?.value || 'rsi2', l = live(M);
  const files = {}, args = ['--csv', '/bars.csv', '--costs', M, '--strategy', chosen, '--backend', 'serial'];
  const full = $('opt-live')?.checked && l?.params;
  if (full) {  // the live strategy with every parameter it trades with
    args.splice(args.indexOf('--strategy') + 1, 1, l.strategy);
    for (const [k, v] of Object.entries(l.params)) args.push('--param', `${k}=${v}`);
  } else if ($('opt-psar')?.checked) args.push('--param', 'psar=1', '--param', 'rebuy=0');
  if (['EU', 'APAC', 'INDEX'].includes(M)) { files['venues.txt'] = await (await fetch('lab/venues.txt')).text(); args.push('--venues', '/venues.txt'); }
  if (M === 'UK') { files['aim.txt'] = await (await fetch('lab/aim.txt')).text(); args.push('--stamp-free', '/aim.txt'); }
  status.textContent = `Running on ${(csv.length / 1e6).toFixed(1)} MB of bars…`;
  if (worker) worker.terminate();
  worker = new Worker('lab/worker.js', { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.progress) { status.textContent = data.progress.replace(/^strategy: +/, '').slice(0, 110); return; }
    btn.disabled = false;
    if (data.error) { status.textContent = `The engine stopped: ${data.error}`; $('log').textContent = data.log.join('\n'); return; }
    status.textContent = `Done in ${(data.ms / 1000).toFixed(1)} s on this device${data.ran === 'on the CPU' ? '' : `: ${data.ran}`}.`;
    showResult(data);
  };
  worker.postMessage({ csv, args, env: {}, files, gpu: !!full }, [csv.buffer]);
});

// ---- the menu and the bridge's status
function menuOpen(on) { $('menu-list').hidden = !on; $('menu-toggle').setAttribute('aria-expanded', on ? 'true' : 'false'); }
$('menu-toggle').addEventListener('click', (e) => { e.stopPropagation(); menuOpen($('menu-list').hidden); });
document.addEventListener('click', (e) => { if (!e.target.closest('.menu')) menuOpen(false); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') menuOpen(false); });
async function status() {
  const c = $('conn');
  if (!bridgeKey()) { c.dataset.state = 'bad'; c.textContent = 'Set your bridge on the desk'; return; }
  try {
    const ctl = new AbortController(); setTimeout(() => ctl.abort(), 4000);
    const s = await fetch(`${bridgeUrl()}/status`, { signal: ctl.signal }).then((r) => r.json());
    c.dataset.state = s.gateway ? 'ok' : 'bad';
    c.textContent = s.gateway ? `IBKR connected · bridge ${s.version}` : 'IB Gateway not answering';
  } catch { c.dataset.state = 'bad'; c.textContent = 'Bridge not answering'; }
}

fetch('data/strategies.json', { cache: 'no-cache' }).then((r) => (r.ok ? r.json() : null)).catch(() => null).then((x) => { index = x; renderMarkets(); renderChoices(); });
renderMarkets(); renderChoices(); status();
setInterval(() => { if (!document.hidden) status(); }, 60000);
