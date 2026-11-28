// The trading desk: the IBKR account, each market's session and day, and the next session's trades, live through the
// bridge on the NAS (the same address and key as the Daily Trades page, kept in this browser only). The next
// session's trades and their chances come from the page's published results (data/<market>.json, no prices).
const $ = (id) => document.getElementById(id);
const MARKETS = ['UK', 'EU', 'US', 'APAC', 'INDEX'];
// Each market's regular session in its own time zone (the bridge trades the opening and the closing auction).
const SESSIONS = {
  UK: ['Europe/London', '08:00', '16:30', 'London'],
  EU: ['Europe/Paris', '09:00', '17:30', 'Amsterdam · Xetra'],
  US: ['America/New_York', '09:30', '16:00', 'New York'],
  APAC: ['Australia/Sydney', '10:00', '16:00', 'Sydney · Singapore'],
  INDEX: ['Europe/Berlin', '09:00', '17:30', 'Xetra ETFs'],
};
const stored = (k, d = '') => { try { return localStorage.getItem(k) || d; } catch { return d; } };
const store = (k, v) => { try { localStorage.setItem(k, v); } catch { /* not kept */ } };
const bridgeUrl = () => stored('bridgeUrl', 'http://127.0.0.1:8765').replace(/\/+$/, '');
const bridgeKey = () => stored('bridgeKey');
const bridge = (path, body) => fetch(bridgeUrl() + path, {
  method: body ? 'POST' : 'GET',
  headers: { 'X-Bridge-Key': bridgeKey(), ...(body ? { 'Content-Type': 'application/json' } : {}) },
  body: body ? JSON.stringify(body) : undefined,
}).then(async (r) => { const j = await r.json().catch(() => ({})); if (!r.ok && !j.log && !j.items && !j.positions) throw new Error(j.error || `HTTP ${r.status}`); return j; });

const state = { market: stored('deskMarket', 'UK'), status: null, portfolio: null, watch: null, trades: {}, next: {}, shadow: {}, chart: null };
if (!MARKETS.includes(state.market)) state.market = 'UK';

// ---- small helpers
const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
const num = (x, d = 2) => (x == null || Number.isNaN(+x) ? '–' : (+x).toLocaleString('en-GB', { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (x, d = 1) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${(100 * x).toFixed(d)}%`);
const signed = (x, d = 2) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${num(x, d)}`);
const tone = (x) => (x > 0 ? 'pos' : x < 0 ? 'neg' : '');
function table(t, head, rows, empty) {
  t.replaceChildren();
  const th = t.createTHead().insertRow();
  for (const [h, c] of head) th.append(el('th', h, c || ''));
  const tb = t.createTBody();
  if (!rows.length) { const r = tb.insertRow(); const c = r.insertCell(); c.colSpan = head.length; c.className = 'muted'; c.textContent = empty; return; }
  for (const row of rows) {
    const r = tb.insertRow();
    for (const [v, c, onClick] of row) {
      const td = r.insertCell();
      if (v instanceof Node) td.append(v); else td.textContent = v;
      if (c) td.className = c;
      if (onClick) { td.classList.add('link'); td.addEventListener('click', onClick); }
    }
  }
}
function log(text, ok) {
  const li = el('li', `${new Date().toLocaleTimeString('en-GB')}  ${text}`, ok === true ? 'ok' : ok === false ? 'bad' : '');
  $('log').prepend(li);
}

// ---- the markets' clocks
function session(m, now = new Date()) {
  const [zone, open, close] = SESSIONS[m];
  const parts = Object.fromEntries(new Intl.DateTimeFormat('en-GB', { timeZone: zone, weekday: 'short', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' })
    .formatToParts(now).map((p) => [p.type, p.value]));
  const mins = (+parts.hour) * 60 + (+parts.minute);
  const at = (s) => { const [h, mi] = s.split(':').map(Number); return h * 60 + mi; };
  const weekday = !['Sat', 'Sun'].includes(parts.weekday);
  const o = at(open), c = at(close);
  const isOpen = weekday && mins >= o && mins < c;
  const span = (x) => (x >= 60 ? `${Math.floor(x / 60)} h ${x % 60} min` : `${x} min`);
  const note = isOpen ? `closes in ${span(c - mins)}` : weekday && mins < o ? `opens in ${span(o - mins)}` : 'opens next trading day';
  return { time: `${parts.hour}:${parts.minute}`, isOpen, note };
}
function dayOf(m) {
  const t = state.trades[m];
  const rows = t && Array.isArray(t.shares) ? t.shares : [];
  let pnl = 0, cur = '', bought = 0, held = 0;
  for (const r of rows) { pnl += (r.pnl || 0) + (r.open || 0); cur = cur || r.currency || ''; if (r.bought > 0) bought += 1; if (r.held > 0) held += 1; }
  const n = (state.next[m]?.orders || []).length;
  return { pnl, cur, bought, held, n, known: Boolean(t) };
}
function renderMarkets() {
  const box = $('markets');
  if (!box.children.length) {
    for (const m of MARKETS) {
      const b = el('button', null, 'mkt');
      b.type = 'button';
      b.dataset.m = m;
      b.addEventListener('click', () => { state.market = m; store('deskMarket', m); renderMarkets(); renderMarket(); refreshTrades(m); refreshShadow(m); });
      box.append(b);
    }
  }
  for (const b of box.children) {
    const m = b.dataset.m, s = session(m), d = dayOf(m);
    b.setAttribute('aria-pressed', String(m === state.market));
    b.replaceChildren(
      Object.assign(el('div', null, 'row'), {}),
      el('div', SESSIONS[m][3], 'clock'),
      Object.assign(el('div', null, 'row'), {}),
      el('div', d.known ? `${d.bought} bought today · ${d.held} held · ${d.n} next` : `${d.n} next session`, 'day'),
    );
    b.children[0].append(el('span', m, 'name'), el('span', s.time, 'clock'));
    b.children[2].append(el('span', s.isOpen ? 'open' : 'closed', `sess ${s.isOpen ? 'open' : 'closed'}`), el('span', d.known && (d.bought || d.held) ? `${signed(d.pnl)} ${d.cur}` : s.note, `pnl ${tone(d.pnl)}`));
    b.title = s.note;
  }
}

// ---- the account
function renderAccount() {
  const box = $('account');
  box.replaceChildren();
  const acct = state.portfolio?.account || {};
  const pick = (k) => Object.entries(acct).find(([key]) => key.startsWith(`${k}:`) && !key.endsWith(':BASE')) || Object.entries(acct).find(([key]) => key === k || key.startsWith(`${k}:`));
  const tiles = [['NetLiquidation', 'Net liquidation'], ['TotalCashValue', 'Cash'], ['GrossPositionValue', 'In positions'], ['UnrealizedPnL', 'Open profit'], ['RealizedPnL', 'Realised today'], ['BuyingPower', 'Buying power']];
  for (const [k, label] of tiles) {
    const hit = pick(k);
    const t = el('div', null, 'tile');
    const v = hit ? +hit[1] : null;
    t.append(el('span', label, 'k'), el('span', hit ? num(v) : '–', `v ${/PnL/.test(k) ? tone(v) : ''}`), el('span', hit && hit[0].includes(':') ? hit[0].split(':')[1] : '', 's'));
    box.append(t);
  }
}
function marketOfPosition(p) {
  if (p.currency === 'GBP') return ['UK'];
  if (p.currency === 'USD') return ['US'];
  if (p.currency === 'AUD' || p.currency === 'SGD') return ['APAC'];
  return ['EU', 'INDEX'];
}
function renderPositions() {
  const p = state.portfolio;
  const rows = (p?.positions || []).slice().sort((a, b) => (b.unrealized || 0) - (a.unrealized || 0));
  table($('positions'), [['Share'], ['Exch.'], ['Shares', 'num'], ['Avg cost', 'num'], ['Price', 'num'], ['Value', 'num'], ['Open P&L', 'num']],
    rows.map((r) => {
      const cost = r.avgCost * r.qty, ret = cost ? r.unrealized / Math.abs(cost) : null;
      return [[r.symbol, '', () => openChart(marketOfPosition(r), r.symbol, null)], [r.exchange || ''], [num(r.qty, 0), 'num'], [num(r.avgCost, 3), 'num'], [num(r.price, 3), 'num'],
        [`${num(r.value)} ${r.currency}`, 'num'], [`${signed(r.unrealized)} (${pct(ret)})`, `num ${tone(r.unrealized)}`]];
    }), p ? 'No positions: everything is in cash.' : 'Reading the account from IBKR…');
  $('pos-note').textContent = p ? `${rows.length} position${rows.length === 1 ? '' : 's'} in all markets${p.age != null ? ` · read ${p.age} s ago` : ''}` : '';
}
function renderAlerts() {
  const w = state.watch, box = $('alerts');
  const items = w && Array.isArray(w.items) ? w.items : [];
  box.hidden = !(w && w.ok === false && items.length);
  box.replaceChildren(el('b', 'Unexpected at Interactive Brokers'), ...items.map((x) => el('div', x)));
}
function renderConn() {
  const s = state.status, c = $('conn');
  if (!s) { c.dataset.state = 'bad'; c.textContent = bridgeKey() ? 'Bridge not answering' : 'Set your bridge'; return; }
  c.dataset.state = s.gateway ? 'ok' : 'bad';
  c.textContent = s.gateway ? `IBKR connected · bridge ${s.version}` : 'IB Gateway not answering';
}

// ---- the selected market: today and the next session
function renderMarket() {
  const m = state.market;
  $('mkt-title').textContent = `${m} · ${SESSIONS[m][3]}`;
  // the research page opens on this market: its Lab (the strategies run here, on the bridge's bars) and its picks
  $('mkt-lab').href = `lab.html?market=${m.toLowerCase()}`;
  $('mkt-lab').textContent = `Explore ${m} in the Lab`;
  $('mkt-research').href = `index.html?market=${m.toLowerCase()}#orders`;
  $('lab-link').href = `lab.html?market=${m.toLowerCase()}`;
  const a = state.status?.auto?.[m];
  $('mkt-note').textContent = a ? `orders placed ${a.place}, sold ${a.close} (${a.zone})` : '';
  const t = state.trades[m];
  table($('today'), [['Share'], ['State'], ['Bought', 'num'], ['Sold', 'num'], ['Held', 'num'], ['P&L', 'num']],
    (t?.shares || []).map((r) => {
      const pnl = (r.pnl || 0) + (r.open || 0);
      return [[r.ticker, '', () => openChart([m], r.ticker, null)], [r.state || ''], [r.bought ? `${num(r.bought, 0)} @ ${num(r.buy, 3)}` : '–', 'num'],
        [r.sold ? `${num(r.sold, 0)} @ ${num(r.sell, 3)}` : '–', 'num'], [r.held ? num(r.held, 0) : '–', 'num'], [r.bought || r.held ? `${signed(pnl)} ${r.currency || ''}` : '–', `num ${tone(pnl)}`]];
    }), t ? (t.error || 'Nothing traded today.') : 'Reading today from IBKR…');
  const nx = state.next[m];
  $('next-note').textContent = nx ? `After the close of ${nx.asOf || '?'}, ranked by the expected return after costs; P(net gain) is each order's chance, calibrated on the strategy's earlier orders.` : 'Loading…';
  renderShadow();
  const bar = (p) => { const s = el('span', null, 'prob'); const b = el('span', null, 'bar'); const i = el('i'); i.style.width = `${Math.round(100 * (p || 0))}%`; b.append(i); s.append(b, el('span', p == null ? '–' : `${Math.round(100 * p)}%`)); return s; };
  table($('next'), [['Share'], ['Buy', 'num'], ['Take-profit', 'num'], ['Stop-loss', 'num'], ['Buy-back', 'num'], ['P(net gain)'], ['Expected', 'num']],
    (nx?.orders || []).map((o) => [[o.ticker, '', () => openChart([m], o.ticker, o)],
      [`${o.limit == null ? 'open' : pct(o.limit, 1)} ${o.entryDay ? 'all day' : 'auction'}`, 'num'], [o.takeRel == null ? 'none' : pct(o.takeRel, 1), 'num pos'],
      [o.stopRel == null ? 'none' : pct(o.stopRel, 1), 'num neg'], [o.rebuyRel == null ? 'none' : pct(o.rebuyRel, 1), 'num'], [bar(o.probs?.final)], [pct(o.expected, 2), `num ${tone(o.expected)}`]]),
    nx ? 'No trade for the next session: none expected to gain after its costs.' : 'Loading…');
}

// ---- shadow mode (GET /shadow): every 5 minutes of the session, each of the day's orders in its state and what the learnt
// intraday policy would do there. Recorded only: no order is changed.
const ACTIONS = { hold: 'hold', keep: 'keep the order', sell: 'sell now', even: 'stop to the buy price', trail0_5: 'trail the stop ½ σ', trail1_0: 'trail the stop 1 σ', notake: 'drop the take-profit', cancel: 'cancel the order', buy: 'buy now' };
const actionName = (a) => ACTIONS[String(a || '').replace('.', '_')] || a || '–';
function renderShadow() {
  const m = state.market, sh = state.shadow[m];
  if (!sh) { $('shadow-note').textContent = bridgeKey() ? 'Reading…' : 'Set your bridge to see the checks.'; table($('shadow'), [['Share']], [], ''); return; }
  if (sh.error) { $('shadow-note').textContent = `Not read: ${sh.error}`; table($('shadow'), [['Share']], [], ''); return; }
  const rows = sh.rows || [], last = new Map(), acted = new Map();
  for (const r of rows) {
    last.set(r.ticker, r);
    if (r.action && !['hold', 'keep'].includes(r.action)) acted.set(r.ticker, [...(acted.get(r.ticker) || []), `${r.time} ${actionName(r.action)}`]);
  }
  $('shadow-note').textContent = `${sh.hasPolicy ? 'What the learnt policy would do (recorded only, nothing changed)' : `No policy for ${m} yet: the states are recorded for learning`}. ${sh.note || ''}`.trim();
  const sig = (x) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${(+x).toFixed(2)} σ`);
  table($('shadow'), [['Share'], ['At', 'num'], ['State'], ['Move', 'num'], ['Now'], ['Earlier today']],
    [...last.values()].map((r) => [[r.ticker], [r.time || '', 'num'], [r.status || ''], [sig(r.moveSigma ?? r.gapSigma), 'num'],
      [actionName(r.action), ['hold', 'keep', undefined].includes(r.action) ? '' : 'neg'], [(acted.get(r.ticker) || []).slice(-3).join(' · ') || '–']]),
    `No check today yet (every 5 minutes of the ${m} session).`);
}
async function refreshShadow(m) {
  if (!bridgeKey()) return;
  try { state.shadow[m] = await bridge(`/shadow?market=${m}`); } catch (e) { state.shadow[m] = { error: e.message }; }
  if (m === state.market) renderShadow();
}

// ---- the chart: daily bars from IBKR (through the bridge), the order's levels on it
async function openChart(markets, ticker, order) {
  $('chart-card').hidden = false;
  $('chart-title').textContent = `${ticker}`;
  $('chart-note').textContent = 'Loading…';
  for (const m of markets) {
    try {
      const r = await bridge(`/chart?market=${m}&ticker=${encodeURIComponent(ticker)}&days=90`);
      state.chart = { bars: r.bars, order, ticker, m };
      drawChart();
      $('chart-note').textContent = `${m} · ${r.bars.length} daily bars from IBKR${order ? ' · the next session’s levels from the last close: buy limit, take-profit, stop-loss, buy-back' : ''}${m === 'UK' ? ' · prices in pence' : ''}`;
      $('chart-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      return;
    } catch (e) { $('chart-note').textContent = `No chart: ${e.message}`; }
  }
}
function drawChart() {
  const c = $('chart'), ch = state.chart;
  if (!ch || $('chart-card').hidden) return;
  const css = getComputedStyle(document.documentElement);
  const col = (v, d) => css.getPropertyValue(v).trim() || d;
  const dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = 320;
  c.width = W * dpr, c.height = H * dpr;
  const g = c.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, W, H);
  const bars = ch.bars, last = bars[bars.length - 1]?.c;
  const levels = [];
  if (ch.order && last) {
    const o = ch.order;
    if (o.limit != null) levels.push(['buy', last * (1 + o.limit), col('--accent', '#1f5f8b')]);
    if (o.takeRel != null) levels.push(['take-profit', last * (1 + o.takeRel), col('--good', '#1d7a4f')]);
    if (o.stopRel != null) levels.push(['stop-loss', last * (1 + o.stopRel), col('--bad', '#b3362f')]);
    if (o.rebuyRel != null) levels.push(['buy-back', last * (1 + o.rebuyRel), col('--hedged', '#8a5a00')]);
  }
  let lo = Math.min(...bars.map((b) => b.l), ...levels.map((x) => x[1])), hi = Math.max(...bars.map((b) => b.h), ...levels.map((x) => x[1]));
  const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
  const L = 8, R = 64, T = 8, B = 22, w = (W - L - R) / bars.length;
  const y = (p) => T + (H - T - B) * (1 - (p - lo) / (hi - lo));
  g.font = '11px "IBM Plex Mono", monospace';
  g.fillStyle = col('--muted', '#5a6575');
  g.strokeStyle = col('--rule', '#d8dde5');
  for (let k = 0; k <= 4; k++) {
    const p = lo + (hi - lo) * k / 4, yy = y(p);
    g.beginPath(); g.moveTo(L, yy); g.lineTo(W - R, yy); g.stroke();
    g.fillText(p >= 100 ? p.toFixed(1) : p.toFixed(3), W - R + 6, yy + 4);
  }
  bars.forEach((b, k) => {
    const x = L + k * w + w / 2, up = b.c >= b.o;
    g.strokeStyle = g.fillStyle = up ? col('--good', '#1d7a4f') : col('--bad', '#b3362f');
    g.beginPath(); g.moveTo(x, y(b.h)); g.lineTo(x, y(b.l)); g.stroke();
    const top = y(Math.max(b.o, b.c)), bot = y(Math.min(b.o, b.c));
    g.fillRect(x - Math.max(1, w * 0.35), top, Math.max(2, w * 0.7), Math.max(1, bot - top));
  });
  g.fillStyle = col('--muted', '#5a6575');
  g.fillText(bars[0]?.d || '', L, H - 6);
  g.fillText(bars[bars.length - 1]?.d || '', W - R - 70, H - 6);
  for (const [name, p, c2] of levels) {
    g.strokeStyle = g.fillStyle = c2; g.setLineDash([5, 4]);
    g.beginPath(); g.moveTo(L, y(p)); g.lineTo(W - R, y(p)); g.stroke(); g.setLineDash([]);
    g.fillText(name, L + 4, y(p) - 4);
  }
}
$('chart-close').addEventListener('click', () => { $('chart-card').hidden = true; });
window.addEventListener('resize', drawChart);

// ---- actions on the selected market
for (const b of document.querySelectorAll('[data-act]')) {
  b.addEventListener('click', async () => {
    const m = state.market, path = b.dataset.act, dry = Boolean(b.dataset.dry);
    const ask = { '/place': dry ? null : `Place the ${m} orders with Interactive Brokers now?`, '/close': `Cancel today's unfilled ${m} orders and sell what is left at the close?`,
      '/liquidate': `Sell now every ${m} position the bridge holds from an earlier day (its stops cancelled first)? Today's trades are left alone.`, '/report': null,
      '/intraday': `Fetch the last 120 sessions of 5-minute bars for ${m}'s candidates from IBKR into the store? It runs in the background for about an hour and pauses for the bridge's own steps.` }[path];
    if (ask && !window.confirm(ask)) return;
    b.disabled = true;
    log(`${m} ${b.textContent}…`);
    try {
      const r = await bridge(path, dry ? { market: m, dryRun: true } : path === '/intraday' ? { market: m, days: 120 } : { market: m });
      log(`${m} ${b.textContent}: ${r.log || (r.ok ? 'done' : JSON.stringify(r))}`, r.ok !== false);
      refreshTrades(m);
      refreshAccount();
    } catch (e) { log(`${m} ${b.textContent} failed: ${e.message}`, false); }
    b.disabled = false;
  });
}

// ---- the sections: Today and History (the menu, as on the research page)
function menuOpen(on) { $('menu-list').hidden = !on; $('menu-toggle').setAttribute('aria-expanded', on ? 'true' : 'false'); }
function showView(v) {
  if (!['today', 'history'].includes(v)) v = 'today';
  $('v-today').hidden = v !== 'today';
  $('v-history').hidden = v !== 'history';
  $('menu-current').textContent = v === 'today' ? 'Today' : 'History';
  for (const b of document.querySelectorAll('.views button[data-view]')) b.toggleAttribute('aria-current', b.dataset.view === v);
  store('deskView', v);
  if (location.hash !== `#${v}`) history.replaceState(null, '', `#${v}`);
  menuOpen(false);
  if (v === 'history') refreshHistory();
}
for (const b of document.querySelectorAll('.views button[data-view]')) b.addEventListener('click', () => showView(b.dataset.view));
$('menu-toggle').addEventListener('click', (e) => { e.stopPropagation(); menuOpen($('menu-list').hidden); });
document.addEventListener('click', (e) => { if (!e.target.closest('.menu')) menuOpen(false); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') menuOpen(false); });

// ---- history: each day's orders and trades in the selected market, against the model (GET /history)
let histData = null;
function renderHistoryMarkets() {
  const box = $('hist-markets');
  box.replaceChildren(...MARKETS.map((m) => {
    const b = el('button', m); b.type = 'button'; b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', String(m === state.market));
    b.addEventListener('click', () => { state.market = m; store('deskMarket', m); renderHistoryMarkets(); renderHistory(); renderMarkets(); renderMarket(); });
    return b;
  }));
}
async function refreshHistory() {
  renderHistoryMarkets();
  if (!bridgeKey()) { $('hist-summary').textContent = 'Set your bridge to see your history.'; return; }
  try { histData = await bridge('/history'); } catch (e) { $('hist-summary').textContent = `History not available: ${e.message}`; return; }
  renderHistory();
}
function renderHistory() {
  const box = $('hist-days');
  box.replaceChildren();
  if (!histData) return;
  const m = state.market;
  const code = (x) => x.split(':').pop().replace(' ', '.');
  const ratio = (a, b) => { const r = a / b; const s = [1, 0.01, 100].reduce((p, x) => (Math.abs(Math.log(r / x)) < Math.abs(Math.log(r / p)) ? x : p), 1); return r / s; };
  const perDay = {};
  let n = 0, wins = 0, placed = 0;
  const total = {};
  for (const d of (histData.days || []).filter((x) => x.market === m && x.date)) {
    const day = (perDay[d.date] = perDay[d.date] || { rows: [], orders: 0, bought: 0, pnl: {} });
    day.orders += d.orders.length; placed += d.orders.length;
    const traded = Object.fromEntries(d.trades.map((t) => [code(t.ticker), t]));
    const model = Object.fromEntries((d.model || []).map((t) => [code(t.ticker), t]));
    const reported = d.trades.length > 0 || (d.model || []).length > 0;
    for (const name of new Set([...d.orders.map((o) => code(o.ticker)), ...Object.keys(traded)])) {
      const t = traded[name], mo = model[name], filled = t && t.bought > 0;
      if (filled) { n += 1; if (t.ret > 0) wins += 1; day.bought += 1; day.pnl[t.currency] = (day.pnl[t.currency] || 0) + t.pnl; total[t.currency] = (total[t.currency] || 0) + t.pnl; }
      const gap = filled && mo && mo.filled > 0 ? ratio(t.buy, mo.entry) - 1 : null;
      const modelSays = mo ? (mo.filled > 0 ? pct(mo.ret, 2) : 'no buy') : '';
      const yours = filled && t.ret != null ? pct(t.ret, 2) : filled ? 'open' : reported ? 'no buy' : 'waiting';
      day.rows.push([[name], [`${yours}${modelSays ? ` (model ${modelSays})` : ''}`, `num ${filled && t.ret != null ? tone(t.ret) : ''}`],
        [gap == null ? '' : pct(gap, 2), `num ${gap == null ? '' : gap <= 0 ? 'pos' : 'neg'}`],
        [filled ? `${signed(t.pnl)} (fees ${num(t.commission)})` : '', `num ${filled ? tone(t.pnl) : ''}`]]);
    }
  }
  const dates = Object.keys(perDay).filter((x) => perDay[x].orders > 0 || perDay[x].bought > 0).sort().reverse();
  const money = (o) => Object.entries(o).map(([c, v]) => `${signed(v)} ${c}`).join(', ');
  $('hist-summary').textContent = dates.length ? `${dates.length} trading days · ${placed} orders · ${n} bought · ${n ? Math.round(100 * wins / n) : 0}% of them gained · total ${money(total) || '0'}` : 'Nothing yet: each trading day with an order appears here once it is over.';
  for (const date of dates) {
    const day = perDay[date], tot = Object.values(day.pnl).reduce((a, b) => a + b, 0);
    const f = el('details'), s = el('summary');
    s.append(el('span', new Date(`${date}T12:00:00Z`).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }), 'when'),
      el('span', `${day.orders} order${day.orders === 1 ? '' : 's'} · ${day.bought} bought`, 'cnt'), el('span', money(day.pnl) || '–', `money ${tone(tot)}`));
    const wrap = el('div', null, 'table-scroll'), t = el('table');
    wrap.append(t);
    table(t, [['Share'], ['Return (model)', 'num'], ['Paid vs model', 'num'], ['Profit (fees)', 'num']], day.rows, '');
    f.append(s, wrap);
    box.append(f);
  }
}

// ---- settings
$('settings-open').addEventListener('click', () => { menuOpen(false); $('set-url').value = bridgeUrl(); $('set-key').value = bridgeKey(); $('settings').showModal(); });
$('settings').addEventListener('close', () => {
  if ($('settings').returnValue !== 'save') return;
  store('bridgeUrl', $('set-url').value.trim()); store('bridgeKey', $('set-key').value.trim());
  log('Bridge settings saved.', true);
  refreshAll();
});

// ---- the reads
async function refreshStatus() {
  try {
    const ctl = new AbortController(); setTimeout(() => ctl.abort(), 4000);
    state.status = await fetch(`${bridgeUrl()}/status`, { signal: ctl.signal }).then((r) => r.json());
  } catch { state.status = null; }
  renderConn(); renderMarket();
}
async function refreshAccount() {
  if (!bridgeKey()) return;
  try { state.portfolio = await bridge('/portfolio'); } catch (e) { state.portfolio = state.portfolio || null; $('pos-note').textContent = `Account not read: ${e.message}`; }
  try { state.watch = await bridge('/watch'); } catch { state.watch = null; }
  renderAccount(); renderPositions(); renderAlerts();
}
async function refreshTrades(m) {
  if (!bridgeKey()) return;
  try { state.trades[m] = await bridge(`/trades?market=${m}`); } catch (e) { state.trades[m] = { error: e.message, shares: [] }; }
  renderMarkets(); if (m === state.market) renderMarket();
}
async function refreshNext() {
  await Promise.all(MARKETS.map(async (m) => {
    try { state.next[m] = await fetch(`data/${m.toLowerCase()}.json`, { cache: 'no-cache' }).then((r) => r.json()); } catch { state.next[m] = { orders: [] }; }
  }));
  renderMarkets(); renderMarket();
}
function refreshAll() {
  refreshStatus(); refreshAccount(); refreshNext();
  for (const m of MARKETS) refreshTrades(m);
  refreshShadow(state.market);
}

renderMarkets(); renderMarket(); renderAccount(); renderPositions();
showView(location.hash.slice(1) || stored('deskView', 'today'));
if (!bridgeKey()) { log('Set your bridge (top right) to see the account and trade.'); $('settings-open').click(); }
refreshAll();
setInterval(() => { if (!document.hidden) renderMarkets(); }, 15000);
setInterval(() => { if (!document.hidden) { refreshStatus(); refreshAccount(); } }, 60000);
setInterval(() => { if (!document.hidden) for (const m of MARKETS) refreshTrades(m); }, 60000);
setInterval(() => { if (!document.hidden) refreshShadow(state.market); }, 60000);
setInterval(() => { if (!document.hidden) refreshNext(); }, 10 * 60000);
