// The page: the latest scan of each market, published by the scheduled workflow (results only, no
// prices): the next open's trades and the model's walk-forward backtest. Rendering only; every
// number comes from the model (apps/ofm_research), the chart laid out there too.
const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${(100 * x).toFixed(d)}%`);
const SERIES = [['daily trades', '--daybook']];

function cell(tag, text, cls) {
  const c = document.createElement(tag);
  if (cls) c.className = cls;
  c.textContent = text;
  return c;
}
function table(el, head, rows) {
  el.replaceChildren();
  const thead = el.createTHead().insertRow();
  head.forEach(([t, cls]) => thead.append(cell('th', t, cls)));
  const body = el.createTBody();
  rows.forEach((r) => {
    const tr = body.insertRow();
    if (r.gap) { tr.className = 'gap'; const td = cell('td', r.gap); td.colSpan = head.length; tr.append(td); return; }
    r.forEach((c) => tr.append(cell('td', c[0], c[1])));
  });
}
const signed = (x, d = 2) => [pct(x, d), `num ${x >= 0 ? 'pos' : 'neg'}`];

let shown = null;
// The costs the backtest charges (r.costs, from the model's --costs), or none.
const costs = (r) => {
  const c = r.costs;
  if (!c) return 'no fees or costs';
  if (c.net === false && r.model === 'uk') return `the backtest before costs; each order's expected return after its own ${c.broker} costs (commission and stamp duty; the auctions cross no spread)`;
  if (c.net === false) return `before costs (gross; ${c.broker} costs not deducted${c.venues ? '' : `, ${{ UK: '£', US: '$', APAC: 'A$', EU: '€' }[c.market] || ''}${c.capital.toLocaleString('en-GB')} a trade`})`;
  if (c.venues) return `after ${c.broker} costs and each exchange's transaction taxes (${c.market === 'EU' ? `€${c.capital.toLocaleString('en-GB')} a trade${r.exchanges && !r.exchanges.includes('OSL') ? '' : ', its value in NOK in Oslo'}` : `${c.capital.toLocaleString('en-GB')} a trade in its currency`})`;
  const cur = { UK: '£', US: '$', APAC: 'A$', EU: '€' }[c.market] || '';
  return `after ${c.broker} costs (${cur}${c.capital.toLocaleString('en-GB')} a trade${c.stamp ? `, ${(100 * c.stamp).toFixed(1)}% stamp duty on buying${c.stampFree ? ', none on AIM shares' : ''}` : ''})`;
};

// The exchanges scanned (r.exchanges, written by the daily run), in words.
const EXCHANGE = { LSE: 'London Stock Exchange', NYSE: 'NYSE', NASDAQ: 'NASDAQ', PAR: 'Paris', AMS: 'Amsterdam', BRU: 'Brussels', LIS: 'Lisbon', MSE: 'Madrid', FRA: 'Frankfurt', OSL: 'Oslo', ASX: 'Australia (ASX)', SGX: 'Singapore (SGX)' };
const places = (r, source) => {
  const ex = r.exchanges;
  if (!ex?.length) return { UK: 'London Stock Exchange', US: 'New York (NYSE and NASDAQ)', EU: 'Europe', APAC: 'Australia (ASX) and Singapore (SGX)' }[source] || source;
  const names = ex.map((e) => EXCHANGE[e] || e);
  return names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
};

// The markets traded automatically: the bridge's AUTO_MARKETS once it answers, remembered in this
// browser for when it cannot be reached (away from home); UK and EU until it has answered once.
const LIVE_KEY = 'trading.live';
let live = new Set(['UK', 'EU']);
try { const v = JSON.parse(localStorage.getItem(LIVE_KEY)); if (Array.isArray(v) && v.length) live = new Set(v); } catch { /* none kept */ }
const SIZE = (c) => (c ? `${{ UK: '£', US: '$', EU: '€' }[c.market] || ''}${c.capital.toLocaleString('en-GB')}${c.venues ? ' (each exchange\'s currency)' : ''}` : '');

// The plan in plain words, the orders, and the backtest (folded away).
function render(r, source) {
  shown = r;
  const where = places(r, source);
  const n = r.orders.length, c = r.costs;
  const isLive = live.has(source);
  const day = new Date(`${r.asOf}T12:00:00Z`).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' });
  $('summary').className = `summary ${isLive ? 'live' : 'research'}`;
  $('summary').innerHTML = isLive
    ? `<strong>${n ? `${n} order${n === 1 ? '' : 's'}, ${SIZE(c)} each` : 'No orders'}</strong> for the next trading day, placed automatically before the open; `
      + (r.book === 'hold' ? `each held until its first close at its exit level, at most ${r.days} days.` : 'anything bought is sold by the close.')
    : (r.backtest?.sharpe ?? 0) < 1
      ? `<strong>Research only: not traded.</strong> ${c?.net === false ? 'Before costs' : 'After costs'} the model shows no reliable edge here${n ? '; the orders below are what it would place' : ''}.`
      : `<strong>Not traded automatically:</strong> the bridge does not trade this market (AUTO_MARKETS)${n ? '; the orders below are what it would place' : ''}.`;
  const buys = r.orders.reduce((a, o) => a + (o.fill ?? 0), 0);  // the orders' chances added up: the buys to expect
  $('asof').textContent = `${where} · ${r.stocks} most traded shares · after the close of ${day} · ${costs(r)}`
    + (r.orders.some((o) => o.fill != null) ? ` · about ${buys.toFixed(1)} of these orders expected to buy` : '');
  const levels = r.orders.some((o) => o.stop != null || o.take != null);  // without: every trade sold at the close
  const holds = r.orders.some((o) => o.days != null);  // a multi-day book: sold at the first close at its exit level, else on its last day
  const head = [['Share'], ['Buy at', 'num']];
  if (levels) head.push(['Stop', 'num'], ['Target', 'num']);
  if (holds) head.push(['Sold at', 'num']);
  const rebuys = r.orders.some((o) => o.rebuy > 0);  // a buy-back after a stop (learnt; none on most days)
  if (rebuys) head.push(['Buy-back', 'num']);
  const chances = r.orders.some((o) => o.fill != null);  // each order's chance to buy (the share's own days so far)
  if (chances) head.push(['Chance to buy', 'num']);
  const ifs = r.orders.some((o) => o.ifFilled != null);  // the gain if it buys; expected = the chance to buy times it
  const costed = r.orders.some((o) => o.cost != null);  // the UK model: each order's own costs, already off its expected return
  if (costed) head.push(['Costs', 'num']);
  if (ifs) head.push(['If it buys', 'num']);
  head.push(['Expected', 'num']);
  table($('plan'), head,
    n ? r.orders.map((o) => {
      const row = [[o.ticker], [o.limit == null ? 'open' : o.entry === 'open' ? `open, at most ${pct(o.limit, 1)}` : pct(o.limit, 1), 'num']];
      // The stop from the price bought at: fixed, trailing under the day's high, or to the price paid once up `breakeven`.
      const how = o.stopMode === 'trail' ? ' trailing' : o.stopMode === 'breakeven' && o.breakeven != null ? `, to 0 once ${pct(o.breakeven, 1)}` : '';
      if (levels) row.push([o.stop == null ? 'none' : pct(o.stop, 1) + how, o.stop == null ? 'num' : 'num neg'], [o.take == null ? 'none' : pct(o.take, 1), o.take == null ? 'num' : 'num pos']);
      if (holds) row.push([`first close ${pct(o.exitAbove, 2)} or more, else day ${o.days}${o.holdDays ? ` (about ${o.holdDays.toFixed(1)} days)` : ''}`, 'num pos']);
      // The buy-back's price from the buy price: rebuy of the way from the stop back up to it.
      if (rebuys) row.push(o.rebuy > 0 && o.stop != null ? [pct(o.stop * (1 - o.rebuy), 1), 'num neg'] : ['none', 'num']);
      if (chances) row.push([o.fill == null ? '–' : `${Math.round(100 * o.fill)}%`, 'num']);
      if (costed) row.push(o.cost == null ? ['–', 'num'] : [pct(-o.cost, 2), 'num neg']);
      if (ifs) row.push(o.ifFilled == null ? ['–', 'num'] : signed(o.ifFilled, 2));
      row.push(signed(o.expected, Math.abs(o.expected) < 0.0005 ? 3 : 2));  // the order's: the chance to buy times the gain if it buys
      return row;
    }) : [{ gap: `No orders: none is expected to gain${c?.net === false ? '' : ' after costs'}, so the money stays in cash.` }]);
  // The notes under the orders: the research model's (in the page), or the UK model's.
  const hint = $('planhint');
  if (hint) {
    hint.dataset.base ??= hint.innerHTML;
    hint.innerHTML = r.model === 'uk' && r.book === 'hold'
      ? '<b>Buy at</b>: in the opening auction only, at most this far above the last close; an order that does not buy there is cancelled, so it never buys a dip later. '
        + `<b>Sold at</b>: in the closing auction of the first day the share closes this far above the price paid, else at the close of day ${r.days}: held as few days as it needs. `
        + '<b>Costs</b>: the order\'s own commission, stamp duty (none on AIM shares), already taken off; it trades in the auctions, so it pays no spread. '
        + '<b>If it buys</b>: what similar past trades made before costs, less these costs. <b>Expected</b>: that times the chance to buy. <span id="odds"></span>'
      : r.model === 'uk'
      ? '<b>Buy at</b>: in the opening auction only, at most this far above the last close; an order that does not buy there is cancelled, so it never buys a dip later in the day. '
        + '<b>Stop</b>: from the price bought at; <i>trailing</i> follows the day\'s high at that distance; <i>to 0 once +x%</i> moves to the price paid once the share is up that much. '
        + '<b>Target</b>: sold if it rises that far. Whatever is still held is sold at the close, so nothing is held overnight. '
        + '<b>Costs</b>: the order\'s own commission, stamp duty (none on AIM shares), already taken off; it trades in the auctions, so it pays no spread. '
        + '<b>If it buys</b>: what similar past orders made before costs, less these costs. <b>Expected</b>: that times the chance to buy. <span id="odds"></span>'
      : hint.dataset.base;
  }
  const B = r.backtest;
  // The same trades in the expected gain's terms: per trade that bought, from its buy price.
  const perTrade = B.trades ? `
    <div class="tile"><span class="k">Per trade that bought</span><span class="v">${pct(B.avgTrade, 2)}</span><span class="s">average from the buy price · ${Math.round(100 * B.trades / B.orders)}% of orders bought · ${Math.round(100 * B.winRate)}% of trades gained</span></div>` : '';
  $('verdict').innerHTML = `
    <div class="tile"><span class="k">Simulated, ${B.from} – ${B.to}</span><span class="v">${pct(B.annualReturn)}</span><span class="s">a year · Sharpe ${B.sharpe.toFixed(2)} · worst fall ${pct(B.maxDrawdown, 0).slice(1)}</span></div>${perTrade}`;
  $('odds').textContent = B.trades
    ? `In the backtest ${Math.round(100 * B.trades / B.orders)}% of orders bought (the rest stayed in cash), their average was ${pct(B.avgTrade, 2)} and ${Math.round(100 * B.winRate)}% of them gained.`
    : '';
  table($('recent'), [['Day'], ['Simulated result', 'num']],
    r.recent.slice().reverse().map((d) => [[d.date], signed(d.day)]));
  drawChart(r);
}

// The chart as laid out by the model (r.chart: heights and positions as shares of the chart).
function drawChart(r) {
  const css = getComputedStyle(document.documentElement);
  const color = Object.fromEntries(SERIES.map(([name, c]) => [name, css.getPropertyValue(c).trim()]));
  const canvas = $('chart'), legend = $('legend'), C = r.chart;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr, canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!C) { legend.innerHTML = ''; return; }
  const pad = { l: 46, r: 10, t: 10, b: 24 };
  const X = (x) => pad.l + x * (w - pad.l - pad.r);
  const Y = (y) => pad.t + (1 - y) * (h - pad.t - pad.b);
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.fillStyle = css.getPropertyValue('--muted').trim();
  ctx.strokeStyle = css.getPropertyValue('--rule').trim();
  ctx.lineWidth = 1;
  C.ticks.forEach((t) => {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(t.y)); ctx.lineTo(w - pad.r, Y(t.y)); ctx.stroke();
    ctx.fillText(t.label, 4, Y(t.y) + 4);
  });
  let lastX = -Infinity;
  C.years.forEach((yr) => { if (X(yr.x) - lastX >= 60) { ctx.fillText(yr.label, X(yr.x) - 12, h - 6); lastX = X(yr.x); } });
  const series = C.series.filter((s) => color[s.name]);  // the daily trades only (an older scan also carries the market)
  series.forEach((s) => {
    ctx.strokeStyle = color[s.name] || ctx.strokeStyle;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.y.forEach((y, i) => (i ? ctx.lineTo(X(i / (s.y.length - 1)), Y(y)) : ctx.moveTo(X(0), Y(y))));
    ctx.stroke();
  });
  legend.innerHTML = series.map((s) => `<span><i style="background:${color[s.name]}"></i>${s.name} ${s.final}</span>`).join('');
}
window.addEventListener('resize', () => { if (shown) drawChart(shown); });
// The sections, one at a time (the menu; #orders, #results, ... in the address, remembered).
const VIEWS = ['orders', 'results', 'backtest', 'broker', 'about'];
let bridgeOk = false;
function showView(v) {
  if (!VIEWS.includes(v)) v = 'orders';
  for (const w of VIEWS) $(`view-${w}`).hidden = w !== v;
  for (const b of document.querySelectorAll('.views button')) b.toggleAttribute('aria-current', b.dataset.view === v);
  try { localStorage.setItem('view', v); } catch { /* no storage */ }
  if (location.hash !== `#${v}`) history.replaceState(null, '', `${location.pathname}${location.search}#${v}`);
  if (v === 'backtest' && shown) drawChart(shown);  // a hidden canvas has no width
  if ((v === 'results' || v === 'broker') && !bridgeOk) brokerProbe(true);
  if (typeof todayLoop === 'function') todayLoop(v === 'results' && bridgeOk);
}
for (const b of document.querySelectorAll('.views button')) b.addEventListener('click', () => showView(b.dataset.view));
document.addEventListener('click', (e) => { const a = e.target.closest('[data-go]'); if (a) { e.preventDefault(); showView(a.dataset.go); } });

const MARKETS = { uk: 'UK', eu: 'EU', us: 'US', apac: 'APAC' };
let market = 'uk';
try { market = new URLSearchParams(location.search).get('market') || localStorage.getItem('market') || 'uk'; } catch { /* no storage */ }
if (!MARKETS[market]) market = 'uk';
function showMarket(m) {
  market = m;
  try { localStorage.setItem('market', m); } catch { /* no storage */ }
  for (const b of document.querySelectorAll('.market-switch button')) b.setAttribute('aria-selected', String(b.dataset.market === m));
  const name = MARKETS[m];
  $('asof').textContent = `Loading the latest ${name} scan…`;
  $('plan').innerHTML = '', $('verdict').innerHTML = '', $('recent').innerHTML = '', $('legend').innerHTML = '';
  fetch(`data/${m}.json`, { cache: 'no-cache' })
    .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
    .then((r) => { if (market === m) render(r, name); })
    .catch(() => { if (market === m) $('asof').textContent = `No ${name} scan has been published yet; it appears after the next scheduled run.`; });
  sideBySide(m);
}
// The market's strategies (data/strategies.json, from config/markets.json): the live one and its
// shadows, each with its backtest and its record since tracking began (the same days for each).
function sideBySide(m) {
  const el = $('sidebyside');
  el.replaceChildren();
  const get = (f) => fetch(`data/${f}`, { cache: 'no-cache' }).then((res) => (res.ok ? res.json() : null)).catch(() => null);
  get('strategies.json').then(async (idx) => {
    const list = idx?.markets?.[m] || [];
    if (!list.length) { table(el, [['Strategy']], [{ gap: 'one strategy: the live one (no shadows published yet)' }]); return; }
    const liveTrack = m === 'uk' ? 'track.json' : `track-${m}.json`;
    const rows = await Promise.all(list.map(async (s) => {
      const live = s.role === 'live';
      const [r, t] = await Promise.all([get(live ? `${m}.json` : `${m}-${s.name}.json`), get(live ? liveTrack : `track-${m}-${s.name}.json`)]);
      return { s, r, t };
    }));
    const since = rows.map((x) => x.t?.days?.[0]?.date).filter(Boolean).sort().pop();  // the days all have
    const record = (t) => {
      const d = (t?.days || []).filter((x) => !since || x.date >= since).map((x) => x.day ?? 0);
      return d.length ? [d.reduce((w, x) => w * (1 + x), 1) - 1, d.length] : [null, 0];
    };
    if (market !== m) return;
    table(el, [['Strategy'], ['Expected a day', 'num'], ['Expected a year', 'num'], ['Backtest a year', 'num'], ['Sharpe', 'num'], ['Orders bought', 'num'], ['Tracked', 'num'], ['Next open', 'num']],
      rows.map(({ s, r, t }) => {
        const B = r?.backtest || {};
        const [rec, n] = record(t);
        const E = r?.expected;  // a day's recommended orders on average (unfilled 0), and compounded over 252 days
        return [[`${s.name}${s.role === 'live' ? ' (live)' : ' (shadow)'}`], E ? signed(E.daily, 3) : ['–', 'num'], E ? signed(E.yearly, 1) : ['–', 'num'], r ? signed(B.annualReturn, 1) : ['–', 'num'], [B.sharpe != null ? B.sharpe.toFixed(2) : '–', 'num'],
          [B.orders ? `${Math.round(100 * B.trades / B.orders)}%` : '–', 'num'], n ? [`${pct(rec, 2)} over ${n} day${n === 1 ? '' : 's'}`, `num ${rec >= 0 ? 'pos' : 'neg'}`] : ['from the next scan', 'num'],
          [r ? String((r.orders || []).length) : '–', 'num']];
      }));
  });
}
for (const b of document.querySelectorAll('.market-switch button')) b.addEventListener('click', () => showMarket(b.dataset.market));
showMarket(market);

// Your broker: a bridge on this computer (tools/ibkr/bridge.py, docker/ibkr) next to IB Gateway. The
// panel appears only when it answers; the orders and their prices come from it, never from the page.
// The bridge's address: on this computer by default, or a server's HTTPS address (set in the panel, or
// once with ?bridge=https://...), kept in this browser with the key.
const stored = (k, d = '') => { try { return localStorage.getItem(k) || d; } catch { return d; } };
const store = (k, v) => { try { localStorage.setItem(k, v); } catch { /* no storage */ } };
// ?bridge= is taken once and dropped from the address, so a bookmark or a reload with it does not
// overwrite an address saved later in the panel.
try {
  const u = new URL(location.href), q = u.searchParams.get('bridge');
  if (q) { store('bridgeUrl', q); u.searchParams.delete('bridge'); history.replaceState(null, '', u.pathname + u.search + u.hash); }
} catch { /* no URL */ }
const bridgeUrl = () => stored('bridgeUrl', 'http://127.0.0.1:8765').replace(/\/+$/, '');
const bridgeKey = () => stored('bridgeKey');
const bridge = (path, body) => fetch(bridgeUrl() + path, {
  method: body ? 'POST' : 'GET',
  headers: { 'X-Bridge-Key': bridgeKey(), ...(body ? { 'Content-Type': 'application/json' } : {}) },
  body: body ? JSON.stringify(body) : undefined,
}).then(async (r) => { const j = await r.json().catch(() => ({})); if (!r.ok && !j.log) throw new Error(j.error || `HTTP ${r.status}`); return j; });
const brokerMarket = () => MARKETS[market];
function brokerLog(text) { $('broker-log').textContent = text; }
function brokerOrders() {
  if (!bridgeKey()) { $('broker-orders').innerHTML = ''; brokerLog('Enter your bridge key to see the orders.'); return; }
  const m = brokerMarket();
  bridge(`/orders?market=${m}`).then((b) => {
    const px = (x) => (x == null ? 'none' : x.toFixed(m === 'UK' ? 2 : 2));
    table($('broker-orders'), [['Share'], ['Quantity', 'num'], ['Buy limit', 'num'], ['Stop-loss', 'num'], ['Take-profit', 'num']],
      b.orders.length ? b.orders.map((o) => [[o.ticker], [String(o.quantity), 'num'], [px(o.limit), 'num'], [px(o.stop), 'num neg'], [px(o.take), 'num pos']])
        : [{ gap: 'no orders for the next open' }]);
    brokerLog(`${m}: ${b.orders.length} orders for the open after the close of ${b.asOf}, ${b.capital} a trade${m === 'UK' ? ' (prices in pence)' : ''}.`);
    brokerHistory();
  }).catch((e) => { $('broker-orders').innerHTML = ''; brokerLog(`Orders not available: ${e.message}`); brokerHistory(); });
}
function brokerAct(path, body, question) {
  if (question && !window.confirm(question)) return;
  brokerLog('Working…');
  bridge(path, body).then((r) => { brokerLog(r.log || JSON.stringify(r)); brokerHistory(); }).catch((e) => brokerLog(`Failed: ${e.message}`));
}
// Looks for the bridge; the panel shows when it answers (or when opened by hand, to set the address).
function brokerProbe(opened) {
  $('broker-url').value = bridgeUrl();
  $('broker-key').value = bridgeKey();
  const ctl = new AbortController();
  setTimeout(() => ctl.abort(), 3000);
  fetch(`${bridgeUrl()}/status`, { signal: ctl.signal }).then((r) => r.json()).then((s) => {
    bridgeOk = true;
    $('results-empty').hidden = true;
    const auto = Object.entries(s.auto || {}).map(([m, a]) => `${m}: orders at ${a.place}, sold at ${a.close}`).join('; ');
    const last = (s.last || [])[0];
    $('broker-status').textContent = `${s.gateway ? 'Connected to Interactive Brokers' : 'IB Gateway is not answering: start it and log in'} · automatic ${auto || 'off'}`
      + (last ? ` · last action: ${last.market} ${last.action} ${last.when}, ${last.ok ? 'ok' : 'FAILED'}` : '');
    // The markets the bridge trades are the live ones; the others are research.
    const auto2 = new Set(Object.keys(s.auto || {}));
    if (auto2.size) try { localStorage.setItem(LIVE_KEY, JSON.stringify([...auto2])); } catch { /* not kept */ }
    if (auto2.size && [...auto2].join() !== [...live].join()) { live = auto2; if (shown) render(shown, MARKETS[market]); }
    brokerOrders();
    $('history-title').hidden = false;
    if (!$('view-results').hidden) todayLoop(true);
  }).catch(() => {
    bridgeOk = false;
    $('results-empty').hidden = false;
    if (!opened) return;
    $('broker-status').textContent = `No bridge answered at ${bridgeUrl()}. Set its address and key below, then Save.`;
    $('broker-orders').innerHTML = '';
  });
}
$('broker-save').addEventListener('click', () => {
  store('bridgeUrl', $('broker-url').value.trim() || 'http://127.0.0.1:8765');
  store('bridgeKey', $('broker-key').value.trim());
  brokerProbe(true);
});
$('broker-dry').addEventListener('click', () => brokerAct('/place', { market: brokerMarket(), dryRun: true }));
$('broker-place').addEventListener('click', () => brokerAct('/place', { market: brokerMarket() }, `Place the ${brokerMarket()} orders with Interactive Brokers?`));
// The bridge's checks (GET /diag): the token, the orders, IB Gateway, the account, the data, the
// schedule, the recordings; each passed, failed or a warning.
$('broker-diag').addEventListener('click', () => {
  brokerLog('Checking… (IB Gateway is asked too: up to a minute)');
  $('broker-checks').replaceChildren();
  bridge('/diag').then((r) => {
    const mark = (ok) => (ok === true ? ['passed', 'pos'] : ok === false ? ['failed', 'neg'] : ['warning', '']);
    table($('broker-checks'), [['Check'], ['Result'], ['Detail']], r.checks.map((c) => [[c.check], mark(c.ok), [c.detail]]));
    const bad = r.checks.filter((c) => c.ok === false).length, warn = r.checks.filter((c) => c.ok == null).length;
    brokerLog(bad ? `${bad} failed, ${warn} warnings.` : warn ? `All passed, ${warn} warnings.` : 'All passed.');
  }).catch((e) => brokerLog(`Diagnostics failed: ${e.message} (an older bridge has none: redeploy it)`));
});
// Today at IBKR (GET /trades): every one of today's orders and every holding of the model's shares,
// live, with what was bought and sold and the profit; refreshed every minute while Results is open.
let todayTimer = null;
function brokerToday() {
  if (!bridgeKey()) return;
  const m = brokerMarket();
  $('today-box').hidden = false;
  $('today-summary').textContent = 'Asking Interactive Brokers…';
  bridge(`/trades?market=${m}`).then((t) => {
    if (brokerMarket() !== m) return;
    // The bridge reads IBKR in the background and answers at once: before its first read, a note.
    if (!t.at) { $('today-summary').textContent = t.error || 'Being read from Interactive Brokers: in a minute.'; setTimeout(() => { if (brokerMarket() === m) brokerToday(); }, 20000); return; }
    const px = (x) => (x == null ? '' : x >= 100 ? x.toFixed(1) : x.toFixed(x >= 10 ? 2 : 3));
    const money = (x) => `${x >= 0 ? '+' : ''}${x.toFixed(2)}`;
    const cls = (x) => (x > 0 ? 'num pos' : x < 0 ? 'num neg' : 'num');
    const main = (t.shares.find((r) => r.bought || r.held) || {}).currency || '';
    const rows = t.shares.map((r) => {
      const profit = (r.pnl || 0) + (r.open || 0);
      const counted = r.bought || r.held;
      const qty = r.bought || r.held;
      const trade = counted
        ? `${qty}${r.ordered && r.bought && r.bought < r.ordered ? ` of ${r.ordered}` : ''} at ${px(r.buy)}\n${r.held ? (r.price != null ? `now ${px(r.price)}` : 'held') : `sold at ${px(r.sell)}`}`
        : r.limit != null ? `buy at ${px(r.limit)}` : '';
      const tone = /refused|earlier/.test(r.state) ? 'neg' : /held/.test(r.state) ? 'warn' : '';
      return [[`${r.ticker}\n${r.state}`, `share ${tone}`], [trade, 'num'],
        counted ? [`${money(profit)}${r.currency !== main ? ` ${r.currency}` : ''}`, cls(profit)] : ['', 'num']];
    });
    table($('today'), [['Share'], ['Trade', 'num'], [`Profit${main ? ` (${main})` : ''}`, 'num']],
      rows.length ? rows : [{ gap: 'No orders today and nothing held.' }]);
    const n = (f) => t.shares.filter(f).length;
    const tot = {};
    for (const r of t.shares) if (r.bought || r.held) tot[r.currency] = (tot[r.currency] || 0) + (r.pnl || 0) + (r.open || 0);
    const held = n((r) => r.held > 0), old = n((r) => r.held > 0 && !r.bought);
    const parts = [`${n((r) => r.bought > 0)} bought`, `${held} held now${old ? ` (${old} from an earlier day)` : ''}`,
      `${n((r) => /^sold/.test(r.state))} sold`, `${n((r) => r.state === 'waiting to buy')} waiting to buy`];
    const bad = n((r) => r.state === 'refused by IBKR');
    if (bad) parts.push(`${bad} refused`);
    const totals = Object.entries(tot).map(([c, v]) => `${money(v)} ${c}`).join(', ');
    $('today-summary').innerHTML = `<b>${parts.join(' · ')}</b>${totals ? ` · profit today ${totals}` : ''} · ${t.at.slice(11, 16)}${t.age > 90 ? ` (${Math.round(t.age / 60)} min ago)` : ''}${t.note ? ` · last read failed: ${t.note}` : ''}`;
  }).catch((e) => { if (brokerMarket() === m) $('today-summary').textContent = `Not available now: ${e.message}`; });
}
$('today-refresh').addEventListener('click', brokerToday);
function todayLoop(on) {
  clearInterval(todayTimer);
  todayTimer = on ? setInterval(() => { if (!document.hidden) brokerToday(); }, 60000) : null;
  if (on) brokerToday();
}
// The history kept by the bridge: each day's orders placed and what was traded (fills from IBKR).
function brokerHistory() {
  if (!bridgeKey()) return;
  bridge('/history').then((h) => {
    const code = (x) => x.split(':').pop().replace(' ', '.');  // EXCHANGE:CODE in multi-exchange markets; IBKR's "BT A"
    // A broker's price against the model's (EODData's): LSE prices may be in pence on one side and pounds on the other.
    const ratio = (a, b) => { const r = a / b; const s = [1, 0.01, 100].reduce((p, x) => (Math.abs(Math.log(r / x)) < Math.abs(Math.log(r / p)) ? x : p), 1); return r / s; };
    let n = 0, wins = 0, pnl = {}, placedN = 0, checked = 0, agree = 0, gapN = 0, gapS = 0, retN = 0, retA = 0, retM = 0;
    // Today is in "Today at Interactive Brokers" (live); the days before, one fold each, newest first.
    const today = new Date().toLocaleDateString('en-CA');
    const perDay = {};
    for (const d of h.days.filter((x) => x.market === brokerMarket() && x.date && x.date !== today)) {
      const day = (perDay[d.date] = perDay[d.date] || { rows: [], bought: 0, orders: 0, pnl: {} });
      day.orders += d.orders.length;
      const traded = Object.fromEntries(d.trades.map((t) => [code(t.ticker), t]));
      const model = Object.fromEntries((d.model || []).map((t) => [code(t.ticker), t]));
      const reported = d.trades.length > 0 || (d.model || []).length > 0;
      placedN += d.orders.length;
      const names = new Set([...d.orders.map((o) => code(o.ticker)), ...Object.keys(traded)]);
      for (const name of names) {
        const o = d.orders.find((x) => code(x.ticker) === name) || {};
        const t = traded[name], mo = model[name];
        const filled = t && t.bought > 0;
        if (filled) {
          n += 1;
          if (t.ret > 0) wins += 1;
          pnl[t.currency] = (pnl[t.currency] || 0) + t.pnl;  // IBKR's realised P&L, after its commissions
          day.bought += 1, day.pnl[t.currency] = (day.pnl[t.currency] || 0) + t.pnl;
        }
        // The comparison: did both fill, the entry's gap (positive: paid more than the model assumed), the returns.
        let gap = null;
        if (mo && d.trades.length) {
          checked += 1;
          if ((mo.filled > 0) === Boolean(filled)) agree += 1;
          if (filled && mo.filled > 0) {
            gap = ratio(t.buy, mo.entry) - 1, gapN += 1, gapS += gap;
            if (t.ret != null) retN += 1, retA += t.ret, retM += mo.ret;
          }
        }
        const modelSays = mo ? (mo.filled > 0 ? pct(mo.ret, 2) : 'no buy') : '';
        const yours = filled && t.ret != null ? pct(t.ret, 2) : (filled ? 'open' : (reported ? 'no buy' : 'waiting'));
        day.rows.push([[name],
          [`${yours}${modelSays ? ` (model ${modelSays})` : ''}`, `num ${filled && t.ret != null ? (t.ret >= 0 ? 'pos' : 'neg') : ''}`],
          gap != null ? [pct(gap, 2), `num ${gap <= 0 ? 'pos' : 'neg'}`] : ['', 'num'],
          filled ? [`${t.pnl.toFixed(2)} (fees ${t.commission.toFixed(2)})`, `num ${t.pnl > 0 ? 'pos' : t.pnl < 0 ? 'neg' : ''}`] : ['', 'num']]);
      }
    }
    const box = $('broker-history');
    box.replaceChildren();
    const dates = Object.keys(perDay).sort().reverse();
    if (!dates.length) box.append(cell('p', 'Nothing yet: each trading day appears here once it is over.', 'muted'));
    for (const date of dates) {
      const day = perDay[date];
      const money = Object.entries(day.pnl).map(([c, v]) => `${v >= 0 ? '+' : ''}${v.toFixed(2)} ${c}`).join(', ');
      const total = Object.values(day.pnl).reduce((a, b) => a + b, 0);
      const fold = document.createElement('details');
      fold.className = 'day';
      const head = document.createElement('summary');
      const when = new Date(`${date}T12:00:00Z`).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' });
      head.append(cell('span', when, 'day-date'), cell('span', `${day.orders} orders · ${day.bought} bought`, 'day-count'),
        cell('span', money || '–', `day-pnl ${total > 0 ? 'pos' : total < 0 ? 'neg' : ''}`));
      const wrap = document.createElement('div');
      wrap.className = 'table-scroll';
      const t = document.createElement('table');
      wrap.append(t);
      table(t, [['Share'], ['Return (model)', 'num'], ['Paid vs model', 'num'], ['Profit (fees)', 'num']], day.rows);
      fold.append(head, wrap);
      box.append(fold);
    }
    const money = Object.entries(pnl).map(([c, v]) => `${v >= 0 ? '+' : ''}${v.toFixed(2)} ${c}`).join(', ');
    const vs = checked
      ? ` · against the model: fills agree on ${agree} of ${checked}` + (gapN ? `, entry ${pct(gapS / gapN, 2)} vs the model's on average` : '')
        + (retN ? `, return ${pct(retA / retN, 2)} vs the model's ${pct(retM / retN, 2)} a trade` : '')
      : '';
    const days = new Set(h.days.filter((x) => x.market === brokerMarket()).map((x) => x.date)).size;
    $('broker-summary').textContent = days
      ? `${days} day${days === 1 ? '' : 's'} · ${placedN} orders · ${n} bought (${placedN ? Math.round(100 * n / placedN) : 0}%) · ${n ? Math.round(100 * wins / n) : 0}% of them gained · total ${money || '0'}${vs}`
      : '';
  }).catch(() => { /* an older bridge without history */ });
}
$('broker-report').addEventListener('click', () => brokerAct('/report', { market: brokerMarket() }));
$('broker-close').addEventListener('click', () => brokerAct('/close', { market: brokerMarket() }, `Cancel today's unfilled ${brokerMarket()} orders and sell what is left at the close?`));
for (const b of document.querySelectorAll('.market-switch button')) b.addEventListener('click', () => { if (bridgeOk) { brokerOrders(); if (!$('view-results').hidden) brokerToday(); } });
brokerProbe();
let firstView = location.hash.slice(1);
try { firstView = firstView || localStorage.getItem('view') || 'orders'; } catch { /* no storage */ }
showView(firstView);

