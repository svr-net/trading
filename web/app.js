// The page: the latest scan of each market, published by the scheduled workflow (results only, no
// prices): the next open's trades and the model's walk-forward backtest. Rendering only; every
// number comes from the model (apps/ofm_research), the chart laid out there too.
const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${(100 * x).toFixed(d)}%`);
const SERIES = [['daily trades', '--daybook'], ['market (bought and held)', '--market']];

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
  const cur = c.market === 'UK' ? '£' : '$';
  return `after ${c.broker} costs (${cur}${c.capital.toLocaleString('en-GB')} a trade${c.stamp ? `, ${(100 * c.stamp).toFixed(1)}% stamp duty on buying` : ''})`;
};

// The next open's trades (a buy limit from the last close; sold at the stop-loss, the take-profit or the
// close), and the backtest.
function render(r, source) {
  shown = r;
  const where = { UK: 'UK (London Stock Exchange)', US: 'US (NYSE and NASDAQ)' }[source] || source;
  const n = r.orders.length;
  $('asof').textContent = `${where}, ${r.stocks} most traded shares · close of ${r.asOf} · ${n} trade${n === 1 ? '' : 's'} for the next open · ${costs(r)}`;
  const level = (x) => (x == null ? 'none' : pct(x, 2));
  const levels = r.orders.some((o) => o.stop != null || o.take != null);  // without: every trade sold at the close
  const head = [['Share'], ['Buy limit (vs last close)', 'num'], ['Expected return', 'num']];
  if (levels) head.push(['Stop-loss', 'num'], ['Take-profit', 'num']);
  table($('plan'), head,
    n ? r.orders.map((o) => {
      const row = [[o.ticker], [o.limit == null ? 'at the open' : pct(o.limit, 2), 'num'], signed(o.expected, 3)];
      if (levels) row.push([level(o.stop), o.stop == null ? 'num' : 'num neg'], [level(o.take), o.take == null ? 'num' : 'num pos']);
      return row;
    }) : [{ gap: 'no trades: none is expected to gain; stay in cash' }]);
  const B = r.backtest, M = r.market;
  $('verdict').innerHTML = `
    <div class="tile"><span class="k">Daily trades, ${B.from} – ${B.to}</span><span class="v">${pct(B.annualReturn)}</span><span class="s">a year · Sharpe ${B.sharpe.toFixed(2)} · max drawdown ${pct(B.maxDrawdown, 0).slice(1)}</span></div>
    <div class="tile"><span class="k">Market, bought and held</span><span class="v">${pct(M.annualReturn)}</span><span class="s">a year · Sharpe ${M.sharpe.toFixed(2)} · max drawdown ${pct(M.maxDrawdown, 0).slice(1)}</span></div>`;
  table($('recent'), [['Day'], ['Daily trades', 'num'], ['Market', 'num']],
    r.recent.slice().reverse().map((d) => [[d.date], signed(d.day), signed(d.market)]));
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
  C.series.forEach((s) => {
    ctx.strokeStyle = color[s.name] || ctx.strokeStyle;
    ctx.lineWidth = s.name.startsWith('market') ? 1.5 : 2;
    ctx.beginPath();
    s.y.forEach((y, i) => (i ? ctx.lineTo(X(i / (s.y.length - 1)), Y(y)) : ctx.moveTo(X(0), Y(y))));
    ctx.stroke();
  });
  legend.innerHTML = C.series.map((s) => `<span><i style="background:${color[s.name]}"></i>${s.name} ${s.final}</span>`).join('');
}
window.addEventListener('resize', () => { if (shown) drawChart(shown); });

const MARKETS = { uk: 'UK', us: 'US' };
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
}
for (const b of document.querySelectorAll('.market-switch button')) b.addEventListener('click', () => showMarket(b.dataset.market));
showMarket(market);
