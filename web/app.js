// The page: loads the C++ library (WebAssembly), runs it on the chosen engine and shows the
// daily trades it recommends. Rendering only; every number comes from the library.
import createOfm from './wasm/ofm.js';
import { runKernels } from './gpu.js';

const $ = (id) => document.getElementById(id);
const status = (t) => { $('status').textContent = t; };
const pct = (x, d = 1) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${(100 * x).toFixed(d)}%`);
const yieldToPaint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const MODEL_COSTS = [10, 10, 1];  // the model portfolio's costs (bp); not shown, the forecasts do not depend on them
const SERIES = [['daily trades', '--daybook'], ['intraday strategy', '--daybook'], ['market (bought and held)', '--market']];

let ofm = null;
let shown = null;  // [report, source]

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

async function readFile(input) {
  const f = input.files && input.files[0];
  return f ? f.text() : '';
}

async function load(session) {
  if ($('src-files').checked) {
    const stocks = await readFile($('stocks-file'));
    if (!stocks) throw new Error('choose a stocks CSV file first');
    session.loadCsv(stocks, await readFile($('series-file')));
  } else {
    session.loadSample();
  }
  session.setDayCost(100 * (+$('cost-pct').value || 0));
}

async function run() {
  $('run').disabled = true;
  try {
    const session = new ofm.Session();
    status('Reading the data…');
    await yieldToPaint();
    await load(session);
    const t0 = performance.now();
    status('Running the kernels on WebGPU…');
    const g = await runKernels(session, (p) => status(`Running the kernels on WebGPU… ${Math.round(100 * p)}%`));
    const json = session.finishGpu(...MODEL_COSTS, g.ms, g.adapter);
    session.delete();
    const r = JSON.parse(json);
    status(`Done in ${Math.round(performance.now() - t0).toLocaleString('en-GB')} ms · ${r.engine}`);
    render(r, $('src-files').checked ? 'Your data' : 'Sample market (synthetic)');
  } catch (e) {
    status(`Could not run: ${e.message}`);
  } finally {
    $('run').disabled = false;
  }
}

// Up to 10 day trades for the next open: buy at the open, sell at the stop, the take-profit or the close.
function render(r, source = 'UK') {
  shown = [r, source];
  const D = r.day;
  if (!D) { $('asof').textContent = 'This scan has no daily trades yet; they appear after the next scheduled run.'; return; }
  const n = D.orders.length;
  const where = { UK: 'UK (London Stock Exchange)', US: 'US (NYSE and NASDAQ)' }[source];
  const among = D.universe ? (D.universeFrom ? `shares ranked ${D.universeFrom + 1}–${D.universeFrom + D.universe} by traded value` : `${D.universe} most traded shares`) : `${r.stocks} most traded shares`;
  $('asof').textContent = `${where ? `${where}, ${among}` : `${source}, ${r.stocks} stocks`} · close of ${r.asOf} · ${n} trade${n === 1 ? '' : 's'} for the next open · costs ${(D.buyBps / 100).toFixed(2)}% per side`;
  const level = (x) => (x == null ? 'none' : pct(x, 2));
  table($('plan'), [['Buy at the open'], ['Expected return', 'num'], ['Stop-loss', 'num'], ['Take-profit', 'num']],
    n ? D.orders.map((o) => [[o.ticker], signed(o.tradeReturn, 3), [level(o.stop), o.stop == null ? 'num' : 'num neg'], [level(o.take), o.take == null ? 'num' : 'num pos']])
      : [{ gap: D.backtestLoses ? 'no trades: the daily trades\' backtest gain in this market is not significant (t-statistic below 2); stay in cash' : 'no trades: none is expected to make money after costs; stay in cash' }]);

  // A market whose daily trades have no edge but whose intraday strategy has (US): the intraday
  // strategy leads, with its own chart and days.
  if (r.intraday && D.backtestLoses) { renderIntraday(r.intraday, true); return; }
  const whole = r.parts[0].rows, b = whole.find((x) => x.name === D.name), k = whole.find((x) => x.name.startsWith('market'));
  const half = (i) => r.parts[i].rows.find((x) => x.name === D.name);
  $('verdict').innerHTML = `
    <div class="tile"><span class="k">Daily trades, ${r.parts[0].from} – ${r.parts[0].to}</span><span class="v">${pct(b.annualReturn)}</span><span class="s">a year · Sharpe ${b.sharpe.toFixed(2)} · max drawdown ${pct(b.maxDrawdown, 0).slice(1)} · halves ${pct(half(1).annualReturn)} and ${pct(half(2).annualReturn)}</span></div>
    <div class="tile"><span class="k">Market, bought and held</span><span class="v">${pct(k.annualReturn)}</span><span class="s">a year · Sharpe ${k.sharpe.toFixed(2)} · max drawdown ${pct(k.maxDrawdown, 0).slice(1)}</span></div>`;

  const recent = r.recent || [];
  table($('recent'), [['Day'], ['Daily trades', 'num'], ['Market', 'num']],
    recent.slice().reverse().map((d) => [[d.date], d.day == null ? ['–', 'num'] : signed(d.day), signed(d.market)]));
  drawChart(r);
  renderIntraday(r.intraday);
}

// The US intraday strategy's backtest (r.intraday, from the library). lead: it takes the backtest
// section (tiles, chart, last days) in place of the daily trades.
function renderIntraday(I, lead = false) {
  $('intraday').hidden = !I;
  if (!I) return;
  const W = I.whole, [h1, h2] = I.halves;
  $('intra-asof').textContent = `${W.from} – ${W.to} · ${I.tradesPerDay.toFixed(1)} trades a day · ${pct(I.stopShare, 0)} of trades stopped, ${pct(I.takeShare, 0)} took profit · entry in use: ${I.entryMinutes ? `${I.entryMinutes} minutes after the open` : 'cash'}`;
  $('intra-verdict').innerHTML = `
    <div class="tile"><span class="k">Intraday strategy</span><span class="v">${pct(W.annualReturn)}</span><span class="s">a year · Sharpe ${W.sharpe.toFixed(2)} · max drawdown ${pct(W.maxDrawdown, 0).slice(1)} · halves ${pct(h1.annualReturn)} and ${pct(h2.annualReturn)}</span></div>
    <div class="tile"><span class="k">Fill test</span><span class="v">${pct(I.fillOptimistic)}</span><span class="s">a year if a bar that touched both levels took the profit first (above: the stop first)</span></div>`;
  table($('intra-years'), [['Year'], ['Return', 'num'], ['Days', 'num']], I.years.map((y) => [[y.year], signed(y.ret, 1), [String(y.days), 'num']]));
  if (!lead) return;
  table($('plan'), [['Buy after the open'], ['Expected return', 'num'], ['Stop-loss', 'num'], ['Take-profit', 'num']],
    [{ gap: 'The daily trades have no edge among these shares; the intraday strategy below chooses its trades 5 to 40 minutes after the open, from the day\'s 100 most traded.' }]);
  $('verdict').innerHTML = `
    <div class="tile"><span class="k">Intraday strategy, ${W.from} – ${W.to}</span><span class="v">${pct(W.annualReturn)}</span><span class="s">a year · Sharpe ${W.sharpe.toFixed(2)} · max drawdown ${pct(W.maxDrawdown, 0).slice(1)} · halves ${pct(h1.annualReturn)} and ${pct(h2.annualReturn)}</span></div>
    <div class="tile"><span class="k">The same shares, bought and held</span><span class="v">${pct(I.market.annualReturn)}</span><span class="s">a year · Sharpe ${I.market.sharpe.toFixed(2)} · max drawdown ${pct(I.market.maxDrawdown, 0).slice(1)}</span></div>`;
  table($('recent'), [['Day'], ['Intraday strategy', 'num'], ['Market', 'num']],
    I.recent.slice().reverse().map((d) => [[d.date], signed(d.day), signed(d.market)]));
  drawChart({ chart: I.chart });
}

// The chart as laid out by the library (r.chart: heights and positions as shares of the chart);
// the page only places them on the canvas.
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

$('src-files').addEventListener('change', () => { $('files').hidden = false; });
$('src-sample').addEventListener('change', () => { $('files').hidden = true; });
$('run').addEventListener('click', run);
window.addEventListener('resize', () => { if (shown && shown[0].day) render(...shown); });

// The latest scans, published by the scheduled workflow (results only, no prices): one per market.
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
  $('plan').innerHTML = '', $('verdict').innerHTML = '', $('recent').innerHTML = '', $('legend').innerHTML = '', $('intraday').hidden = true;
  fetch(`data/${m}.json`, { cache: 'no-cache' })
    .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
    .then((r) => { if (market === m) render(r, name); })
    .catch(() => { if (market === m) $('asof').textContent = `No ${name} scan has been published yet; it appears after the next scheduled run.`; });
}
for (const b of document.querySelectorAll('.market-switch button')) b.addEventListener('click', () => showMarket(b.dataset.market));
showMarket(market);

createOfm().then((m) => {
  ofm = m;
  status(`Model ${m.version()} ready. Choose the data and run.`);
  if (new URLSearchParams(location.search).has('autorun') || location.hash === '#run') run();
}).catch((e) => status(`Could not load the model: ${e.message}`));
