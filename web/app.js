// The page: loads the C++ library (WebAssembly), runs it on the chosen engine and shows the
// daily trades it recommends. Rendering only; every number comes from the library.
import createOfm from './wasm/ofm.js';
import { runKernels } from './gpu.js';

const $ = (id) => document.getElementById(id);
const status = (t) => { $('status').textContent = t; };
const pct = (x, d = 1) => (x == null ? '–' : `${x >= 0 ? '+' : ''}${(100 * x).toFixed(d)}%`);
const yieldToPaint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const MODEL_COSTS = [10, 10, 1];  // the model portfolio's costs (bp); not shown, the forecasts do not depend on them
const SERIES = [['daily trades', '--daybook'], ['market (bought and held)', '--market']];

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
    let session = new ofm.Session();
    status('Reading the data…');
    await yieldToPaint();
    await load(session);
    const engine = $('engine').value;
    let json, note = '';
    const t0 = performance.now();
    if (engine === 'emulated' || engine === 'reference') {
      status(`Running on the ${engine === 'emulated' ? 'emulated GPU' : 'reference engine'}…`);
      await yieldToPaint();
      json = session.runCpu(engine, ...MODEL_COSTS);
    } else {
      try {
        status('Running the kernels on WebGPU…');
        const g = await runKernels(session, (p) => status(`Running the kernels on WebGPU… ${Math.round(100 * p)}%`), engine === 'gpu-cpu' ? 'software' : 'auto');
        json = session.finishGpu(...MODEL_COSTS, g.ms, g.adapter);
      } catch (e) {
        if (engine === 'gpu' || engine === 'gpu-cpu') throw e;
        session.delete();
        session = new ofm.Session();
        await load(session);
        status('WebGPU unavailable; running the emulated GPU…');
        await yieldToPaint();
        json = session.runCpu('emulated', ...MODEL_COSTS);
        note = ` · WebGPU unavailable (${e.message}); the same kernels ran on the CPU`;
      }
    }
    session.delete();
    const r = JSON.parse(json);
    status(`Done in ${Math.round(performance.now() - t0).toLocaleString('en-GB')} ms · ${r.engine}${note}`);
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
  $('asof').textContent = `${where ? `${where}, ${r.stocks} most traded shares` : `${source}, ${r.stocks} stocks`} · close of ${r.asOf} · ${n} trade${n === 1 ? '' : 's'} for the next open · costs ${(D.buyBps / 100).toFixed(2)}% per side`;
  const level = (x) => (x == null ? 'none' : pct(x, 2));
  table($('plan'), [['Buy at the open'], ['Capital', 'num'], ['Expected return, after costs', 'num'], ['Stop-loss', 'num'], ['Take-profit', 'num']],
    n ? D.orders.map((o) => [[o.ticker], [o.weight == null ? '–' : `${(100 * o.weight).toFixed(1)}%`, 'num'], signed(o.tradeReturn, 2), [level(o.stop), o.stop == null ? 'num' : 'num neg'], [level(o.take), o.take == null ? 'num' : 'num pos']])
      : [{ gap: D.backtestLoses ? 'no trades: the daily trades lost money in this market\'s backtest; stay in cash' : 'no trades: none is expected to make money after costs; stay in cash' }]);

  const whole = r.parts[0].rows, b = whole.find((x) => x.name === D.name), k = whole.find((x) => x.name.startsWith('market'));
  const half = (i) => r.parts[i].rows.find((x) => x.name === D.name);
  $('verdict').innerHTML = `
    <div class="tile"><span class="k">Daily trades, ${r.parts[0].from} – ${r.parts[0].to}</span><span class="v">${pct(b.annualReturn)}</span><span class="s">a year · Sharpe ${b.sharpe.toFixed(2)} · max drawdown ${pct(b.maxDrawdown, 0).slice(1)} · halves ${pct(half(1).annualReturn)} and ${pct(half(2).annualReturn)}</span></div>
    <div class="tile"><span class="k">Market, bought and held</span><span class="v">${pct(k.annualReturn)}</span><span class="s">a year · Sharpe ${k.sharpe.toFixed(2)} · max drawdown ${pct(k.maxDrawdown, 0).slice(1)}</span></div>`;

  const recent = r.recent || [];
  table($('recent'), [['Day'], ['Daily trades', 'num'], ['Market', 'num']],
    recent.slice().reverse().map((d) => [[d.date], d.day == null ? ['–', 'num'] : signed(d.day), signed(d.market)]));
  drawChart(r);
}

function drawChart(r) {
  const css = getComputedStyle(document.documentElement);
  drawCurves($('chart'), $('legend'), r.dates, SERIES.filter(([name]) => r.curves[name]).map(([name, color]) => ({ name, color: css.getPropertyValue(color).trim(), v: r.curves[name], bold: name === 'daily trades' })));
}

function drawCurves(canvas, legend, dates, curves) {
  const css = getComputedStyle(document.documentElement);
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr, canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const all = curves.flatMap((c) => c.v);
  const lo = Math.log(Math.min(...all)), hi = Math.log(Math.max(...all));
  const pad = { l: 46, r: 10, t: 10, b: 24 };
  const X = (i) => pad.l + (i / (dates.length - 1)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (Math.log(v) - lo) / Math.max(hi - lo, 1e-9)) * (h - pad.t - pad.b);
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.fillStyle = css.getPropertyValue('--muted').trim();
  ctx.strokeStyle = css.getPropertyValue('--rule').trim();
  ctx.lineWidth = 1;
  const ticks = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64].filter((v) => Math.log(v) >= lo - 1e-9 && Math.log(v) <= hi + 1e-9);
  ticks.forEach((v) => {
    const y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(`${v}×`, 4, y + 4);
  });
  const years = [];
  dates.forEach((d, i) => { if (i === 0 || d.slice(0, 4) !== dates[i - 1].slice(0, 4)) years.push([i, d.slice(0, 4)]); });
  const every = Math.ceil(years.length / Math.max(1, Math.floor((w - pad.l) / 60)));
  years.forEach(([i, y], k) => { if (k % every === 0) ctx.fillText(y, X(i) - 12, h - 6); });
  curves.forEach((c) => {
    ctx.strokeStyle = c.color;
    ctx.lineWidth = c.bold ? 2 : 1.5;
    ctx.beginPath();
    c.v.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v))));
    ctx.stroke();
  });
  legend.innerHTML = curves.map((c) => `<span><i style="background:${c.color}"></i>${c.name} ${c.v[c.v.length - 1].toFixed(2)}×</span>`).join('');
}


$('src-files').addEventListener('change', () => { $('files').hidden = false; });
$('src-sample').addEventListener('change', () => { $('files').hidden = true; });
$('run').addEventListener('click', run);
window.addEventListener('resize', () => { if (shown && shown[0].day) drawChart(shown[0]); });

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
  $('plan').innerHTML = '', $('verdict').innerHTML = '', $('recent').innerHTML = '', $('legend').innerHTML = '';
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
