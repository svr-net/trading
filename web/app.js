// The page: loads the C++ library (WebAssembly), runs it on the chosen engine and shows the
// report it returns. Rendering only; every number comes from the library.
import createOfm from './wasm/ofm.js';
import { runKernels } from './gpu.js';

const $ = (id) => document.getElementById(id);
const status = (t) => { $('status').textContent = t; };
const pct = (x, d = 1) => (x == null ? '–' : `${(100 * x).toFixed(d)}%`);
const bp = (x, d = 1) => (x == null ? '–' : `${(1e4 * x >= 0 ? '+' : '')}${(1e4 * x).toFixed(d)}`);
const yieldToPaint = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
const SERIES = [['model', '--model'], ['model + futures hedge', '--hedged'], ['market (bought and held)', '--market']];

let ofm = null;
let last = null;
let showAll = false;
let scan = null;

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
    r.forEach((c) => (c instanceof Node ? tr.append(c) : tr.append(cell('td', c[0], c[1]))));
  });
}

async function readFile(input) {
  const f = input.files && input.files[0];
  return f ? f.text() : '';
}

async function run() {
  $('run').disabled = true;
  try {
    const session = new ofm.Session();
    status('Reading the data…');
    await yieldToPaint();
    if ($('src-files').checked) {
      const stocks = await readFile($('stocks-file'));
      if (!stocks) throw new Error('choose a stocks CSV file first');
      session.loadCsv(stocks, await readFile($('series-file')));
    } else {
      session.loadSample();
    }
    const dealing = +$('cost-pct').value || 0, stamp = +$('stamp-pct').value || 0;
    const costs = [100 * (dealing + stamp), 100 * dealing, +$('fut-bps').value || 0];
    session.setUniverseHedge($('universe-hedge').checked);
    const engine = $('engine').value;
    let json, note;
    const t0 = performance.now();
    const cpu = async (which, why) => {
      status(`Running on the ${which === 'emulated' ? 'emulated GPU' : 'reference engine'}…`);
      await yieldToPaint();
      json = session.runCpu(which, ...costs);
      note = why;
    };
    if (engine === 'emulated' || engine === 'reference') {
      await cpu(engine, '');
    } else {
      try {
        status('Running the kernels on WebGPU…');
        const g = await runKernels(session, (p) => status(`Running the kernels on WebGPU… ${Math.round(100 * p)}%`));
        json = session.finishGpu(...costs, g.ms, g.adapter);
        note = '';
      } catch (e) {
        if (engine === 'gpu') throw e;
        session.delete();
        const fresh = new ofm.Session();
        if ($('src-files').checked) fresh.loadCsv(await readFile($('stocks-file')), await readFile($('series-file')));
        else fresh.loadSample();
        fresh.setUniverseHedge($('universe-hedge').checked);
        status('WebGPU unavailable; running the emulated GPU…');
        await yieldToPaint();
        json = fresh.runCpu('emulated', ...costs);
        note = `WebGPU unavailable (${e.message}); the same kernels ran on the CPU.`;
        fresh.delete();
      }
    }
    last = JSON.parse(json);
    render(last, note, performance.now() - t0);
  } catch (e) {
    status(`Could not run: ${e.message}`);
  } finally {
    $('run').disabled = false;
  }
}

function render(r, note, ms) {
  $('results').hidden = false;
  if (ms > 0) status(`Done in ${Math.round(ms).toLocaleString('en-GB')} ms · ${r.engine}${note ? ' · ' + note : ''}`);
  const whole = r.parts[0].rows;
  const m = whole.find((x) => x.name === 'model'), k = whole.find((x) => x.name.startsWith('market'));
  const h = whole.find((x) => x.name === 'model + futures hedge');
  const pass = (ok) => `<span class="pill ${ok ? 'pass' : 'fail'}">${ok ? 'passes' : 'fails'}</span>`;
  $('verdict').innerHTML = `
    <div class="tile"><span class="k">Model, ${r.parts[0].from} – ${r.parts[0].to}</span><span class="v">${m.sharpe.toFixed(2)}</span><span class="s">Sharpe ratio, ${pct(m.annualReturn)} a year · ${pass(r.pass)}</span></div>
    <div class="tile"><span class="k">Market, bought and held</span><span class="v">${k.sharpe.toFixed(2)}</span><span class="s">Sharpe ratio, ${pct(k.annualReturn)} a year</span></div>
    <div class="tile"><span class="k">Model + futures hedge</span><span class="v">${h.sharpe.toFixed(2)}</span><span class="s">${r.hedgeSeries ? `hedged with ${r.hedgeSeries}, on average ${pct(r.trades.hedgeShare, 0)} of the book` : 'no futures series'} · ${pass(r.passHedged)}</span></div>`;

  // Expected returns.
  $('expected-note').textContent = `As of the close on ${r.asOf}: expected return over the market for the next day, and over a forecast's expected life of ${r.life.toFixed(1)} days. A stock is bought when that beats the round trip of ${(r.costs.buyBps + r.costs.sellBps).toFixed(0)} bp, and sold when it lags by more.`;
  const ex = r.expected;
  const maxAbs = Math.max(...ex.map((x) => Math.abs(x.overLife)), 1e-12);
  const row = (x) => {
    const bar = document.createElement('td');
    bar.className = 'bar';
    const box = document.createElement('div');
    box.className = 'barbox';
    const s = document.createElement('span');
    s.className = x.overLife >= 0 ? 'up' : 'down';
    s.style.width = `${(50 * Math.abs(x.overLife)) / maxAbs}%`;
    box.append(s);
    bar.append(box);
    return [[x.ticker], [bp(x.daily, 2), `num ${x.daily >= 0 ? 'pos' : 'neg'}`], [bp(x.overLife), `num ${x.overLife >= 0 ? 'pos' : 'neg'}`], bar,
      [x.held ? 'held' : ''], [x.order > 0 ? 'buy at next open' : x.order < 0 ? 'sell at next open' : '']];
  };
  const n = 15;
  const rows = showAll || ex.length <= 2 * n ? ex.map(row) : [...ex.slice(0, n).map(row), { gap: `${ex.length - 2 * n} more stocks` }, ...ex.slice(-n).map(row)];
  table($('expected'), [['Stock'], ['Next day (bp)', 'num'], ['Over life (bp)', 'num'], [''], ['Position'], ['Order']], rows);
  $('expected-all').textContent = showAll ? 'Show the top and bottom 15' : `Show all ${ex.length} stocks`;
  $('expected-all').hidden = ex.length <= 2 * n;

  // Backtest.
  drawChart(r);
  const partRows = [];
  r.parts.forEach((p) => {
    partRows.push({ gap: `${p.name}: ${p.from} – ${p.to}` });
    p.rows.forEach((x) => partRows.push([[x.name], [pct(x.annualReturn), 'num'], [pct(x.volatility), 'num'], [x.sharpe.toFixed(2), 'num'], [pct(x.maxDrawdown), 'num']]));
  });
  table($('parts'), [['Strategy'], ['Return / yr', 'num'], ['Volatility', 'num'], ['Sharpe', 'num'], ['Max drawdown', 'num']], partRows);
  const t = r.trades;
  $('trades').textContent = `${t.count} closed trades: ${t.wins} gained (mean ${pct(t.meanWin, 2)}), ${t.losses} lost (mean ${pct(t.meanLoss, 2)}). Turnover ${t.turnoverPerYear.toFixed(1)}× a year; ${t.meanHoldings.toFixed(0)} stocks held on average.`;
  table($('years'), [['Year'], ['Model', 'num'], ['Hedged', 'num'], ['Market', 'num']],
    r.years.map((y) => [[y.year], ...y.returns.map((v) => [pct(v), `num ${v >= 0 ? 'pos' : 'neg'}`])]));
  const used = r.factors.filter((f) => f.premium !== 0 && f.premium != null).sort((a, b) => Math.abs(b.t) - Math.abs(a.t));
  table($('factors'), [['Signal'], ['t', 'num'], ['Premium (bp/day)', 'num']],
    used.length ? used.map((f) => [[f.name], [f.t.toFixed(2), 'num'], [bp(f.premium, 2), `num ${f.premium >= 0 ? 'pos' : 'neg'}`]]) : [{ gap: 'no factor has enough evidence yet' }]);
  renderStructure(r);
  $('facts').textContent = `${r.stocks} stocks, ${r.days} days (${r.from} – ${r.to}); horizons ${r.horizons.join(', ')} days; ${r.records} daily factor records; ${r.engine}.`;
}

function renderStructure(r) {
  const st = r.structure;
  const box = $('structure').closest('.two');
  box.hidden = !st;
  if (!st) return;
  const f = (x, d = 3) => (x == null ? '–' : x.toFixed(d));
  table($('structure'), [['Half-life'], ['Tree length', 'num'], ['Dimension', 'num'], ['Geodesic', 'num']],
    st.latest.map((x) => [[`${x.halfLife} days`], [f(x.treeLength), 'num'], [f(x.dimension), 'num'], [f(x.geodesic, 2), 'num']]));
  const canvas = $('structure-chart');
  const css = getComputedStyle(document.documentElement);
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr, canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const lines = [['treeLength', '--model', 'tree length'], ['dimension', '--hedged', 'effective dimension']];
  const pad = { l: 8, r: 8, t: 8, b: 20 };
  lines.forEach(([key, color]) => {
    const v = st.series[key];
    const ok = v.filter((x) => x != null);
    if (!ok.length) return;
    const lo = Math.min(...ok), hi = Math.max(...ok);
    ctx.strokeStyle = css.getPropertyValue(color).trim();
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let started = false;
    v.forEach((x, i) => {
      if (x == null) { started = false; return; }
      const X = pad.l + (i / Math.max(1, v.length - 1)) * (w - pad.l - pad.r);
      const Y = pad.t + (1 - (x - lo) / Math.max(hi - lo, 1e-12)) * (h - pad.t - pad.b);
      if (started) ctx.lineTo(X, Y); else ctx.moveTo(X, Y), started = true;
    });
    ctx.stroke();
  });
  ctx.fillStyle = css.getPropertyValue('--muted').trim();
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.fillText(r.dates[0].slice(0, 4), pad.l, h - 5);
  ctx.fillText(r.dates[r.dates.length - 1].slice(0, 4), w - pad.r - 30, h - 5);
  $('structure-legend').innerHTML = lines.map(([, color, name]) => `<span><i style="background:${css.getPropertyValue(color).trim()}"></i>${name} (half-life ${st.series.halfLife} days, each scaled to its range)</span>`).join('');
}

function drawChart(r) {
  const css = getComputedStyle(document.documentElement);
  drawCurves($('chart'), $('legend'), r.dates, SERIES.map(([name, color]) => ({ name, color: css.getPropertyValue(color).trim(), v: r.curves[name], bold: name === 'model' })));
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
  const ticks = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8].filter((v) => Math.log(v) >= lo - 1e-9 && Math.log(v) <= hi + 1e-9);
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

// The daily trades against the day's actual bars (open, high, low, close, volume).
function renderCheck(x, sgn, prob) {
  $('top10-check-wrap').hidden = !x.dayTrades || !x.eval;
  if (!x.dayTrades || !x.eval) return;
  const e = x.eval;
  $('top10-eval').innerHTML = `
    <div class="tile"><span class="k">Take-profit touched</span><span class="v">${pct(e.realTake)}</span><span class="s">expected ${pct(e.predTake)} · ${e.trades.toLocaleString('en-GB')} trades on ${e.days.toLocaleString('en-GB')} days</span></div>
    <div class="tile"><span class="k">Stop-loss touched</span><span class="v">${pct(e.realStop)}</span><span class="s">expected ${pct(e.predStop)}</span></div>
    <div class="tile"><span class="k">Expected vs realised, per trade</span><span class="v">${bp(e.realised, 1)} bp</span><span class="s">open to close, over all ranked stocks; expected ${bp(e.expected, 1)} bp · correlation ${e.ic.toFixed(3)} · same sign ${pct(e.signHit, 0)}</span></div>
    <div class="tile"><span class="k">Before costs, per day</span><span class="v">${bp(e.grossTop, 1)} bp</span><span class="s">the 10, open to close; all ranked ${bp(e.grossAll, 1)} bp. Round trip ${(last.costs.buyBps + last.costs.sellBps).toFixed(0)} bp.</span></div>`;
  $('top10-check-note').textContent = `The trades of ${x.lastDate}, planned at the close before (whether or not the policy traded), against that day's bars. Levels and prices relative to the day's open; volume relative to its usual (the mean over the forecast's life).`;
  table($('top10-check'), [['Stock'], ['Stop', 'num'], ['Take', 'num'], ['Expected', 'num'], ['P(stop)', 'num'], ['P(take)', 'num'], ['High', 'num'], ['Low', 'num'], ['Close', 'num'], ['Volume', 'num'], ['Exit'], ['Net', 'num']],
    x.lastDay.map((c) => [[c.ticker], [c.stop <= -0.999 ? 'none' : sgn(c.stop), 'num'], [c.take == null ? 'none' : sgn(c.take), 'num'], [sgn(c.expected, 2), 'num'],
      [prob(c.pStop), 'num'], [prob(c.pTake), 'num'], [sgn(c.high), 'num pos'], [sgn(c.low), 'num neg'], [sgn(c.close, 2), `num ${c.close >= 0 ? 'pos' : 'neg'}`],
      [c.volume == null ? '–' : `${c.volume.toFixed(2)}×`, 'num'], [c.exit === 'still open' ? 'not traded' : c.exit], [sgn(c.ret, 2), `num ${c.ret >= 0 ? 'pos' : 'neg'}`]]));
}

// Rolling top-10 portfolios (from the published UK scan).
const TOP_SERIES = [['with adaptive stops', '--model'], ['without stops', '--hedged'], ['market (bought and held)', '--market']];
let topPick = 0;
function renderTop(r) {
  const all = r.top10 && r.top10.strategies;
  if (!all || !all.length) return;
  topPick = Math.min(topPick, all.length - 1);
  const kName = (k) => (k == null ? 'none' : `${(+k).toFixed(2)} sd`);
  const x = all[topPick];
  const TS = Object.keys(x.curves).map((n, k) => [n, ['--model', '--hedged', '--market'][k]]);
  x.plan = x.plan || [];
  x.life = x.life || 1;
  $('top10-tabs').innerHTML = '';
  $('top10-tabs').hidden = all.length < 2;
  all.forEach((s, k) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = s.name.replace(/^top 10 /, '');
    b.setAttribute('aria-pressed', k === topPick ? 'true' : 'false');
    b.addEventListener('click', () => { topPick = k; renderTop(r); });
    $('top10-tabs').append(b);
  });
  $('top10-asof').textContent = `UK, ${r.stocks} most traded shares · close of ${r.asOf} · orders for the next open. Levels are relative to the last close.`;
  $('top10-note').textContent = `Each close the 10 stocks ranked best are held in equal shares, bought at the next open (${r.costs.buyBps} bp on purchases, ${r.costs.sellBps} bp on sales). Each position has a stop-loss and a take-profit at multiples of the stock's volatility ${x.oneDay ? 'over one day (the holding period)' : `over the forecast's life (${x.life.toFixed(1)} days)`}, checked against each day's high and low; ${x.dayTrades ? 'each stock learns its own: each morning, the stop and take-profit with the best return per unit of risk over all its earlier days, from any value the data allow, none included, and it is traded only if that beats cash after costs' : `the multiples are the pair whose shadow portfolio has grown most so far (now: stop ${kName(x.stopNow)}, take-profit ${kName(x.takeNow)})`}.${x.dayTrades ? '' : ' "Largest" ranks by traded value, the nearest measure of size in the bars.'}`;
  const sgn = (v, d = 1) => (v == null ? '–' : `${v > 0 ? '+' : ''}${(100 * v).toFixed(d)}%`);
  const prob = (v) => (v == null ? '–' : v < 0.0005 ? '<0.1%' : v > 0.9995 ? '>99.9%' : `${(100 * v).toFixed(1)}%`);
  const act = x.dayTrades ? { 1: 'buy at open, sell by close', 0: 'no trade' } : { 1: 'buy at open', 0: 'hold', '-1': 'sell at open' };
  const lifeCols = !x.oneDay;
  const head = [['Stock'], [x.name.includes('largest') ? 'Size rank' : 'Rank today', 'num'], ...(x.dayTrades ? [['SAR · RSI · ADX']] : []), ['Action'], ['Open', 'num'], ['Stop-loss', 'num'], ['Take-profit', 'num'], ['Expected close', 'num'],
    ['P(stop) 1 d', 'num'], ['P(take) 1 d', 'num'], ...(x.dayTrades ? [['Return / risk', 'num']] : []), ...(lifeCols ? [[`P(stop) ${x.life.toFixed(1)} d`, 'num'], [`P(take) ${x.life.toFixed(1)} d`, 'num']] : [])];
  // Only the trades to book (same-day trades); the stocks left in cash are not listed.
  const book = x.dayTrades ? x.plan.filter((p) => p.action === 1) : x.plan;
  table($('top10-plan'), head,
    book.length ? book.map((p) => [[p.ticker], [p.rank ? String(p.rank) : '–', 'num'], ...(x.dayTrades ? [[p.state == null || p.state < 0 ? '–' : `${p.state & 4 ? '↑' : '↓'} · ${p.state & 2 ? '>50' : '<50'} · ${p.state & 1 ? 'strong' : 'weak'}`]] : []), [act[p.action]], [sgn(p.open), 'num'],
      [`${sgn(p.stop)}${p.advisoryStop && p.stop != null ? '*' : ''}`, 'num neg'], [`${sgn(p.take)}${p.advisoryTake && p.take != null ? '*' : ''}`, 'num pos'],
      [sgn(p.close, 2), `num ${p.close >= 0 ? 'pos' : 'neg'}`], [prob(p.pStopDay), 'num'], [prob(p.pTakeDay), 'num'],
      ...(x.dayTrades ? [[p.ratio == null ? '–' : p.ratio.toFixed(3), `num ${p.ratio > 0 ? 'pos' : 'neg'}`]] : []),
      ...(lifeCols ? [[prob(p.pStopLife), 'num'], [prob(p.pTakeLife), 'num']] : [])])
      : [{ gap: x.dayTrades ? 'no trades to book: none of the 10 has learnt levels that beat cash after costs' : 'no positions' }]);
  renderCheck(x, sgn, prob);
  if (x.dayTrades) {
    const n = x.plan.filter((p) => p.action === 1).length;
    const ways = (x.wayRatios || []).map((w) => `${w.name} ${w.sharpe == null ? '–' : w.sharpe.toFixed(2)}`).join(', ');
    if (x.wayNow) $('top10-asof').textContent += ` Levels learnt from ${x.wayNow}, the way with the best record so far (return per unit of risk, a year: ${ways}).`;
    $('top10-asof').textContent += ` ${n} trade${n === 1 ? '' : 's'} to book: going down the ranking from the best expected return, the first (up to 10) whose own learnt levels beat cash after the ${(r.costs.buyBps + r.costs.sellBps).toFixed(0)} bp round trip; stocks that would not are skipped. Trades were booked on ${x.tradedDays} of ${x.eval.days} days.`;
  }
  const adv = x.plan.some((p) => p.advisoryStop || p.advisoryTake);
  $('top10-plan-note').textContent = `Prices relative to the last close (prices themselves are not published under the data licence; apply the percentages to the actual open for a purchase). Open: the expected open, taken as the last close. Expected close: the model's expected return for the day, relative to the market. ${x.dayTrades ? '' : 'Probabilities: of touching the stop or the take-profit first, for the stock\'s daily volatility (simulated, 20,000 paths).'} ${x.oneDay ? 'Every position is a one-day trade: bought at the open and sold the same day at its stop-loss, its take-profit (the stop first when both are touched) or the close, each paying the round trip. Levels are multiples of the stock\'s open-to-close volatility, learnt from its own earlier days; probabilities are how often those levels were touched on them; return / risk is the learnt levels\' mean net return over its deviation per trade.' : 'Held stocks keep the levels set at entry.'}${adv ? ' * The rule has no level here at present (the data favoured none); shown is the best finite level so far, set from the last close, as advice.' : ''} ${x.turnoverPerYear.toFixed(1)}× turnover a year.`;
  const whole = x.parts[0].rows;
  const [a, b, m] = TS.map(([n]) => whole.find((y) => y.name === n));
  $('top10-verdict').innerHTML = `
    <div class="tile"><span class="k">${x.dayTrades ? 'Learnt policy (trades only when worth the risk)' : 'With adaptive stops'}, ${x.parts[0].from} – ${x.parts[0].to}</span><span class="v">${a.sharpe.toFixed(2)}</span><span class="s">Sharpe ratio, ${pct(a.annualReturn)} a year, max drawdown ${pct(a.maxDrawdown)}</span></div>
    <div class="tile"><span class="k">Same portfolio without stops</span><span class="v">${b.sharpe.toFixed(2)}</span><span class="s">Sharpe ratio, ${pct(b.annualReturn)} a year, max drawdown ${pct(b.maxDrawdown)}</span></div>
    <div class="tile"><span class="k">Market, bought and held</span><span class="v">${m.sharpe.toFixed(2)}</span><span class="s">Sharpe ratio, ${pct(m.annualReturn)} a year, max drawdown ${pct(m.maxDrawdown)}</span></div>${x.gross && !x.dayTrades ? `
    <div class="tile"><span class="k">Before costs: the day's 10 vs all ranked</span><span class="v">${bp(x.gross.excess, 2)} bp</span><span class="s">a day, open to open, over ${x.gross.days} days (t ${x.gross.t.toFixed(1)}); ahead on ${pct(x.gross.hitRate, 0)} of days. Round trip ${(r.costs.buyBps + r.costs.sellBps).toFixed(0)} bp.</span></div>` : ''}`;
  const css = getComputedStyle(document.documentElement);
  drawCurves($('top10-chart'), $('top10-legend'), x.dates, TS.map(([n, c], k) => ({ name: n, color: css.getPropertyValue(c).trim(), v: x.curves[n], bold: k === 0 })));
  const rows = [];
  x.parts.forEach((p) => {
    rows.push({ gap: `${p.name}: ${p.from} – ${p.to}` });
    p.rows.forEach((y) => rows.push([[y.name], [pct(y.annualReturn), 'num'], [y.sharpe.toFixed(2), 'num'], [pct(y.maxDrawdown), 'num']]));
  });
  table($('top10-parts'), [['Strategy'], ['Return / yr', 'num'], ['Sharpe', 'num'], ['Max DD', 'num']], rows);
  table($('top10-exits'), [['Exit'], ['Positions', 'num'], ['Gained', 'num'], ['Mean return', 'num'], ['Days held', 'num']],
    x.exits.filter((e) => e.count > 0).map((e) => [[e.reason], [String(e.count), 'num'], [e.count ? pct(e.wins / e.count, 0) : '–', 'num'], [e.count ? pct(e.meanReturn, 2) : '–', `num ${e.meanReturn >= 0 ? 'pos' : 'neg'}`], [e.count ? e.meanDays.toFixed(0) : '–', 'num']]));
  table($('top10-years'), [['Year'], ['With stops', 'num'], ['Without', 'num'], ['Market', 'num']],
    x.years.map((y) => [[y.year], ...y.returns.map((v) => [pct(v), `num ${v >= 0 ? 'pos' : 'neg'}`])]));
  $('top10-grid-wrap').hidden = !x.gridK.length;
  const G = x.gridK.length;
  table($('top10-grid'), [['Stop \\ take'], ...x.gridK.map((k) => [kName(k), 'num'])],
    x.gridK.map((ks, p) => [[kName(ks)], ...x.gridK.map((_, q) => { const v = x.grid[p * G + q]; return [v == null ? '–' : v.toFixed(2), `num ${v >= m.sharpe ? 'pos' : ''}`]; })]));
}

$('src-files').addEventListener('change', () => { $('files').hidden = false; });
$('src-sample').addEventListener('change', () => { $('files').hidden = true; });
$('run').addEventListener('click', run);
$('expected-all').addEventListener('click', () => { showAll = !showAll; if (last) render(last, '', 0); });
window.addEventListener('resize', () => { if (last) { drawChart(last); renderStructure(last); } if (scan) renderTop(scan); });

// The latest UK scan, published by the scheduled UK workflow (results only, no prices).
fetch('data/uk.json', { cache: 'no-cache' })
  .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
  .then((r) => {
    last = scan = r;
    renderTop(r);
    render(r, '', 0);
  })
  .catch(() => { $('top10-asof').textContent = 'No UK scan has been published yet; it appears after the next scheduled run.'; });

createOfm().then((m) => {
  ofm = m;
  status(`Model ${m.version()} ready. Choose the data and run.`);
  if (new URLSearchParams(location.search).has('autorun') || location.hash === '#run') run();
}).catch((e) => status(`Could not load the model: ${e.message}`));
