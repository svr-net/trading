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
  if (c.net === false && r.model === 'uk') return `the backtest before costs; each order's expected return after its own ${c.broker} costs (commission and ${c.market === 'UK' ? 'stamp duty' : c.market === 'US' ? "the regulators' fees" : c.market === 'FOREX' ? "IBKR's FX commission" : c.market === 'APAC' ? "its exchange's fees" : "its exchange's taxes"}; the auctions cross no spread)`;
  if (c.net === false) return `before costs (gross; ${c.broker} costs not deducted${c.venues ? '' : `, ${{ UK: '£', US: '$', APAC: 'A$', EU: '€', INDEX: '€', FOREX: '$' }[c.market] || ''}${c.capital.toLocaleString('en-GB')} a trade`})`;
  if (c.venues) return `after ${c.broker} costs and each exchange's transaction taxes (${c.market === 'EU' ? `€${c.capital.toLocaleString('en-GB')} a trade${r.exchanges && !r.exchanges.includes('OSL') ? '' : ', its value in NOK in Oslo'}` : `${c.capital.toLocaleString('en-GB')} a trade in its currency`})`;
  const cur = { UK: '£', US: '$', APAC: 'A$', EU: '€', INDEX: '€', FOREX: '$' }[c.market] || '';
  return `after ${c.broker} costs (${cur}${c.capital.toLocaleString('en-GB')} a trade${c.stamp ? `, ${(100 * c.stamp).toFixed(1)}% stamp duty on buying${c.stampFree ? ', none on AIM shares' : ''}` : ''})`;
};

// The exchanges scanned (r.exchanges, written by the daily run), in words.
const EXCHANGE = { LSE: 'London Stock Exchange', NYSE: 'NYSE', NASDAQ: 'NASDAQ', PAR: 'Paris', AMS: 'Amsterdam', BRU: 'Brussels', LIS: 'Lisbon', MSE: 'Madrid', FRA: 'Frankfurt', OSL: 'Oslo', ASX: 'Australia (ASX)', SGX: 'Singapore (SGX)' };
const places = (r, source) => {
  const ex = r.exchanges;
  if (source === 'INDEX') return 'UCITS index ETFs on Xetra, in euros (world, regional, country and sector indices)';
  if (source === 'FOREX') return 'Currency pairs (each with its inverse)';
  if (!ex?.length) return { UK: 'London Stock Exchange', US: 'New York (NYSE and NASDAQ)', EU: 'Europe', APAC: 'Australia (ASX) and Singapore (SGX)' }[source] || source;
  const names = ex.map((e) => EXCHANGE[e] || e);
  return names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
};

// The markets traded automatically: the bridge's AUTO_MARKETS once it answers, remembered in this
// browser for when it cannot be reached (away from home); UK and EU until it has answered once.
const LIVE_KEY = 'trading.live';
let live = new Set(['UK', 'EU']);
try { const v = JSON.parse(localStorage.getItem(LIVE_KEY)); if (Array.isArray(v) && v.length) live = new Set(v); } catch { /* none kept */ }
const SIZE = (c) => (c ? `${{ UK: '£', US: '$', EU: '€', INDEX: '€', FOREX: '$' }[c.market] || ''}${c.capital.toLocaleString('en-GB')}${c.venues ? ' (each exchange\'s currency)' : ''}` : '');

// The plan in plain words, the orders, and the backtest (folded away).
function render(r, source) {
  shown = r;
  const where = places(r, source);
  const n = r.orders.length, c = r.costs;
  const isLive = live.has(source);
  const day = new Date(`${r.asOf}T12:00:00Z`).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' });
  const unfunded = isLive && r.budget && !(r.budget.total > 0);  // the account gives this market no money today: nothing is placed
  $('summary').className = `summary ${isLive && !unfunded ? 'live' : 'research'}`;
  if (source === 'FOREX') {  // research only: modelled, never traded
    $('summary').className = 'summary research';
  }
  $('summary').innerHTML = source === 'FOREX'
    ? `<strong>Research only: currency pairs are modelled, never traded.</strong>${n ? ` The ${n} order${n === 1 ? '' : 's'} below ${n === 1 ? 'is' : 'are'} what the model would place (a pair ending in “-” is its inverse: selling the pair).` : ''}`
    : unfunded
    ? `<strong>No orders placed for the next trading day:</strong> this market's share of the account is 0 today (its record over the last year is not reliably positive)${n ? `; the ${n} order${n === 1 ? '' : 's'} below ${n === 1 ? 'is' : 'are'} what the model would place` : ''}.`
    : isLive
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
  const atOpen = r.orders.length > 0 && r.orders.every((o) => o.entry === 'open');  // bought in the opening auction only: the column is the most it pays
  const head = [['Share'], [atOpen ? 'Buy at open' : 'Buy at', 'num']];
  if (levels) head.push(['Stop', 'num'], ['Target', 'num']);
  if (holds) head.push(['Sell at', 'num']);
  const rebuys = r.orders.some((o) => o.rebuy > 0);  // a buy-back after a stop (learnt; none on most days)
  if (rebuys) head.push(['Buy-back', 'num']);
  const chances = r.orders.some((o) => o.fill != null);  // each order's chance to buy (the share's own days so far)
  if (chances) head.push(['Chance to buy', 'num opt']);
  const ifs = r.orders.some((o) => o.ifFilled != null);  // the gain if it buys; expected = the chance to buy times it
  const costed = r.orders.some((o) => o.cost != null);  // the UK model: each order's own costs, already off its expected return
  if (costed) head.push(['Costs', 'num opt']);
  if (ifs) head.push(['If it buys', 'num opt']);
  head.push(['Expected', 'num']);
  table($('plan'), head,
    n ? r.orders.map((o) => {
      const row = [[o.ticker], [o.limit == null ? 'open' : o.entry === 'open' ? (atOpen ? `≤ ${pct(o.limit, 1)}` : `open, ≤ ${pct(o.limit, 1)}`) : pct(o.limit, 1), 'num']];
      // The stop from the price bought at: fixed, trailing under the day's high, or to the price paid once up `breakeven`.
      const how = o.stopMode === 'trail' ? ' trailing' : o.stopMode === 'breakeven' && o.breakeven != null ? `, to 0 once ${pct(o.breakeven, 1)}` : '';
      if (levels) row.push([o.stop == null ? 'none' : pct(o.stop, 1) + how, o.stop == null ? 'num' : 'num neg'], [o.take == null ? 'none' : pct(o.take, 1), o.take == null ? 'num' : 'num pos']);
      if (holds) row.push([`${pct(o.exitAbove, 2)}, ≤ ${o.days} d`, 'num pos']);  // short enough for a phone: the notes say what it means
      // The buy-back's price from the buy price: rebuy of the way from the stop back up to it.
      if (rebuys) row.push(o.rebuy > 0 && o.stop != null ? [pct(o.stop * (1 - o.rebuy), 1), 'num neg'] : ['none', 'num']);
      if (chances) row.push([o.fill == null ? '–' : `${Math.round(100 * o.fill)}%`, 'num opt']);
      if (costed) row.push(o.cost == null ? ['–', 'num opt'] : [pct(-o.cost, 2), 'num neg opt']);
      if (ifs) row.push(o.ifFilled == null ? ['–', 'num opt'] : [signed(o.ifFilled, 2)[0], `${signed(o.ifFilled, 2)[1]} opt`]);
      row.push(signed(o.expected, Math.abs(o.expected) < 0.0005 ? 3 : 2));  // the order's: the chance to buy times the gain if it buys
      return row;
    }) : [{ gap: `No orders: none is expected to gain${c?.net === false ? '' : ' after costs'}, so the money stays in cash.` }]);
  // Each share opens its chart (daily bars from IBKR through the bridge: never published).
  if (n) [...$('plan').tBodies[0].rows].forEach((tr, k) => {
    const o = r.orders[k], td = tr.cells[0];
    if (!o || !td) return;
    td.classList.add('sym'), td.tabIndex = 0, td.setAttribute('role', 'button'), td.title = `${o.ticker}: chart`;
    td.addEventListener('click', () => openChart(o, source));
    td.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openChart(o, source); } });
  });
  // The notes under the orders: the research model's (in the page), or the UK model's.
  const hint = $('planhint');
  const taxes = c?.market === 'UK' ? 'stamp duty (none on AIM shares)'
    : c?.market === 'US' ? "the regulators' fees on the sale (no tax on buying)"
    : c?.market === 'FOREX' ? "IBKR's FX commission (0.2 basis points, at least USD 2 an order)"
    : c?.market === 'APAC' ? "its exchange's fees (no tax on buying in Sydney or Singapore)"
    : "its exchange's taxes on a share held (the French FTT 0.4% and the Spanish 0.2% on buying, the Portuguese 0.3%, the Belgian TOB 0.35% each side; none in Amsterdam or Frankfurt)"
  if (hint) {
    hint.dataset.base ??= hint.innerHTML;
    hint.innerHTML = r.model === 'uk' && r.book === 'hold'
      ? '<b>Buy at open</b>: in the opening auction only, paying at most this far above the last close; an order that does not buy there is cancelled, so it never buys a dip later. '
        + `<b>Sell at</b>: in the closing auction of the first day the share closes this far above the price paid, else at the close of day ${r.days} (≤ ${r.days} d): held as few days as it needs`
        + (r.orders.some((o) => o.holdDays) ? `, about ${(r.orders.reduce((a, o) => a + (o.holdDays || 0), 0) / r.orders.length).toFixed(1)} days on average. ` : '. ')
        + 'Each morning a share held is looked at again: one the model no longer wants is sold at the open. '
        + `<b>Costs</b>: the order's own commission and ${taxes}, already taken off; it trades in the auctions, so it pays no spread. `
        + '<b>If it buys</b>: what similar past trades made before costs, less these costs. <b>Expected</b>: that times the chance to buy. <span id="odds"></span>'
      : r.model === 'uk'
      ? '<b>Buy at</b>: in the opening auction only, at most this far above the last close; an order that does not buy there is cancelled, so it never buys a dip later in the day. '
        + '<b>Stop</b>: from the price bought at; <i>trailing</i> follows the day\'s high at that distance; <i>to 0 once +x%</i> moves to the price paid once the share is up that much. '
        + '<b>Target</b>: sold if it rises that far. Whatever is still held is sold at the close, so nothing is held overnight. '
        + `<b>Costs</b>: the order's own commission and ${taxes}, already taken off; it trades in the auctions, so it pays no spread. `
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
  // Stretches with no trade (the line flat for 15 points or more, about 30 sessions): shaded and named, so they
  // do not read as missing data. At the start the model is still learning (its first fit and calibration); later
  // it stood aside because its out-of-sample record showed no edge (14 Nov: the US, mid-2023 to early 2024).
  const flat = [];
  if (series[0]) {
    const y = series[0].y;
    let a = 0;
    for (let i = 1; i <= y.length; i++) {
      if (i < y.length && y[i] === y[i - 1]) continue;
      if (i - 1 - a >= 15) flat.push([a, i - 1]);
      a = i;
    }
    const n = y.length - 1 || 1;
    ctx.save();
    flat.forEach(([a0, b0]) => {
      const x0 = X(a0 / n), x1 = X(b0 / n);
      ctx.globalAlpha = 0.12;
      ctx.fillStyle = css.getPropertyValue('--muted').trim();
      ctx.fillRect(x0, pad.t, x1 - x0, h - pad.t - pad.b);
      ctx.globalAlpha = 1;
      const label = a0 === 0 ? 'learning' : 'no trades';
      if (x1 - x0 >= ctx.measureText(label).width + 6) ctx.fillText(label, (x0 + x1) / 2 - ctx.measureText(label).width / 2, pad.t + 12);
    });
    ctx.restore();
  }
  series.forEach((s) => {
    ctx.strokeStyle = color[s.name] || ctx.strokeStyle;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.y.forEach((y, i) => (i ? ctx.lineTo(X(i / (s.y.length - 1)), Y(y)) : ctx.moveTo(X(0), Y(y))));
    ctx.stroke();
  });
  legend.innerHTML = series.map((s) => `<span><i style="background:${color[s.name]}"></i>${s.name} ${s.final}</span>`).join('')
    + (flat.length ? '<span class="flat-key">shaded: no trades (the model stood aside: no edge in its record)</span>' : '');
}
window.addEventListener('resize', () => { if (shown) drawChart(shown); if (chartShown) drawOHLC(...chartShown); });
// A share's chart (Orders: tap a share): its daily bars from IBKR through the bridge (prices are never
// published here: the data licence), with Bollinger bands (20 days, 2 standard deviations), the Parabolic SAR
// (Wilder: 0.02 a step, at most 0.2), the volume, and the order's buy cap and exit level.
let chartShown = null;
// The model's reasons in words: each input that added most to the share's score at the last close, with its
// own value (apps/ofm_uk whyOf: a signal's value is oriented so that higher is bought).
const REASON = {
  'loser 1d': (v) => `Fell ${pct(v, 1).slice(1)} on the last day`,
  'loser 5d': (v) => `Fell ${pct(v, 1).slice(1)} over 5 days`,
  'loser 20d': (v) => `Fell ${pct(v, 1).slice(1)} over 20 days`,
  'residual loser 1d': (v) => `Fell ${pct(v, 1).slice(1)} more than its market on the last day`,
  'residual loser 5d': (v) => `Fell ${pct(v, 1).slice(1)} more than its market over 5 days`,
  'residual loser 20d': (v) => `Fell ${pct(v, 1).slice(1)} more than its market over 20 days`,
  'day fell open-close': (v) => `Fell ${pct(v, 1).slice(1)} from its open to its close`,
  'gapped down': (v) => `Opened ${pct(v, 1).slice(1)} under the close before`,
  'close near low': () => "Closed near the day's low",
  'below 20d mean': (v) => `${pct(v, 1).slice(1)} under its 20-day average`,
  'Bollinger z (below the mean)': (v) => `${v.toFixed(1)} standard deviations under its 20-day average (Bollinger)`,
  'under the lower Bollinger band': () => 'Closed under its lower Bollinger band',
  'loser 1d x market volatility': () => 'A fall in a turbulent market (reversals pay more then)',
  'loser 5d x market volatility': () => 'A 5-day fall in a turbulent market (reversals pay more then)',
  'loser 5d, low vol': () => 'A 5-day fall that is large for how much it usually moves',
  'low volatility': () => 'A calm share (low volatility)',
  'low max day': () => 'No extreme up day lately',
  'momentum 12-1m': (v) => `Up ${pct(v, 0).slice(1)} over the year before last month`,
  'near 52w high': () => 'Near its 52-week high',
  'volume surge': () => 'Volume well above usual',
  'overnight rev 20d': () => 'Its overnight moves have reversed lately',
  'overnight rev 60d': () => 'Its overnight moves have reversed over 3 months',
  'overnight rev 250d': () => 'Its overnight moves have reversed over the year',
  'intraday mom 20d': () => 'Strong intraday moves lately',
  'intraday mom 60d': () => 'Strong intraday moves over 3 months',
  'intraday mom 250d': () => 'Strong intraday moves over the year',
  'intraday less overnight 60d': () => 'Gains made in the day rather than overnight (3 months)',
  'intraday less overnight 250d': () => 'Gains made in the day rather than overnight (the year)',
  volatility: (v) => `Its volatility (${pct(v, 1).slice(1)} a day)`,
  liquidity: () => 'Its trading volume',
  'stamp-free': () => 'No stamp duty (AIM)',
};
function chartWhy(o) {
  const facts = [
    ['Expected', signed(o.expected, 2)[0]], ['Chance to buy', o.fill != null ? `${Math.round(100 * o.fill)}%` : '–'],
    ['Buy cap', o.limit != null ? `≤ ${pct(o.limit, 1)}` : '–'], ['Exit', o.exitAbove != null ? `${pct(o.exitAbove, 2)}, ≤ ${o.days} d` : '–'],
    ['Daily move', o.sigma ? pct(o.sigma, 1).slice(1) : '–'], ['Rank today', o.rank ? `${o.rank} of ${o.of}` : '–'],
  ];
  const why = (o.why || []).map((w) => (REASON[w.k] ? REASON[w.k](w.v) : w.k));
  // Where its expected return comes from: the forecast's part, and the average trade's for a share that moves
  // this much a day (the order list is chosen by the expected return, which scales with the daily move).
  const split = o.fromForecast != null && o.fromAverage != null
    ? `<p class="split">Its expected return before costs, ${pct(o.fromForecast + o.fromAverage, 2)}: <b>${pct(o.fromForecast, 2)}</b> from the forecast${o.rank ? ` (rank ${o.rank} of ${o.of} today)` : ''}, and <b>${pct(o.fromAverage, 2)}</b> the average trade's for a share that moves ${pct(o.sigma, 1).slice(1)} a day. ${
      Math.abs(o.fromForecast) < Math.abs(o.fromAverage) ? 'It is on the list mostly for how much it moves, less for the forecast.' : 'It is on the list mostly for the forecast.'}</p>`
    : '';
  $('chart-dialog-why').innerHTML = `<div class="facts">${facts.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join('')}</div>`
    + split + (why.length ? `<strong>What the forecast saw</strong><ul>${why.map((x) => `<li>${x}</li>`).join('')}</ul>` : '');
}
function openChart(o, source) {
  const dlg = $('chart-dialog');
  $('chart-dialog-title').textContent = o.ticker;
  $('chart-dialog-profile').textContent = '';
  chartWhy(o);
  $('chart-dialog-note').textContent = 'Loading its daily bars from your bridge…';
  $('ohlc-legend').innerHTML = '';
  chartShown = null;
  const cv = $('ohlc');
  cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
  if (typeof dlg.showModal === 'function') dlg.showModal(); else dlg.setAttribute('open', '');
  if (source === 'FOREX') { $('chart-dialog-note').textContent = 'Currency pairs are research only: the bridge keeps no bars for them.'; return; }
  bridge(`/chart?market=${encodeURIComponent(source)}&ticker=${encodeURIComponent(o.ticker)}&days=140`)
    .then((j) => {
      if (!j.bars?.length) throw new Error(j.error || 'no bars');
      $('chart-dialog-note').textContent = `${j.bars.length} daily bars to ${j.bars[j.bars.length - 1].d}${source === 'UK' ? ' (pence)' : ''}`;
      const pf = j.profile || {};
      $('chart-dialog-profile').textContent = [pf.exchange, pf.sector, pf.industry].filter(Boolean).join(' · ');
      chartShown = [j.bars, o];
      drawOHLC(j.bars, o);
    })
    .catch((e) => {
      $('chart-dialog-note').textContent = /key|401|Failed to fetch|NetworkError|Load failed/i.test(String(e.message))
        ? 'Connect your bridge under Broker (its address and key) to see this chart: the prices come from your IBKR account.'
        : `No chart: ${e.message}.`;
    });
}
$('chart-dialog-close').addEventListener('click', () => { const d = $('chart-dialog'); if (d.close) d.close(); else d.removeAttribute('open'); chartShown = null; });
$('chart-dialog').addEventListener('click', (e) => { if (e.target === e.currentTarget && e.currentTarget.close) { e.currentTarget.close(); chartShown = null; } });  // a tap outside
function indicators(bars) {
  const n = bars.length, bb = new Array(n).fill(null), sar = new Array(n).fill(null);
  for (let i = 19; i < n; i++) {  // Bollinger: 20 closes, 2 standard deviations
    let s = 0, s2 = 0;
    for (let k = i - 19; k <= i; k++) s += bars[k].c, s2 += bars[k].c * bars[k].c;
    const m = s / 20, sd = Math.sqrt(Math.max(0, s2 / 20 - m * m));
    bb[i] = { m, up: m + 2 * sd, lo: m - 2 * sd };
  }
  if (n >= 3) {  // Parabolic SAR (Wilder 1978), both ways
    let up = bars[1].c >= bars[0].c, af = 0.02;
    let ep = up ? Math.max(bars[0].h, bars[1].h) : Math.min(bars[0].l, bars[1].l);
    let s = up ? Math.min(bars[0].l, bars[1].l) : Math.max(bars[0].h, bars[1].h);
    sar[1] = { v: s, up };
    for (let i = 2; i < n; i++) {
      s += af * (ep - s);
      const b = bars[i];
      if (up) {
        s = Math.min(s, bars[i - 1].l, bars[i - 2].l);
        if (b.l < s) { up = false; s = ep; ep = b.l; af = 0.02; } else if (b.h > ep) { ep = b.h; af = Math.min(0.2, af + 0.02); }
      } else {
        s = Math.max(s, bars[i - 1].h, bars[i - 2].h);
        if (b.h > s) { up = true; s = ep; ep = b.h; af = 0.02; } else if (b.l < ep) { ep = b.l; af = Math.min(0.2, af + 0.02); }
      }
      sar[i] = { v: s, up };
    }
  }
  return { bb, sar };
}
function drawOHLC(bars, o) {
  const css = getComputedStyle(document.documentElement), col = (v) => css.getPropertyValue(v).trim();
  const canvas = $('ohlc'), dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr, canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const { bb, sar } = indicators(bars);
  const show = Math.min(bars.length, w < 500 ? 60 : 90), from = bars.length - show;
  const last = bars[bars.length - 1].c;
  const cap = o.limit != null && o.limit < 1 ? last * (1 + o.limit) : null;  // the order's limit, from the last close
  const exit = o.exitAbove != null ? last * (1 + o.exitAbove) : null;
  let lo = Infinity, hi = -Infinity, vmax = 0;
  for (let i = from; i < bars.length; i++) {
    const b = bars[i];
    lo = Math.min(lo, b.l, bb[i]?.lo ?? Infinity, sar[i]?.v ?? Infinity), hi = Math.max(hi, b.h, bb[i]?.up ?? -Infinity, sar[i]?.v ?? -Infinity), vmax = Math.max(vmax, b.v);
  }
  for (const x of [cap, exit]) if (x != null) lo = Math.min(lo, x), hi = Math.max(hi, x);
  const pad = { l: 8, r: 54, t: 8, b: 20 }, volH = (h - pad.t - pad.b) * 0.2, priceH = h - pad.t - pad.b - volH - 6;
  const span = hi - lo || 1;
  const Y = (p) => pad.t + (hi - p) / span * priceH, step = (w - pad.l - pad.r) / show, X = (i) => pad.l + (i - from + 0.5) * step;
  ctx.font = '11px "IBM Plex Mono", monospace';
  ctx.strokeStyle = col('--rule'), ctx.fillStyle = col('--muted'), ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {  // the price grid, its labels on the right
    const p = lo + span * k / 4, y = Y(p);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(p >= 100 ? p.toFixed(0) : p.toFixed(2), w - pad.r + 4, y + 4);
  }
  let lastMonth = '';
  for (let i = from; i < bars.length; i++) {  // the months along the bottom
    const mth = bars[i].d.slice(0, 7);
    if (mth !== lastMonth) { if (lastMonth) ctx.fillText(new Date(`${bars[i].d}T12:00:00Z`).toLocaleDateString('en-GB', { month: 'short' }), X(i) - 10, h - 5); lastMonth = mth; }
  }
  const band = (key, dash) => {  // a Bollinger line
    ctx.beginPath(); ctx.setLineDash(dash); ctx.strokeStyle = col('--accent'); ctx.lineWidth = 1.2;
    let on = false;
    for (let i = from; i < bars.length; i++) { if (!bb[i]) continue; const y = Y(bb[i][key]); if (on) ctx.lineTo(X(i), y); else ctx.moveTo(X(i), y), on = true; }
    ctx.stroke(); ctx.setLineDash([]);
  };
  ctx.globalAlpha = 0.08; ctx.fillStyle = col('--accent'); ctx.beginPath();  // the band between them, shaded
  let started = false;
  for (let i = from; i < bars.length; i++) if (bb[i]) { if (started) ctx.lineTo(X(i), Y(bb[i].up)); else ctx.moveTo(X(i), Y(bb[i].up)), started = true; }
  for (let i = bars.length - 1; i >= from; i--) if (bb[i]) ctx.lineTo(X(i), Y(bb[i].lo));
  ctx.fill(); ctx.globalAlpha = 1;
  band('up', []), band('lo', []), band('m', [4, 3]);
  const vTop = pad.t + priceH + 6;
  for (let i = from; i < bars.length; i++) {  // the candles and the volume
    const b = bars[i], x = X(i), upDay = b.c >= b.o, c = upDay ? col('--good') : col('--bad');
    ctx.strokeStyle = c; ctx.fillStyle = c; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, Y(b.h)); ctx.lineTo(x, Y(b.l)); ctx.stroke();
    const bw = Math.max(1, step * 0.6), y0 = Y(Math.max(b.o, b.c)), y1 = Y(Math.min(b.o, b.c));
    ctx.fillRect(x - bw / 2, y0, bw, Math.max(1, y1 - y0));
    if (vmax > 0) { ctx.globalAlpha = 0.45; ctx.fillRect(x - bw / 2, vTop + volH * (1 - b.v / vmax), bw, volH * b.v / vmax); ctx.globalAlpha = 1; }
  }
  ctx.fillStyle = col('--hedged');  // the PSAR's dots
  for (let i = from; i < bars.length; i++) if (sar[i]) { ctx.beginPath(); ctx.arc(X(i), Y(sar[i].v), Math.max(1.5, Math.min(2.5, step * 0.18)), 0, 2 * Math.PI); ctx.fill(); }
  const level = (p, c, label, left) => {  // the order's levels, dashed, labelled at either end (they are often close)
    if (p == null) return;
    ctx.setLineDash([6, 4]); ctx.strokeStyle = c; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(p)); ctx.lineTo(w - pad.r, Y(p)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = c; ctx.fillText(label, left ? pad.l + 4 : w - pad.r - ctx.measureText(label).width - 4, Y(p) - 4);
  };
  level(cap, col('--ink'), 'buy cap', true);
  level(exit, col('--good'), 'exit', false);
  const legendKey = (c, t) => `<span><i style="background:${c}"></i>${t}</span>`;
  $('ohlc-legend').innerHTML = legendKey(col('--accent'), 'Bollinger 20, 2') + legendKey(col('--hedged'), 'PSAR') + legendKey(col('--ink'), 'buy cap') + legendKey(col('--good'), 'exit');
}
// The sections, one at a time (the menu; #orders, #results, ... in the address, remembered).
const VIEWS = ['orders', 'results', 'backtest', 'broker', 'lab', 'about'];
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

const MARKETS = { uk: 'UK', eu: 'EU', us: 'US', apac: 'APAC', index: 'INDEX', forex: 'FOREX' };
let market = 'uk';
try { market = new URLSearchParams(location.search).get('market') || localStorage.getItem('market') || 'uk'; } catch { /* no storage */ }
if (!MARKETS[market]) market = 'uk';
function showMarket(m) {
  market = m;
  try { localStorage.setItem('market', m); } catch { /* no storage */ }
  for (const b of document.querySelectorAll('.market-switch button')) b.setAttribute('aria-selected', String(b.dataset.market === m));
  const name = MARKETS[m];
  // FOREX is research only, never traded: its tab has no broker views (Results and Broker are the bridge's).
  const research = m === 'forex';
  for (const b of document.querySelectorAll('.views button')) if (['results', 'broker', 'lab'].includes(b.dataset.view)) b.hidden = research;
  if (research && (!$('view-results').hidden || !$('view-broker').hidden || !$('view-lab').hidden)) showView('orders');
  labSwitches(m);
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
// What the uk model's shadows leave out of the live one (config/markets.json; docs/uk-research.md).
const SHADOW_NOTE = {
  'hold5-base': 'without the research switches',
  'resid-target': 'without the volatility regime',
  noroll: 'without rolling missing bars',
  adapt: 'with adaptive learning',
  norisk: "without the book's volatility target",
  cap1: 'buy cap: the close plus 1 daily volatility',
  cap2: 'buy cap: the close plus 2 daily volatilities',
  moo: "buy at the auction's price, whatever it is",
  caplearn: "each order's own buy cap, learnt",
  nocaplearn: 'without the learnt buy cap (the close plus 0.5 volatility)',
  boll: 'with Bollinger bands',
  psar: 'with a PSAR stop at the close',
  robustvol: "with a volatility that leaves out one-off jumps",
  topfifth: "only the forecast's top fifth",
  'robust-top': "the forecast's top fifth, with the robust volatility",
  previous: 'the live model before 14 Nov (without the robust volatility / top fifth)',
};
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
        return [[`${s.name}${s.role === 'live' ? ' (live)' : ` (shadow${SHADOW_NOTE[s.name] ? `: ${SHADOW_NOTE[s.name]}` : ''})`}`], E ? signed(E.daily, 3) : ['–', 'num'], E ? signed(E.yearly, 1) : ['–', 'num'], r ? signed(B.annualReturn, 1) : ['–', 'num'], [B.sharpe != null ? B.sharpe.toFixed(2) : '–', 'num'],
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


// The Lab: the model compiled to WebAssembly (web/lab/, docker/wasm/Dockerfile) run here, in a Web Worker, on the
// market's daily bars from the bridge (GET /bars.csv, keyed: the prices never leave this device). Its switches
// start as the market's live strategy's (data/strategies.json).
const LAB_SWITCHES = [
  ['X_RESID', '1', 'Residual reversal (Blitz et al. 2013)'],
  ['X_VOLREGIME', '1', 'Reversal scaled by market volatility (Nagel 2012)'],
  ['X_TARGET', '5', "Fitted on the 5-day hold's return"],
  ['X_RISK', '1', "The book's volatility target"],
  ['X_CAPLEARN', '1', "Each order's own learnt buy cap"],
  ['X_BOLL', '1', 'Bollinger bands as inputs'],
  ['X_PSAR', '1', 'A PSAR exit at the close'],
  ['X_ADAPT', '1', 'Adaptive learning (three windows)'],
];
function labSwitches(m) {
  const box = $('lab-switches');
  if (!box) return;
  fetch('data/strategies.json', { cache: 'no-cache' }).then((r) => (r.ok ? r.json() : null)).catch(() => null).then((idx) => {
    const env = (idx?.markets?.[m] || []).find((s) => s.role === 'live')?.env || {};
    box.replaceChildren(...LAB_SWITCHES.map(([k, v, t]) => {
      const l = document.createElement('label'), c = document.createElement('input');
      c.type = 'checkbox', c.dataset.key = k, c.dataset.value = v, c.checked = env[k] === v;
      l.append(c, document.createTextNode(`${t}${env[k] === v ? ' (live)' : ''}`));
      return l;
    }));
  });
}
let labWorker = null;
$('lab-run').addEventListener('click', async () => {
  const M = MARKETS[market], status = $('lab-status');
  $('lab-verdict').innerHTML = '', $('lab-orders').replaceChildren(), $('lab-log').hidden = true;
  status.textContent = 'Fetching the bars from your bridge…';
  let csv;
  try {
    const r = await fetch(`${bridgeUrl()}/bars.csv?market=${encodeURIComponent(M)}`, { headers: { 'X-Bridge-Key': bridgeKey() } });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
    csv = new Uint8Array(await r.arrayBuffer());
  } catch (e) {
    status.textContent = /key|401|Failed to fetch|NetworkError|Load failed/i.test(String(e.message)) ? 'Connect your bridge under Broker (its address and key): the bars come from your IBKR account.' : `No bars: ${e.message}.`;
    return;
  }
  const env = {};
  for (const c of $('lab-switches').querySelectorAll('input')) if (c.checked) env[c.dataset.key] = c.dataset.value;
  const files = {}, args = ['--csv', '/bars.csv', '--costs', M, '--hold', '5'];
  if (['EU', 'APAC', 'INDEX'].includes(M)) { files['venues.txt'] = await (await fetch('lab/venues.txt')).text(); args.push('--venues', '/venues.txt'); }
  if (M === 'UK') { files['aim.txt'] = await (await fetch('lab/aim.txt')).text(); args.push('--stamp-free', '/aim.txt'); }
  status.textContent = `Running the model on ${(csv.length / 1e6).toFixed(1)} MB of bars…`;
  if (labWorker) labWorker.terminate();
  labWorker = new Worker('lab/worker.js', { type: 'module' });
  labWorker.onmessage = ({ data }) => {
    if (data.progress) { status.textContent = data.progress.replace(/^uk: +/, '').slice(0, 120); return; }
    if (data.error) { status.textContent = `The model stopped: ${data.error}`; $('lab-log').textContent = data.log.join('\n'); $('lab-log').hidden = false; return; }
    const b = data.report?.backtest, n = data.orders?.orders?.length || 0;
    status.textContent = `Done in ${(data.ms / 1000).toFixed(1)} s on this device.`;
    $('lab-verdict').innerHTML = b && b.trades
      ? `<div class="tile"><span class="k">On this device, ${b.from} – ${b.to}</span><span class="v">${pct(b.annualReturn)}</span><span class="s">a year · Sharpe ${b.sharpe.toFixed(2)} · worst fall ${pct(b.maxDrawdown, 0).slice(1)} · ${b.trades} trades</span></div>`
      : '<div class="tile"><span class="k">On this device</span><span class="v">–</span><span class="s">too little history in the bars to learn and calibrate</span></div>';
    table($('lab-orders'), [['Share'], ['Buy at open', 'num'], ['Sell at', 'num'], ['Expected', 'num']],
      n ? data.orders.orders.map((o) => {
        const last = o.close, cap = o.limit / last - 1;
        return [[o.ticker.split(':').pop()], [`≤ ${pct(cap, 1)}`, 'num'], [`${pct(o.exitAbove, 2)}, ≤ ${o.days} d`, 'num pos'], signed(o.expected, 2)];
      }) : [{ gap: 'No orders: none expected to gain after its costs.' }]);
    $('lab-log').textContent = data.log.filter((l) => /^uk: /.test(l)).join('\n'), $('lab-log').hidden = false;
  };
  labWorker.postMessage({ csv, args, env, files }, [csv.buffer]);
});
