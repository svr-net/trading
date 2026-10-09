// Shared page scaffolding: navigation, environment status, cards, tiles, tables,
// run buttons and the specification editor used by every page.
import { EXTRA_FEATURES, MODEL_DEFAULTS, MODEL_NAMES, PAPER_ALPHAS, loadSpec, resetSpec, saveSpec, strategyLabel } from './spec.js';
import { fmt } from './charts.js';

export const PAGES = [
  { group: 'Overview', items: [{ id: 'index', href: 'index.html', title: 'Overview' }] },
  {
    group: 'Data',
    items: [
      { id: 'core', href: 'core.html', title: 'Core numerics' },
      { id: 'data', href: 'data.html', title: 'Market data' },
      { id: 'factors', href: 'factors.html', title: 'Alpha factors' },
      { id: 'labels', href: 'labels.html', title: 'Labels' },
    ],
  },
  { group: 'Prediction', items: [{ id: 'models', href: 'models.html', title: 'Machine-learning models' }] },
  {
    group: 'Trading',
    items: [
      { id: 'strategies', href: 'strategies.html', title: 'Fixed strategies' },
      { id: 'adaptive', href: 'adaptive.html', title: 'Self-adaptive strategy' },
      { id: 'robustness', href: 'robustness.html', title: 'Robustness' },
    ],
  },
  {
    group: 'Advances in Financial ML',
    items: [
      { id: 'bars', href: 'bars.html', title: 'Information-driven bars' },
      { id: 'fracdiff', href: 'fracdiff.html', title: 'Fractional differentiation' },
      { id: 'labeling', href: 'labeling.html', title: 'Triple barrier & meta-labels' },
      { id: 'validation', href: 'validation.html', title: 'Purged CV & importance' },
      { id: 'portfolio', href: 'portfolio.html', title: 'Hierarchical risk parity' },
      { id: 'overfitting', href: 'overfitting.html', title: 'Backtest overfitting' },
    ],
  },
  {
    group: 'Hedging & algorithmic trading',
    items: [
      { id: 'hedging', href: 'hedging.html', title: 'Beta hedging & sizing' },
      { id: 'options', href: 'options.html', title: 'Option hedges' },
      { id: 'pairs', href: 'pairs.html', title: 'Cointegrated pairs' },
      { id: 'trend', href: 'trend.html', title: 'Trend following' },
      { id: 'regimes', href: 'regimes.html', title: 'Regime switching' },
      { id: 'execution', href: 'execution.html', title: 'Optimal execution' },
      { id: 'tournament', href: 'tournament.html', title: 'Tournament & meta-allocation' },
    ],
  },
  { group: 'Acceleration', items: [{ id: 'gpu', href: 'gpu.html', title: 'WebGPU strategy search' }] },
];

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

/** Builds the page shell. Returns { main, toolbar, status, content }. */
export function initPage({ id, title, context, description }) {
  document.title = `${title} · trading SAT`;
  // On narrow screens the links collapse behind a Menu button so the page content shows first.
  const links = el('div', { class: 'nav-links', id: 'nav-links' });
  const toggle = el('button', { class: 'menu-toggle', type: 'button', 'aria-expanded': 'false', 'aria-controls': 'nav-links', text: '☰ Menu' });
  const nav = el('nav', { class: 'sidebar' },
    el('div', { class: 'nav-head' },
      el('div', {}, el('div', { class: 'brand', text: 'trading · SAT' }),
        el('div', { class: 'sub', text: 'Self-adaptive ML trading in WebAssembly + WebGPU' })),
      toggle),
    links);
  toggle.addEventListener('click', () => {
    const open = nav.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
    toggle.textContent = open ? '✕ Close' : '☰ Menu';
  });
  for (const g of PAGES) {
    links.append(el('div', { class: 'group', text: g.group }));
    for (const p of g.items) links.append(el('a', { href: p.href, class: p.id === id ? 'active' : '', text: p.title }));
  }
  const wasmDot = el('span', { class: 'dot' });
  const gpuDot = el('span', { class: 'dot' });
  const wasmText = el('span', { text: 'WASM: loading…' });
  const gpuText = el('span', { text: 'WebGPU: checking…' });
  const emuDot = el('span', { class: 'dot' });
  const emuText = el('span', { text: 'GPU emulator: checking…' });
  links.append(el('div', { class: 'env' }, el('div', {}, wasmDot, wasmText), el('div', {}, gpuDot, gpuText), el('div', { id: 'emulator-status' }, emuDot, emuText)));

  const status = el('span', { class: 'status', id: 'run-status' });
  const toolbar = el('div', { class: 'toolbar' });
  const content = el('div');
  const main = el('main', {},
    el('header', {}, el('div', { class: 'context', text: context }), el('h1', { text: title }), description ? el('p', { html: description }) : null),
    toolbar, content);
  toolbar.append(status);
  const app = document.getElementById('app');
  app.className = 'layout';
  app.append(nav, main);

  import('./sat-client.js').then(({ run }) => run('version', {}))
    .then((v) => { wasmDot.className = 'dot ok'; wasmText.textContent = `WASM: ${v.library}`; })
    .catch((e) => { wasmDot.className = 'dot bad'; wasmText.textContent = 'WASM: ' + e.message; });
  gpuStatus().then((s) => {
    gpuDot.className = 'dot ' + (s.ok ? 'ok' : 'warn');
    gpuText.textContent = 'WebGPU: ' + s.text;
    // The emulated GPU (the kernels on the CPU, in the worker) is always available.
    let mode = 'auto';
    try { mode = localStorage.getItem('sat-engine') || 'auto'; } catch (_) { /* storage unavailable */ }
    emuDot.className = 'dot ok';
    emuText.textContent = mode === 'emulator' ? 'GPU emulator: selected' : mode === 'wasm' ? 'GPU emulator: available'
      : s.ok ? 'GPU emulator: standby (fallback)' : 'GPU emulator: in use (no WebGPU)';
  });

  return { main, toolbar, status, content, spec: loadSpec() };
}

export async function gpuStatus() {
  if (!('gpu' in navigator)) return { ok: false, text: 'not available' };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, text: 'no adapter' };
    const info = adapter.info || {};
    return { ok: true, text: [info.vendor, info.architecture].filter(Boolean).join(' ') || 'available' };
  } catch (e) {
    return { ok: false, text: e.message };
  }
}

export function card(parent, title, note) {
  const body = el('div');
  const c = el('section', { class: 'card' }, el('h2', { text: title }), note ? el('p', { class: 'note', html: note }) : null, body);
  parent.append(c);
  return body;
}

export function grid(parent, wide = false) {
  const g = el('div', { class: wide ? 'grid wide' : 'grid' });
  parent.append(g);
  return g;
}

export function tiles(parent, items) {
  const box = el('div', { class: 'tiles' });
  for (const it of items)
    box.append(el('div', { class: 'tile' },
      el('div', { class: 'label', text: it.label }), el('div', { class: 'value', text: it.value }),
      it.hint ? el('div', { class: 'hint', text: it.hint }) : null));
  parent.append(box);
  return box;
}

export function table(parent, headers, rows) {
  const t = el('table', { class: 'data' }, el('tr', {}, headers.map((h) => el('th', { text: h }))));
  for (const r of rows) t.append(el('tr', {}, r.map((c) => (c instanceof Node ? el('td', {}, c) : el('td', { text: c })))));
  const wrap = el('div', { class: 'table-wrap' }, t);
  parent.append(wrap);
  return wrap;
}

export const passFail = (ok, text) => el('span', { class: ok ? 'pass' : 'fail', text: (ok ? '✓ ' : '✗ ') + (text ?? (ok ? 'pass' : 'fail')) });

/** Adds a primary run button. handler(spec) may be async; status shows timing or errors. */
export function runButton(page, label, handler, { auto = true } = {}) {
  const btn = el('button', { class: 'primary', text: label });
  page.toolbar.insertBefore(btn, page.status);
  const go = async () => {
    btn.disabled = true;
    page.status.className = 'status';
    page.status.textContent = 'Running…';
    const t0 = performance.now();
    try {
      page.content.innerHTML = '';
      const msg = await handler(page.spec);
      page.status.textContent = msg || `Done in ${fmt.num(performance.now() - t0)} ms`;
    } catch (e) {
      console.error(e);
      page.status.className = 'status error';
      page.status.textContent = 'Error: ' + e.message;
    } finally {
      btn.disabled = false;
    }
  };
  btn.addEventListener('click', go);
  if (auto) setTimeout(go, 0);
  return btn;
}

// ------------------------------------------------------------------ spec editor

function numInput(get, set, { scale = 1, step = 'any', digits = 6 } = {}) {
  const v = get();
  const input = el('input', { type: 'number', step, value: v === undefined || v === null ? '' : +(v * scale).toFixed(digits) });
  input.addEventListener('change', () => set(input.value === '' ? null : Number(input.value) / scale));
  return input;
}

function textInput(get, set) {
  const input = el('input', { type: 'text', value: get() ?? '' });
  input.addEventListener('change', () => set(input.value));
  return input;
}

function selectInput(options, get, set, labels = {}) {
  const s = el('select', {}, options.map((o) => el('option', { value: o, text: labels[o] || o, selected: o === get() })));
  s.addEventListener('change', () => set(s.value));
  return s;
}

function checkInput(get, set) {
  const c = el('input', { type: 'checkbox', checked: !!get() });
  c.addEventListener('change', () => set(c.checked));
  return c;
}

function field(label, input) { return el('label', { class: 'field' }, el('span', { text: label }), input); }
function checkField(label, input) { return el('label', { class: 'field check' }, el('span', { text: label }), input); }
const listOf = (text) => text.split(/[\s,;]+/).filter(Boolean).map(Number).filter(Number.isFinite);

// Model settings shown in the editor, per family: [key, label, scale].
const MODEL_FIELDS = {
  logistic: [['l2', 'L2 penalty']],
  svm: [['l2', 'λ'], ['epochs', 'epochs']],
  tree: [['maxDepth', 'depth'], ['minLeaf', 'min leaf']],
  forest: [['trees', 'trees'], ['maxDepth', 'depth'], ['colsample', 'features/split']],
  xgboost: [['trees', 'trees'], ['maxDepth', 'depth'], ['learningRate', 'η']],
  lightgbm: [['trees', 'trees'], ['maxLeaves', 'leaves'], ['learningRate', 'η']],
  mlp: [['hidden', 'hidden'], ['epochs', 'epochs'], ['learningRate', 'Adam step']],
  lstm: [['hidden', 'hidden'], ['seqLen', 'sequence (days)'], ['epochs', 'epochs']],
};

/** Same names as sat::algo::allocationName. */
export const ALLOCATION_NAMES = { exponential: 'exponential weights', best: 'follow the leader', sharpe: 'Sharpe-weighted', riskadjusted: 'Sharpe / volatility', inversevol: 'inverse volatility', equal: 'equal weight' };

const STRATEGY_KINDS = { topk: 'Long top k', longshort: 'Long-short k', threshold: 'Long if P > θ', probweighted: 'P-weighted > θ', betsize: 'Bet size ≥ m' };

/**
 * Collapsible editor for the shared specification.
 * sections: subset of ['market','csv','factors','labels','models','walkForward','strategies','costs','selector','robustness',
 * 'bars','labeling','validation','portfolio','overfitting','hedging','options','pairs','trend','regimes','execution','tournament'].
 */
export function specEditor(page, sections, { open = false } = {}) {
  page.spec = loadSpec();
  const details = el('details', { class: 'spec', open });
  const body = el('div', { class: 'spec-sections' });
  details.append(el('summary', { text: 'Specification (shared by all pages)' }), body);
  page.main.insertBefore(details, page.toolbar);
  const changed = () => saveSpec(page.spec);
  const render = () => {
    body.innerHTML = '';
    const spec = page.spec;
    const sec = (title, wide) => { const s = el('div', { class: 'spec-section' }, el('h3', { text: title })); if (wide) s.style.gridColumn = '1 / -1'; body.append(s); return s; };
    const rerender = () => { changed(); render(); };

    if (sections.includes('market')) {
      const s = sec(spec.csv ? 'Market data: uploaded CSV' : 'Synthetic regime-switching market');
      const m = spec.market;
      if (spec.csv) {
        s.append(el('p', { class: 'note', text: `Using ${fmt.num(spec.csv.length / 1024)} KiB of uploaded daily bars. Remove the file to return to the synthetic market.` }));
      } else {
        s.append(field('stocks', numInput(() => m.numAssets, (v) => { m.numAssets = Math.round(v); changed(); }, { step: 1 })));
        s.append(field('trading days', numInput(() => m.numDates, (v) => { m.numDates = Math.round(v); changed(); }, { step: 1 })));
        s.append(field('seed', numInput(() => m.seed, (v) => { m.seed = Math.round(v); changed(); }, { step: 1 })));
        s.append(field('regime persistence', numInput(() => m.persistence, (v) => { m.persistence = v; changed(); }, { step: 0.005 })));
        s.append(field('idiosyncratic vol (daily %)', numInput(() => m.idiosyncraticVol, (v) => { m.idiosyncraticVol = v; changed(); }, { scale: 100 })));
        const t = el('table', { class: 'edit' }, el('tr', {}, ['regime', 'drift bp/d', 'vol %/d', 'momentum', 'vol. reversal'].map((h) => el('th', { text: h }))));
        m.regimes.forEach((g) => t.append(el('tr', {},
          el('td', {}, textInput(() => g.name, (v) => { g.name = v; changed(); })),
          el('td', {}, numInput(() => g.drift, (v) => { g.drift = v; changed(); }, { scale: 1e4, digits: 3 })),
          el('td', {}, numInput(() => g.volatility, (v) => { g.volatility = v; changed(); }, { scale: 100 })),
          el('td', {}, numInput(() => g.momentum, (v) => { g.momentum = v; changed(); }, { step: 0.01 })),
          el('td', {}, numInput(() => g.volumeReversal, (v) => { g.volumeReversal = v; changed(); }, { step: 0.01 }))))); 
        s.append(t);
      }
    }
    if (sections.includes('csv')) {
      const s = sec('Your own data (CSV)');
      s.append(el('p', { class: 'note', html: 'Daily bars in long format with a header: <code>date,ticker,open,high,low,close,volume[,vwap]</code>, e.g. Hong Kong stocks. Everything stays in this browser.' }));
      const file = el('input', { type: 'file', accept: '.csv,text/csv' });
      file.addEventListener('change', async () => { if (file.files[0]) { spec.csv = await file.files[0].text(); rerender(); } });
      s.append(file);
      if (spec.csv) s.append(el('div', { class: 'spec-actions' }, el('button', { text: 'Remove CSV (back to synthetic)', onclick: () => { delete spec.csv; rerender(); } })));
    }
    if (sections.includes('factors')) {
      const s = sec('Alpha factors (features)');
      const box = el('div', { style: 'display:flex;flex-wrap:wrap;gap:4px 10px;font-size:12px' });
      for (const id of PAPER_ALPHAS) {
        const c = el('input', { type: 'checkbox', checked: spec.alphas.includes(id) });
        c.addEventListener('change', () => { spec.alphas = PAPER_ALPHAS.filter((a) => (a === id ? c.checked : spec.alphas.includes(a))); changed(); });
        box.append(el('label', {}, c, ` #${id}`));
      }
      s.append(box);
      s.append(field('normalisation', selectInput(['rank', 'zscore', 'none'], () => spec.normalisation, (v) => { spec.normalisation = v; changed(); }, { rank: 'cross-sectional rank', zscore: 'cross-sectional z-score', none: 'raw' })));
      const extra = el('div', { style: 'display:flex;flex-wrap:wrap;gap:4px 10px;font-size:12px;margin-top:6px' }, el('span', { style: 'color:var(--text-secondary)', text: 'extra features:' }));
      spec.extraFeatures = spec.extraFeatures || [];
      for (const [k, label] of Object.entries(EXTRA_FEATURES)) {
        const c = el('input', { type: 'checkbox', checked: spec.extraFeatures.includes(k) });
        c.addEventListener('change', () => { spec.extraFeatures = Object.keys(EXTRA_FEATURES).filter((f) => (f === k ? c.checked : spec.extraFeatures.includes(f))); changed(); });
        extra.append(el('label', { title: label }, c, ` ${k}`));
      }
      s.append(extra);
      s.append(field('order d of the ffd feature', numInput(() => spec.ffdOrder ?? 0.4, (v) => { spec.ffdOrder = Math.min(1, Math.max(0, v)); changed(); }, { step: 0.05 })));
    }
    if (sections.includes('labels')) {
      const s = sec('Labels');
      s.append(field('label', selectInput(['direction', 'excess', 'minmax', 'triple'], () => spec.label.kind, (v) => { spec.label.kind = v; changed(); },
        { direction: 'up/down over the horizon', excess: 'beats the median', minmax: 'N-period min-max', triple: 'triple barrier' })));
      s.append(field('horizon / max holding (days)', numInput(() => spec.label.horizon, (v) => { spec.label.horizon = Math.max(1, Math.round(v)); changed(); }, { step: 1 })));
      s.append(field('min-max window (days)', numInput(() => spec.label.window, (v) => { spec.label.window = Math.max(2, Math.round(v)); changed(); }, { step: 1 })));
      s.append(field('triple barrier width (× vol)', numInput(() => spec.label.barrierWidth, (v) => { spec.label.barrierWidth = Math.max(0.05, v); changed(); }, { step: 0.1 })));
    }
    if (sections.includes('walkForward')) {
      const s = sec('Walk-forward training');
      const w = spec.walkForward;
      s.append(field('training window (days)', numInput(() => w.trainWindow, (v) => { w.trainWindow = Math.round(v); changed(); }, { step: 1 })));
      s.append(field('re-fit every (days)', numInput(() => w.retrainEvery, (v) => { w.retrainEvery = Math.max(1, Math.round(v)); changed(); }, { step: 1 })));
      s.append(field('max training rows', numInput(() => w.maxTrainRows, (v) => { w.maxTrainRows = Math.round(v); changed(); }, { step: 500 })));
      s.append(field('sample weights', selectInput(['none', 'uniqueness', 'decay'], () => w.weighting || 'none', (v) => { w.weighting = v; changed(); },
        { none: 'equal', uniqueness: 'average uniqueness', decay: 'uniqueness + time decay' })));
      s.append(field('oldest weight (decay)', numInput(() => w.decayOldest ?? 0.5, (v) => { w.decayOldest = v; changed(); }, { step: 0.1 })));
      s.append(field('train on CUSUM events (× vol, 0 = all days)', numInput(() => spec.cusumMultiple || 0, (v) => { spec.cusumMultiple = Math.max(0, v); changed(); }, { step: 0.5 })));
    }
    if (sections.includes('models')) {
      const s = sec('Models', true);
      spec.models.forEach((m, i) => {
        const row = el('div', { style: 'display:flex;flex-wrap:wrap;gap:6px;align-items:end;margin-bottom:6px;padding-bottom:6px;border-bottom:1px solid var(--grid)' });
        const lab = (text, input) => el('label', { style: 'display:grid;font-size:11px;color:var(--text-secondary);min-width:80px' }, text, input);
        row.append(lab('model', selectInput(Object.keys(MODEL_DEFAULTS), () => m.type, (v) => { spec.models[i] = { ...MODEL_DEFAULTS[v] }; rerender(); }, MODEL_NAMES)));
        for (const [key, label] of MODEL_FIELDS[m.type] || []) row.append(lab(label, numInput(() => m[key], (v) => { m[key] = v; changed(); })));
        row.append(el('button', { text: 'remove', onclick: () => { spec.models.splice(i, 1); rerender(); } }));
        s.append(row);
      });
      s.append(el('div', { class: 'spec-actions' }, Object.keys(MODEL_DEFAULTS).map((t) => el('button', {
        text: '+ ' + MODEL_NAMES[t], onclick: () => { spec.models.push({ ...MODEL_DEFAULTS[t] }); rerender(); },
      }))));
    }
    if (sections.includes('strategies')) {
      const s = sec('Fixed strategies (the candidate pool, for every model)');
      const t = el('table', { class: 'edit' }, el('tr', {}, ['rule', 'k or θ', 'hold (days)', ''].map((h) => el('th', { text: h }))));
      spec.strategies.forEach((st, i) => t.append(el('tr', {},
        el('td', {}, selectInput(Object.keys(STRATEGY_KINDS), () => st.kind, (v) => { st.kind = v; st.param = v === 'topk' || v === 'longshort' ? 5 : v === 'betsize' ? 0.1 : 0.55; rerender(); }, STRATEGY_KINDS)),
        el('td', {}, numInput(() => st.param, (v) => { st.param = v; changed(); })),
        el('td', {}, numInput(() => st.holding, (v) => { st.holding = Math.max(1, Math.round(v)); changed(); }, { step: 1 })),
        el('td', {}, el('button', { text: '×', title: strategyLabel(st), onclick: () => { spec.strategies.splice(i, 1); rerender(); } })))));
      s.append(t, el('button', { text: '+ strategy', onclick: () => { spec.strategies.push({ kind: 'topk', param: 5, holding: 1 }); rerender(); } }));
    }
    if (sections.includes('costs')) {
      const s = sec('Trading costs');
      s.append(field('cost per unit turnover (bp)', numInput(() => spec.costBps, (v) => { spec.costBps = Math.max(0, v); changed(); })));
    }
    if (sections.includes('selector')) {
      const s = sec('Self-adaptive selector');
      const q = spec.selector;
      s.append(field('look-back (days)', numInput(() => q.lookback, (v) => { q.lookback = Math.max(2, Math.round(v)); changed(); }, { step: 1 })));
      s.append(field('adapt every (days)', numInput(() => q.adaptEvery, (v) => { q.adaptEvery = Math.max(1, Math.round(v)); changed(); }, { step: 1 })));
      s.append(field('score', selectInput(['sharpe', 'sortino', 'return'], () => q.metric, (v) => { q.metric = v; changed(); }, { sharpe: 'Sharpe ratio', sortino: 'Sortino ratio', return: 'mean return' })));
      s.append(field('hold top M (1 on the GPU)', numInput(() => q.topM, (v) => { q.topM = Math.max(1, Math.round(v)); changed(); }, { step: 1 })));
      s.append(checkField('cash when no candidate scores > min', checkInput(() => q.allowCash, (v) => { q.allowCash = v; changed(); })));
      s.append(field('min score', numInput(() => q.minScore, (v) => { q.minScore = v; changed(); })));
    }
    if (sections.includes('robustness')) {
      const s = sec('Robustness grid');
      const r = spec.robustness;
      const list = (label, key) => s.append(field(label, textInput(() => r[key].join(', '), (v) => { r[key] = listOf(v); changed(); })));
      list('look-backs (days)', 'lookbacks');
      list('adaptation steps (days)', 'steps');
      list('costs (bp)', 'costs');
    }
    const AFML_SECTIONS = {
      bars: ['Synthetic trade stream', [['days', 'days'], ['tradesPerDay', 'trades per day (average)'], ['activityDispersion', 'activity dispersion (log sd)'], ['persistence', 'order-flow persistence'], ['barsPerDay', 'target bars per day'], ['seed', 'seed']]],
      labeling: ['Events and barriers', [['cusumMultiple', 'CUSUM threshold (× median |return|)'], ['profitTaking', 'profit taking (× target)'], ['stopLoss', 'stop loss (× target)'], ['maxHolding', 'vertical barrier (days)'], ['volSpan', 'volatility span (days)'], ['momentum', 'primary: momentum look-back (days)'], ['metaHolding', 'meta-labels: holding (days)'], ['metaCusumMultiple', 'meta-labels: CUSUM multiple']]],
      validation: ['Cross-validation', [['trees', 'trees'], ['horizon', 'label horizon (days)'], ['folds', 'folds k'], ['embargo', 'embargo (days)'], ['groups', 'CPCV groups N'], ['testGroups', 'CPCV test groups k'], ['maxRows', 'max samples']]],
      portfolio: ['Allocation', [['window', 'estimation window (days)'], ['rebalance', 'rebalance every (days)'], ['trials', 'Monte Carlo trials'], ['simAssets', 'simulated assets']]],
      overfitting: ['Overfitting', [['blocks', 'CSCV blocks S (even)']]],
    };
    for (const [key, [title, fields]] of Object.entries(AFML_SECTIONS)) {
      if (!sections.includes(key)) continue;
      const s = sec(title);
      const o = spec[key];
      for (const [f, label] of fields) s.append(field(label, numInput(() => o[f], (v) => { o[f] = v; changed(); })));
      if (key === 'validation') s.append(field('model', selectInput(['xgboost', 'lightgbm', 'forest', 'tree'], () => o.model, (v) => { o.model = v; changed(); }, MODEL_NAMES)));
      if (key === 'labeling') s.append(field('meta-labels: secondary model', selectInput(['logistic', 'xgboost', 'forest'], () => o.metaModel, (v) => { o.metaModel = v; changed(); }, MODEL_NAMES)));
    }
    const ALGO_SECTIONS = {
      hedging: ['Hedges and sizing', [['betaWindow', 'rolling beta window (days)'], ['delta', 'Kalman state noise δ'], ['hedgeCostBps', 'hedge cost (bp)'], ['targetVol', 'volatility target (annual)'], ['volSpan', 'volatility span (days)'], ['maxLeverage', 'max leverage'], ['kellyFraction', 'Kelly fraction'], ['kellyWindow', 'Kelly window (days)']]],
      options: ['Options', [['vol', 'true volatility'], ['impliedVol', 'implied (hedging) volatility'], ['years', 'option life (years)'], ['strike', 'strike / spot'], ['costBps', 'hedge cost (bp)'], ['paths', 'Monte Carlo paths'], ['seed', 'seed'], ['tenorDays', 'overlay tenor (days)'], ['putMoneyness', 'put strike / spot'], ['callMoneyness', 'call strike / spot'], ['volPremium', 'implied minus realised vol']]],
      pairs: ['Pairs', [['days', 'generated days'], ['beta', 'true hedge ratio'], ['halfLife', 'spread half-life (days)'], ['spreadVol', 'spread volatility'], ['seed', 'seed'], ['delta', 'Kalman state noise δ'], ['observationVariance', 'observation variance'], ['entryZ', 'entry |z|'], ['exitZ', 'exit |z|'], ['costBps', 'cost (bp)']]],
      trend: ['Trend following', [['targetVol', 'volatility target (annual)'], ['volSpan', 'volatility span (days)'], ['buffer', 'position buffer'], ['costBps', 'cost (bp)'], ['reweightEvery', 'reweight every (days)'], ['reweightWindow', 'reweight window (days)']]],
      regimes: ['Regimes', [['states', 'HMM states'], ['window', 'estimation window (days)'], ['refitEvery', 'refit every (days)'], ['threshold', 'risk-off above P(volatile)'], ['riskOffExposure', 'risk-off exposure'], ['costBps', 'cost (bp)'], ['targetVol', 'comparison: vol target']]],
      execution: ['Execution', [['shares', 'shares to sell'], ['price', 'price'], ['horizonDays', 'horizon (days)'], ['periods', 'trading periods'], ['sigma', 'volatility (per share per √day)'], ['epsilon', 'fixed cost ε (per share)'], ['eta', 'temporary impact η'], ['gamma', 'permanent impact γ'], ['riskAversion', 'risk aversion λ'], ['paths', 'Monte Carlo paths']]],
      tournament: ['Meta-allocation', [['lookback', 'look-back (days)'], ['rebalanceEvery', 'rebalance every (days)'], ['eta', 'exponential weights: η'], ['topN', 'follow the leader: top N'], ['costBps', 'reallocation cost (bp)']]],
    };
    for (const [key, [title, fields]] of Object.entries(ALGO_SECTIONS)) {
      if (!sections.includes(key)) continue;
      const s = sec(title);
      const o = spec[key];
      for (const [f, label] of fields) s.append(field(label, numInput(() => o[f], (v) => { o[f] = v; changed(); }, { digits: 12 })));
      if (key === 'trend') s.append(checkField('long only', checkInput(() => o.longOnly, (v) => { o.longOnly = v; changed(); })));
      if (key === 'tournament') {
        s.append(field('method', selectInput(['exponential', 'best', 'sharpe', 'riskadjusted', 'inversevol', 'equal'], () => o.method, (v) => { o.method = v; changed(); }, ALLOCATION_NAMES)));
        s.append(checkField('cash when nothing scores > 0', checkInput(() => o.allowCash, (v) => { o.allowCash = v; changed(); })));
      }
    }
    body.append(el('div', { class: 'spec-actions', style: 'grid-column:1/-1' },
      el('button', { text: 'Reset to defaults', onclick: () => { page.spec = resetSpec(); render(); } }),
      el('span', { class: 'status', text: 'Changes are saved automatically and shared across pages. Press Run to recompute.' })));
  };
  render();
  return details;
}

/** Index of the item of `values` with the largest value. */
export const argmax = (values) => values.reduce((b, v, i) => (v > values[b] ? i : b), 0);

/** One row of the standard performance table. */
export const perfRow = (name, m) => [name, fmt.pct(m.annualReturn, 1), fmt.pct(m.annualVolatility, 1), fmt.ratio(m.sharpe), fmt.ratio(m.sortino),
  fmt.pct(m.maxDrawdown, 1), fmt.ratio(m.calmar), fmt.pct(m.winRate, 1), fmt.ratio(m.averageTurnover)];
export const PERF_HEADERS = ['strategy', 'ann. return', 'ann. vol', 'Sharpe', 'Sortino', 'max DD', 'Calmar', 'win rate', 'turnover/day'];
