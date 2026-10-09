// Default specification shared by all pages: market data, factors, labels, models,
// walk-forward schedule, trading strategies, costs and the self-adaptive selector. Pages
// edit a copy through the spec editor; the current spec is kept in localStorage so every
// page works on the same configuration. An uploaded CSV is kept separately (it can be large).

export const PAPER_ALPHAS = [1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 17, 20, 29, 33, 34, 35, 40, 41, 44, 62, 65, 81];

export const MODEL_DEFAULTS = {
  logistic: { type: 'logistic', l2: 1 },
  svm: { type: 'svm', l2: 1, epochs: 3, seed: 1 },
  tree: { type: 'tree', maxDepth: 5, minLeaf: 50, bins: 32 },
  forest: { type: 'forest', trees: 40, maxDepth: 6, minLeaf: 20, subsample: 0.7, colsample: 0.4, bins: 32, seed: 1 },
  xgboost: { type: 'xgboost', trees: 60, maxDepth: 3, learningRate: 0.1, lambda: 1, gamma: 0, minLeaf: 20, subsample: 0.8, colsample: 0.8, bins: 32, seed: 1 },
  lightgbm: { type: 'lightgbm', trees: 60, maxDepth: 8, maxLeaves: 12, learningRate: 0.1, lambda: 1, minLeaf: 20, subsample: 0.8, colsample: 0.8, bins: 32, seed: 1 },
  mlp: { type: 'mlp', hidden: 16, epochs: 5, batch: 64, learningRate: 0.005, maxSamples: 4000, weightDecay: 1e-4, seed: 1 },
  lstm: { type: 'lstm', hidden: 8, seqLen: 5, epochs: 2, batch: 64, learningRate: 0.01, maxSamples: 3000, weightDecay: 1e-4, seed: 1 },
};

export const MODEL_NAMES = {
  logistic: 'Logistic regression', svm: 'Linear SVM', tree: 'Decision tree', forest: 'Random forest',
  xgboost: 'XGBoost', lightgbm: 'LightGBM', mlp: 'MLP', lstm: 'LSTM',
};

export const DEFAULT_MODELS = ['logistic', 'forest'];
export const DEFAULT_STRATEGIES = [
  { kind: 'topk', param: 5, holding: 1 },
  { kind: 'topk', param: 10, holding: 5 },
  { kind: 'longshort', param: 3, holding: 1 },
  { kind: 'threshold', param: 0.52, holding: 1 },
  { kind: 'probweighted', param: 0.5, holding: 1 },
];

// The pool before the consolidation (v0.3.3 and earlier). A stored spec still holding exactly
// this pool, never edited, moves to the consolidated one; an edited pool is kept as it is.
const LEGACY_MODELS = ['logistic', 'forest', 'xgboost', 'lightgbm', 'mlp', 'lstm'];
const LEGACY_STRATEGIES = [
  ['topk', 3, 1], ['topk', 5, 1], ['topk', 10, 5], ['longshort', 3, 1], ['longshort', 5, 1],
  ['threshold', 0.52, 1], ['threshold', 0.55, 1], ['probweighted', 0.5, 1], ['betsize', 0.1, 1],
];
function isLegacyPool(stored) {
  const m = stored.models, s = stored.strategies;
  return Array.isArray(m) && Array.isArray(s) && m.length === LEGACY_MODELS.length && s.length === LEGACY_STRATEGIES.length &&
    m.every((x, i) => JSON.stringify(x) === JSON.stringify(MODEL_DEFAULTS[LEGACY_MODELS[i]])) &&
    s.every((x, i) => x.kind === LEGACY_STRATEGIES[i][0] && x.param === LEGACY_STRATEGIES[i][1] && x.holding === LEGACY_STRATEGIES[i][2]);
}

export function defaultSpec() {
  return {
    market: {
      numAssets: 30, numDates: 1260, seed: 7, persistence: 0.985, idiosyncraticVol: 0.016, betaDispersion: 0.3,
      regimes: [
        { name: 'bull', drift: 0.0008, volatility: 0.009, momentum: 0.12, volumeReversal: 0.15 },
        { name: 'bear', drift: -0.001, volatility: 0.018, momentum: -0.06, volumeReversal: 0.22 },
        { name: 'range', drift: 0, volatility: 0.008, momentum: -0.16, volumeReversal: 0.3 },
      ],
    },
    alphas: [...PAPER_ALPHAS],
    normalisation: 'rank',
    extraFeatures: [],
    ffdOrder: 0.4,
    cusumMultiple: 0,
    label: { kind: 'direction', horizon: 1, window: 10, barrierWidth: 1, volSpan: 50 },
    // The consolidated pool (examples/pool_ablation): the models and rules that add value to the
    // self-adaptive strategy. Every other family and rule kind can still be added in the editor.
    models: DEFAULT_MODELS.map((t) => ({ ...MODEL_DEFAULTS[t] })),
    walkForward: { trainWindow: 504, retrainEvery: 63, maxTrainRows: 8000, seed: 11, weighting: 'none', decayOldest: 0.5 },
    strategies: DEFAULT_STRATEGIES.map((x) => ({ ...x })),
    costBps: 10,
    selector: { lookback: 63, adaptEvery: 21, metric: 'sharpe', topM: 1, allowCash: true, minScore: 0 },
    robustness: { lookbacks: [21, 42, 63, 126, 252], steps: [5, 10, 21, 42, 63], metrics: ['return', 'sharpe', 'sortino'], costs: [0, 5, 10, 20, 40] },
    // Four more logistic regressions, each trained only on calm, turbulent, rising or falling
    // market days: diverse candidates for the composite forecast.
    stateSpecialists: true,
    // Composite forecast: the models' forecasts combined into one (none | average | adaptive),
    // traded in their place or, with keepMembers, alongside them. examples/forecast_study chose
    // the self-adaptive forecast with market-state ADWIN memory and the evidence rule over the
    // generalists and the state specialists.
    composite: { method: 'adaptive', keepMembers: false, lookback: 63, adaptEvery: 21, window: 'market-adwin', decision: 'evidence', halfLife: 63, adwinDelta: 1e-4, minT: 2, stateBandwidth: 1 },
  };
}

export const EXTRA_FEATURES = { ffd: 'fractionally differentiated log price', vol: 'EWM volatility', roll: 'Roll spread', corwin: 'Corwin-Schultz spread', amihud: 'Amihud illiquidity', kyle: "Kyle's lambda" };

const STORAGE_KEY = 'sat-spec-v1';
const CSV_KEY = 'sat-csv-v1';

export function loadSpec() {
  let spec = defaultSpec();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      // New sections added in later versions keep their defaults when an older spec is stored.
      const stored = JSON.parse(raw);
      for (const k of ['label', 'walkForward', 'composite'])
        if (stored[k] && typeof stored[k] === 'object') stored[k] = { ...spec[k], ...stored[k] };
      if (isLegacyPool(stored)) { delete stored.models; delete stored.strategies; }
      // Combinations and sections removed in v0.6 fall back to the defaults.
      if (!['none', 'average', 'adaptive'].includes(stored.composite?.method)) stored.composite = { ...spec.composite };
      // The v0.5 default (the average of the separate models, unedited) moves to the new default.
      if (stored.composite?.method === 'average' && stored.composite.keepMembers === false && !('window' in stored.composite)) stored.composite = { ...spec.composite };
      for (const k of ['bars', 'labeling', 'validation', 'portfolio', 'overfitting', 'hedging', 'options', 'pairs', 'trend', 'regimes', 'execution', 'tournament', 'multiSignal'])
        delete stored[k];
      spec = { ...spec, ...stored };
    }
  } catch (_) { /* storage unavailable: fall back to defaults */ }
  try {
    const csv = localStorage.getItem(CSV_KEY);
    if (csv) spec.csv = csv;
  } catch (_) { /* ignore */ }
  return spec;
}

export function saveSpec(spec) {
  const { csv, ...rest } = spec;
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(rest)); } catch (_) { /* ignore */ }
  try {
    if (csv) localStorage.setItem(CSV_KEY, csv); else localStorage.removeItem(CSV_KEY);
  } catch (_) { /* too large for storage: kept for this page only */ }
}

export function resetSpec() {
  try { localStorage.removeItem(STORAGE_KEY); localStorage.removeItem(CSV_KEY); } catch (_) { /* ignore */ }
  return defaultSpec();
}

const STRATEGY_TEXT = { topk: (p) => `Top-${Math.round(p)}`, longshort: (p) => `Long-short ${Math.round(p)}`, threshold: (p) => `P>${p.toFixed(2)}`, probweighted: (p) => `P-weighted>${p.toFixed(2)}`, betsize: (p) => `Bet size>${p.toFixed(2)}` };
/** Same text as StrategySpec::label in the library. */
export function strategyLabel(s) {
  return STRATEGY_TEXT[s.kind](s.param) + (s.holding > 1 ? ` /${s.holding}d` : '');
}
