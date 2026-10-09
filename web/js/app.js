// Entry point of the standalone bundle: every page is bundled into one script and the
// page to run is chosen by <body data-page="...">. Each page module only runs when loaded.
const pages = {
  index: () => import('./pages/index.js'),
  core: () => import('./pages/core.js'),
  data: () => import('./pages/data.js'),
  factors: () => import('./pages/factors.js'),
  labels: () => import('./pages/labels.js'),
  models: () => import('./pages/models.js'),
  strategies: () => import('./pages/strategies.js'),
  adaptive: () => import('./pages/adaptive.js'),
  robustness: () => import('./pages/robustness.js'),
  bars: () => import('./pages/bars.js'),
  fracdiff: () => import('./pages/fracdiff.js'),
  labeling: () => import('./pages/labeling.js'),
  validation: () => import('./pages/validation.js'),
  portfolio: () => import('./pages/portfolio.js'),
  overfitting: () => import('./pages/overfitting.js'),
  hedging: () => import('./pages/hedging.js'),
  options: () => import('./pages/options.js'),
  pairs: () => import('./pages/pairs.js'),
  trend: () => import('./pages/trend.js'),
  regimes: () => import('./pages/regimes.js'),
  execution: () => import('./pages/execution.js'),
  tournament: () => import('./pages/tournament.js'),
  composite: () => import('./pages/composite.js'),
  gpu: () => import('./pages/gpu.js'),
};

const id = document.body.dataset.page || 'index';
(pages[id] || pages.index)();
