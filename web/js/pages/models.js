import { run } from '../sat-client.js';
import { barChart, fmt, heatmap, lineChart, scatterChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'models',
  gpu: false,
  title: 'Machine-learning models',
  context: 'sat/ml: Classifier · walkForward · LogisticRegression · LinearSvm · RandomForest · GradientBoosting · Mlp · Lstm',
  description: 'Each model forecasts the probability that a stock\'s label is 1, using the alpha factors. ' +
    'Training is walk-forward: every <i>k</i> days the model is re-fitted on the previous window of labelled samples, minus the label look-ahead, and predicts until the next re-fit. ' +
    'Every forecast is therefore out of sample. The tree ensembles grow histogram-split, second-order trees. XGBoost-style grows them depth-wise, LightGBM-style leaf-wise. ' +
    'The LSTM reads the last few days\' factor vectors as a sequence. All models are implemented in the C++ library and trained here in WebAssembly.',
});
specEditor(page, ['market', 'csv', 'factors', 'labels', 'walkForward', 'models']);

runButton(page, 'Train walk-forward', async (spec) => {
  const t0 = performance.now();
  const r = await run('models', spec);
  const ms = r.models;
  const bestAuc = ms.reduce((b, m) => (m.oos.auc > b.oos.auc ? m : b), ms[0]);
  tiles(page.content, [
    { label: 'Best AUC', value: bestAuc.oos.auc.toFixed(3), hint: bestAuc.name },
    { label: 'Base rate (label 1)', value: fmt.pct(ms[0].oos.baseRate, 1), hint: `${fmt.num(ms[0].oos.count)} out-of-sample samples` },
    { label: 'Out of sample', value: `${ms[0].start} → ${ms[0].end}`, hint: `window ${r.trainWindow} days, re-fit every ${r.retrainEvery}, purge ${r.lookahead}` },
    { label: 'Training', value: `${fmt.num(r.trainMs)} ms`, hint: r.cached ? 'cached from an earlier run' : `${ms.reduce((s, m) => s + m.retrains.length, 0)} fits in WebAssembly` },
  ]);
  const g = grid(page.content);
  barChart(card(g, 'Out-of-sample accuracy and AUC', 'Daily stock direction is hard to forecast: a few points above 50% is typical, and enough to trade on.'), {
    labels: ms.map((m) => m.name), series: [{ name: 'accuracy', values: ms.map((m) => m.oos.accuracy) }, { name: 'AUC', values: ms.map((m) => m.oos.auc) }],
    yFormat: (v) => v.toFixed(3),
  });
  scatterChart(card(g, 'ROC curves', 'True against false positive rate as the probability threshold moves; the diagonal is no skill.'), {
    xLabel: 'false positive rate', yLabel: 'true positive rate', diagonal: true, square: true,
    series: ms.map((m, i) => ({ name: m.name, x: Array.from(m.rocFpr), y: Array.from(m.rocTpr), colorIndex: i })),
  });
  scatterChart(card(g, 'Calibration', 'Observed share of label 1 against the predicted probability, by decile of predictions.'), {
    xLabel: 'predicted', yLabel: 'observed', diagonal: true,
    series: ms.map((m, i) => ({ name: m.name, x: Array.from(m.calibrationPredicted), y: Array.from(m.calibrationObserved), colorIndex: i })),
  });
  const fitX = ms[0].retrains.map((_, i) => i);
  lineChart(card(g, 'Accuracy of each fit', 'Out-of-sample accuracy over the period each fit predicted. It moves with the market regime.'), {
    x: fitX, xFormat: (v) => ms[0].retrains[Math.round(v)]?.date ?? '', xLabel: 'fit', yFormat: (v) => v.toFixed(3),
    series: ms.map((m, i) => ({ name: m.name, y: m.retrains.map((f) => f.accuracy), colorIndex: i, markers: true })),
  });
  const withImp = ms.filter((m) => m.importance.length);
  if (withImp.length)
    heatmap(card(g, 'Feature importance', 'Share of total split gain (trees) or of absolute coefficients (linear models), by alpha.'), {
      rows: withImp.map((m) => m.name), cols: r.features.map((f) => f.replace('alpha', '#')), values: withImp.map((m) => Array.from(m.importance)),
      format: (v) => (100 * v).toFixed(0), tooltipFormat: (v) => fmt.pct(v, 1), xLabel: 'alpha', rowWidth: 130, center: 0,
    });
  if (r.regimeNames.length)
    heatmap(card(g, 'Accuracy by regime', 'The models see no regime label, yet their accuracy depends on it.'), {
      rows: ms.map((m) => m.name), cols: r.regimeNames, values: ms.map((m) => Array.from(m.accuracyByRegime)), format: (v) => fmt.pct(v, 1), center: 0.5, rowWidth: 130,
    });
  const box = card(page.content, 'Out-of-sample classification metrics', '');
  table(box, ['model', 'accuracy', 'precision', 'recall', 'F1', 'AUC', 'log loss', 'fits', 'time'],
    ms.map((m) => [m.name, fmt.pct(m.oos.accuracy, 2), fmt.pct(m.oos.precision, 1), fmt.pct(m.oos.recall, 1), m.oos.f1.toFixed(3), m.oos.auc.toFixed(3), m.oos.logLoss.toFixed(4), String(m.retrains.length), `${fmt.num(m.elapsedMs)} ms`]));
  return `Done in ${fmt.num(performance.now() - t0)} ms · WebAssembly`;
});
