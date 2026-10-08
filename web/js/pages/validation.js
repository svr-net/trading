import { run } from '../sat-client.js';
import { barChart, fmt, heatmap, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'validation',
  title: 'Purged cross-validation and feature importance',
  context: 'sat/afml/sampling · importance: purgedKFold · combinatorialPurgedSplits · MDI · MDA · SFI',
  description: 'Financial labels overlap in time: a 5-day label on Monday shares four days of returns with Tuesday\'s. ' +
    'Ordinary k-fold cross-validation then trains on the information it tests on, and its score is optimistic. ' +
    '<b>Purging</b> removes training samples whose span overlaps a test fold, and an <b>embargo</b> drops those just after it (chapter 7). ' +
    '<b>Combinatorial purged cross-validation</b> tests every choice of <i>k</i> of <i>N</i> groups and assembles several complete out-of-sample backtest paths, not just one (chapter 12). ' +
    'Feature importance is measured three ways on the purged folds (chapter 8): <b>MDI</b> (in-sample split gain), <b>MDA</b> (out-of-sample loss when a feature is shuffled) and <b>SFI</b> (each feature on its own).',
});
specEditor(page, ['market', 'csv', 'factors', 'validation'], { open: false });

runButton(page, 'Cross-validate', async (spec) => {
  const r = await run('afmlValidation', spec);
  const [shuffled, blocked, purged] = r.cv;
  tiles(page.content, [
    { label: 'Samples', value: fmt.num(r.rows), hint: `every ${r.stride} day(s), labels look ${r.horizon} days ahead` },
    { label: 'Shuffled k-fold accuracy', value: fmt.pct(shuffled.accuracy, 2), hint: 'leaks overlapping labels' },
    { label: 'Purged k-fold accuracy', value: fmt.pct(purged.accuracy, 2), hint: `embargo ${r.embargo} days` },
    { label: 'Optimism of shuffling', value: `${fmt.num(100 * (shuffled.accuracy - purged.accuracy), 2)} pts`, hint: 'accuracy points' },
    { label: 'CPCV paths', value: fmt.num(r.cpcv.paths.length), hint: `${r.cpcv.splits} splits of N = ${r.cpcv.groups}, k = ${r.cpcv.testGroups}` },
  ]);
  const g = grid(page.content);
  barChart(card(g, 'Accuracy by validation scheme', `${r.model}, ${r.folds} folds. Only the purged scheme estimates out-of-sample accuracy fairly.`), {
    labels: r.cv.map((c) => c.name), series: [{ name: 'accuracy', values: r.cv.map((c) => c.accuracy) }, { name: 'AUC', values: r.cv.map((c) => c.auc) }],
    yFormat: (v) => v.toFixed(3),
  });
  const paths = r.cpcv.paths;
  lineChart(card(g, 'Combinatorial purged CV: backtest paths', 'Long the top fifth and short the bottom fifth of the predictions on each path. The spread of the paths shows how much a single backtest can mislead.'), {
    x: Array.from(paths[0].equity, (_, i) => i), xLabel: 'sampled date', yFormat: (v) => v.toFixed(2),
    series: paths.map((p, i) => ({ name: `path ${i + 1} (SR ${p.sharpe.toFixed(2)})`, y: Array.from(p.equity), colorIndex: i, width: 1.5 })),
  });
  const names = r.features.map((f) => f.replace('alpha', '#'));
  const norm = (v) => { const a = Array.from(v); const s = a.reduce((x, y) => x + Math.abs(y), 0) || 1; return a.map((x) => x / s); };
  heatmap(card(g, 'Feature importance', 'Each method normalised to sum 1 (SFI: AUC above 0.5). Agreement between methods is reassuring; MDI alone is not.'), {
    rows: ['MDI', 'MDA', 'SFI'], cols: names,
    values: [norm(r.mdi), norm(r.mda), norm(Array.from(r.sfi, (v) => Math.max(0, v - 0.5)))],
    format: (v) => (100 * v).toFixed(0), tooltipFormat: (v) => fmt.pct(v, 1), xLabel: 'feature', rowWidth: 48,
  });
  const box = card(page.content, 'Schemes', 'Training rows are averaged over folds; purging and the embargo remove some.');
  table(box, ['scheme', 'accuracy', 'AUC', 'log loss', 'training rows per fold'],
    r.cv.map((c) => [c.name, fmt.pct(c.accuracy, 2), c.auc.toFixed(3), c.logLoss.toFixed(4), fmt.num(c.trainRows)]));
  const imp = card(page.content, 'Importance by feature', 'MDA: rise in out-of-fold log loss when the feature is shuffled (± standard error). SFI: cross-validated AUC of the feature alone.');
  table(imp, ['feature', 'MDI', 'MDA', '± se', 'SFI (AUC)'],
    names.map((n, i) => [n, fmt.pct(r.mdi[i], 1), r.mda[i].toExponential(2), r.mdaSe[i].toExponential(1), r.sfi[i].toFixed(3)]));
  return `Done in ${fmt.num(r.elapsedMs)} ms · WebAssembly`;
});
