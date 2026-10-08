import { run } from '../sat-client.js';
import { barChart, dateAxis, fmt, heatmap, lineChart } from '../charts.js';
import { card, el, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'factors',
  title: 'Alpha factors',
  context: 'sat/factors: AlphaInputs · computeAlpha · paperAlphaIds',
  description: 'The models\' inputs are 23 of the "101 Formulaic Alphas" (Kakushadze, 2016): price-volume signals built from the operator algebra, listed in the paper\'s factor table. ' +
    'Each factor is normalised across stocks on every date (rank by default) before learning. ' +
    'The <b>information coefficient</b> (IC) is the rank correlation between a factor and the next day\'s returns across stocks. Its mean and stability (ICIR) show how much a factor predicts on its own. ' +
    'Coverage is the share of defined raw values. Factors that correlate ranks of price levels, such as #3 and #81, are often undefined because those ranks rarely move.',
});
specEditor(page, ['market', 'csv', 'factors']);

runButton(page, 'Compute factors', async (spec) => {
  const r = await run('factors', spec);
  const fs = r.factors;
  const best = [...fs].sort((a, b) => Math.abs(b.icir) - Math.abs(a.icir));
  tiles(page.content, [
    { label: 'Factors', value: fmt.num(fs.length), hint: `warm-up ${r.warmup} days` },
    { label: 'Strongest |ICIR|', value: `#${best[0].id}`, hint: `IC ${best[0].icMean.toFixed(3)}, ICIR ${best[0].icir.toFixed(2)}` },
    { label: 'Mean |IC|', value: (fs.reduce((s, f) => s + Math.abs(f.icMean), 0) / fs.length).toFixed(4) },
    { label: 'Computed in', value: `${fmt.num(r.elapsedMs)} ms`, hint: 'WebAssembly' },
  ]);
  const g = grid(page.content);
  barChart(card(g, 'Mean information coefficient', 'Rank correlation with the next day\'s return, averaged over dates.'), {
    labels: fs.map((f) => '#' + f.id), series: [{ name: 'mean IC', values: fs.map((f) => f.icMean) }], yFormat: (v) => v.toFixed(3),
  });
  const x = r.dates.map((_, i) => i);
  lineChart(card(g, 'Cumulative IC of the six strongest factors', 'A steady slope means a stable signal; turns show where the regime changes its sign.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date',
    series: best.slice(0, 6).map((f, i) => ({ name: '#' + f.id, y: Array.from(f.cumulativeIc), colorIndex: i })),
  });
  if (r.regimeNames.length)
    heatmap(card(g, 'Mean IC by regime', 'The same factor can predict in one regime and mislead in another.'), {
      rows: fs.map((f) => '#' + f.id), cols: r.regimeNames, values: fs.map((f) => Array.from(f.icByRegime)), format: (v) => v.toFixed(3), xLabel: 'regime', yLabel: 'alpha', rowWidth: 44,
    }).canvas.style.height = '520px';
  heatmap(card(g, 'Correlation between factors', 'Average cross-sectional correlation of the normalised factors.'), {
    rows: fs.map((f) => '#' + f.id), cols: fs.map((f) => '#' + f.id), values: r.correlation.map((row) => Array.from(row)), format: (v) => v.toFixed(1), rowWidth: 44,
  }).canvas.style.height = '520px';
  const box = card(page.content, 'Factor table', 'Formulas in the operator notation of the core numerics page.');
  table(box, ['alpha', 'formula', 'coverage', 'mean IC', 'IC sd', 'ICIR', 'IC > 0'],
    fs.map((f) => [`#${f.id}`, el('code', { text: f.formula }), fmt.pct(f.coverage, 0), f.icMean.toFixed(4), f.icStd.toFixed(3), f.icir.toFixed(2), fmt.pct(f.icPositive, 0)]));
});
