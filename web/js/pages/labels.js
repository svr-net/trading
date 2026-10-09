import { run } from '../sat-client.js';
import { dateAxis, fmt, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'labels',
  gpu: false,
  title: 'Labels',
  context: 'sat/features: LabelSpec · makeLabels · assemble',
  description: 'What the classifiers learn to predict. <b>Direction</b> labels a stock 1 if it rises over the next <i>h</i> days. ' +
    '<b>Excess</b> labels it 1 if it beats the median stock, a relative target that suits long-short trading. ' +
    '<b>N-period min-max</b> (Han, Kim and Enke, 2023) labels only turning points: a close that is the lowest of its centred window is a buying point (1), the highest a selling point (0). ' +
    'Every label looks into the future, so the walk-forward training stops that many days before each prediction date (purging).',
});
specEditor(page, ['market', 'csv', 'labels']);

runButton(page, 'Compute labels', async (spec) => {
  const r = await run('labels', spec);
  tiles(page.content, r.kinds.map((k) => ({ label: k.name, value: fmt.pct(k.positive, 1) + ' up', hint: `${fmt.pct(k.labelled, 0)} of samples labelled, looks ${k.lookahead} day(s) ahead` })));
  const g = grid(page.content, true);
  const x = r.dates.map((_, i) => i);
  lineChart(card(g, `N-period min-max labels: ${r.ticker}`, 'Buying points (window minima) and selling points (window maxima) on the close.'), {
    x, xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [
      { name: 'close', y: Array.from(r.close), colorIndex: 0, width: 1.5 },
      { name: 'buy (label 1)', y: Array.from(r.buy), colorIndex: 2, pointsOnly: true },
      { name: 'sell (label 0)', y: Array.from(r.sell), colorIndex: 7, pointsOnly: true },
    ],
  });
  const box = card(page.content, 'Labelling schemes', 'Agreement is with the next-day direction label on the samples both define.');
  table(box, ['label', 'labelled', 'share of 1', 'agreement with next-day direction', 'look-ahead (days)'],
    r.kinds.map((k) => [k.name, fmt.pct(k.labelled, 1), fmt.pct(k.positive, 1), fmt.pct(k.agreement, 1), String(k.lookahead)]));
});
