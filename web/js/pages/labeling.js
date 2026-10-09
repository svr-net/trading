import { run } from '../sat-client.js';
import { barChart, dateAxis, fmt, histogram, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'labeling',
  gpu: false,
  title: 'Triple barrier and meta-labels',
  context: 'sat/afml/labeling · sampling: cusumFilter · tripleBarrier · averageUniqueness · sequentialBootstrap',
  description: 'Instead of a label for every day, a <b>CUSUM filter</b> samples the days on which the price has drifted by more than a threshold since the last event. ' +
    'Each event is then labelled by the <b>triple-barrier method</b>: profit-taking and stop-loss barriers scaled by recent volatility, and a vertical barrier that limits the holding period (chapter 3). ' +
    'Overlapping labels share returns, so each label\'s <b>average uniqueness</b> measures how much independent information it carries. The <b>sequential bootstrap</b> draws samples that overlap less (chapter 4). ' +
    'With <b>meta-labeling</b>, a primary rule (here, momentum) chooses the side and a secondary model learns when to act on it, trading recall for precision.',
});
specEditor(page, ['market', 'csv', 'labeling'], { open: false });

runButton(page, 'Label events', async (spec) => {
  const r = await run('afmlLabeling', spec);
  const m = r.meta, c = r.counts, n = r.events.length;
  const mean = (v) => (v.length ? Array.from(v).reduce((a, b) => a + b, 0) / v.length : NaN);
  tiles(page.content, [
    { label: `CUSUM events: ${r.ticker}`, value: fmt.num(n), hint: `threshold ${fmt.pct(r.cusumThreshold, 2)} of log price` },
    { label: 'Barrier hit first', value: `${fmt.pct(c.upper / n, 0)} / ${fmt.pct(c.lower / n, 0)} / ${fmt.pct(c.vertical / n, 0)}`, hint: 'profit / stop / vertical' },
    { label: 'Mean uniqueness', value: fmt.ratio(mean(r.uniqueness)), hint: 'of the event labels' },
    { label: 'Bootstrap uniqueness', value: `${fmt.ratio(mean(r.bootstrapSequential), 3)} vs ${fmt.ratio(mean(r.bootstrapStandard), 3)}`, hint: 'sequential vs standard' },
    { label: 'Meta-labels: precision', value: `${fmt.pct(m.primary.precision, 1)} → ${fmt.pct(m.filtered.precision, 1)}`, hint: `${m.model}, AUC ${fmt.ratio(m.auc, 3)}` },
  ]);
  const g = grid(page.content);
  const x = r.dates.map((_, i) => i), xf = dateAxis(r.dates);
  const last = r.events.slice(-12);
  const from = Math.max(0, (last[0]?.t0 ?? 0) - 10), to = Math.min(r.close.length, (last[last.length - 1]?.t1 ?? r.close.length) + 5);
  const seg = (f) => x.slice(from, to).map(f);
  const barrier = (key) => seg((t) => { const e = last.find((v) => t >= v.t0 && t <= v.t1); return e ? e[key] : NaN; });
  lineChart(card(g, `Barriers of the last events: ${r.ticker}`, 'Each event opens a box: upper and lower barriers until the first touch or the vertical barrier.'), {
    x: x.slice(from, to), xFormat: xf, xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [
      { name: 'close', y: Array.from(r.close).slice(from, to) },
      { name: 'upper barrier', y: barrier('upper'), colorIndex: 2, step: true },
      { name: 'lower barrier', y: barrier('lower'), colorIndex: 7, step: true },
      { name: 'event', y: seg((t) => (last.some((e) => e.t0 === t) ? r.close[t] : NaN)), colorIndex: 3, pointsOnly: true },
    ],
  });
  lineChart(card(g, 'Concurrency of labels', 'Number of event labels whose span covers each date.'), {
    x, xFormat: xf, xLabel: 'date', series: [{ name: 'concurrent labels', y: Array.from(r.concurrency), step: true }],
  });
  histogram(card(g, 'Average uniqueness of the labels', '1 = shares no return with any other label.'), Array.from(r.uniqueness), { bins: 20, name: 'labels', xFormat: (v) => v.toFixed(2) });
  {
    // Both samplers on shared bins: the sequential bootstrap's samples sit further right.
    const seq = Array.from(r.bootstrapSequential), std = Array.from(r.bootstrapStandard), both = [...seq, ...std];
    const lo = Math.min(...both), hi = Math.max(...both), bins = 12, w = (hi - lo) / bins || 1;
    const countOf = (v) => { const c = new Array(bins).fill(0); for (const x of v) c[Math.min(bins - 1, Math.floor((x - lo) / w))]++; return c.map((n) => n / v.length); };
    barChart(card(g, 'Sequential vs standard bootstrap', `Mean uniqueness of 30 bootstrap samples of the labels. Sequential: ${fmt.ratio(mean(seq), 3)}, standard: ${fmt.ratio(mean(std), 3)}.`), {
      labels: Array.from({ length: bins }, (_, k) => (lo + (k + 0.5) * w).toFixed(3)),
      series: [{ name: 'standard', values: countOf(std) }, { name: 'sequential', values: countOf(seq) }],
      yFormat: (v) => fmt.pct(v, 0),
    });
  }
  const box = card(page.content, 'Meta-labeling on all stocks', `Primary: ${m.momentum}-day momentum side at CUSUM events, held to a ${m.holding}-day triple barrier. ` +
    `Secondary: ${m.model} on the alpha factors in the direction of the bet, trained on events resolved before ${m.splitDate} (${fmt.num(m.trainEvents)}), tested on the ${fmt.num(m.testEvents)} after it.`);
  table(box, ['bets', 'taken', 'precision', 'recall', 'F1', 'mean return per bet', 'bet-sized return per event'], [
    ['primary rule alone', fmt.num(m.primary.bets), fmt.pct(m.primary.precision, 1), fmt.pct(m.primary.recall, 1), fmt.ratio(m.primary.f1, 3), fmt.pct(m.primary.meanReturn, 3), '–'],
    ['filtered by the secondary model', fmt.num(m.filtered.bets), fmt.pct(m.filtered.precision, 1), fmt.pct(m.filtered.recall, 1), fmt.ratio(m.filtered.f1, 3), fmt.pct(m.filtered.meanReturn, 3), fmt.pct(m.filtered.sizedMeanReturn, 4)],
  ]);
});
