import { run } from '../sat-client.js';
import { fmt, histogram, lineChart, scatterChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'overfitting',
  title: 'Backtest overfitting',
  context: 'sat/afml/backtest_stats · overfitting: deflatedSharpe · probabilityOfBacktestOverfitting · betSize',
  description: 'Every model × rule candidate in the pool is a trial. The more trials are run, the higher the best backtest Sharpe ratio climbs by luck alone. ' +
    'The <b>deflated Sharpe ratio</b> is the probability that a strategy\'s true Sharpe ratio beats the expected maximum of that many unskilled trials, allowing for non-normal returns (chapter 14). ' +
    '<b>Combinatorially symmetric cross-validation</b> estimates the <b>probability of backtest overfitting</b>: how often the candidate that looks best in one half of the data falls below the median in the other half (chapter 11). ' +
    'The self-adaptive strategy also chooses among the candidates, but out of sample, which is exactly what these tests reward. The last chart shows how a forecast\'s probability becomes a bet size (chapter 10).',
});
specEditor(page, ['market', 'csv', 'models', 'strategies', 'costs', 'selector', 'overfitting'], { open: false });

runButton(page, 'Assess', async (spec) => {
  const r = await run('afmlOverfitting', spec);
  const ann = Math.sqrt(252);
  tiles(page.content, [
    { label: 'Trials (candidates)', value: fmt.num(r.trials), hint: `${fmt.num(r.days)} evaluation days` },
    { label: 'Expected max Sharpe by luck', value: fmt.ratio(r.expectedMaxSharpe * ann), hint: 'annualised, for this many trials' },
    { label: 'Best fixed: deflated Sharpe', value: fmt.pct(r.best.dsr, 1), hint: `Sharpe ${fmt.ratio(r.best.annualSharpe)}, PSR ${fmt.pct(r.best.psr, 1)}` },
    { label: 'Self-adaptive: deflated Sharpe', value: fmt.pct(r.adaptive.dsr, 1), hint: `Sharpe ${fmt.ratio(r.adaptive.annualSharpe)}, PSR ${fmt.pct(r.adaptive.psr, 1)}` },
    { label: 'Probability of backtest overfitting', value: fmt.pct(r.pbo.pbo, 1), hint: `${fmt.num(r.pbo.combinations)} CSCV combinations` },
  ]);
  const g = grid(page.content);
  histogram(card(g, 'Sharpe ratios of the candidates', 'Annualised, over the evaluation days. The best of these is selected with hindsight.'),
    Array.from(r.candidateSharpe, (v) => v * ann), { bins: 20, name: 'candidates', xFormat: (v) => v.toFixed(2) });
  lineChart(card(g, 'Expected maximum Sharpe ratio of unskilled trials', 'With the variance of the candidates\' Sharpe ratios, against the number of trials.'), {
    x: Array.from(r.trialCounts), xLabel: 'trials', yFormat: (v) => v.toFixed(2),
    series: [{ name: 'expected max (annualised)', y: Array.from(r.expectedMaxAnnual), markers: true }],
  });
  histogram(card(g, 'Logits of the in-sample winner\'s out-of-sample rank', 'Negative: the best in-sample candidate ranked below the median out of sample. PBO is the share below zero.'),
    Array.from(r.pbo.logits), { bins: 25, name: 'combinations', xFormat: (v) => v.toFixed(1) });
  scatterChart(card(g, 'In-sample vs out-of-sample Sharpe of the winner', `Slope ${fmt.ratio(r.pbo.degradationSlope)}: how much in-sample performance carries over. ${fmt.pct(r.pbo.probabilityOfLoss, 0)} of combinations lose money out of sample.`), {
    xLabel: 'in-sample Sharpe (per day)', yLabel: 'out-of-sample', series: [{ name: 'combination', x: Array.from(r.pbo.inSample), y: Array.from(r.pbo.outOfSample) }],
  });
  lineChart(card(g, 'Bet size from probability', 'Size 2N(z) − 1 with z = (p − ½) / √(p(1 − p)), and rounded to steps of 0.1. The bet-sized rule in the strategy pool uses it.'), {
    x: Array.from(r.betProbability), xLabel: 'P(up)', yFormat: (v) => v.toFixed(1),
    series: [{ name: 'bet size', y: Array.from(r.betSize) }, { name: 'discretised', y: Array.from(r.betDiscrete), step: true, colorIndex: 1 }],
  });
  const box = card(page.content, 'Strategy statistics', 'PSR: probability the true Sharpe ratio is above zero. DSR: above the expected maximum of the trials.');
  const row = (name, s) => [name, fmt.ratio(s.annualSharpe), fmt.ratio(s.skew), fmt.ratio(s.kurtosis), fmt.pct(s.psr, 1), fmt.pct(s.dsr, 1),
    fmt.pct(s.maxDrawdown, 1), fmt.pct(s.drawdown95, 1), fmt.num(s.longestUnderWater), fmt.ratio(s.concentration, 3)];
  table(box, ['strategy', 'Sharpe', 'skew', 'kurtosis', 'PSR', 'DSR', 'max DD', 'DD 95%', 'longest under water (days)', 'HHI of gains'], [
    row(`Best fixed (hindsight): ${r.bestFixedLabel}`, r.best),
    row(r.adaptiveLabel, r.adaptive),
  ]);
  return `Done · models ${r.cachedPredictions ? 'cached' : `trained in ${fmt.num(r.trainMs)} ms`} · WebAssembly`;
});
