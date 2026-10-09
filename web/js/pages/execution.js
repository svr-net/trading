import { run } from '../sat-client.js';
import { fmt, histogram, lineChart } from '../charts.js';
import { card, grid, initPage, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'execution',
  title: 'Optimal execution',
  context: 'sat/algo/execution: almgrenChriss · efficientFrontier · simulateShortfall',
  description: 'Any strategy has to trade, and large orders move prices. The <b>Almgren–Chriss</b> model (as presented in Cartea, Jaimungal &amp; Penalva\'s <i>Algorithmic and High-Frequency Trading</i>) liquidates a block over a fixed horizon. Selling fast pays <b>temporary impact</b> (a cost that grows with the trading rate); selling slowly leaves the position exposed to <b>price risk</b>. ' +
    'Minimising expected cost plus λ × variance gives a trajectory that decays like sinh(κ(T − t)): λ = 0 is the straight line (TWAP), and more risk aversion front-loads the selling. Sweeping λ traces the <b>efficient frontier</b> of execution. A Monte Carlo of the same dynamics checks the closed-form cost and risk of the implementation shortfall.',
});
specEditor(page, ['execution'], { open: true });

runButton(page, 'Optimise', async (spec) => {
  const r = await run('algoExecution', spec);
  const [twap, ac, fast, now] = r.plans;
  const bp = (v) => fmt.num((v / r.notional) * 1e4, 1) + ' bp';
  tiles(page.content, [
    { label: 'Urgency κ', value: `${fmt.ratio(r.kappa, 3)} / day`, hint: `half of the position gone in ≈ ${fmt.ratio(Math.log(2) / Math.max(r.kappa, 1e-12), 1)} days` },
    { label: 'TWAP: cost ± sd', value: `${bp(twap.expectedCost)} ± ${bp(twap.sd)}`, hint: `of notional ${fmt.compact(r.notional)}` },
    { label: 'Almgren–Chriss: cost ± sd', value: `${bp(ac.expectedCost)} ± ${bp(ac.sd)}`, hint: `Monte Carlo ${bp(ac.simMean)} ± ${bp(ac.simSd)}` },
    { label: 'Immediate sale', value: bp(now.expectedCost), hint: 'no risk, all impact' },
  ]);
  const g = grid(page.content);
  lineChart(card(g, 'Holdings over the horizon', 'Shares still to sell.'), {
    x: Array.from(r.times), xLabel: 'day', xFormat: (v) => v.toFixed(1), yFormat: (v) => fmt.compact(v),
    series: [twap, ac, fast].map((p, i) => ({ name: p.name, y: Array.from(p.holdings), colorIndex: i, markers: r.times.length <= 30 })),
  });
  lineChart(card(g, 'Efficient frontier', 'Expected cost against its standard deviation as λ varies over six orders of magnitude.'), {
    x: Array.from(r.frontierSd), xLabel: 'sd of shortfall', xFormat: (v) => fmt.compact(v), yFormat: (v) => fmt.compact(v),
    series: [{ name: 'expected cost', y: Array.from(r.frontierCost), markers: true }],
  });
  histogram(card(g, 'Shortfall distribution: TWAP', `${fmt.num(r.paths)} simulated paths.`), Array.from(twap.shortfall), { bins: 40, xFormat: (v) => fmt.compact(v) });
  histogram(card(g, 'Shortfall distribution: Almgren–Chriss', 'Higher mean, much narrower.'), Array.from(ac.shortfall), { bins: 40, xFormat: (v) => fmt.compact(v) });
  table(card(page.content, 'Schedules', 'Implementation shortfall = arrival value − proceeds. Closed form vs simulation.'),
    ['schedule', 'expected cost', 'sd', 'E + λ·Var', 'simulated mean', 'simulated sd', '95% quantile'],
    r.plans.map((p) => [p.name, fmt.compact(p.expectedCost), fmt.compact(p.sd), fmt.compact(p.expectedCost + spec.execution.riskAversion * p.sd * p.sd),
      fmt.compact(p.simMean), fmt.compact(p.simSd), fmt.compact(p.simQ95)]));
});
