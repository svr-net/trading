// Node test of the WebAssembly build: node web/tests/wasm.test.mjs
import assert from 'node:assert/strict';
import createOfm from '../wasm/ofm.js';

const ofm = await createOfm();
const s = new ofm.Session();
const meta = JSON.parse(s.loadSample());
assert.equal(meta.stocks, 120);
assert.ok(meta.series.length >= 1);

const plan = s.plan();
assert.equal(plan.K, 10 * plan.H);
assert.ok(plan.sources.zscore.includes('fn main') && plan.sources.gram.includes('fn main') && plan.sources.expect.includes('fn main'));
assert.equal(s.tables().length, 10 * plan.T * plan.N);
assert.equal(s.header(0).length, 32);

const ref = JSON.parse(s.runCpu('reference', 60, 10, 1));
const emu = JSON.parse(s.runCpu('emulated', 60, 10, 1));
for (const r of [ref, emu]) {
  assert.ok(r.expected.length > 50, 'expected returns for the universe');
  assert.equal(r.parts.length, 3);
  assert.ok(r.curves.model.length === r.dates.length);
  assert.ok(r.structure && r.structure.latest.length === plan.H && r.structure.series.treeLength.length === r.dates.length);
  // The daily book: its curve, a row in every part, its costs and at most 10 orders with levels.
  assert.ok(r.day && r.curves[r.day.name].length === r.dates.length, 'daily book curve');
  assert.ok(r.parts.every((p) => p.rows.some((x) => x.name === r.day.name)));
  assert.equal(r.day.buyBps, 5);
  assert.ok(r.day.orders.length <= 10 && r.day.orders.every((o) => o.ticker && o.tradeReturn > 0 && o.expected > 0 && (o.take == null || o.take > 0) && (o.stop == null || o.stop < 0)));
  assert.ok(r.recent.every((d) => typeof d.day === 'number'));
}
// The daily book's cost is its own setting.
s.setDayCost(20);
const dear = JSON.parse(s.runCpu('reference', 60, 10, 1));
assert.equal(dear.day.buyBps, 20);
const dayRet = (r) => r.parts[0].rows.find((x) => x.name === r.day.name).annualReturn;
assert.ok(dayRet(dear) < dayRet(ref), 'a dearer daily book earns less');
// The emulated kernels and the reference agree on the ranking of today's expected returns.
const top = (r) => r.expected.slice(0, 10).map((x) => x.ticker);
const overlap = top(ref).filter((t) => top(emu).includes(t)).length;
assert.ok(overlap >= 8, `top-10 overlap ${overlap}`);
const sharpe = (r) => r.parts[0].rows.find((x) => x.name === 'model').sharpe;
assert.ok(Math.abs(sharpe(ref) - sharpe(emu)) < 0.1, `Sharpe ${sharpe(ref)} vs ${sharpe(emu)}`);

// The host protocol, driven with the emulated read-backs a GPU would return: a bad size is refused.
s.loadSample();
assert.throws(() => s.absorb(0, new Float32Array(3)));

console.log(`ok: reference Sharpe ${sharpe(ref).toFixed(2)}, emulated ${sharpe(emu).toFixed(2)}, top-10 overlap ${overlap}/10`);
