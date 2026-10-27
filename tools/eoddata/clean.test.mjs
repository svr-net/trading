// node tools/eoddata/clean.test.mjs: the cleaning rules (clean.mjs) on made-up bars.
import assert from 'node:assert/strict';
import { cleanBars } from './clean.mjs';

const day = (k) => new Date(Date.UTC(2024, 0, 1) + k * 864e5).toISOString().slice(0, 10);
const bar = (k, c, v = 1000) => ({ d: day(k), o: c, h: c * 1.01, l: c * 0.99, c, v });
const series = (n, f) => Array.from({ length: n }, (_, k) => f(k));

// A x100 unit swap: the earlier bars are rescaled, nothing is dropped.
{
  const b = series(40, (k) => bar(k, k < 20 ? 4 + 0.01 * k : 400 + k));
  const { segments, counts } = cleanBars('A', b);
  assert.equal(counts.unit, 1);
  assert.equal(segments.length, 1);
  assert.ok(Math.abs(segments[0].bars[0].c - 400) < 1e-9 && segments[0].bars.length === 40);
}
// A hole of 200 days: two series, the earlier as A~1.
{
  const b = [...series(30, (k) => bar(k, 10)), ...series(30, (k) => bar(230 + k, 50))];
  const { segments, counts } = cleanBars('A', b);
  assert.equal(counts.segment, 1);
  assert.deepEqual(segments.map((s) => s.code), ['A~1', 'A']);
}
// No volume, a bad wick, a bad open and a one-day spike; a lasting crash is kept.
{
  const b = series(60, (k) => bar(k, 100));
  b[20].v = 0;
  b[25].h = 100 * 50;  // a tick 50x the body
  b[30].o = 160;       // the open 60% away, the close unchanged
  b[35].c = 300, b[35].h = 303;  // a spike that is gone the next day
  for (let k = 50; k < 60; ++k) b[k] = bar(k, 35);  // a crash that lasts (x2.86: no split ratio, under x3)
  const { segments, counts } = cleanBars('A', b);
  assert.deepEqual([counts.novol, counts.wick, counts.badopen, counts.spike, counts.bigmove], [1, 1, 1, 1, 1]);
  const kept = segments[0].bars;
  assert.equal(kept.length, 57);
  assert.ok(kept.every((x) => x.h <= Math.max(x.o, x.c) * 1.3));
  assert.ok(kept.some((x) => x.c === 35));
}
// A 1-for-10 reverse split missing from the list: adjusted, one series.
{
  const b = series(40, (k) => bar(k, k < 20 ? 1 : 10.1));
  const { segments, counts } = cleanBars('A', b);
  assert.equal(counts.split, 1);
  assert.equal(segments.length, 1);
  assert.ok(Math.abs(segments[0].bars[0].c - 10) < 1e-9 && Math.abs(segments[0].bars[0].v - 100) < 1e-9);
}
// An overnight jump of x4.3 (no split ratio): two series.
{
  const b = series(40, (k) => bar(k, k < 20 ? 10 : 43.4));
  const { segments, counts } = cleanBars('A', b);
  assert.equal(counts.split + counts.jump, 1);
  assert.equal(counts.jump, 1);
  assert.deepEqual(segments.map((s) => s.code), ['A~1', 'A']);
}
// Too short: dropped.
assert.equal(cleanBars('A', series(10, (k) => bar(k, 10)), { minDays: 20 }).segments.length, 0);
console.log('clean.mjs: all tests passed');
