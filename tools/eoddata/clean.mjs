// Cleans one share's daily bars (sorted by date, splits already applied) before the model sees
// them. Each rule fixes a defect the data audit (tools/audit/data_audit.py) found in EODData's
// bars; the counts say how often each one fired. Prices are never printed.
//
//   unit     a close-to-close move within 25% of x100 or /100 (pence and pounds swapped): the
//            earlier bars are rescaled to the later unit.
//   segment  a hole of more than 90 calendar days (a ticker reused, or a listing suspended and
//            back): the share is cut there; the earlier part keeps its own series (code~1, ~2, ...)
//            and the latest the code, so two companies are never stitched together.
//   novol    a bar without volume (a quote, no trade): left out.
//   wick     a high or low further from the body (open and close) than 25% and 6 times the share's
//            median day range over its last 60 bars: a bad tick, cut back to the body.
//   badopen  an open more than 20% from the previous close while the close is back within 5%: a
//            bad opening print; the bar is left out (its entry price is unknown).
//   spike    a close more than 60% from the previous close that is back within 10% the next day: a
//            bad print; the bar is left out.
// Moves over 60% that last are kept: they happened, and dropping the share would leave out the
// shares that crashed (survivorship bias).
const LOG = Math.log;

export function cleanBars(code, input, { minDays = 0 } = {}) {
  const counts = { unit: 0, segment: 0, novol: 0, wick: 0, badopen: 0, spike: 0, bigmove: 0 };
  let bars = input.map((b) => ({ ...b }));
  // Units: rescale the bars before a x100 or /100 move, from the latest backwards.
  for (let k = bars.length - 1; k > 0; --k) {
    const r = bars[k].c / bars[k - 1].c;
    for (const f of [100, 0.01])
      if (Math.abs(LOG(r / f)) < LOG(1.25)) {
        for (let j = 0; j < k; ++j) for (const p of ['o', 'h', 'l', 'c']) bars[j][p] *= f;
        ++counts.unit;
      }
  }
  // Segments at holes of more than 90 calendar days.
  const parts = [[]];
  for (let k = 0; k < bars.length; ++k) {
    if (k > 0 && (Date.parse(bars[k].d) - Date.parse(bars[k - 1].d)) / 864e5 > 90) parts.push([]), ++counts.segment;
    parts[parts.length - 1].push(bars[k]);
  }
  const segments = [];
  parts.forEach((p, n) => {
    const kept = cleanSegment(p, counts);
    if (kept.length >= minDays && kept.length > 0) segments.push({ code: n === parts.length - 1 ? code : `${code}~${n + 1}`, bars: kept });
  });
  return { segments, counts };
}

function median(v) {
  if (!v.length) return NaN;
  const s = [...v].sort((a, b) => a - b);
  return s[s.length >> 1];
}

function cleanSegment(bars, counts) {
  const out = [], ranges = [];
  for (let k = 0; k < bars.length; ++k) {
    const b = { ...bars[k] };
    if (!(b.v > 0)) { ++counts.novol; continue; }
    const top = Math.max(b.o, b.c), bot = Math.min(b.o, b.c);
    const med = ranges.length >= 10 ? median(ranges.slice(-60)) : 0.03;
    const limit = Math.max(LOG(1.25), 6 * med);
    if (LOG(b.h / top) > limit) b.h = top, ++counts.wick;
    if (LOG(bot / b.l) > limit) b.l = bot, ++counts.wick;
    const prev = out[out.length - 1];
    if (prev) {
      const gap = LOG(b.o / prev.c), move = LOG(b.c / prev.c);
      if (Math.abs(gap) > LOG(1.2) && Math.abs(move) < LOG(1.05)) { ++counts.badopen; continue; }
      if (Math.abs(move) > LOG(1.6)) {
        const next = bars[k + 1];
        if (next && next.v > 0 && Math.abs(LOG(next.c / prev.c)) < LOG(1.1)) { ++counts.spike; continue; }
        ++counts.bigmove;
      }
    }
    out.push(b);
    if (b.h > b.l) ranges.push(LOG(b.h / b.l));
  }
  return out;
}
