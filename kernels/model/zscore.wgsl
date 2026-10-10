// #include header.wgsl
// Signal k of every stock on day c0 + td and its rank z-score among the day's eligible stocks (ties:
// by stock index; 0 where not eligible), fused: one workgroup per (signal, day), grid K x days.
// Phase 1 writes the 64 lanes' signals; phase 2 ranks each stock against the day's stocks staged
// tile by tile (64 at a time) in workgroup memory.
@group(0) @binding(2) var<storage, read_write> sig : array<f32>;
@group(0) @binding(3) var<storage, read_write> z : array<f32>;

const WG : u32 = 64u;
var<workgroup> tv : array<f32, 64>;
var<workgroup> tok : array<u32, 64>;

// Families: 0 return, 1 low volatility, 2 nearness to the high, 3 trend quality, 4 small size,
// 5 RSI (share of up moves in the absolute moves), 6 Bollinger z-score (close against the mean and
// spread of the previous h closes), 7 stochastic %K (close within the low-high range), 8 money flow
// (share of traded value on up days), 9 intraday range (negative mean high-low span).
fn signal(fam : u32, h : u32, t : u32, a : u32) -> f32 {
  let fh = f32(h);
  let lc = at(0u, t, a);
  if (fam == 0u) { return lc - at(0u, t - h, a); }
  if (fam == 1u || fam == 3u) {
    let d1 = at(1u, t, a) - at(1u, t - h, a);
    let d2 = at(2u, t, a) - at(2u, t - h, a);
    let vol = sqrt(max(0.0, (d2 - d1 * d1 / fh) / (fh - 1.0)));
    if (fam == 1u) { return -vol; }
    return select(0.0, (lc - at(0u, t - h, a)) / (vol * sqrt(fh)), vol > 0.0);
  }
  if (fam == 2u) {
    var mx = lc;
    for (var u = 1u; u < h; u = u + 1u) { mx = max(mx, at(0u, t - u, a)); }
    return lc - mx;
  }
  if (fam == 4u) { return -(at(3u, t, a) - at(3u, t - h, a)) / fh; }
  if (fam == 5u) {
    var up = 0.0;
    var tot = 0.0;
    for (var u = 0u; u < h; u = u + 1u) {
      let d = at(0u, t - u, a) - at(0u, t - u - 1u, a);
      up = up + max(d, 0.0);
      tot = tot + abs(d);
    }
    return select(0.5, up / tot, tot > 0.0);
  }
  if (fam == 6u) {
    // Against the previous h closes (with today's close in the window, a 2-day z-score is always
    // +-0.707: only the sign of the last move).
    var s = 0.0;
    for (var u = 1u; u <= h; u = u + 1u) { s = s + at(0u, t - u, a); }
    let m = s / fh;
    var v = 0.0;
    for (var u = 1u; u <= h; u = u + 1u) { v = v + (at(0u, t - u, a) - m) * (at(0u, t - u, a) - m); }
    let sd = sqrt(v / (fh - 1.0));
    return select(0.0, (lc - m) / sd, sd > 0.0);
  }
  if (fam == 7u) {
    var lo = at(8u, t, a);
    var hi = at(7u, t, a);
    for (var u = 1u; u < h; u = u + 1u) { lo = min(lo, at(8u, t - u, a)); hi = max(hi, at(7u, t - u, a)); }
    return select(0.5, (lc - lo) / (hi - lo), hi > lo);
  }
  if (fam == 8u) {
    var up = 0.0;
    var tot = 0.0;
    for (var u = 0u; u < h; u = u + 1u) {
      let v = at(9u, t - u, a);
      if (at(0u, t - u, a) > at(0u, t - u - 1u, a)) { up = up + v; }
      tot = tot + v;
    }
    return select(0.5, up / tot, tot > 0.0);
  }
  var s = 0.0;
  for (var u = 0u; u < h; u = u + 1u) { s = s + (at(7u, t - u, a) - at(8u, t - u, a)); }
  return -s / fh;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let k : u32 = wid.x;
  let td : u32 = wid.y;
  let t : u32 = P.c0 + td;
  let fam : u32 = k / P.H;
  let h : u32 = horizon(k % P.H);
  for (var a = lane; a < P.N; a = a + WG) {
    var x = 0.0;
    if (at(4u, t, a) > 0.5) { x = signal(fam, h, t, a); }
    sig[(td * P.N + a) * P.K + k] = x;
  }
  storageBarrier();
  workgroupBarrier();
  let blocks : u32 = (P.N + WG - 1u) / WG;
  for (var ib = 0u; ib < blocks; ib = ib + 1u) {
    let a : u32 = ib * WG + lane;
    var x : f32 = 0.0;
    if (a < P.N) { x = sig[(td * P.N + a) * P.K + k]; }
    var n : u32 = 0u;
    var r : u32 = 0u;
    for (var jb = 0u; jb < blocks; jb = jb + 1u) {
      let b : u32 = jb * WG + lane;
      var y : f32 = 0.0;
      var e : u32 = 0u;
      if (b < P.N && at(4u, t, b) > 0.5) { y = sig[(td * P.N + b) * P.K + k]; e = 1u; }
      workgroupBarrier();
      tv[lane] = y;
      tok[lane] = e;
      workgroupBarrier();
      for (var j = 0u; j < WG; j = j + 1u) {
        if (tok[j] == 1u) {
          n = n + 1u;
          let bj = jb * WG + j;
          if (tv[j] < x || (tv[j] == x && bj < a)) { r = r + 1u; }
        }
      }
    }
    if (a < P.N) {
      var out = 0.0;
      if (at(4u, t, a) > 0.5) {
        let fn1 = f32(n);
        out = (f32(r) - 0.5 * (fn1 - 1.0)) / sqrt(max((fn1 * fn1 - 1.0) / 12.0, 1.0e-12));
      }
      z[(td * P.N + a) * P.K + k] = out;
    }
  }
}
