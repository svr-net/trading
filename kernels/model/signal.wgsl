// #include header.wgsl
// One stock's signal k on day c0 + td (0 where the stock is not eligible).
// Grid ceil(N/64) x K x days: id = (stock, signal, day of the chunk).
@group(0) @binding(2) var<storage, read_write> sig : array<f32>;

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
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let a = id.x;
  let k = id.y;
  let td = id.z;
  if (a >= P.N || k >= P.K || td >= P.cd) { return; }
  let t = P.c0 + td;
  var x = 0.0;
  if (at(4u, t, a) > 0.5) { x = signal(k / P.H, horizon(k % P.H), t, a); }
  sig[(td * P.N + a) * P.K + k] = x;
}
