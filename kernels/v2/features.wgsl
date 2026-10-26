// #include header.wgsl
// Each stock's features at every close from its own daily OHLCV, and the next day's targets. One
// invocation per stock walks its days in order (the averages are recursive, so only the past is
// used). bars: 5 tables [o, h, l, c, v][T][N], 0 where there is no bar. out [T][N][P.K]:
//   0..5   momentum: the log return over the last 2, 4, ..., 64 days, over sigma sqrt(days)
//   6..8   volume shock: log volume less its average over 4, 16, 64 days
//   9..11  range shock: log of the day's range over its average over 4, 16, 64 days
//   12     the day's open-to-close over sigma      13  close-to-close over sigma
//   14     the gap (open over the previous close) over sigma
//   15     the close's place in the day's range (-0.5 low .. 0.5 high)
//   16     upper wick, 17 lower wick (shares of the range)
//   18     signed volume shock: the 16-day volume shock times the sign of the open-to-close
//   19     liquidity: the log of the traded value (close x volume), averaged over 16 days
//   20     the log of sigma (the stock's volatility level)
//   21..24 the next day's targets: open-to-close, open-to-high, open-to-low (as a fall), close-to-close, over sigma
//   25     1 where every feature and target is defined
// sigma: the exponentially weighted standard deviation of close-to-close log returns (half-life 16 days).
@group(0) @binding(1) var<storage, read> bars : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

var<private> ret : array<f32, 65>;
var<private> lv : array<f32, 3>;
var<private> lr : array<f32, 3>;
var<private> ltv : f32;

fn bar(f : u32, t : u32, a : u32) -> f32 { return bars[(f * P.T + t) * P.N + a]; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let a : u32 = wid.x * 64u + lane;
  if (a < P.N) {
    var n : u32 = 0u;
    var var16 : f32 = 0.0;
    var pc : f32 = 0.0;
    for (var j = 0u; j < 65u; j = j + 1u) { ret[j] = 0.0; }
    for (var j = 0u; j < 3u; j = j + 1u) { lv[j] = 0.0; lr[j] = 0.0; }
    ltv = 0.0;
    let l16 : f32 = pow(2.0, -1.0 / 16.0);
    for (var t = 0u; t < P.T; t = t + 1u) {
      let row : u32 = (t * P.N + a) * P.K;
      for (var j = 0u; j < P.K; j = j + 1u) { out[row + j] = 0.0; }
      let o : f32 = bar(0u, t, a);
      let h : f32 = bar(1u, t, a);
      let l : f32 = bar(2u, t, a);
      let c : f32 = bar(3u, t, a);
      let v : f32 = bar(4u, t, a);
      if (!(o > 0.0 && h >= l && l > 0.0 && c > 0.0)) { pc = 0.0; n = 0u; continue; }
      var r : f32 = 0.0;
      if (pc > 0.0) { r = log(c / pc); }
      // The return history (newest first).
      for (var j = 64u; j > 0u; j = j - 1u) { ret[j] = ret[j - 1u]; }
      ret[0] = r;
      if (n == 0u) { var16 = r * r; } else { var16 = l16 * var16 + (1.0 - l16) * r * r; }
      let rg : f32 = log(max(h / l, 1.0 + 1e-6));
      var lvol : f32 = 0.0;
      if (v > 0.0) { lvol = log(v); }
      let sig : f32 = sqrt(var16);
      if (n >= 64u && sig > 0.0 && v > 0.0) {
        var cum : f32 = 0.0;
        var k : u32 = 0u;
        for (var j = 0u; j < 6u; j = j + 1u) {
          var hz : u32 = 2u;
          for (var b = 0u; b < j; b = b + 1u) { hz = hz * 2u; }
          loop {
            if (k >= hz) { break; }
            cum = cum + ret[k];
            k = k + 1u;
          }
          out[row + j] = cum / (sig * sqrt(f32(hz)));
        }
        for (var j = 0u; j < 3u; j = j + 1u) {
          out[row + 6u + j] = lvol - lv[j];
          out[row + 9u + j] = log(rg / max(lr[j], 1e-9));
        }
        out[row + 12u] = log(c / o) / sig;
        out[row + 13u] = r / sig;
        out[row + 14u] = log(o / pc) / sig;
        let span : f32 = max(h - l, 1e-12);
        out[row + 15u] = (c - l) / span - 0.5;
        out[row + 16u] = (h - max(o, c)) / span;
        out[row + 17u] = (min(o, c) - l) / span;
        var sgn : f32 = 0.0;
        if (c > o) { sgn = 1.0; } else if (c < o) { sgn = -1.0; }
        out[row + 18u] = sgn * (lvol - lv[1]);
        out[row + 19u] = ltv;
        out[row + 20u] = log(sig);
        // The next day's targets.
        if (t + 1u < P.T) {
          let o1 : f32 = bar(0u, t + 1u, a);
          let h1 : f32 = bar(1u, t + 1u, a);
          let l1 : f32 = bar(2u, t + 1u, a);
          let c1 : f32 = bar(3u, t + 1u, a);
          if (o1 > 0.0 && h1 >= l1 && l1 > 0.0 && c1 > 0.0) {
            out[row + 21u] = log(c1 / o1) / sig;
            out[row + 22u] = log(h1 / o1) / sig;
            out[row + 23u] = log(o1 / l1) / sig;
            out[row + 24u] = log(c1 / c) / sig;
            out[row + 25u] = 1.0;
          }
        }
      }
      // Update the averages after the day's features (only the past enters them).
      for (var j = 0u; j < 3u; j = j + 1u) {
        let lam : f32 = pow(2.0, -1.0 / (4.0 * pow(4.0, f32(j))));
        if (n == 0u) { lv[j] = lvol; lr[j] = rg; } else {
          if (v > 0.0) { lv[j] = lam * lv[j] + (1.0 - lam) * lvol; }
          lr[j] = lam * lr[j] + (1.0 - lam) * rg;
        }
      }
      if (v > 0.0) {
        let x : f32 = log(c * v);
        if (n == 0u) { ltv = x; } else { ltv = l16 * ltv + (1.0 - l16) * x; }
      }
      pc = c;
      n = n + 1u;
    }
  }
}
