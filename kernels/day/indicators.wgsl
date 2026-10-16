// #include header.wgsl
// Technical indicators of every stock at the closes t0 .. t0 + cd - 1, at each horizon h (P.hz,
// P.nH of them): momentum (the h-day log move over its realised volatility), the close's place in
// the h-day range, RSI (net share of up moves), volume against its h-day mean (log) and the day's
// range against its h-day mean (log). Bars: 5 tables [o, h, l, c, v][T][N], 0 where there is no
// bar. Grid ceil(N/64) x cd; out [cd][N][FI], FI = 5 nH.
@group(0) @binding(1) var<storage, read> bars : array<f32>;
@group(0) @binding(2) var<storage, read_write> ind : array<f32>;

fn bar(f : u32, t : u32, a : u32) -> f32 { return bars[(f * P.T + t) * P.N + a]; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let a : u32 = wid.x * 64u + lane;
  let td : u32 = wid.y;
  if (a < P.N && td < P.cd) {
    let t : u32 = P.t0 + td;
    let row : u32 = (td * P.N + a) * P.FI;
    for (var j = 0u; j < P.FI; j = j + 1u) { ind[row + j] = 0.0; }
    let ct : f32 = bar(3u, t, a);
    if (ct > 0.0) {
      for (var k = 0u; k < P.nH; k = k + 1u) {
        let h : u32 = horizon(k);
        if (t >= h) {
          var hi : f32 = 0.0;
          var lo : f32 = 0.0;
          var up : f32 = 0.0;
          var dn : f32 = 0.0;
          var ss : f32 = 0.0;
          var lv : f32 = 0.0;
          var lr : f32 = 0.0;
          var n : f32 = 0.0;
          var prev : f32 = 0.0;
          var first : f32 = 0.0;
          for (var u = t - h; u <= t; u = u + 1u) {
            let c : f32 = bar(3u, u, a);
            let bh : f32 = bar(1u, u, a);
            let bl : f32 = bar(2u, u, a);
            if (c > 0.0 && bl > 0.0 && bh > bl) {
              if (n > 0.0) {
                let r : f32 = log(c / prev);
                if (r > 0.0) { up = up + r; } else { dn = dn - r; }
                ss = ss + r * r;
                hi = max(hi, bh);
                lo = min(lo, bl);
              } else {
                first = c;
                hi = bh;
                lo = bl;
              }
              prev = c;
              lv = lv + log(bar(4u, u, a));
              lr = lr + log(bh / bl);
              n = n + 1.0;
            }
          }
          if (n >= 2.0 && hi > lo) {
            let o : u32 = row + 5u * k;
            if (ss > 0.0) { ind[o] = log(ct / first) / sqrt(ss); }
            ind[o + 1u] = (ct - lo) / (hi - lo) - 0.5;
            if (up + dn > 0.0) { ind[o + 2u] = (up - dn) / (up + dn); }
            let vt : f32 = bar(4u, t, a);
            if (vt > 0.0) { ind[o + 3u] = log(vt) - lv / n; }
            let rt : f32 = log(bar(1u, t, a) / bar(2u, t, a));
            if (rt > 0.0 && lr > 0.0) { ind[o + 4u] = log(rt / (lr / n)); }
          }
        }
      }
    }
  }
}
