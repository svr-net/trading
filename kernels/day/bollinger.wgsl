// #include header.wgsl
// Bollinger bands of every stock at every close, at each horizon h (P.hz, P.nH of them): the mean m
// and standard deviation s of the last h closes. One invocation per stock runs through its days in
// order with running sums; days without a bar are skipped (the window is the last h bars) and left
// 0. Bars: 5 tables [o, h, l, c, v][T][N], 0 where there is no bar. Grid ceil(N/64); out
// [T][N][3 nH], per horizon:
//   0  the close's place in the bands, (c - m) / s (%B in standard deviations)
//   1  the bands' width, log(s / c) (narrow before a "squeeze" ends)
//   2  the band unit s / c, the stop and take-profit unit of the Bollinger levels
// Each is 0 until the stock has h + 1 bars.
@group(0) @binding(1) var<storage, read> bars : array<f32>;
@group(0) @binding(2) var<storage, read_write> ind : array<f32>;

fn bar(f : u32, t : u32, a : u32) -> f32 { return bars[(f * P.T + t) * P.N + a]; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let a : u32 = wid.x * 64u + lane;
  if (a < P.N) {
    let w : u32 = 3u * P.nH;
    for (var k = 0u; k < P.nH; k = k + 1u) {
      let h : u32 = horizon(k);
      var s1 : f32 = 0.0;
      var s2 : f32 = 0.0;
      var n : u32 = 0u;
      var tail : u32 = 0u;
      for (var t = 0u; t < P.T; t = t + 1u) {
        let row : u32 = (t * P.N + a) * w + 3u * k;
        ind[row] = 0.0;
        ind[row + 1u] = 0.0;
        ind[row + 2u] = 0.0;
        let c : f32 = bar(3u, t, a);
        if (c > 0.0) {
          s1 = s1 + c;
          s2 = s2 + c * c;
          n = n + 1u;
          if (n > h) {
            // Drop the oldest bar of the window: the next day with a bar from the tail.
            loop {
              let ct : f32 = bar(3u, tail, a);
              tail = tail + 1u;
              if (ct > 0.0) {
                s1 = s1 - ct;
                s2 = s2 - ct * ct;
                break;
              }
            }
            let m : f32 = s1 / f32(h);
            let v : f32 = s2 / f32(h) - m * m;
            if (v > 0.0) {
              let s : f32 = sqrt(v);
              ind[row] = (c - m) / s;
              ind[row + 1u] = log(s / c);
              ind[row + 2u] = s / c;
            }
          }
        }
      }
    }
  }
}
