// #include header.wgsl
// Each feature's predictive power on each target, day by day: the cross-sectional correlation over
// the stocks defined that day (feat as written by features.wgsl, P.K values per stock-day, the
// first P.NT - 4 features, then 4 targets and the defined flag). One invocation per (day, feature,
// target). out [T][F][4]: the correlation (0 where fewer than 30 stocks).
@group(0) @binding(1) var<storage, read> feat : array<f32>;
@group(0) @binding(2) var<storage, read_write> ic : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 64u + lane;
  let nf : u32 = P.NT;
  if (s < P.T * nf * 4u) {
    let k : u32 = s % 4u;
    let j : u32 = (s / 4u) % nf;
    let t : u32 = s / (4u * nf);
    var n : f32 = 0.0;
    var sx : f32 = 0.0;
    var sy : f32 = 0.0;
    var sxx : f32 = 0.0;
    var syy : f32 = 0.0;
    var sxy : f32 = 0.0;
    for (var a = 0u; a < P.N; a = a + 1u) {
      let row : u32 = (t * P.N + a) * P.K;
      if (feat[row + P.K - 1u] > 0.0) {
        let x : f32 = feat[row + j];
        let y : f32 = min(max(feat[row + nf + k], -6.0), 6.0);
        n = n + 1.0; sx = sx + x; sy = sy + y; sxx = sxx + x * x; syy = syy + y * y; sxy = sxy + x * y;
      }
    }
    var r : f32 = 0.0;
    if (n >= 30.0) {
      let vx : f32 = sxx / n - (sx / n) * (sx / n);
      let vy : f32 = syy / n - (sy / n) * (sy / n);
      if (vx > 0.0 && vy > 0.0) { r = (sxy / n - (sx / n) * (sy / n)) / sqrt(vx * vy); }
    }
    ic[s] = r;
  }
}
