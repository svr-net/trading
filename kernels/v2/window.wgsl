// #include header.wgsl
// Each share's last 60 days up to the close t (the outlier screen in apps/ofm_research.cpp): over the
// days with a close, high and low, the mean log traded value, the share of days its close did not
// move from the day before's, and the mean log range (high / low); ok when at least 40 days had a bar
// and one had a close before it. bars: 5 tables [o, h, l, c, v][T][N], 0 where there is no bar.
// out: 3 tables [value, unmoved, range][T][N]; ok [T][N] (1 or 0). One invocation a share's day.
@group(0) @binding(1) var<storage, read> bars : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<storage, read_write> ok : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let x : u32 = wid.x * 64u + lane;
  if (x < P.T * P.N) {
    let t : u32 = x / P.N;
    let i : u32 = x % P.N;
    var n : f32 = 0.0;
    var lv : f32 = 0.0;
    var st : f32 = 0.0;
    var sn : f32 = 0.0;
    var rg : f32 = 0.0;
    var u : u32 = 0u;
    if (t >= 59u) { u = t - 59u; }
    for (; u <= t; u = u + 1u) {
      let h : f32 = bars[(1u * P.T + u) * P.N + i];
      let l : f32 = bars[(2u * P.T + u) * P.N + i];
      let c : f32 = bars[(3u * P.T + u) * P.N + i];
      let v : f32 = bars[(4u * P.T + u) * P.N + i];
      if (!(c > 0.0) || !(h > 0.0) || !(l > 0.0)) { continue; }
      n = n + 1.0;
      lv = lv + log(c * max(v, 1.0));
      rg = rg + log(h / l);
      if (u > 0u) {
        let c0 : f32 = bars[(3u * P.T + u - 1u) * P.N + i];
        if (c0 > 0.0) {
          if (c == c0) { st = st + 1.0; }
          sn = sn + 1.0;
        }
      }
    }
    ok[x] = 0.0;
    out[x] = 0.0;
    out[P.T * P.N + x] = 0.0;
    out[2u * P.T * P.N + x] = 0.0;
    if (n >= 40.0 && sn > 0.0) {
      ok[x] = 1.0;
      out[x] = lv / n;
      out[P.T * P.N + x] = st / sn;
      out[2u * P.T * P.N + x] = rg / n;
    }
  }
}
