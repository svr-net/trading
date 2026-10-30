// #include header.wgsl
// The market's daily return: the average close-to-close log return of the stocks with a bar on
// the day and the day before. bars: 5 tables [o, h, l, c, v][T][N], 0 where there is no bar.
// out [T]. One invocation per day.
@group(0) @binding(1) var<storage, read> bars : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let t : u32 = wid.x * 64u + lane;
  if (t < P.T) {
    var s : f32 = 0.0;
    var n : f32 = 0.0;
    if (t > 0u) {
      for (var a = 0u; a < P.N; a = a + 1u) {
        let c0 : f32 = bars[(3u * P.T + t - 1u) * P.N + a];
        let c1 : f32 = bars[(3u * P.T + t) * P.N + a];
        if (c0 > 0.0 && c1 > 0.0) { s = s + log(c1 / c0); n = n + 1.0; }
      }
    }
    out[t] = 0.0;
    if (n > 0.0) { out[t] = s / n; }
  }
}
