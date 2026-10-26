// #include header.wgsl
// Market-wide features at every close from named series (futures, indices; some close after the
// shares, so their last move carries news the shares have not priced yet): each series' log move
// over its own volatility (exponentially weighted, half-life 16 days), and its level against its
// 64-day average. One invocation per series walks its days. closes [M][T] (0 before a series
// starts; carried forward over its missing days). out [T][2 M], 0 until 16 days of history.
// P.u0 = M.
@group(0) @binding(1) var<storage, read> closes : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 64u + lane;
  let M : u32 = P.u0;
  if (s < M) {
    let l16 : f32 = pow(2.0, -1.0 / 16.0);
    let l64 : f32 = pow(2.0, -1.0 / 64.0);
    var pc : f32 = 0.0;
    var v : f32 = 0.0;
    var avg : f32 = 0.0;
    var n : u32 = 0u;
    for (var t = 0u; t < P.T; t = t + 1u) {
      out[t * 2u * M + 2u * s] = 0.0;
      out[t * 2u * M + 2u * s + 1u] = 0.0;
      let c : f32 = closes[s * P.T + t];
      if (!(c > 0.0)) { continue; }
      if (pc > 0.0) {
        let r : f32 = log(c / pc);
        if (n >= 16u && v > 0.0) {
          out[t * 2u * M + 2u * s] = r / sqrt(v);
          out[t * 2u * M + 2u * s + 1u] = log(c / avg);
        }
        if (n == 0u) { v = r * r; } else { v = l16 * v + (1.0 - l16) * r * r; }
        n = n + 1u;
      }
      if (avg > 0.0) { avg = l64 * avg + (1.0 - l64) * c; } else { avg = c; }
      pc = c;
    }
  }
}
