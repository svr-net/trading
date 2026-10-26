// #include header.wgsl
// Interactions, so a linear model can let a signal's strength depend on the stock's liquidity and
// volatility: the product of each listed signal's within-day standardised value with the
// liquidity's and with the volatility level's (features.wgsl 19 and 20). sig [P.u0]: the signals;
// feat [T][N][P.K] with P.NT features; zs [T][P.NT][2] (the day's means and deviations). out
// [T][N][2 P.u0]. One invocation per (day, stock).
@group(0) @binding(1) var<storage, read> feat : array<f32>;
@group(0) @binding(2) var<storage, read> zs : array<f32>;
@group(0) @binding(3) var<storage, read> sig : array<f32>;
@group(0) @binding(4) var<storage, read_write> out : array<f32>;

fn z(t : u32, a : u32, f : u32) -> f32 {
  let x : f32 = feat[(t * P.N + a) * P.K + f];
  return min(max((x - zs[(t * P.NT + f) * 2u]) / zs[(t * P.NT + f) * 2u + 1u], -4.0), 4.0);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 64u + lane;
  if (s < P.T * P.N) {
    let a : u32 = s % P.N;
    let t : u32 = s / P.N;
    let M : u32 = P.u0;
    let zl : f32 = z(t, a, 19u);
    let zv : f32 = z(t, a, 20u);
    for (var j = 0u; j < M; j = j + 1u) {
      let x : f32 = z(t, a, u32(sig[j]));
      out[s * 2u * M + j] = x * zl;
      out[s * 2u * M + M + j] = x * zv;
    }
  }
}
