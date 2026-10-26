// #include header.wgsl
// The walk-forward model's predictions at every close: each target's coefficients (fitted on the
// days before, beta [T][4][D]) applied to the stock's standardised inputs. One invocation per
// (day, stock). out [T][N][4] (0 where the stock is not defined).
@group(0) @binding(1) var<storage, read> feat : array<f32>;
@group(0) @binding(2) var<storage, read> zs : array<f32>;
@group(0) @binding(3) var<storage, read> beta : array<f32>;
@group(0) @binding(4) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 64u + lane;
  let D : u32 = P.NT + 1u;
  if (s < P.T * P.N) {
    let a : u32 = s % P.N;
    let t : u32 = s / P.N;
    let row : u32 = (t * P.N + a) * P.K;
    for (var k = 0u; k < 4u; k = k + 1u) {
      var y : f32 = 0.0;
      if (feat[row + P.K - 1u] > 0.0) {
        y = beta[(t * 4u + k) * D];
        for (var f = 0u; f < P.NT; f = f + 1u) {
          let x : f32 = min(max((feat[row + f] - zs[(t * P.NT + f) * 2u]) / zs[(t * P.NT + f) * 2u + 1u], -4.0), 4.0);
          y = y + beta[(t * 4u + k) * D + 1u + f] * x;
        }
      }
      out[s * 4u + k] = y;
    }
  }
}
