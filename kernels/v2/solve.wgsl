// #include header.wgsl
// The walk-forward model's coefficients for every day and target at once: each (day, target) solves
// its normal equations (the sums of the days before it, acc, with a ridge of rk[0] times the trace)
// by Gauss-Jordan elimination. The fields of P: T the days, u0 D (the inputs and the constant, at
// most 64), u1 E = D (D + 4) (a day's sums: the D x D matrix, then the 4 targets' vectors). acc [T][E];
// beta [T][4][D] (left as it is where the matrix is empty). One invocation per (day, target).
@group(0) @binding(1) var<storage, read> acc : array<f32>;
@group(0) @binding(2) var<storage, read> rk : array<f32>;
@group(0) @binding(3) var<storage, read_write> beta : array<f32>;

var<private> M : array<f32, 4096>;
var<private> v : array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 64u + lane;
  if (s < P.T * 4u) {
    let t : u32 = s / 4u;
    let k : u32 = s % 4u;
    let D : u32 = P.u0;
    let base : u32 = t * P.u1;
    var tr : f32 = 0.0;
    for (var q = 0u; q < D; q = q + 1u) { tr = tr + acc[base + q * D + q]; }
    if (tr > 0.0) {
      for (var q = 0u; q < D * D; q = q + 1u) { M[q] = acc[base + q]; }
      for (var q = 0u; q < D; q = q + 1u) {
        v[q] = acc[base + D * D + k * D + q];
        M[q * D + q] = M[q * D + q] + rk[0] * tr;
      }
      for (var c = 0u; c < D; c = c + 1u) {
        for (var r = 0u; r < D; r = r + 1u) {
          if (r != c) {
            let q2 : f32 = M[r * D + c] / M[c * D + c];
            for (var q = 0u; q < D; q = q + 1u) { M[r * D + q] = M[r * D + q] - q2 * M[c * D + q]; }
            v[r] = v[r] - q2 * v[c];
          }
        }
      }
      for (var q = 0u; q < D; q = q + 1u) { beta[(t * 4u + k) * D + q] = v[q] / M[q * D + q]; }
    }
  }
}
