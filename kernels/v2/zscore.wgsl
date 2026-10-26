// #include header.wgsl
// Each day's mean and standard deviation of each feature over the stocks defined that day (the
// model's inputs are standardised within the day). One invocation per (day, feature). out
// [T][P.NT][2].
@group(0) @binding(1) var<storage, read> feat : array<f32>;
@group(0) @binding(2) var<storage, read_write> zs : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 64u + lane;
  if (s < P.T * P.NT) {
    let f : u32 = s % P.NT;
    let t : u32 = s / P.NT;
    var n : f32 = 0.0;
    var s1 : f32 = 0.0;
    var s2 : f32 = 0.0;
    for (var a = 0u; a < P.N; a = a + 1u) {
      let row : u32 = (t * P.N + a) * P.K;
      if (feat[row + P.K - 1u] > 0.0) { let x : f32 = feat[row + f]; n = n + 1.0; s1 = s1 + x; s2 = s2 + x * x; }
    }
    var mu : f32 = 0.0;
    var sd : f32 = 1.0;
    if (n > 1.0) { mu = s1 / n; sd = sqrt(max(s2 / n - mu * mu, 1e-12)); }
    zs[s * 2u] = mu;
    zs[s * 2u + 1u] = sd;
  }
}
