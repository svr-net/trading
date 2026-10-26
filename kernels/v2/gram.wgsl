// #include header.wgsl
// Each day's sums for the walk-forward model: over the stocks defined that day, the products of
// the inputs [1, features standardised within the day] with each other and with each target. One
// invocation per (day, entry): entries 0 .. D D - 1 are X'X (D = P.NT + 1), then D 4 entries X'y
// for the 4 targets. out [T][D (D + 4)].
@group(0) @binding(1) var<storage, read> feat : array<f32>;
@group(0) @binding(2) var<storage, read> zs : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;

fn xin(t : u32, a : u32, j : u32) -> f32 {
  if (j == 0u) { return 1.0; }
  let f : u32 = j - 1u;
  let row : u32 = (t * P.N + a) * P.K;
  let mu : f32 = zs[(t * P.NT + f) * 2u];
  let sd : f32 = zs[(t * P.NT + f) * 2u + 1u];
  return min(max((feat[row + f] - mu) / sd, -4.0), 4.0);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 64u + lane;
  let D : u32 = P.NT + 1u;
  let E : u32 = D * (D + 4u);
  if (s < P.T * E) {
    let e : u32 = s % E;
    let t : u32 = s / E;
    var acc : f32 = 0.0;
    for (var a = 0u; a < P.N; a = a + 1u) {
      let row : u32 = (t * P.N + a) * P.K;
      if (feat[row + P.K - 1u] > 0.0) {
        if (e < D * D) {
          acc = acc + xin(t, a, e / D) * xin(t, a, e % D);
        } else {
          let k : u32 = (e - D * D) / D;
          let j : u32 = (e - D * D) % D;
          acc = acc + xin(t, a, j) * min(max(feat[row + P.NT + k], -6.0), 6.0);
        }
      }
    }
    out[s] = acc;
  }
}
