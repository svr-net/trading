// #include header.wgsl
// Each day's sums for the walk-forward model: over the stocks defined that day, the products of
// the inputs [1, features standardised within the day] with each other and with each target.
// Entries 0 .. D D - 1 are X'X (D = P.NT + 1, at most 64), then D 4 entries X'y for the 4 targets.
// out [T][D (D + 4)]. Tiled: one workgroup per day; each tile of 32 stocks is standardised once into
// workgroup memory (an undefined stock as zeros), then every lane adds the tile to its own entries
// (lane, lane + 64, ...), stock by stock in order, so the sums are the plain loop's.
@group(0) @binding(1) var<storage, read> feat : array<f32>;
@group(0) @binding(2) var<storage, read> zs : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;

var<workgroup> xs : array<f32, 2048>;
var<workgroup> ys : array<f32, 128>;
var<private> acc : array<f32, 72>;

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
  let t : u32 = wid.x;
  let D : u32 = P.NT + 1u;
  let E : u32 = D * (D + 4u);
  for (var q = 0u; q < 72u; q = q + 1u) { acc[q] = 0.0; }
  for (var a0 = 0u; a0 < P.N; a0 = a0 + 32u) {
    if (lane < 32u) {
      let a : u32 = a0 + lane;
      var ok : bool = false;
      if (a < P.N) { ok = feat[(t * P.N + a) * P.K + P.K - 1u] > 0.0; }
      for (var j = 0u; j < D; j = j + 1u) {
        var x : f32 = 0.0;
        if (ok) { x = xin(t, a, j); }
        xs[lane * 64u + j] = x;
      }
      for (var k = 0u; k < 4u; k = k + 1u) {
        var y : f32 = 0.0;
        if (ok) { y = min(max(feat[(t * P.N + a) * P.K + P.NT + k], -6.0), 6.0); }
        ys[lane * 4u + k] = y;
      }
    }
    workgroupBarrier();
    let m : u32 = min(32u, P.N - a0);
    for (var q = 0u; q < 72u; q = q + 1u) {
      let e : u32 = lane + q * 64u;
      if (e >= E) { break; }
      var s : f32 = acc[q];
      if (e < D * D) {
        let i : u32 = e / D;
        let j : u32 = e % D;
        for (var b = 0u; b < m; b = b + 1u) { s = s + xs[b * 64u + i] * xs[b * 64u + j]; }
      } else {
        let k : u32 = (e - D * D) / D;
        let j : u32 = (e - D * D) % D;
        for (var b = 0u; b < m; b = b + 1u) { s = s + xs[b * 64u + j] * ys[b * 4u + k]; }
      }
      acc[q] = s;
    }
    workgroupBarrier();
  }
  for (var q = 0u; q < 72u; q = q + 1u) {
    let e : u32 = lane + q * 64u;
    if (e >= E) { break; }
    out[t * E + e] = acc[q];
  }
}
