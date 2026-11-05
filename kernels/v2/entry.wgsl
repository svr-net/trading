// #include header.wgsl
// The records' means over the exit settings, the selection's step after records.wgsl: for each pool,
// ranking and entry (an entry r of P.T), the mean over its P.u0 level settings (g = r * u0 + level)
// of each record's two scores, sc (the mean shrunk by its t-statistic) and so (Sortino). One lane an
// entry, adding in the levels' order, so the sums are the ones a loop over the settings makes.
@group(0) @binding(1) var<storage, read> sc : array<f32>;
@group(0) @binding(2) var<storage, read> so : array<f32>;
@group(0) @binding(3) var<storage, read_write> es : array<f32>;
@group(0) @binding(4) var<storage, read_write> eo : array<f32>;

@compute @workgroup_size(16)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let r : u32 = wid.x * 16u + lane;
  if (r < P.T) {
    var a : f32 = 0.0;
    var b : f32 = 0.0;
    for (var g = r * P.u0; g < (r + 1u) * P.u0; g = g + 1u) {
      a = a + sc[g];
      b = b + so[g];
    }
    es[r] = a / f32(P.u0);
    eo[r] = b / f32(P.u0);
  }
}
