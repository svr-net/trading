// #include header.wgsl
// One Adam step: each parameter's gradient averaged over the P.nS samples of the mini-batch.
// g [nS][nP], w, m, v [nP]. Grid ceil(nP/64).
@group(0) @binding(1) var<storage, read> g : array<f32>;
@group(0) @binding(2) var<storage, read_write> w : array<f32>;
@group(0) @binding(3) var<storage, read_write> m : array<f32>;
@group(0) @binding(4) var<storage, read_write> v : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let p : u32 = wid.x * 64u + lane;
  if (p < P.nP && P.nS > 0u) {
    var s : f32 = 0.0;
    for (var k = 0u; k < P.nS; k = k + 1u) { s = s + g[k * P.nP + p]; }
    let gk : f32 = s / f32(P.nS);
    m[p] = P.b1 * m[p] + (1.0 - P.b1) * gk;
    v[p] = P.b2 * v[p] + (1.0 - P.b2) * gk * gk;
    let mh : f32 = m[p] / (1.0 - pow(P.b1, P.steps));
    let vh : f32 = v[p] / (1.0 - pow(P.b2, P.steps));
    w[p] = w[p] - P.lr * mh / (sqrt(vh) + 1e-8);
  }
}
