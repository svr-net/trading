// #include header.wgsl
// The network's outputs for P.nS samples: two hidden layers of P.Hd rectified linear units (at
// most 64), P.Out linear outputs. x [nS][In], y [nS][Out]. Grid ceil(nS/8).
@group(0) @binding(1) var<storage, read> w : array<f32>;
@group(0) @binding(2) var<storage, read> x : array<f32>;
@group(0) @binding(3) var<storage, read_write> y : array<f32>;

var<private> h1 : array<f32, 64>;
var<private> h2 : array<f32, 64>;

@compute @workgroup_size(8)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 8u + lane;
  if (s < P.nS) {
    for (var o = 0u; o < P.Hd; o = o + 1u) {
      var z : f32 = w[oB1() + o];
      for (var k = 0u; k < P.In; k = k + 1u) { z = z + w[o * P.In + k] * x[s * P.In + k]; }
      h1[o] = max(z, 0.0);
    }
    for (var o = 0u; o < P.Hd; o = o + 1u) {
      var z : f32 = w[oB2() + o];
      for (var k = 0u; k < P.Hd; k = k + 1u) { z = z + w[oW2() + o * P.Hd + k] * h1[k]; }
      h2[o] = max(z, 0.0);
    }
    for (var o = 0u; o < P.Out; o = o + 1u) {
      var z : f32 = w[oB3() + o];
      for (var k = 0u; k < P.Hd; k = k + 1u) { z = z + w[oW3() + o * P.Hd + k] * h2[k]; }
      y[s * P.Out + o] = z;
    }
  }
}
