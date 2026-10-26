// #include header.wgsl
// Each sample's gradient of the quantile (pinball) loss: output o is the quantile tau[o] of target
// y[s][o]. x [nS][In], y [nS][Out], g [nS][nP]. Grid ceil(nS/8).
@group(0) @binding(1) var<storage, read> w : array<f32>;
@group(0) @binding(2) var<storage, read> x : array<f32>;
@group(0) @binding(3) var<storage, read> y : array<f32>;
@group(0) @binding(4) var<storage, read> tau : array<f32>;
@group(0) @binding(5) var<storage, read_write> g : array<f32>;

var<private> h1 : array<f32, 64>;
var<private> h2 : array<f32, 64>;
var<private> d2 : array<f32, 64>;
var<private> d3 : array<f32, 64>;

@compute @workgroup_size(8)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 8u + lane;
  if (s < P.nS) {
    let gs : u32 = s * P.nP;
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
      if (z > y[s * P.Out + o]) { d3[o] = 1.0 - tau[o]; } else { d3[o] = 0.0 - tau[o]; }
      g[gs + oB3() + o] = d3[o];
      for (var k = 0u; k < P.Hd; k = k + 1u) { g[gs + oW3() + o * P.Hd + k] = d3[o] * h2[k]; }
    }
    for (var k = 0u; k < P.Hd; k = k + 1u) {
      var d : f32 = 0.0;
      if (h2[k] > 0.0) {
        for (var o = 0u; o < P.Out; o = o + 1u) { d = d + w[oW3() + o * P.Hd + k] * d3[o]; }
      }
      d2[k] = d;
      g[gs + oB2() + k] = d;
      for (var j = 0u; j < P.Hd; j = j + 1u) { g[gs + oW2() + k * P.Hd + j] = d * h1[j]; }
    }
    for (var j = 0u; j < P.Hd; j = j + 1u) {
      var d : f32 = 0.0;
      if (h1[j] > 0.0) {
        for (var k = 0u; k < P.Hd; k = k + 1u) { d = d + w[oW2() + k * P.Hd + j] * d2[k]; }
      }
      g[gs + oB1() + j] = d;
      for (var k = 0u; k < P.In; k = k + 1u) { g[gs + j * P.In + k] = d * x[s * P.In + k]; }
    }
  }
}
