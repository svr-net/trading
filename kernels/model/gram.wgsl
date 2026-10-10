// #include header.wgsl
// Per day: the Gram matrix of the z-scores (upper triangle), each signal's projection on the
// next-day targets, and the counts of eligible and targeted stocks. Grid ceil(stride/64) x days.
@group(0) @binding(2) var<storage, read> z : array<f32>;
@group(0) @binding(3) var<storage, read_write> g : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let p = id.x;
  let td = id.y;
  if (p >= P.stride || td >= P.cd) { return; }
  let t = P.c0 + td;
  var acc = 0.0;
  if (p < P.nPairs) {
    var k = 0u;
    var rem = p;
    loop {
      let len = P.K - k;
      if (rem < len) { break; }
      rem = rem - len;
      k = k + 1u;
    }
    let l = k + rem;
    for (var a = 0u; a < P.N; a = a + 1u) {
      acc = acc + z[(td * P.N + a) * P.K + k] * z[(td * P.N + a) * P.K + l];
    }
  } else if (p < P.nPairs + P.K) {
    let k = p - P.nPairs;
    for (var a = 0u; a < P.N; a = a + 1u) {
      acc = acc + z[(td * P.N + a) * P.K + k] * (at(6u, t, a) * at(5u, t, a));
    }
  } else if (p == P.nPairs + P.K) {
    for (var a = 0u; a < P.N; a = a + 1u) { acc = acc + at(4u, t, a); }
  } else {
    for (var a = 0u; a < P.N; a = a + 1u) { acc = acc + at(4u, t, a) * at(6u, t, a); }
  }
  g[td * P.stride + p] = acc;
}
