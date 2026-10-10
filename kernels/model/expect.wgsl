// #include header.wgsl
// E = z . v per stock and day. Grid ceil(N/64) x days.
@group(0) @binding(2) var<storage, read> z : array<f32>;
@group(0) @binding(3) var<storage, read> v : array<f32>;
@group(0) @binding(4) var<storage, read_write> e : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let a = id.x;
  let td = id.y;
  if (a >= P.N || td >= P.cd) { return; }
  var acc = 0.0;
  for (var k = 0u; k < P.K; k = k + 1u) { acc = acc + z[(td * P.N + a) * P.K + k] * v[td * P.K + k]; }
  e[td * P.N + a] = acc;
}
