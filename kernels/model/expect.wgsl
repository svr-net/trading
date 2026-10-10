// #include header.wgsl
// E = z . v per stock and day: one workgroup per 64 stocks of a day; the day's v is staged in
// workgroup memory, the dot product runs four signals at a time. Grid ceil(N/64) x days.
@group(0) @binding(2) var<storage, read> z : array<f32>;
@group(0) @binding(3) var<storage, read> v : array<f32>;
@group(0) @binding(4) var<storage, read_write> e : array<f32>;

var<workgroup> vs : array<f32, 160>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let td : u32 = wid.y;
  let a : u32 = wid.x * 64u + lane;
  if (td < P.cd) {
    for (var k = lane; k < P.K; k = k + 64u) { vs[k] = v[td * P.K + k]; }
    workgroupBarrier();
    if (a < P.N) {
      let row = (td * P.N + a) * P.K;
      var acc = vec4<f32>(0.0);
      var k = 0u;
      for (; k + 4u <= P.K; k = k + 4u) {
        acc = acc + vec4<f32>(z[row + k], z[row + k + 1u], z[row + k + 2u], z[row + k + 3u]) * vec4<f32>(vs[k], vs[k + 1u], vs[k + 2u], vs[k + 3u]);
      }
      var s = acc.x + acc.y + acc.z + acc.w;
      for (; k < P.K; k = k + 1u) { s = s + z[row + k] * vs[k]; }
      e[td * P.N + a] = s;
    }
  }
}
