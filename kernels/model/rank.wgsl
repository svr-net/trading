// #include header.wgsl
// The stock's rank z-score of signal k among the day's eligible stocks (ties: by stock index), 0
// where it is not eligible. Grid ceil(N/64) x K x days, like signal.
@group(0) @binding(2) var<storage, read> sig : array<f32>;
@group(0) @binding(3) var<storage, read_write> z : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let a = id.x;
  let k = id.y;
  let td = id.z;
  if (a >= P.N || k >= P.K || td >= P.cd) { return; }
  let t = P.c0 + td;
  var out = 0.0;
  if (at(4u, t, a) > 0.5) {
    let x = sig[(td * P.N + a) * P.K + k];
    var n = 0u;
    var r = 0u;
    for (var b = 0u; b < P.N; b = b + 1u) {
      if (at(4u, t, b) > 0.5) {
        n = n + 1u;
        let y = sig[(td * P.N + b) * P.K + k];
        if (y < x || (y == x && b < a)) { r = r + 1u; }
      }
    }
    let fn1 = f32(n);
    out = (f32(r) - 0.5 * (fn1 - 1.0)) / sqrt(max((fn1 * fn1 - 1.0) / 12.0, 1.0e-12));
  }
  z[(td * P.N + a) * P.K + k] = out;
}
