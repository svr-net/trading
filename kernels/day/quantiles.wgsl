// #include header.wgsl
// Rolling windows' quantiles: each of P.nS windows of at most P.nP values that has a new value
// (has[w] > 0) takes x[w], drops its oldest value once full, and gives its P.nH quantiles at the
// levels tau[(w % 2) P.nH + k], interpolated between the order statistics at tau (n - 1). Each window
// is kept sorted (srt) beside its values in arrival order (ring, a circle starting at head), so a
// day's update is one removal and one insertion. st [nS][2] (count, head); out [nS][nH], written
// once the window holds more than nH values. Grid ceil(nS/64).
@group(0) @binding(1) var<storage, read> has : array<f32>;
@group(0) @binding(2) var<storage, read> x : array<f32>;
@group(0) @binding(3) var<storage, read> tau : array<f32>;
@group(0) @binding(4) var<storage, read_write> ring : array<f32>;
@group(0) @binding(5) var<storage, read_write> srt : array<f32>;
@group(0) @binding(6) var<storage, read_write> st : array<f32>;
@group(0) @binding(7) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let w : u32 = wid.x * 64u + lane;
  if (w < P.nS && has[w] > 0.0) {
    let b : u32 = w * P.nP;
    var n : u32 = u32(st[w * 2u]);
    var h : u32 = u32(st[w * 2u + 1u]);
    let v : f32 = x[w];
    if (n == P.nP) {
      let old : f32 = ring[b + h];
      var j : u32 = 0u;
      loop {
        if (j + 1u >= n || srt[b + j] == old) { break; }
        j = j + 1u;
      }
      for (var k = j; k + 1u < n; k = k + 1u) { srt[b + k] = srt[b + k + 1u]; }
      n = n - 1u;
      ring[b + h] = v;
      h = (h + 1u) % P.nP;
    } else {
      ring[b + (h + n) % P.nP] = v;
    }
    var p : u32 = n;
    loop {
      if (p == 0u) { break; }
      if (srt[b + p - 1u] <= v) { break; }
      srt[b + p] = srt[b + p - 1u];
      p = p - 1u;
    }
    srt[b + p] = v;
    n = n + 1u;
    st[w * 2u] = f32(n);
    st[w * 2u + 1u] = f32(h);
    if (n > P.nH) {
      for (var k = 0u; k < P.nH; k = k + 1u) {
        let pos : f32 = tau[(w % 2u) * P.nH + k] * f32(n - 1u);
        let lo : u32 = u32(pos);
        let fr : f32 = pos - f32(lo);
        if (lo + 1u < n) {
          out[w * P.nH + k] = srt[b + lo] * (1.0 - fr) + srt[b + lo + 1u] * fr;
        } else {
          out[w * P.nH + k] = srt[b + lo];
        }
      }
    }
  }
}
