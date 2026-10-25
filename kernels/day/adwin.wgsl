// #include header.wgsl
// ADWIN windows (AdwinBank in daybook.hpp; Bifet and Gavalda 2007) of P.nS records at once: the
// record in slot list[s] takes the value x[s], then drops its older part while some split of its
// window, each side at least the square root of the window, has means further apart than the
// scale-free bound (confidence 1 / P.hzn). p1, p2 [slots][P.nP]: prefix sums of the values and of
// their squares; st [slots][2]: the window's first and end. Grid ceil(nS/8).
@group(0) @binding(1) var<storage, read> list : array<f32>;
@group(0) @binding(2) var<storage, read> x : array<f32>;
@group(0) @binding(3) var<storage, read_write> p1 : array<f32>;
@group(0) @binding(4) var<storage, read_write> p2 : array<f32>;
@group(0) @binding(5) var<storage, read_write> st : array<f32>;

@compute @workgroup_size(8)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 8u + lane;
  if (s < P.nS) {
    let r : u32 = u32(list[s]);
    let b : u32 = r * P.nP;
    var first : u32 = u32(st[r * 2u]);
    let end : u32 = u32(st[r * 2u + 1u]) + 1u;
    p1[b + end] = p1[b + end - 1u] + x[s];
    p2[b + end] = p2[b + end - 1u] + x[s] * x[s];
    st[r * 2u + 1u] = f32(end);
    loop {
      let n : u32 = end - first;
      let side : u32 = max(2u, u32(ceil(sqrt(f32(n)))));
      if (n < 2u * side) { break; }
      let lt : f32 = log(2.0 * log(f32(n)) * P.hzn);
      var moved : bool = false;
      for (var k = first + side; k + side <= end; k = k + 1u) {
        let n0 : f32 = f32(k - first);
        let n1 : f32 = f32(end - k);
        let m0 : f32 = (p1[b + k] - p1[b + first]) / n0;
        let m1 : f32 = (p1[b + end] - p1[b + k]) / n1;
        let v0 : f32 = max(0.0, (p2[b + k] - p2[b + first]) / n0 - m0 * m0);
        let v1 : f32 = max(0.0, (p2[b + end] - p2[b + k]) / n1 - m1 * m1);
        if (abs(m0 - m1) > sqrt(2.0 * lt * (v0 / n0 + v1 / n1))) {
          first = k;
          moved = true;
          break;
        }
      }
      if (!moved) { break; }
    }
    st[r * 2u] = f32(first);
  }
}
