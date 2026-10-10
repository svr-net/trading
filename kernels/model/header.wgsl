// Shared by the model's kernels (kernels.hpp).
struct Header {
  T : u32, N : u32, K : u32, H : u32,
  c0 : u32, cd : u32, nPairs : u32, stride : u32,
  u0 : u32, u1 : u32, u2 : u32, u3 : u32,
  u4 : u32, u5 : u32, u6 : u32, u7 : u32,
  hz : array<vec4<u32>, 4>,
};
@group(0) @binding(0) var<uniform> P : Header;
@group(0) @binding(1) var<storage, read> tab : array<f32>;
// Tables: 0 log close, 1 prefix log returns, 2 prefix squared, 3 prefix log traded value,
// 4 eligible, 5 target, 6 target mask, 7 log high, 8 log low, 9 traded value.
fn horizon(i : u32) -> u32 { return P.hz[i / 4u][i % 4u]; }
fn at(tb : u32, t : u32, a : u32) -> f32 { return tab[tb * P.T * P.N + t * P.N + a]; }
