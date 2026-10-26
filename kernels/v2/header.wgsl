// Shared by the predictive model's kernels (src/v2.hpp).
struct V2Header {
  T : u32, N : u32, K : u32, NT : u32,
  t0 : u32, u0 : u32, u1 : u32, u2 : u32,
};
@group(0) @binding(0) var<uniform> P : V2Header;
