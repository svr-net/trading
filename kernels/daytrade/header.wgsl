// Shared by the day-trade kernels (daytrade.hpp).
struct Header {
  T : u32, N : u32, K : u32, s : u32,
  t0 : u32, t1 : u32, H : u32, u0 : u32,
  buy : f32, sell : f32, f0 : f32, f1 : f32,
  hz : array<vec4<u32>, 4>,
};
@group(0) @binding(0) var<uniform> P : Header;
@group(0) @binding(1) var<storage, read> bars : array<f32>;
const NONE : f32 = 3.0e38;
const KS : u32 = 24u;
const KL : u32 = 10u;
const KT : u32 = 6u;
const KD : u32 = 24u;
fn bar(f : u32, t : u32, i : u32) -> f32 { return bars[(f * P.T + t) * P.N + i]; }
