// Shared by the daily trades' kernels (daykernels.hpp).
struct DayHeader {
  T : u32, N : u32, nH : u32, FI : u32,
  t0 : u32, cd : u32, nS : u32, In : u32,
  Hd : u32, Out : u32, nP : u32, u0 : u32,
  lr : f32, b1 : f32, b2 : f32, steps : f32,
  nC : u32, G : u32, NK : u32, u1 : u32,
  buy : f32, sell : f32, hzn : f32, f1 : f32,
  ent : array<vec4<u32>, 1>,
  hz : array<vec4<u32>, 4>,
};
@group(0) @binding(0) var<uniform> P : DayHeader;
fn horizon(i : u32) -> u32 { return P.hz[i / 4u][i % 4u]; }
// Network parameters: W1 [Hd][In], b1 [Hd], W2 [Hd][Hd], b2 [Hd], W3 [Out][Hd], b3 [Out].
fn oB1() -> u32 { return P.Hd * P.In; }
fn oW2() -> u32 { return oB1() + P.Hd; }
fn oB2() -> u32 { return oW2() + P.Hd * P.Hd; }
fn oW3() -> u32 { return oB2() + P.Hd; }
fn oB3() -> u32 { return oW3() + P.Out * P.Hd; }
