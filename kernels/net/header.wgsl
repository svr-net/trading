// Shared by the neural network's kernels (src/net.hpp).
struct NetHeader {
  nS : u32, In : u32, Hd : u32, Out : u32,
  nP : u32, u0 : u32, u1 : u32, u2 : u32,
  lr : f32, b1 : f32, b2 : f32, steps : f32,
};
@group(0) @binding(0) var<uniform> P : NetHeader;
// Network parameters: W1 [Hd][In], b1 [Hd], W2 [Hd][Hd], b2 [Hd], W3 [Out][Hd], b3 [Out].
fn oB1() -> u32 { return P.Hd * P.In; }
fn oW2() -> u32 { return oB1() + P.Hd; }
fn oB2() -> u32 { return oW2() + P.Hd * P.Hd; }
fn oW3() -> u32 { return oB2() + P.Hd; }
fn oB3() -> u32 { return oW3() + P.Out * P.Hd; }
