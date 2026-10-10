// #include header.wgsl
// Per day, the Gram matrix of the columns [z_0 .. z_{K-1}, y] (y: the stocks' next-day targets) as a
// tiled matrix product: one workgroup per 8 x 8 tile of outputs (upper triangle only), stocks
// streamed 32 at a time through workgroup memory as vec4. Writes the upper triangle of Z'Z (pairs),
// Z'y (projections) and, from the first tile, the counts of eligible and targeted stocks.
// Grid (tiles x tiles) x days.
@group(0) @binding(2) var<storage, read> z : array<f32>;
@group(0) @binding(3) var<storage, read_write> g : array<f32>;

const TS : u32 = 8u;   // outputs per tile side
const TQ : u32 = 8u;   // vec4 per tile row: 32 stocks per step
var<workgroup> A : array<vec4<f32>, 64>;
var<workgroup> B : array<vec4<f32>, 64>;
var<workgroup> part : array<f32, 128>;

fn column(c : u32, td : u32, t : u32, a : u32) -> f32 {
  if (a >= P.N) { return 0.0; }
  if (c < P.K) { return z[(td * P.N + a) * P.K + c]; }
  if (c == P.K) { return at(6u, t, a) * at(5u, t, a); }
  return 0.0;
}

fn quad(c : u32, td : u32, t : u32, a : u32) -> vec4<f32> {
  return vec4<f32>(column(c, td, t, a), column(c, td, t, a + 1u), column(c, td, t, a + 2u), column(c, td, t, a + 3u));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let cols : u32 = P.K + 1u;
  let nt : u32 = (cols + TS - 1u) / TS;
  let tr : u32 = wid.x / nt;
  let tc : u32 = wid.x % nt;
  let td : u32 = wid.y;
  let t : u32 = P.c0 + td;
  if (tr <= tc && td < P.cd) {
    let lx : u32 = lane % TS;
    let ly : u32 = lane / TS;
    var acc : f32 = 0.0;
    for (var a0 = 0u; a0 < P.N; a0 = a0 + 4u * TQ) {
      let a : u32 = a0 + 4u * (lane % TQ);
      A[lane] = quad(tr * TS + lane / TQ, td, t, a);
      B[lane] = quad(tc * TS + lane / TQ, td, t, a);
      workgroupBarrier();
      for (var q = 0u; q < TQ; q = q + 1u) { acc = acc + dot(A[ly * TQ + q], B[lx * TQ + q]); }
      workgroupBarrier();
    }
    let r : u32 = tr * TS + ly;
    let c : u32 = tc * TS + lx;
    if (r <= c && r < P.K && c < cols) {
      if (c < P.K) { g[td * P.stride + r * (2u * P.K + 1u - r) / 2u + c - r] = acc; }
      else { g[td * P.stride + P.nPairs + r] = acc; }
    }
  }
  if (wid.x == 0u && td < P.cd) {
    var ne : f32 = 0.0;
    var nm : f32 = 0.0;
    for (var a = lane; a < P.N; a = a + 64u) { ne = ne + at(4u, t, a); nm = nm + at(4u, t, a) * at(6u, t, a); }
    part[lane] = ne;
    part[64u + lane] = nm;
    workgroupBarrier();
    if (lane == 0u) {
      var s1 = 0.0;
      var s2 = 0.0;
      for (var j = 0u; j < 64u; j = j + 1u) { s1 = s1 + part[j]; s2 = s2 + part[64u + j]; }
      g[td * P.stride + P.nPairs + P.K] = s1;
      g[td * P.stride + P.nPairs + P.K + 1u] = s2;
    }
  }
}
