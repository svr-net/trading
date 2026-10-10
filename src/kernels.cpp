// WGSL sources of the fused kernels and their CPU emulation (see kernels.hpp). The emulation
// mirrors the WGSL line by line in 32-bit floats; ranks use the same order (value, then stock
// index), so the emulated GPU returns what the GPU returns, up to the last bit of sqrt and
// division, which WGSL leaves to the device.
#include "ofm/kernels.hpp"

#include "signals.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ofm::kernels {

namespace {

const char* kHeader = R"wgsl(
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
)wgsl";

const char* kZscore = R"wgsl(
@group(0) @binding(2) var<storage, read_write> z : array<f32>;
var<workgroup> val : array<f32, 256>;

// The stock's signal (src/signals.hpp computes the same on the CPU).
fn signal(fam : u32, h : u32, t : u32, a : u32) -> f32 {
  let fh = f32(h);
  let lc = at(0u, t, a);
  if (fam == 0u) { return lc - at(0u, t - h, a); }
  if (fam == 1u || fam == 3u) {
    let d1 = at(1u, t, a) - at(1u, t - h, a);
    let d2 = at(2u, t, a) - at(2u, t - h, a);
    let vol = sqrt(max(0.0, (d2 - d1 * d1 / fh) / (fh - 1.0)));
    if (fam == 1u) { return -vol; }
    return select(0.0, (lc - at(0u, t - h, a)) / (vol * sqrt(fh)), vol > 0.0);
  }
  if (fam == 2u) {
    var mx = lc;
    for (var u = 1u; u < h; u = u + 1u) { mx = max(mx, at(0u, t - u, a)); }
    return lc - mx;
  }
  if (fam == 4u) { return -(at(3u, t, a) - at(3u, t - h, a)) / fh; }
  if (fam == 5u) {
    var up = 0.0;
    var tot = 0.0;
    for (var u = 0u; u < h; u = u + 1u) {
      let d = at(0u, t - u, a) - at(0u, t - u - 1u, a);
      up = up + max(d, 0.0);
      tot = tot + abs(d);
    }
    return select(0.5, up / tot, tot > 0.0);
  }
  if (fam == 6u) {
    var s = 0.0;
    for (var u = 1u; u <= h; u = u + 1u) { s = s + at(0u, t - u, a); }
    let m = s / fh;
    var v = 0.0;
    for (var u = 1u; u <= h; u = u + 1u) { v = v + (at(0u, t - u, a) - m) * (at(0u, t - u, a) - m); }
    let sd = sqrt(v / (fh - 1.0));
    return select(0.0, (lc - m) / sd, sd > 0.0);
  }
  if (fam == 7u) {
    var lo = at(8u, t, a);
    var hi = at(7u, t, a);
    for (var u = 1u; u < h; u = u + 1u) { lo = min(lo, at(8u, t - u, a)); hi = max(hi, at(7u, t - u, a)); }
    return select(0.5, (lc - lo) / (hi - lo), hi > lo);
  }
  if (fam == 8u) {
    var up = 0.0;
    var tot = 0.0;
    for (var u = 0u; u < h; u = u + 1u) {
      let v = at(9u, t - u, a);
      if (at(0u, t - u, a) > at(0u, t - u - 1u, a)) { up = up + v; }
      tot = tot + v;
    }
    return select(0.5, up / tot, tot > 0.0);
  }
  var s = 0.0;
  for (var u = 0u; u < h; u = u + 1u) { s = s + (at(7u, t - u, a) - at(8u, t - u, a)); }
  return -s / fh;
}
var<workgroup> ok : array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_index) a : u32) {
  let k = wg.x;
  let td = wg.y;
  let t = P.c0 + td;
  let fam = k / P.H;
  let h = horizon(k % P.H);
  var x = 0.0;
  var e = 0u;
  if (a < P.N && at(4u, t, a) > 0.5) {
    e = 1u;
    x = signal(fam, h, t, a);
  }
  val[a] = x;
  ok[a] = e;
  workgroupBarrier();
  if (a < P.N) {
    var out = 0.0;
    if (e == 1u) {
      var n = 0u;
      var r = 0u;
      for (var b = 0u; b < P.N; b = b + 1u) {
        if (ok[b] == 1u) {
          n = n + 1u;
          let y = val[b];
          if (y < x || (y == x && b < a)) { r = r + 1u; }
        }
      }
      let fn1 = f32(n);
      out = (f32(r) - 0.5 * (fn1 - 1.0)) / sqrt(max((fn1 * fn1 - 1.0) / 12.0, 1.0e-12));
    }
    z[(td * P.N + a) * P.K + k] = out;
  }
}
)wgsl";

const char* kGram = R"wgsl(
@group(0) @binding(2) var<storage, read> z : array<f32>;
@group(0) @binding(3) var<storage, read_write> g : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let p = id.x;
  let td = id.y;
  if (p >= P.stride || td >= P.cd) { return; }
  let t = P.c0 + td;
  var acc = 0.0;
  if (p < P.nPairs) {
    var k = 0u;
    var rem = p;
    loop {
      let len = P.K - k;
      if (rem < len) { break; }
      rem = rem - len;
      k = k + 1u;
    }
    let l = k + rem;
    for (var a = 0u; a < P.N; a = a + 1u) {
      acc = acc + z[(td * P.N + a) * P.K + k] * z[(td * P.N + a) * P.K + l];
    }
  } else if (p < P.nPairs + P.K) {
    let k = p - P.nPairs;
    for (var a = 0u; a < P.N; a = a + 1u) {
      acc = acc + z[(td * P.N + a) * P.K + k] * (at(6u, t, a) * at(5u, t, a));
    }
  } else if (p == P.nPairs + P.K) {
    for (var a = 0u; a < P.N; a = a + 1u) { acc = acc + at(4u, t, a); }
  } else {
    for (var a = 0u; a < P.N; a = a + 1u) { acc = acc + at(4u, t, a) * at(6u, t, a); }
  }
  g[td * P.stride + p] = acc;
}
)wgsl";

const char* kExpect = R"wgsl(
@group(0) @binding(2) var<storage, read> z : array<f32>;
@group(0) @binding(3) var<storage, read> v : array<f32>;
@group(0) @binding(4) var<storage, read_write> e : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  let a = id.x;
  let td = id.y;
  if (a >= P.N || td >= P.cd) { return; }
  var acc = 0.0;
  for (var k = 0u; k < P.K; k = k + 1u) { acc = acc + z[(td * P.N + a) * P.K + k] * v[td * P.K + k]; }
  e[td * P.N + a] = acc;
}
)wgsl";

}  // namespace

const std::string& zscoreSource() {
  static const std::string s = std::string(kHeader) + kZscore;
  return s;
}
const std::string& gramSource() {
  static const std::string s = std::string(kHeader) + kGram;
  return s;
}
const std::string& expectSource() {
  static const std::string s = std::string(kHeader) + kExpect;
  return s;
}

void emulateZscore(const Plan& p, std::size_t c, std::vector<float>& z) {
  const std::size_t N = p.N, K = p.K, H = p.H, TN = p.T * N, c0 = p.chunkStart(c), cd = p.chunkLength(c);
  const float* tab = p.tables.data();
  auto at = [&](std::size_t table, std::size_t t, std::size_t a) { return tab[table * TN + t * N + a]; };
  z.assign(cd * N * K, 0.0f);
  std::vector<float> val(N);
  std::vector<unsigned> ok(N);
  std::vector<std::size_t> order;
  for (std::size_t td = 0; td < cd; ++td) {
    const std::size_t t = c0 + td;
    for (std::size_t k = 0; k < K; ++k) {
      const std::size_t fam = k / H, h = p.horizons[k % H];
      // Phase 1: one invocation per stock.
      for (std::size_t a = 0; a < N; ++a) {
        float x = 0.0f;
        unsigned e = 0;
        if (at(4, t, a) > 0.5f) {
          e = 1;
          x = detail::signal<float>(fam, h, t, [&](std::size_t tb, std::size_t u) { return at(tb, u, a); });
        }
        val[a] = x, ok[a] = e;
      }
      // Phase 2 (after the barrier): rank by (value, stock index), as the counting loop does.
      order.clear();
      for (std::size_t a = 0; a < N; ++a)
        if (ok[a]) order.push_back(a);
      std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return val[x] < val[y]; });
      const float n = static_cast<float>(order.size());
      const float sd = std::sqrt(std::max((n * n - 1.0f) / 12.0f, 1.0e-12f));
      for (std::size_t r = 0; r < order.size(); ++r) z[(td * N + order[r]) * K + k] = (static_cast<float>(r) - 0.5f * (n - 1.0f)) / sd;
    }
  }
}

void emulateGram(const Plan& p, std::size_t c, const std::vector<float>& z, std::vector<float>& g) {
  const std::size_t N = p.N, K = p.K, TN = p.T * N, c0 = p.chunkStart(c), cd = p.chunkLength(c), P = p.pairs();
  const float* tab = p.tables.data();
  auto at = [&](std::size_t table, std::size_t t, std::size_t a) { return tab[table * TN + t * N + a]; };
  g.assign(cd * p.stride, 0.0f);
  for (std::size_t td = 0; td < cd; ++td) {
    const std::size_t t = c0 + td;
    const float* zt = &z[td * N * K];
    float* gt = &g[td * p.stride];
    for (std::size_t k = 0, q = 0; k < K; ++k)
      for (std::size_t l = k; l < K; ++l, ++q) {
        float acc = 0.0f;
        for (std::size_t a = 0; a < N; ++a) acc = acc + zt[a * K + k] * zt[a * K + l];
        gt[q] = acc;
      }
    for (std::size_t k = 0; k < K; ++k) {
      float acc = 0.0f;
      for (std::size_t a = 0; a < N; ++a) acc = acc + zt[a * K + k] * (at(6, t, a) * at(5, t, a));
      gt[P + k] = acc;
    }
    float ne = 0.0f, nt = 0.0f;
    for (std::size_t a = 0; a < N; ++a) ne = ne + at(4, t, a), nt = nt + at(4, t, a) * at(6, t, a);
    gt[P + K] = ne, gt[P + K + 1] = nt;
  }
}

void emulateExpect(const Plan& p, std::size_t c, const std::vector<float>& z, const std::vector<float>& v, std::vector<float>& e) {
  const std::size_t N = p.N, K = p.K, cd = p.chunkLength(c);
  e.assign(cd * N, 0.0f);
  for (std::size_t td = 0; td < cd; ++td)
    for (std::size_t a = 0; a < N; ++a) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k) acc = acc + z[(td * N + a) * K + k] * v[td * K + k];
      e[td * N + a] = acc;
    }
}

}  // namespace ofm::kernels
