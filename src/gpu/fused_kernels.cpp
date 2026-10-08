// WGSL compute kernels of the strategy-search pipeline (see fused_backtest.hpp).
//
// 1. candidateBacktestKernel: one invocation per candidate (model x trading rule). Every
//    workgroup walks the days in lockstep; for each day the 64 invocations cooperatively
//    stage the next-day returns and every model's predictions and ranks in workgroup
//    memory, then each candidate rebalances if the day is one of its rebalance dates,
//    books its gross return and turnover, and extends the prefix sums of its net return,
//    squared return and squared loss.
// 2. adaptiveSelectKernel: one invocation per selector setting. On each adaptation date it
//    scores every candidate over the look-back from the prefix sums (O(1) per candidate),
//    holds the best one (or cash), and on a switch prices the turnover between the old and
//    the new candidate's portfolios, recomputed from the stored predictions.
// 3. seriesSummaryKernel: one invocation per series (candidates first, then selectors)
//    accumulates the SeriesMoments of the evaluation days.
//
// Buffer layouts are produced by gpu::compile(); runFusedReference() mirrors this code on
// the CPU. The strategy weights and the window score match strategyWeights() and
// windowScore() of the C++ library.
#include "sat/gpu/fused_backtest.hpp"

namespace sat::gpu {

static_assert(kWorkgroupSize == 64 && kMaxAssets == 128 && kMaxModels == 8 && kStats == 8 && kAdaptStride == 4 &&
                  kCandidateStride == 4 && kSelectorStride == 8,
              "the WGSL sources below hard-code these constants");

namespace {

const char* kHeader = R"wgsl(
struct Header {
  nM : u32, nN : u32, nD : u32, nC : u32,
  nS : u32, evalFrom : u32, cost : f32, offCand : u32,
  offSel : u32, offRet : u32, offProb : u32, offRank : u32,
  nSeries : u32, nEval : u32, pad0 : u32, pad1 : u32,
};

const WG : u32 = 64u;
const MAXN : u32 = 128u;

@group(0) @binding(0) var<uniform> H : Header;
)wgsl";

// Target weights of one date (strategyWeights in the library). loadProb / loadRank are
// defined by each kernel: from workgroup memory or from the tables.
const char* kWeights = R"wgsl(
// Standard normal distribution function (Abramowitz-Stegun 7.1.26, error < 1.5e-7).
fn ncdf(x : f32) -> f32 {
  let z = abs(x) * 0.70710678;
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  let erfv = 1.0 - poly * exp(-z * z);
  return select(0.5 * (1.0 - erfv), 0.5 * (1.0 + erfv), x >= 0.0);
}

fn betSize(prob : f32) -> f32 {
  let p = clamp(prob, 1.0e-6, 1.0 - 1.0e-6);
  return 2.0 * ncdf((p - 0.5) / sqrt(p * (1.0 - p))) - 1.0;
}

fn weights(kind : u32, param : f32, base : u32, n : u32, w : ptr<function, array<f32, 128>>) {
  for (var i = 0u; i < n; i++) { (*w)[i] = 0.0; }
  if (kind == 0u) {
    let k = u32(clamp(floor(param + 0.5), 1.0, f32(n)));
    for (var i = 0u; i < n; i++) { if (u32(loadRank(base, i)) < k) { (*w)[i] = 1.0 / f32(k); } }
  } else if (kind == 1u) {
    let k = u32(clamp(floor(param + 0.5), 1.0, max(1.0, f32(n / 2u))));
    for (var i = 0u; i < n; i++) {
      let r = u32(loadRank(base, i));
      if (r < k) { (*w)[i] = 1.0 / f32(k); } else if (r >= n - k) { (*w)[i] = -1.0 / f32(k); }
    }
  } else if (kind == 2u) {
    var c = 0u;
    for (var i = 0u; i < n; i++) { if (loadProb(base, i) > param) { c++; } }
    for (var i = 0u; i < n; i++) { if (loadProb(base, i) > param) { (*w)[i] = 1.0 / f32(c); } }
  } else if (kind == 3u) {
    var total = 0.0;
    for (var i = 0u; i < n; i++) { let p = loadProb(base, i); if (p > param) { total += p - param; } }
    if (total > 0.0) {
      for (var i = 0u; i < n; i++) { let p = loadProb(base, i); if (p > param) { (*w)[i] = (p - param) / total; } }
    }
  } else {
    // Bet sizing: signed size 2 N(z) - 1, z = (p - 1/2) / sqrt(p (1 - p)), sizes below the
    // threshold dropped, gross exposure 1.
    var gross = 0.0;
    for (var i = 0u; i < n; i++) {
      let m = betSize(loadProb(base, i));
      if (abs(m) >= param && m != 0.0) { (*w)[i] = m; gross += abs(m); }
    }
    if (gross > 0.0) {
      for (var i = 0u; i < n; i++) { (*w)[i] = (*w)[i] / gross; }
    }
  }
}
)wgsl";

}  // namespace

const std::string& candidateBacktestKernel() {
  static const std::string source = std::string(kHeader) + R"wgsl(
@group(0) @binding(1) var<storage, read> T : array<f32>;                // tables (see compile)
@group(0) @binding(2) var<storage, read_write> BOOK : array<vec2<f32>>; // [C][D] (gross, turnover)
@group(0) @binding(3) var<storage, read_write> PRE : array<vec4<f32>>;  // [C][D+1] prefix (net, net^2, loss^2, 0)

// One day of the universe, shared by the 64 candidates of the workgroup.
var<workgroup> sRet : array<f32, 128>;
var<workgroup> sProb : array<f32, 1024>;  // [model][stock]
var<workgroup> sRank : array<f32, 1024>;

fn loadProb(base : u32, i : u32) -> f32 { return sProb[base + i]; }
fn loadRank(base : u32, i : u32) -> f32 { return sRank[base + i]; }
)wgsl" + kWeights + R"wgsl(
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(local_invocation_index) li : u32) {
  let c = gid.x;
  let live = c < H.nC;
  let n = H.nN;
  let D = H.nD;
  let M = H.nM;
  var model = 0u;
  var kind = 0u;
  var param = 0.0;
  var hold = 1u;
  if (live) {
    let o = H.offCand + c * 4u;
    model = u32(T[o]);
    kind = u32(T[o + 1u]);
    param = T[o + 2u];
    hold = max(1u, u32(T[o + 3u]));
    PRE[c * (D + 1u)] = vec4<f32>(0.0);
  }
  var w : array<f32, 128>;
  var nw : array<f32, 128>;
  for (var i = 0u; i < n; i++) { w[i] = 0.0; }
  var s1 = 0.0;
  var s2 = 0.0;
  var sd = 0.0;
  let per = n * (2u * M + 1u);
  for (var d = 0u; d < D; d++) {
    workgroupBarrier();
    for (var k = li; k < per; k += WG) {
      if (k < n) {
        sRet[k] = T[H.offRet + d * n + k];
      } else if (k < n * (M + 1u)) {
        let q = k - n;
        let m = q / n;
        sProb[q] = T[H.offProb + (m * D + d) * n + q % n];
      } else {
        let q = k - n * (M + 1u);
        let m = q / n;
        sRank[q] = T[H.offRank + (m * D + d) * n + q % n];
      }
    }
    workgroupBarrier();
    if (live) {
      var turnover = 0.0;
      if (d % hold == 0u) {
        weights(kind, param, model * n, n, &nw);
        for (var i = 0u; i < n; i++) {
          turnover += abs(nw[i] - w[i]);
          w[i] = nw[i];
        }
      }
      var gross = 0.0;
      for (var i = 0u; i < n; i++) { gross += w[i] * sRet[i]; }
      let r = gross - H.cost * turnover;
      BOOK[c * D + d] = vec2<f32>(gross, turnover);
      s1 += r;
      s2 += r * r;
      if (r < 0.0) { sd += r * r; }
      PRE[c * (D + 1u) + d + 1u] = vec4<f32>(s1, s2, sd, 0.0);
    }
  }
}
)wgsl";
  return source;
}

const std::string& adaptiveSelectKernel() {
  static const std::string source = std::string(kHeader) + R"wgsl(
@group(0) @binding(1) var<storage, read> T : array<f32>;
@group(0) @binding(2) var<storage, read> BOOK : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> PRE : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> ADAPT : array<vec4<f32>>;  // [S][E] (net, turnover, selection, gross)

fn loadProb(base : u32, i : u32) -> f32 { return T[H.offProb + base + i]; }
fn loadRank(base : u32, i : u32) -> f32 { return T[H.offRank + base + i]; }
)wgsl" + kWeights + R"wgsl(
// Weights of candidate c held over day `day`: those of its last rebalance date.
fn candidateWeights(c : u32, day : u32, w : ptr<function, array<f32, 128>>) {
  let o = H.offCand + c * 4u;
  let hold = max(1u, u32(T[o + 3u]));
  let rd = (day / hold) * hold;
  weights(u32(T[o + 1u]), T[o + 2u], (u32(T[o]) * H.nD + rd) * H.nN, H.nN, w);
}

// windowScore of the library.
fn score(metric : u32, n : f32, s1 : f32, s2 : f32, sd : f32) -> f32 {
  if (n < 2.0) { return -1.0e30; }
  let mean = s1 / n;
  if (metric == 0u) { return mean; }
  if (metric == 1u) { return mean / sqrt(max(0.0, (s2 - s1 * s1 / n) / (n - 1.0)) + 1.0e-10); }
  return mean / sqrt(sd / n + 1.0e-10);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let s = gid.x;
  if (s >= H.nS) { return; }
  let o = H.offSel + s * 8u;
  let L = u32(T[o]);
  let A = max(1u, u32(T[o + 1u]));
  let metric = u32(T[o + 2u]);
  let allowCash = T[o + 3u] > 0.5;
  let minScore = T[o + 4u];
  let D = H.nD;
  let n = H.nN;
  var sel = -1;
  var prev = -1;
  var wN : array<f32, 128>;
  var wP : array<f32, 128>;
  for (var d = H.evalFrom; d < D; d++) {
    if ((d - H.evalFrom) % A == 0u) {
      let w0 = select(0u, d - L, d > L);
      let len = f32(d - w0);
      var best = 0u;
      var bestScore = -3.0e38;
      for (var c = 0u; c < H.nC; c++) {
        let p0 = PRE[c * (D + 1u) + w0];
        let p1 = PRE[c * (D + 1u) + d];
        let sc = score(metric, len, p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        if (sc > bestScore) {
          best = c;
          bestScore = sc;
        }
      }
      sel = i32(best);
      if (allowCash && !(bestScore > minScore)) { sel = -1; }
    }
    var gross = 0.0;
    var turnover = 0.0;
    if (sel >= 0) { gross = BOOK[u32(sel) * D + d].x; }
    if (d > H.evalFrom && sel == prev) {
      if (sel >= 0) { turnover = BOOK[u32(sel) * D + d].y; }
    } else {
      for (var i = 0u; i < n; i++) {
        wN[i] = 0.0;
        wP[i] = 0.0;
      }
      if (sel >= 0) { candidateWeights(u32(sel), d, &wN); }
      if (d > H.evalFrom && prev >= 0) { candidateWeights(u32(prev), d - 1u, &wP); }
      for (var i = 0u; i < n; i++) { turnover += abs(wN[i] - wP[i]); }
    }
    ADAPT[s * H.nEval + d - H.evalFrom] = vec4<f32>(gross - H.cost * turnover, turnover, f32(sel), gross);
    prev = sel;
  }
}
)wgsl";
  return source;
}

const std::string& seriesSummaryKernel() {
  static const std::string source = std::string(kHeader) + R"wgsl(
@group(0) @binding(1) var<storage, read> BOOK : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> ADAPT : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> STATS : array<f32>;  // [series][8]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let k = gid.x;
  if (k >= H.nSeries) { return; }
  var n = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var sd = 0.0;
  var logW = 0.0;
  var peak = 0.0;
  var mdd = 0.0;
  var wins = 0.0;
  var turn = 0.0;
  for (var d = H.evalFrom; d < H.nD; d++) {
    var r = 0.0;
    var t = 0.0;
    if (k < H.nC) {
      let b = BOOK[k * H.nD + d];
      r = b.x - H.cost * b.y;
      t = b.y;
    } else {
      let a = ADAPT[(k - H.nC) * H.nEval + d - H.evalFrom];
      r = a.x;
      t = a.y;
    }
    n += 1.0;
    s1 += r;
    s2 += r * r;
    if (r < 0.0) { sd += r * r; }
    if (r > 0.0) { wins += 1.0; }
    turn += t;
    logW += log(max(1.0 + r, 1.0e-12));
    peak = max(peak, logW);
    mdd = max(mdd, 1.0 - exp(logW - peak));
  }
  let o = k * 8u;
  STATS[o] = n;
  STATS[o + 1u] = s1;
  STATS[o + 2u] = s2;
  STATS[o + 3u] = sd;
  STATS[o + 4u] = logW;
  STATS[o + 5u] = mdd;
  STATS[o + 6u] = wins;
  STATS[o + 7u] = turn;
}
)wgsl";
  return source;
}

}  // namespace sat::gpu
