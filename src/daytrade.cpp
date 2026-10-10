// The same-day trades' kernels (see daytrade.hpp): one C++ body per kernel, written line by line as
// its WGSL, instantiated in double precision (the reference) and in single precision (the emulated
// GPU, which returns what the GPU returns up to the last bit of exp, log, sqrt and division).
#include "ofm/daytrade.hpp"

#include <algorithm>
#include <cmath>

namespace ofm::daytrade {

namespace {

const char* kHeader = R"wgsl(
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
)wgsl";

const char* kLevels = R"wgsl(
@group(0) @binding(2) var<storage, read> life : array<f32>;
@group(0) @binding(3) var<storage, read_write> st : array<f32>;
@group(0) @binding(4) var<storage, read_write> obs : array<f32>;
@group(0) @binding(5) var<storage, read_write> idx : array<u32>;
@group(0) @binding(6) var<storage, read_write> lev : array<f32>;

var<private> I : u32;
var<private> KC : f32;
var<private> KCS : f32;
var<private> MINMOVE : f32;

fn S(k : u32) -> f32 { return st[I * KS + k]; }
fn setS(k : u32, v : f32) { st[I * KS + k] = v; }
fn lo(q : u32) -> f32 { return obs[I * 4u * P.T + q]; }
fn hi(q : u32) -> f32 { return obs[I * 4u * P.T + P.T + q]; }
fn cl(q : u32) -> f32 { return obs[I * 4u * P.T + 2u * P.T + q]; }
fn sg(q : u32) -> f32 { return obs[I * 4u * P.T + 3u * P.T + q]; }
fn byLo(k : u32) -> u32 { return idx[I * 2u * P.T + k]; }
fn byHi(k : u32) -> u32 { return idx[(I * 2u + 1u) * P.T + k]; }
fn cnt() -> u32 { return u32(S(17u)); }
fn value(f : u32, q : u32) -> f32 { if (f == 0u) { return lo(q); } return hi(q); }
fn sorted(f : u32, k : u32) -> u32 { if (f == 0u) { return byLo(k); } return byHi(k); }

fn erfcApprox(x : f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  let e = y * exp(-z * z);
  return select(2.0 - e, e, x >= 0.0);
}

fn add(vlo : f32, vhi : f32, vc : f32, vs : f32) {
  let q = cnt();
  let base = I * 4u * P.T;
  obs[base + q] = vlo; obs[base + P.T + q] = vhi; obs[base + 2u * P.T + q] = vc; obs[base + 3u * P.T + q] = vs;
  for (var f = 0u; f < 2u; f = f + 1u) {
    let o = (I * 2u + f) * P.T;
    let key = select(vhi, vlo, f == 0u);
    var k = q;
    loop {
      if (k == 0u) { break; }
      if (value(f, idx[o + k - 1u]) < key) { break; }
      idx[o + k] = idx[o + k - 1u];
      k = k - 1u;
    }
    idx[o + k] = q;
  }
  setS(17u, f32(q + 1u)); setS(18u, S(18u) + vs);
  setS(19u, max(S(19u), vlo * vs)); setS(20u, max(S(20u), vhi * vs));
}

fn ratio(s1 : f32, s2 : f32, n : f32) -> f32 {
  if (n < 2.0) { return -NONE; }
  let mu = s1 / n;
  let vr = max(0.0, s2 / n - mu * mu);
  if (vr > 0.0) { return mu / sqrt(vr); }
  return select(-NONE, NONE, mu > 0.0);
}

fn plausibleMax(f : u32) -> f32 {
  let nn = cnt();
  if (nn < 4u) { if (nn > 0u) { return value(f, sorted(f, nn - 1u)); } return NONE; }
  let q1 = value(f, sorted(f, nn / 4u));
  let q2 = value(f, sorted(f, nn / 2u));
  let q3 = value(f, sorted(f, 3u * nn / 4u));
  let sc = (q3 - q1) / 1.349;
  for (var k = nn; k > 0u; k = k - 1u) {
    let v = value(f, sorted(f, k - 1u));
    if (!(sc > 0.0) || f32(nn) * erfcApprox((v - q2) / sc / 1.4142135623730951) >= 0.5) { return v; }
  }
  return value(f, sorted(f, nn - 1u));
}

fn meanSig() -> f32 { return S(18u) / f32(cnt()); }

fn scoreAB(a : f32, bb : f32) -> f32 {
  var s1 = 0.0;
  var s2 = 0.0;
  for (var q = 0u; q < cnt(); q = q + 1u) {
    let v = select(select(cl(q), bb * sg(q), hi(q) >= bb) + KC, -a * sg(q) + KCS, lo(q) >= a);
    s1 = s1 + v; s2 = s2 + v * v;
  }
  return ratio(s1, s2, f32(cnt()));
}

var<private> SCORE : f32;

fn bestStop(bb : f32) -> f32 {
  let nn = cnt();
  let capA = plausibleMax(0u);
  let floorA = max(MINMOVE / meanSig(), lo(byLo(nn / 2u)));
  var t1 = 0.0; var t2 = 0.0; var tm = 0.0; var h1 = 0.0; var h2 = 0.0; var bestA = capA;
  for (var q = 0u; q < nn; q = q + 1u) { t1 = t1 + sg(q); t2 = t2 + sg(q) * sg(q); tm = tm + 1.0; }
  SCORE = -NONE;
  for (var k = 0u; k < nn; k = k + 1u) {
    let x = byLo(k);
    let a = lo(x);
    let base = select(cl(x), bb * sg(x), hi(x) >= bb) + KC;
    if (a > 0.0 && a >= floorA && a <= capA) {
      let s1 = h1 - a * t1 + tm * KCS;
      let s2 = h2 + a * a * t2 - 2.0 * a * KCS * t1 + tm * KCS * KCS;
      let rr = ratio(s1, s2, f32(nn));
      if (rr > SCORE + 1e-12) { SCORE = rr; bestA = a; }
    }
    h1 = h1 + base; h2 = h2 + base * base; t1 = t1 - sg(x); t2 = t2 - sg(x) * sg(x); tm = tm - 1.0;
  }
  return bestA;
}

fn bestTake(a : f32) -> f32 {
  let nn = cnt();
  let capB = plausibleMax(1u);
  let floorB = MINMOVE / meanSig();
  var c1 = 0.0; var c2 = 0.0; var t1 = 0.0; var t2 = 0.0; var tm = 0.0; var h1 = 0.0; var h2 = 0.0;
  var bestB = capB;
  var score = -NONE;
  for (var q = 0u; q < nn; q = q + 1u) {
    if (lo(q) >= a) { let v = -a * sg(q) + KCS; c1 = c1 + v; c2 = c2 + v * v; }
    else { t1 = t1 + sg(q); t2 = t2 + sg(q) * sg(q); tm = tm + 1.0; }
  }
  for (var k = 0u; k < nn; k = k + 1u) {
    let x = byHi(k);
    if (lo(x) >= a) { continue; }
    let bb = hi(x);
    let v = cl(x) + KC;
    if (bb > 0.0 && bb >= floorB && bb <= capB) {
      let s1 = c1 + h1 + bb * t1 + tm * KC;
      let s2 = c2 + h2 + bb * bb * t2 + 2.0 * bb * KC * t1 + tm * KC * KC;
      let rr = ratio(s1, s2, f32(nn));
      if (rr > score + 1e-12) { score = rr; bestB = bb; }
    }
    h1 = h1 + v; h2 = h2 + v * v; t1 = t1 - sg(x); t2 = t2 - sg(x) * sg(x); tm = tm - 1.0;
  }
  return bestB;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  I = gid.x;
  if (I >= P.N) { return; }
  KC = log((1.0 - P.sell) / (1.0 + P.buy));
  KCS = KC + log(1.0 - P.sell);
  MINMOVE = -KC;
  for (var t = P.t0; t < P.t1; t = t + 1u) {
    let o = bar(0u, t, I); let h = bar(1u, t, I); let l = bar(2u, t, I); let c = bar(3u, t, I);
    let ok = o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0;
    let pv = S(21u);
    if (t >= 1u && pv > 0.0 && ok && l <= min(o, c) && h >= max(o, c)) {
      add(-log(l / o) / pv, log(h / o) / pv, log(c / o), pv);
    }
    let lam = exp2(-1.0 / max(1.0, life[t]));
    if (o > 0.0 && c > 0.0) {
      let z = log(c / o);
      setS(0u, select(z * z, lam * S(0u) + (1.0 - lam) * z * z, S(0u) >= 0.0));
    }
    setS(1u, select(0.0, sqrt(S(0u)), S(0u) > 0.0));
    var hz = 1.0;
    if (P.H > 0u) { hz = f32(P.hz[(P.H - 1u) / 4u][(P.H - 1u) % 4u]); }
    for (var k = P.H; k > 0u; k = k - 1u) {
      let hk = f32(P.hz[(k - 1u) / 4u][(k - 1u) % 4u]);
      if (hk >= life[t]) { hz = hk; }
    }
    let al = 1.0 / hz;
    if (t >= 1u) {
      let ph = bar(1u, t - 1u, I); let pl = bar(2u, t - 1u, I); let pc = bar(3u, t - 1u, I);
      if (ok && ph > 0.0 && pl > 0.0 && pc > 0.0) {
        let up = h - ph; let dn = pl - l;
        let gz = max(0.0, c - pc); let lz = max(0.0, pc - c);
        let pz = select(0.0, up, up > dn && up > 0.0); let nz = select(0.0, dn, dn > up && dn > 0.0);
        let tz = max(max(h - l, abs(h - pc)), abs(l - pc));
        if (S(8u) > 0.0) {
          setS(2u, S(2u) + al * (gz - S(2u))); setS(3u, S(3u) + al * (lz - S(3u))); setS(4u, S(4u) + al * (pz - S(4u)));
          setS(5u, S(5u) + al * (nz - S(5u))); setS(6u, S(6u) + al * (tz - S(6u)));
        } else {
          setS(2u, gz); setS(3u, lz); setS(4u, pz); setS(5u, nz); setS(6u, tz); setS(8u, 1.0);
        }
        if (S(6u) > 0.0) {
          let pdi = S(4u) / S(6u); let ndi = S(5u) / S(6u);
          if (pdi + ndi > 0.0) {
            let dx = 100.0 * abs(pdi - ndi) / (pdi + ndi);
            if (S(9u) > 0.0) { setS(7u, S(7u) + al * (dx - S(7u))); } else { setS(7u, dx); setS(9u, 1.0); }
          }
        }
        if (!(S(14u) > 0.0)) {
          setS(13u, select(0.0, 1.0, c >= pc)); setS(10u, select(ph, pl, c >= pc)); setS(11u, select(l, h, c >= pc)); setS(12u, al); setS(14u, 1.0);
        } else if (S(13u) > 0.0) {
          setS(10u, min(S(10u) + S(12u) * (S(11u) - S(10u)), pl));
          if (l < S(10u)) { setS(13u, 0.0); setS(10u, S(11u)); setS(11u, l); setS(12u, al); }
          else if (h > S(11u)) { setS(11u, h); setS(12u, min(1.0, S(12u) + al)); }
        } else {
          setS(10u, max(S(10u) + S(12u) * (S(11u) - S(10u)), ph));
          if (h > S(10u)) { setS(13u, 1.0); setS(10u, S(11u)); setS(11u, h); setS(12u, al); }
          else if (l < S(11u)) { setS(11u, l); setS(12u, min(1.0, S(12u) + al)); }
        }
        if (S(9u) > 0.0) { setS(15u, S(15u) + S(7u)); setS(16u, S(16u) + 1.0); }
      }
    }
    var code = -1.0;
    if (S(9u) > 0.0 && S(8u) > 0.0 && S(2u) + S(3u) > 0.0 && S(16u) >= 1.0) {
      code = select(0.0, 4.0, S(13u) > 0.0) + select(0.0, 2.0, S(2u) / (S(2u) + S(3u)) > 0.5) + select(0.0, 1.0, S(7u) > S(15u) / S(16u));
    }
    setS(22u, code);
    if (t >= P.s) {
      var a = NONE; var bb = NONE; var best = -NONE;
      let nn = cnt();
      if (nn >= 2u) {
        let floorL = MINMOVE / meanSig();
        a = max(lo(byLo(nn / 2u)), floorL); bb = max(hi(byHi(nn / 2u)), floorL);
        best = scoreAB(a, bb);
        for (var it = 0; it < 50; it = it + 1) {
          let nb = bestTake(a);
          let na = bestStop(nb);
          if (!(SCORE > best + 1e-12)) { break; }
          best = SCORE;
          if (na == a && nb == bb) { break; }
          a = na; bb = nb;
        }
      }
      var pS = 0.0; var pT = 0.0;
      for (var q = 0u; q < nn; q = q + 1u) {
        if (lo(q) >= a) { pS = pS + 1.0; }
        if (lo(q) < a && hi(q) >= bb) { pT = pT + 1.0; }
      }
      if (nn > 0u) { pS = pS / f32(nn); pT = pT / f32(nn); }
      let o2 = (t * P.N + I) * KL;
      lev[o2] = a; lev[o2 + 1u] = bb; lev[o2 + 2u] = best; lev[o2 + 3u] = select(0.0, 1.0, nn >= 2u && best > 0.0);
      lev[o2 + 4u] = pS; lev[o2 + 5u] = pT; lev[o2 + 6u] = S(1u); lev[o2 + 7u] = S(19u); lev[o2 + 8u] = S(20u); lev[o2 + 9u] = code;
    }
    setS(21u, S(1u));
  }
}
)wgsl";

const char* kTrades = R"wgsl(
@group(0) @binding(2) var<storage, read> life : array<f32>;
@group(0) @binding(6) var<storage, read> lev : array<f32>;
@group(0) @binding(7) var<storage, read_write> tra : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let t = P.s + gid.y;
  if (i >= P.N || t >= P.T) { return; }
  let ob = (t * P.N + i) * KT;
  for (var k = 0u; k < KT; k = k + 1u) { tra[ob + k] = 0.0; }
  if (t + 1u >= P.T) { return; }
  let d = t + 1u;
  let lb = (t * P.N + i) * KL;
  let o = bar(0u, d, i); let h = bar(1u, d, i); let l = bar(2u, d, i); let c = bar(3u, d, i); let sig = lev[lb + 6u];
  if (!(o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 && sig > 0.0) || l > min(o, c) || h < max(o, c)) { return; }
  let kc = log((1.0 - P.sell) / (1.0 + P.buy));
  let keep = (1.0 - P.sell) / (1.0 + P.buy);
  let hasStop = lev[lb] < NONE * 0.5;
  let hasTake = lev[lb + 1u] < NONE * 0.5;
  let stop = o * exp(-max(select(0.0, lev[lb] * sig, hasStop), -kc));
  let take = o * exp(max(select(0.0, lev[lb + 1u] * sig, hasTake), -kc));
  var px = c; var ex = 0.0;
  if (hasStop && l <= stop) { px = stop * (1.0 - P.sell); ex = 1.0; }
  else if (hasTake && h >= take) { px = take; ex = 2.0; }
  var vs = 0.0; var vn = 0.0;
  let span = u32(ceil(life[t]));
  for (var k = 1u; k <= span && k <= d; k = k + 1u) {
    let v = bar(4u, d - k, i);
    if (v >= 0.0) { vs = vs + v; vn = vn + 1.0; }
  }
  let v0 = bar(4u, d, i);
  tra[ob] = 1.0; tra[ob + 1u] = (px / o) * keep - 1.0; tra[ob + 2u] = ex; tra[ob + 3u] = (c / o) * keep - 1.0;
  tra[ob + 4u] = select(-1.0, v0 / (vs / vn), vn > 0.0 && vs > 0.0 && v0 >= 0.0); tra[ob + 5u] = c / o - 1.0;
}
)wgsl";

const char* kBook = R"wgsl(
@group(0) @binding(6) var<storage, read> lev : array<f32>;
@group(0) @binding(7) var<storage, read> tra : array<f32>;
@group(0) @binding(8) var<storage, read> ex : array<f32>;
@group(0) @binding(9) var<storage, read> elig : array<u32>;
@group(0) @binding(10) var<storage, read_write> days : array<f32>;
@group(0) @binding(11) var<storage, read_write> booked : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let t = P.s + gid.x;
  if (t >= P.T) { return; }
  let od = t * KD;
  for (var k = 0u; k < KD; k = k + 1u) { days[od + k] = 0.0; }
  let plan = t + 1u >= P.T;
  let d = select(t + 1u, t, plan);
  var mean = 0.0; var nm = 0.0;
  if (!plan) {
    for (var i = 0u; i < P.N; i = i + 1u) {
      if (elig[t * P.N + i] != 0u && bar(0u, d, i) > 0.0 && bar(3u, d, i) > 0.0) { mean = mean + bar(3u, d, i) / bar(0u, d, i) - 1.0; nm = nm + 1.0; }
    }
  }
  if (nm > 0.0) { mean = mean / nm; }
  var nb = 0u; var valid = 0u; var rank = 0u;
  var prev = -1; var prevE = 0.0;
  loop {
    var best = -1;
    for (var j = 0u; j < P.N; j = j + 1u) {
      if (elig[t * P.N + j] == 0u) { continue; }
      let e = ex[t * P.N + j];
      if (prev >= 0 && !(e < prevE || (e == prevE && i32(j) > prev))) { continue; }
      if (best < 0 || e > ex[t * P.N + u32(best)]) { best = i32(j); }
    }
    if (best < 0) { break; }
    let i = u32(best);
    prev = best; prevE = ex[t * P.N + i]; rank = rank + 1u;
    let lb = (t * P.N + i) * KL;
    if (plan) {
      if (lev[lb + 3u] > 0.0 && lev[lb + 6u] > 0.0) { booked[t * P.K + nb] = i; booked[P.T * P.K + nb] = rank; nb = nb + 1u; }
      if (nb == P.K) { break; }
      continue;
    }
    let tb = (t * P.N + i) * KT;
    if (!(tra[tb] > 0.0)) { continue; }
    valid = valid + 1u;
    if (valid <= P.K) { days[od + 2u] = days[od + 2u] + tra[tb + 3u]; days[od + 4u] = days[od + 4u] + tra[tb + 5u]; }
    if (nb < P.K && lev[lb + 3u] > 0.0) {
      booked[t * P.K + nb] = i; nb = nb + 1u;
      let e = prevE;
      let xs = tra[tb + 5u] - mean;
      days[od] = days[od] + tra[tb + 1u]; days[od + 6u] = days[od + 6u] + lev[lb]; days[od + 7u] = days[od + 7u] + lev[lb + 1u];
      days[od + 8u] = days[od + 8u] + lev[lb + 4u]; days[od + 9u] = days[od + 9u] + lev[lb + 5u];
      days[od + 10u] = days[od + 10u] + select(0.0, 1.0, tra[tb + 2u] == 1.0);
      days[od + 11u] = days[od + 11u] + select(0.0, 1.0, tra[tb + 2u] == 2.0);
      days[od + 12u] = days[od + 12u] + select(0.0, 1.0, bar(2u, d, i) >= bar(0u, d, i));
      days[od + 13u] = days[od + 13u] + e; days[od + 14u] = days[od + 14u] + xs;
      days[od + 15u] = days[od + 15u] + select(0.0, 1.0, (e > 0.0) == (xs > 0.0));
      days[od + 16u] = days[od + 16u] + e; days[od + 17u] = days[od + 17u] + xs;
      days[od + 18u] = days[od + 18u] + e * e; days[od + 19u] = days[od + 19u] + xs * xs; days[od + 20u] = days[od + 20u] + e * xs;
    }
    if (nb == P.K && valid >= P.K) { break; }
  }
  let slots = f32(min(valid, P.K));
  days[od + 1u] = f32(nb); days[od + 3u] = slots; days[od + 5u] = mean;
  if (!plan && slots > 0.0) { days[od] = days[od] / slots; days[od + 2u] = days[od + 2u] / slots; days[od + 4u] = days[od + 4u] / slots; }
}
)wgsl";

// State slots (per stock).
enum : std::uint32_t {
  sVar = 0, sIv = 1, sG = 2, sL = 3, sPdm = 4, sNdm = 5, sTr = 6, sAdx = 7, sInd = 8, sAdxSet = 9,
  sSar = 10, sEp = 11, sAf = 12, sUp = 13, sSarSet = 14, sAdxSum = 15, sAdxN = 16,
  sN = 17, sSigSum = 18, sMaxDrop = 19, sMaxRise = 20, sPrevIv = 21, sCode = 22
};

template <class R>
R bar(const Buffers<R>& b, std::uint32_t f, std::uint32_t t, std::uint32_t i) { return b.bars[(static_cast<std::size_t>(f) * b.T + t) * b.N + i]; }

// erfc for x >= 0 (Abramowitz and Stegun 7.1.26), and its reflection.
template <class R>
R erfcApprox(R x) {
  const R z = std::fabs(x);
  const R t = R(1) / (R(1) + R(0.3275911) * z);
  const R y = t * (R(0.254829592) + t * (R(-0.284496736) + t * (R(1.421413741) + t * (R(-1.453152027) + t * R(1.061405429)))));
  const R e = y * std::exp(-z * z);
  return x >= R(0) ? e : R(2) - e;
}

template <class R>
struct Stock {
  Buffers<R>& b;
  std::uint32_t i;
  R kc, kcs, minMove;
  std::size_t base() const { return static_cast<std::size_t>(i) * 4 * b.T; }
  R& st(std::uint32_t k) { return b.state[static_cast<std::size_t>(i) * kState + k]; }
  R lo(std::uint32_t q) const { return b.obs[base() + q]; }
  R hi(std::uint32_t q) const { return b.obs[base() + b.T + q]; }
  R cl(std::uint32_t q) const { return b.obs[base() + 2 * static_cast<std::size_t>(b.T) + q]; }
  R sg(std::uint32_t q) const { return b.obs[base() + 3 * static_cast<std::size_t>(b.T) + q]; }
  std::uint32_t byLo(std::uint32_t k) const { return b.idx[static_cast<std::size_t>(i) * 2 * b.T + k]; }
  std::uint32_t byHi(std::uint32_t k) const { return b.idx[(static_cast<std::size_t>(i) * 2 + 1) * b.T + k]; }
  std::uint32_t n() { return static_cast<std::uint32_t>(st(sN)); }

  void add(R vlo, R vhi, R vc, R vs) {
    const std::uint32_t q = n();
    b.obs[base() + q] = vlo, b.obs[base() + b.T + q] = vhi, b.obs[base() + 2 * static_cast<std::size_t>(b.T) + q] = vc,
    b.obs[base() + 3 * static_cast<std::size_t>(b.T) + q] = vs;
    for (std::uint32_t f = 0; f < 2; ++f) {  // insertion, before the equal keys
      const std::size_t o = (static_cast<std::size_t>(i) * 2 + f) * b.T;
      const R key = f == 0 ? vlo : vhi;
      std::uint32_t k = q;
      while (k > 0 && (f == 0 ? lo(b.idx[o + k - 1]) : hi(b.idx[o + k - 1])) >= key) b.idx[o + k] = b.idx[o + k - 1], --k;
      b.idx[o + k] = q;
    }
    st(sN) = R(q + 1), st(sSigSum) += vs;
    st(sMaxDrop) = std::max(st(sMaxDrop), vlo * vs), st(sMaxRise) = std::max(st(sMaxRise), vhi * vs);
  }

  static R ratio(R s1, R s2, R n) {
    if (n < R(2)) return R(-kNone);
    const R mu = s1 / n, vr = std::max(R(0), s2 / n - mu * mu);
    if (vr > R(0)) return mu / std::sqrt(vr);
    return mu > R(0) ? R(kNone) : R(-kNone);
  }
  R value(std::uint32_t f, std::uint32_t q) const { return f == 0 ? lo(q) : hi(q); }
  std::uint32_t sorted(std::uint32_t f, std::uint32_t k) const { return f == 0 ? byLo(k) : byHi(k); }
  // The largest value of the field that is not an outlier by Chauvenet's criterion.
  R plausibleMax(std::uint32_t f) {
    const std::uint32_t nn = n();
    if (nn < 4) return nn > 0 ? value(f, sorted(f, nn - 1)) : R(kNone);
    const R q1 = value(f, sorted(f, nn / 4)), q2 = value(f, sorted(f, nn / 2)), q3 = value(f, sorted(f, 3 * nn / 4));
    const R sc = (q3 - q1) / R(1.349);
    for (std::uint32_t k = nn; k > 0; --k) {
      const R v = value(f, sorted(f, k - 1));
      if (!(sc > R(0)) || R(nn) * erfcApprox((v - q2) / sc / R(1.4142135623730951)) >= R(0.5)) return v;
    }
    return value(f, sorted(f, nn - 1));
  }
  R meanSig() { return st(sSigSum) / R(n()); }
  R score(R a, R bb) {
    R s1 = 0, s2 = 0;
    for (std::uint32_t q = 0; q < n(); ++q) {
      const R v = lo(q) >= a ? -a * sg(q) + kcs : (hi(q) >= bb ? bb * sg(q) : cl(q)) + kc;
      s1 += v, s2 += v * v;
    }
    return ratio(s1, s2, R(n()));
  }
  R bestStop(R bb, R& score) {
    const std::uint32_t nn = n();
    const R capA = plausibleMax(0), floorA = std::max(minMove / meanSig(), lo(byLo(nn / 2)));
    R t1 = 0, t2 = 0, tm = 0, h1 = 0, h2 = 0, bestA = capA;
    for (std::uint32_t q = 0; q < nn; ++q) t1 += sg(q), t2 += sg(q) * sg(q), tm += R(1);
    score = R(-kNone);
    for (std::uint32_t k = 0; k < nn; ++k) {
      const std::uint32_t x = byLo(k);
      const R a = lo(x), base = (hi(x) >= bb ? bb * sg(x) : cl(x)) + kc;
      if (a > R(0) && a >= floorA && a <= capA) {
        const R s1 = h1 - a * t1 + tm * kcs, s2 = h2 + a * a * t2 - R(2) * a * kcs * t1 + tm * kcs * kcs;
        const R rr = ratio(s1, s2, R(nn));
        if (rr > score + R(1e-12)) score = rr, bestA = a;
      }
      h1 += base, h2 += base * base, t1 -= sg(x), t2 -= sg(x) * sg(x), tm -= R(1);
    }
    return bestA;
  }
  R bestTake(R a) {
    const std::uint32_t nn = n();
    const R capB = plausibleMax(1), floorB = minMove / meanSig();
    R c1 = 0, c2 = 0, t1 = 0, t2 = 0, tm = 0, h1 = 0, h2 = 0, bestB = capB, score = R(-kNone);
    for (std::uint32_t q = 0; q < nn; ++q) {
      if (lo(q) >= a) {
        const R v = -a * sg(q) + kcs;
        c1 += v, c2 += v * v;
      } else {
        t1 += sg(q), t2 += sg(q) * sg(q), tm += R(1);
      }
    }
    for (std::uint32_t k = 0; k < nn; ++k) {
      const std::uint32_t x = byHi(k);
      if (lo(x) >= a) continue;
      const R bb = hi(x), v = cl(x) + kc;
      if (bb > R(0) && bb >= floorB && bb <= capB) {
        const R s1 = c1 + h1 + bb * t1 + tm * kc, s2 = c2 + h2 + bb * bb * t2 + R(2) * bb * kc * t1 + tm * kc * kc;
        const R rr = ratio(s1, s2, R(nn));
        if (rr > score + R(1e-12)) score = rr, bestB = bb;
      }
      h1 += v, h2 += v * v, t1 -= sg(x), t2 -= sg(x) * sg(x), tm -= R(1);
    }
    return bestB;
  }
  // The learnt levels from 50% (the median fall and rise), in turns, while the ratio improves.
  bool learn(R& a, R& bb, R& best) {
    a = R(kNone), bb = R(kNone), best = R(-kNone);
    const std::uint32_t nn = n();
    if (nn < 2) return false;
    const R floorL = minMove / meanSig();
    a = std::max(lo(byLo(nn / 2)), floorL), bb = std::max(hi(byHi(nn / 2)), floorL);
    best = score(a, bb);
    for (int it = 0; it < 50; ++it) {
      R s2;
      const R nb = bestTake(a), na = bestStop(nb, s2);
      if (!(s2 > best + R(1e-12))) break;
      best = s2;
      if (na == a && nb == bb) break;
      a = na, bb = nb;
    }
    return best > R(0);
  }
};

}  // namespace

template <class R>
void levels(Buffers<R>& b, std::uint32_t i, std::uint32_t t0, std::uint32_t t1) {
  const R kc = std::log((R(1) - b.sell) / (R(1) + b.buy));
  Stock<R> x{b, i, kc, kc + std::log(R(1) - b.sell), -kc};
  for (std::uint32_t t = t0; t < t1; ++t) {
    const R o = bar(b, 0, t, i), h = bar(b, 1, t, i), l = bar(b, 2, t, i), c = bar(b, 3, t, i);
    const bool ok = o > R(0) && h > R(0) && l > R(0) && c > R(0);
    // The day joins the stock's record, in the volatility known at the close before.
    const R pv = x.st(sPrevIv);
    if (t >= 1 && pv > R(0) && ok && l <= std::min(o, c) && h >= std::max(o, c))
      x.add(-std::log(l / o) / pv, std::log(h / o) / pv, std::log(c / o), pv);
    // Open-to-close volatility, half-life the forecast's life.
    const R lam = std::exp2(R(-1) / std::max(R(1), b.life[t]));
    if (o > R(0) && c > R(0)) {
      const R z = std::log(c / o);
      x.st(sVar) = x.st(sVar) >= R(0) ? lam * x.st(sVar) + (R(1) - lam) * z * z : z * z;
    }
    x.st(sIv) = x.st(sVar) > R(0) ? std::sqrt(x.st(sVar)) : R(0);
    // Parabolic SAR, RSI and ADX over the forecast's life (the model's next horizon).
    R hz = b.H ? R(b.hz[b.H - 1]) : R(1);
    for (std::uint32_t k = b.H; k > 0; --k)
      if (R(b.hz[k - 1]) >= b.life[t]) hz = R(b.hz[k - 1]);
    const R al = R(1) / hz;
    if (t >= 1) {
      const R ph = bar(b, 1, t - 1, i), pl = bar(b, 2, t - 1, i), pc = bar(b, 3, t - 1, i);
      if (ok && ph > R(0) && pl > R(0) && pc > R(0)) {
        const R up = h - ph, dn = pl - l;
        const R gz = std::max(R(0), c - pc), lz = std::max(R(0), pc - c), pz = up > dn && up > R(0) ? up : R(0),
                nz = dn > up && dn > R(0) ? dn : R(0), tz = std::max({h - l, std::fabs(h - pc), std::fabs(l - pc)});
        if (x.st(sInd) > R(0)) {
          x.st(sG) += al * (gz - x.st(sG)), x.st(sL) += al * (lz - x.st(sL)), x.st(sPdm) += al * (pz - x.st(sPdm)),
              x.st(sNdm) += al * (nz - x.st(sNdm)), x.st(sTr) += al * (tz - x.st(sTr));
        } else {
          x.st(sG) = gz, x.st(sL) = lz, x.st(sPdm) = pz, x.st(sNdm) = nz, x.st(sTr) = tz, x.st(sInd) = R(1);
        }
        if (x.st(sTr) > R(0)) {
          const R pdi = x.st(sPdm) / x.st(sTr), ndi = x.st(sNdm) / x.st(sTr);
          if (pdi + ndi > R(0)) {
            const R dx = R(100) * std::fabs(pdi - ndi) / (pdi + ndi);
            if (x.st(sAdxSet) > R(0)) x.st(sAdx) += al * (dx - x.st(sAdx));
            else x.st(sAdx) = dx, x.st(sAdxSet) = R(1);
          }
        }
        if (!(x.st(sSarSet) > R(0))) {
          x.st(sUp) = c >= pc ? R(1) : R(0), x.st(sSar) = c >= pc ? pl : ph, x.st(sEp) = c >= pc ? h : l, x.st(sAf) = al, x.st(sSarSet) = R(1);
        } else if (x.st(sUp) > R(0)) {
          x.st(sSar) = std::min(x.st(sSar) + x.st(sAf) * (x.st(sEp) - x.st(sSar)), pl);
          if (l < x.st(sSar)) x.st(sUp) = R(0), x.st(sSar) = x.st(sEp), x.st(sEp) = l, x.st(sAf) = al;
          else if (h > x.st(sEp)) x.st(sEp) = h, x.st(sAf) = std::min(R(1), x.st(sAf) + al);
        } else {
          x.st(sSar) = std::max(x.st(sSar) + x.st(sAf) * (x.st(sEp) - x.st(sSar)), ph);
          if (h > x.st(sSar)) x.st(sUp) = R(1), x.st(sSar) = x.st(sEp), x.st(sEp) = h, x.st(sAf) = al;
          else if (l < x.st(sEp)) x.st(sEp) = l, x.st(sAf) = std::min(R(1), x.st(sAf) + al);
        }
        if (x.st(sAdxSet) > R(0)) x.st(sAdxSum) += x.st(sAdx), x.st(sAdxN) += R(1);
      }
    }
    R code = R(-1);
    if (x.st(sAdxSet) > R(0) && x.st(sInd) > R(0) && x.st(sG) + x.st(sL) > R(0) && x.st(sAdxN) >= R(1))
      code = (x.st(sUp) > R(0) ? R(4) : R(0)) + (x.st(sG) / (x.st(sG) + x.st(sL)) > R(0.5) ? R(2) : R(0)) +
             (x.st(sAdx) > x.st(sAdxSum) / x.st(sAdxN) ? R(1) : R(0));
    x.st(sCode) = code;
    // The decision at this close: the levels learnt from the record so far.
    if (t >= b.s) {
      R a, bb, best;
      const bool trade = x.learn(a, bb, best);
      R pS = 0, pT = 0;
      const std::uint32_t nn = x.n();
      for (std::uint32_t q = 0; q < nn; ++q) pS += x.lo(q) >= a ? R(1) : R(0), pT += x.lo(q) < a && x.hi(q) >= bb ? R(1) : R(0);
      if (nn > 0) pS /= R(nn), pT /= R(nn);
      R* out = &b.levels[(static_cast<std::size_t>(t) * b.N + i) * kLevel];
      out[0] = a, out[1] = bb, out[2] = best, out[3] = trade ? R(1) : R(0), out[4] = pS, out[5] = pT, out[6] = x.st(sIv),
      out[7] = x.st(sMaxDrop), out[8] = x.st(sMaxRise), out[9] = code;
    }
    x.st(sPrevIv) = x.st(sIv);
  }
}

template <class R>
void trades(Buffers<R>& b, std::uint32_t t, std::uint32_t i) {
  R* out = &b.trades[(static_cast<std::size_t>(t) * b.N + i) * kTrade];
  for (std::size_t k = 0; k < kTrade; ++k) out[k] = R(0);
  if (t + 1 >= b.T) return;
  const std::uint32_t d = t + 1;
  const R* lv = &b.levels[(static_cast<std::size_t>(t) * b.N + i) * kLevel];
  const R o = bar(b, 0, d, i), h = bar(b, 1, d, i), l = bar(b, 2, d, i), c = bar(b, 3, d, i), sig = lv[6];
  if (!(o > R(0) && h > R(0) && l > R(0) && c > R(0) && sig > R(0)) || l > std::min(o, c) || h < std::max(o, c)) return;
  const R kc = std::log((R(1) - b.sell) / (R(1) + b.buy)), keep = (R(1) - b.sell) / (R(1) + b.buy);
  const bool hasStop = lv[0] < R(kNone) * R(0.5), hasTake = lv[1] < R(kNone) * R(0.5);
  const R stop = o * std::exp(-std::max(hasStop ? lv[0] * sig : R(0), -kc)), take = o * std::exp(std::max(hasTake ? lv[1] * sig : R(0), -kc));
  R px = c, ex = R(0);
  if (hasStop && l <= stop) px = stop * (R(1) - b.sell), ex = R(1);
  else if (hasTake && h >= take) px = take, ex = R(2);
  R vs = 0, vn = 0;
  const std::uint32_t span = static_cast<std::uint32_t>(std::ceil(b.life[t]));
  for (std::uint32_t k = 1; k <= span && k <= d; ++k) {
    const R v = bar(b, 4, d - k, i);
    if (v >= R(0)) vs += v, vn += R(1);
  }
  const R v0 = bar(b, 4, d, i);
  out[0] = R(1), out[1] = (px / o) * keep - R(1), out[2] = ex, out[3] = (c / o) * keep - R(1);
  out[4] = vn > R(0) && vs > R(0) && v0 >= R(0) ? v0 / (vs / vn) : R(-1), out[5] = c / o - R(1);
}

template <class R>
void book(Buffers<R>& b, std::uint32_t t) {
  R* out = &b.days[static_cast<std::size_t>(t) * kDay];
  for (std::size_t k = 0; k < kDay; ++k) out[k] = R(0);
  const bool plan = t + 1 >= b.T;
  const std::uint32_t d = plan ? t : t + 1;
  R mean = 0, nm = 0;
  if (!plan)
    for (std::uint32_t i = 0; i < b.N; ++i)
      if (b.elig[static_cast<std::size_t>(t) * b.N + i] && bar(b, 0, d, i) > R(0) && bar(b, 3, d, i) > R(0))
        mean += bar(b, 3, d, i) / bar(b, 0, d, i) - R(1), nm += R(1);
  if (nm > R(0)) mean /= nm;
  // Down the ranking (expected return, then stock index).
  std::uint32_t booked = 0, valid = 0, rank = 0;
  std::int64_t prev = -1;
  R prevE = 0;
  for (;;) {
    std::int64_t best = -1;
    for (std::uint32_t j = 0; j < b.N; ++j) {
      if (!b.elig[static_cast<std::size_t>(t) * b.N + j]) continue;
      const R e = b.expected[static_cast<std::size_t>(t) * b.N + j];
      if (prev >= 0 && !(e < prevE || (e == prevE && static_cast<std::int64_t>(j) > prev))) continue;
      if (best < 0 || e > b.expected[static_cast<std::size_t>(t) * b.N + static_cast<std::size_t>(best)]) best = j;
    }
    if (best < 0) break;
    const std::uint32_t i = static_cast<std::uint32_t>(best);
    prev = best, prevE = b.expected[static_cast<std::size_t>(t) * b.N + i], ++rank;
    const R* lv = &b.levels[(static_cast<std::size_t>(t) * b.N + i) * kLevel];
    if (plan) {
      if (lv[3] > R(0) && lv[6] > R(0)) b.booked[static_cast<std::size_t>(t) * b.K + booked] = i, b.booked[static_cast<std::size_t>(b.T) * b.K + booked] = rank, ++booked;
      if (booked == b.K) break;
      continue;
    }
    const R* tr = &b.trades[(static_cast<std::size_t>(t) * b.N + i) * kTrade];
    if (!(tr[0] > R(0))) continue;
    ++valid;
    if (valid <= b.K) out[2] += tr[3], out[4] += tr[5];
    if (booked < b.K && lv[3] > R(0)) {
      b.booked[static_cast<std::size_t>(t) * b.K + booked] = i, ++booked;
      const R e = prevE, xs = tr[5] - mean;
      out[0] += tr[1], out[6] += lv[0], out[7] += lv[1], out[8] += lv[4], out[9] += lv[5];
      out[10] += tr[2] == R(1) ? R(1) : R(0), out[11] += tr[2] == R(2) ? R(1) : R(0), out[12] += bar(b, 2, d, i) >= bar(b, 0, d, i) ? R(1) : R(0);
      out[13] += e, out[14] += xs, out[15] += (e > R(0)) == (xs > R(0)) ? R(1) : R(0);
      out[16] += e, out[17] += xs, out[18] += e * e, out[19] += xs * xs, out[20] += e * xs;
    }
    if (booked == b.K && valid >= b.K) break;
  }
  const R slots = R(std::min(valid, b.K));
  out[1] = R(booked), out[3] = slots, out[5] = mean;
  if (!plan && slots > R(0)) out[0] /= slots, out[2] /= slots, out[4] /= slots;
}

template <class R>
void run(Buffers<R>& b, std::uint32_t chunk) {
  for (std::uint32_t t0 = 0; t0 < b.T; t0 += chunk)
    for (std::uint32_t i = 0; i < b.N; ++i) levels(b, i, t0, std::min(b.T, t0 + chunk));
  for (std::uint32_t t = b.s; t + 1 < b.T; ++t)
    for (std::uint32_t i = 0; i < b.N; ++i) trades(b, t, i);
  for (std::uint32_t t = b.s; t < b.T; ++t) book(b, t);
}

const std::string& levelsSource() {
  static const std::string s = std::string(kHeader) + kLevels;
  return s;
}
const std::string& tradesSource() {
  static const std::string s = std::string(kHeader) + kTrades;
  return s;
}
const std::string& bookSource() {
  static const std::string s = std::string(kHeader) + kBook;
  return s;
}

template void levels<double>(Buffers<double>&, std::uint32_t, std::uint32_t, std::uint32_t);
template void levels<float>(Buffers<float>&, std::uint32_t, std::uint32_t, std::uint32_t);
template void trades<double>(Buffers<double>&, std::uint32_t, std::uint32_t);
template void trades<float>(Buffers<float>&, std::uint32_t, std::uint32_t);
template void book<double>(Buffers<double>&, std::uint32_t);
template void book<float>(Buffers<float>&, std::uint32_t);
template void run<double>(Buffers<double>&, std::uint32_t);
template void run<float>(Buffers<float>&, std::uint32_t);

}  // namespace ofm::daytrade
