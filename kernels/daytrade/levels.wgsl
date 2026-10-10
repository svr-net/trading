// #include header.wgsl
// One workgroup per stock (grid N; called per chunk of days [t0, t1), state carried over). Lane 0
// keeps the stock's running state (volatility, parabolic SAR / RSI / ADX); the 64 lanes share the
// work on its record of days: the sorted insertion of each new day, and at each decision the
// learnt stop-loss and take-profit (sorted sweeps as two-pass parallel scans, scores as
// reductions), their touch probabilities, and the trade on the next day's bar (fused: bought at the
// open, sold at the stop first, the take-profit or the close).
@group(0) @binding(2) var<storage, read> life : array<f32>;
@group(0) @binding(3) var<storage, read_write> st : array<f32>;
@group(0) @binding(4) var<storage, read_write> obs : array<f32>;
@group(0) @binding(5) var<storage, read_write> idx : array<u32>;
@group(0) @binding(6) var<storage, read_write> lev : array<f32>;
@group(0) @binding(7) var<storage, read_write> tra : array<f32>;

// The lanes per stock: 64 on WebGPU (hardware, or a software adapter running lanes as SIMD). The C++
// translation runs one lane (wgsl2cpp pragma below): there the same code is the sequential sweep,
// without the 64-lane bookkeeping a scalar CPU thread gains nothing from.
// wgsl2cpp: override WG = 1
override WG : u32 = 64u;

var<private> I : u32;
var<private> LID : u32;
var<private> KC : f32;
var<private> KCS : f32;
var<private> MINMOVE : f32;

var<workgroup> R : array<f32, 640>;
var<workgroup> RB : array<f32, 128>;
var<workgroup> RU : array<u32, 64>;
var<workgroup> SC : f32;
var<workgroup> U0 : f32;
var<workgroup> U1 : f32;
var<workgroup> J0 : f32;
var<workgroup> J1 : f32;
var<workgroup> UU : u32;
var<workgroup> UQ : u32;
var<workgroup> UP : u32;

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
fn meanSig() -> f32 { return S(18u) / f32(cnt()); }

fn erfcApprox(x : f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  let e = y * exp(-z * z);
  return select(2.0 - e, e, x >= 0.0);
}

fn ratio(s1 : f32, s2 : f32, n : f32) -> f32 {
  if (n < 2.0) { return -NONE; }
  let mu = s1 / n;
  let vr = max(0.0, s2 / n - mu * mu);
  if (vr > 0.0) { return mu / sqrt(vr); }
  return select(-NONE, NONE, mu > 0.0);
}

// Inserts record q (key: its value of field f) into the order by f: lane 0 finds the position (the
// number of recorded values below the key) by bisection; the entries above it move up one, 64 at a
// time from the top.
fn insert(f : u32, key : f32, q : u32) {
  let o : u32 = (I * 2u + f) * P.T;
  if (LID == 0u) {
    var lo0 = 0u;
    var hi0 = q;
    loop {
      if (lo0 >= hi0) { break; }
      let mid = (lo0 + hi0) / 2u;
      if (value(f, idx[o + mid]) < key) { lo0 = mid + 1u; } else { hi0 = mid; }
    }
    UP = lo0;
  }
  let pos : u32 = workgroupUniformLoad(&UP);
  for (var top = q; top > pos; top = top - min(top - pos, WG)) {
    let n : u32 = min(top - pos, WG);
    var v : u32 = 0u;
    if (LID < n) { v = idx[o + top - 1u - LID]; }
    storageBarrier();
    workgroupBarrier();
    if (LID < n) { idx[o + top - LID] = v; }
    storageBarrier();
    workgroupBarrier();
  }
  if (LID == 0u) { idx[o + pos] = q; }
  storageBarrier();
  workgroupBarrier();
}

// Sharpe ratio of the record's trades with stop a and take bb (in volatilities).
fn scoreAB(a : f32, bb : f32, nn : u32) -> f32 {
  var s1 : f32 = 0.0;
  var s2 : f32 = 0.0;
  for (var q = LID; q < nn; q = q + WG) {
    let v = select(select(cl(q), bb * sg(q), hi(q) >= bb) + KC, -a * sg(q) + KCS, lo(q) >= a);
    s1 = s1 + v;
    s2 = s2 + v * v;
  }
  R[LID] = s1;
  R[WG + LID] = s2;
  workgroupBarrier();
  if (LID == 0u) {
    var t1 = 0.0;
    var t2 = 0.0;
    for (var j = 0u; j < WG; j = j + 1u) { t1 = t1 + R[j]; t2 = t2 + R[WG + j]; }
    U0 = ratio(t1, t2, f32(nn));
  }
  let r : f32 = workgroupUniformLoad(&U0);
  return r;
}

// The largest recorded value of field f that is plausible by Chauvenet's criterion (robust scale).
fn plausibleMax(f : u32, nn : u32) -> f32 {
  if (nn < 4u) {
    if (nn > 0u) { return value(f, sorted(f, nn - 1u)); }
    return NONE;
  }
  let q1 : f32 = value(f, sorted(f, nn / 4u));
  let q2 : f32 = value(f, sorted(f, nn / 2u));
  let q3 : f32 = value(f, sorted(f, 3u * nn / 4u));
  let sc : f32 = (q3 - q1) / 1.349;
  // From the top, 64 values at a time, until a block holds a plausible one.
  var kb : u32 = 0u;
  for (var top = nn; top > 0u; top = top - min(top, WG)) {
    var best : u32 = 0u;
    if (LID < top) {
      let k = top - LID;
      let v = value(f, sorted(f, k - 1u));
      if (!(sc > 0.0) || f32(nn) * erfcApprox((v - q2) / sc / 1.4142135623730951) >= 0.5) { best = k; }
    }
    RU[LID] = best;
    workgroupBarrier();
    if (LID == 0u) {
      var m = 0u;
      for (var j = 0u; j < WG; j = j + 1u) { m = max(m, RU[j]); }
      UP = m;
    }
    let found : u32 = workgroupUniformLoad(&UP);
    if (found > 0u) {
      kb = found;
      break;
    }
  }
  if (kb > 0u) { return value(f, sorted(f, kb - 1u)); }
  return value(f, sorted(f, nn - 1u));
}

// Lane 0 turns the lanes' chunk sums R[j * WG + lane] (j < m) into exclusive prefixes, and puts
// the totals at R[(m + j) * WG].
fn scanLanes(m : u32) {
  workgroupBarrier();
  if (LID == 0u) {
    for (var j = 0u; j < m; j = j + 1u) {
      var c = 0.0;
      for (var l = 0u; l < WG; l = l + 1u) {
        let e = R[j * WG + l];
        R[j * WG + l] = c;
        c = c + e;
      }
      R[(m + j) * WG] = c;
    }
  }
  workgroupBarrier();
}

// The lanes' best (score, level) pairs RB[lane], RB[WG + lane], merged in lane order (each lane's
// chunk follows the previous one's): the score goes to SC, the level is returned.
fn mergeBest(dflt : f32) -> f32 {
  workgroupBarrier();
  if (LID == 0u) {
    var s = -NONE;
    var b = dflt;
    for (var j = 0u; j < WG; j = j + 1u) {
      if (RB[j] > s + 1e-12) { s = RB[j]; b = RB[WG + j]; }
    }
    SC = s;
    U1 = b;
  }
  let r : f32 = workgroupUniformLoad(&U1);
  return r;
}

// The best stop for take bb: a sweep of the record sorted by the fall from the open.
fn bestStop(bb : f32, nn : u32) -> f32 {
  let capA : f32 = plausibleMax(0u, nn);
  let floorA : f32 = max(MINMOVE / meanSig(), lo(byLo(nn / 2u)));
  let ch : u32 = (nn + WG - 1u) / WG;
  let k0 : u32 = min(nn, LID * ch);
  let k1 : u32 = min(nn, k0 + ch);
  var p1 : f32 = 0.0;
  var p2 : f32 = 0.0;
  var p3 : f32 = 0.0;
  var p4 : f32 = 0.0;
  for (var k = k0; k < k1; k = k + 1u) {
    let x = byLo(k);
    let base = select(cl(x), bb * sg(x), hi(x) >= bb) + KC;
    p1 = p1 + base;
    p2 = p2 + base * base;
    p3 = p3 + sg(x);
    p4 = p4 + sg(x) * sg(x);
  }
  R[LID] = p1;
  R[WG + LID] = p2;
  R[2u * WG + LID] = p3;
  R[3u * WG + LID] = p4;
  scanLanes(4u);
  var h1 : f32 = R[LID];
  var h2 : f32 = R[WG + LID];
  var t1 : f32 = R[6u * WG] - R[2u * WG + LID];
  var t2 : f32 = R[7u * WG] - R[3u * WG + LID];
  var tm : f32 = f32(nn - k0);
  var bs : f32 = -NONE;
  var ba : f32 = capA;
  for (var k = k0; k < k1; k = k + 1u) {
    let x = byLo(k);
    let a = lo(x);
    let base = select(cl(x), bb * sg(x), hi(x) >= bb) + KC;
    if (a > 0.0 && a >= floorA && a <= capA) {
      let s1 = h1 - a * t1 + tm * KCS;
      let s2 = h2 + a * a * t2 - 2.0 * a * KCS * t1 + tm * KCS * KCS;
      let rr = ratio(s1, s2, f32(nn));
      if (rr > bs + 1e-12) { bs = rr; ba = a; }
    }
    h1 = h1 + base;
    h2 = h2 + base * base;
    t1 = t1 - sg(x);
    t2 = t2 - sg(x) * sg(x);
    tm = tm - 1.0;
  }
  RB[LID] = bs;
  RB[WG + LID] = ba;
  let best : f32 = mergeBest(capA);
  return best;
}

// The best take for stop a: a sweep of the record sorted by the rise from the open.
fn bestTake(a : f32, nn : u32) -> f32 {
  let capB : f32 = plausibleMax(1u, nn);
  let floorB : f32 = MINMOVE / meanSig();
  let ch : u32 = (nn + WG - 1u) / WG;
  let k0 : u32 = min(nn, LID * ch);
  let k1 : u32 = min(nn, k0 + ch);
  var p1 : f32 = 0.0;
  var p2 : f32 = 0.0;
  var p3 : f32 = 0.0;
  var p4 : f32 = 0.0;
  var p5 : f32 = 0.0;
  var c1 : f32 = 0.0;
  var c2 : f32 = 0.0;
  for (var k = k0; k < k1; k = k + 1u) {
    let x = byHi(k);
    if (lo(x) >= a) {
      let v = -a * sg(x) + KCS;
      c1 = c1 + v;
      c2 = c2 + v * v;
    } else {
      let v = cl(x) + KC;
      p1 = p1 + v;
      p2 = p2 + v * v;
      p3 = p3 + sg(x);
      p4 = p4 + sg(x) * sg(x);
      p5 = p5 + 1.0;
    }
  }
  R[LID] = p1;
  R[WG + LID] = p2;
  R[2u * WG + LID] = p3;
  R[3u * WG + LID] = p4;
  R[4u * WG + LID] = p5;
  R[5u * WG + LID] = c1;
  R[6u * WG + LID] = c2;
  workgroupBarrier();
  if (LID == 0u) {
    var e1 = 0.0;
    var e2 = 0.0;
    for (var l = 0u; l < WG; l = l + 1u) { e1 = e1 + R[5u * WG + l]; e2 = e2 + R[6u * WG + l]; }
    U0 = e1;
    J1 = e2;
  }
  scanLanes(5u);
  let cs1 : f32 = U0;
  let cs2 : f32 = J1;
  var h1 : f32 = R[LID];
  var h2 : f32 = R[WG + LID];
  var t1 : f32 = R[7u * WG] - R[2u * WG + LID];
  var t2 : f32 = R[8u * WG] - R[3u * WG + LID];
  var tm : f32 = R[9u * WG] - R[4u * WG + LID];
  var bs : f32 = -NONE;
  var bv : f32 = capB;
  for (var k = k0; k < k1; k = k + 1u) {
    let x = byHi(k);
    if (lo(x) >= a) { continue; }
    let bb = hi(x);
    let v = cl(x) + KC;
    if (bb > 0.0 && bb >= floorB && bb <= capB) {
      let s1 = cs1 + h1 + bb * t1 + tm * KC;
      let s2 = cs2 + h2 + bb * bb * t2 + 2.0 * bb * KC * t1 + tm * KC * KC;
      let rr = ratio(s1, s2, f32(nn));
      if (rr > bs + 1e-12) { bs = rr; bv = bb; }
    }
    h1 = h1 + v;
    h2 = h2 + v * v;
    t1 = t1 - sg(x);
    t2 = t2 - sg(x) * sg(x);
    tm = tm - 1.0;
  }
  RB[LID] = bs;
  RB[WG + LID] = bv;
  let best : f32 = mergeBest(capB);
  return best;
}

// Day t's levels (just written) against day t + 1's bar.
fn trade(t : u32) {
  let i = I;
  let ob = (t * P.N + i) * KT;
  for (var k = 0u; k < KT; k = k + 1u) { tra[ob + k] = 0.0; }
  if (t + 1u >= P.T) { return; }
  let d = t + 1u;
  let lb = (t * P.N + i) * KL;
  let o = bar(0u, d, i); let h = bar(1u, d, i); let l = bar(2u, d, i); let c = bar(3u, d, i); let sig = lev[lb + 6u];
  if (!(o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 && sig > 0.0) || l > min(o, c) || h < max(o, c)) { return; }
  let keep = (1.0 - P.sell) / (1.0 + P.buy);
  let hasStop = lev[lb] < NONE * 0.5;
  let hasTake = lev[lb + 1u] < NONE * 0.5;
  let stop = o * exp(-max(select(0.0, lev[lb] * sig, hasStop), -KC));
  let take = o * exp(max(select(0.0, lev[lb + 1u] * sig, hasTake), -KC));
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

// Lane 0: the day's volatility and indicator state (after the day joined the record).
fn updateState(t : u32) {
  let o = bar(0u, t, I); let h = bar(1u, t, I); let l = bar(2u, t, I); let c = bar(3u, t, I);
  let ok = o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0;
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
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  I = wid.x;
  LID = lane;
  KC = log((1.0 - P.sell) / (1.0 + P.buy));
  KCS = KC + log(1.0 - P.sell);
  MINMOVE = -KC;
  if (wid.x < P.N) {
    for (var t = P.t0; t < P.t1; t = t + 1u) {
      // Lane 0: the day joins the record, in the volatility known at the close before.
      if (LID == 0u) {
        let o = bar(0u, t, I); let h = bar(1u, t, I); let l = bar(2u, t, I); let c = bar(3u, t, I);
        let ok = o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0;
        let pv = S(21u);
        let q = cnt();
        var join = 0u;
        if (t >= 1u && pv > 0.0 && ok && l <= min(o, c) && h >= max(o, c)) {
          let vlo = -log(l / o) / pv;
          let vhi = log(h / o) / pv;
          let base = I * 4u * P.T;
          obs[base + q] = vlo; obs[base + P.T + q] = vhi; obs[base + 2u * P.T + q] = log(c / o); obs[base + 3u * P.T + q] = pv;
          setS(17u, f32(q + 1u)); setS(18u, S(18u) + pv);
          setS(19u, max(S(19u), vlo * pv)); setS(20u, max(S(20u), vhi * pv));
          J0 = vlo;
          J1 = vhi;
          join = 1u;
        }
        UU = join;
        UQ = q;
      }
      storageBarrier();
      let join : u32 = workgroupUniformLoad(&UU);
      let q : u32 = workgroupUniformLoad(&UQ);
      if (join == 1u) {
        let klo : f32 = workgroupUniformLoad(&J0);
        let khi : f32 = workgroupUniformLoad(&J1);
        insert(0u, klo, q);
        insert(1u, khi, q);
      }
      if (LID == 0u) { updateState(t); }
      storageBarrier();
      workgroupBarrier();
      if (t >= P.s) {
        let nn : u32 = q + join;
        var a : f32 = NONE;
        var bb : f32 = NONE;
        var best : f32 = -NONE;
        if (nn >= 2u) {
          if (LID == 0u) {
            let floorL = MINMOVE / meanSig();
            J0 = max(lo(byLo(nn / 2u)), floorL);
            J1 = max(hi(byHi(nn / 2u)), floorL);
          }
          a = workgroupUniformLoad(&J0);
          bb = workgroupUniformLoad(&J1);
          best = scoreAB(a, bb, nn);
          for (var it = 0; it < 50; it = it + 1) {
            let nb : f32 = bestTake(a, nn);
            let na : f32 = bestStop(nb, nn);
            let sc : f32 = workgroupUniformLoad(&SC);
            if (!(sc > best + 1e-12)) { break; }
            best = sc;
            if (na == a && nb == bb) { break; }
            a = na;
            bb = nb;
          }
        }
        var ps : f32 = 0.0;
        var pt : f32 = 0.0;
        for (var k = LID; k < nn; k = k + WG) {
          if (lo(k) >= a) { ps = ps + 1.0; }
          if (lo(k) < a && hi(k) >= bb) { pt = pt + 1.0; }
        }
        R[LID] = ps;
        R[WG + LID] = pt;
        workgroupBarrier();
        if (LID == 0u) {
          var s1 = 0.0;
          var s2 = 0.0;
          for (var j = 0u; j < WG; j = j + 1u) { s1 = s1 + R[j]; s2 = s2 + R[WG + j]; }
          if (nn > 0u) { s1 = s1 / f32(nn); s2 = s2 / f32(nn); }
          let o2 = (t * P.N + I) * KL;
          lev[o2] = a; lev[o2 + 1u] = bb; lev[o2 + 2u] = best; lev[o2 + 3u] = select(0.0, 1.0, nn >= 2u && best > 0.0);
          lev[o2 + 4u] = s1; lev[o2 + 5u] = s2; lev[o2 + 6u] = S(1u); lev[o2 + 7u] = S(19u); lev[o2 + 8u] = S(20u); lev[o2 + 9u] = S(22u);
          trade(t);
        }
        workgroupBarrier();
      }
      if (LID == 0u) { setS(21u, S(1u)); }
    }
  }
}
