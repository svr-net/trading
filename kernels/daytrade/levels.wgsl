// #include header.wgsl
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
