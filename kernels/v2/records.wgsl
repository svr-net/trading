// #include header.wgsl
// The day's trades, fused, second half: each setting's day (its ranking's candidates' returns under
// its entry and exit setting, from levels.wgsl, averaged over the candidates; one unit each, an
// unfilled one in cash), added to its record, and its scores for the next day. The fields of P: T
// the settings (rankings x u0), u0 the entry and exit settings per share, NT the trades a ranking
// holds, t0 the days in the records after today's. A setting g = ranking * u0 + its entry and exit
// setting. cand [rankings][NT] a ranking's candidates (slot, -1 none); ret [slots][u0]. s1, s2, dn [T]
// the records' sums of the days' results, of their squares, of the losses' squares; sc [T] the
// record's mean shrunk by its t-statistic, so [T] its mean over its downside deviation (Sortino).
@group(0) @binding(1) var<storage, read> cand : array<f32>;
@group(0) @binding(2) var<storage, read> ret : array<f32>;
@group(0) @binding(3) var<storage, read_write> s1 : array<f32>;
@group(0) @binding(4) var<storage, read_write> s2 : array<f32>;
@group(0) @binding(5) var<storage, read_write> dn : array<f32>;
@group(0) @binding(6) var<storage, read_write> sc : array<f32>;
@group(0) @binding(7) var<storage, read_write> so : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let g : u32 = wid.x * 64u + lane;
  if (g < P.T) {
    let k : u32 = g / P.u0;
    let li : u32 = g % P.u0;
    var y : f32 = 0.0;
    var n : f32 = 0.0;
    for (var j = 0u; j < P.NT; j = j + 1u) {
      let s : f32 = cand[k * P.NT + j];
      if (s < 0.0) { continue; }
      n = n + 1.0;
      y = y + ret[u32(s) * P.u0 + li];
    }
    if (n > 0.0) { y = y / n; }
    s1[g] = s1[g] + y;
    s2[g] = s2[g] + y * y;
    let loss : f32 = min(y, 0.0);
    dn[g] = dn[g] + loss * loss;
    let d : f32 = f32(P.t0);
    sc[g] = 0.0;
    so[g] = 0.0;
    if (d >= 2.0) {
      let mu : f32 = s1[g] / d;
      let v : f32 = max(0.0, s2[g] / d - mu * mu);
      if (mu != 0.0) { sc[g] = mu * max(0.0, 1.0 - v / d / (mu * mu)); }
      if (dn[g] > 0.0) { so[g] = mu / sqrt(dn[g] / d); }
    }
  }
}
