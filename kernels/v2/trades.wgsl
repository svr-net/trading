// #include header.wgsl
// One day of the trades under every setting on the lattice, fused: for each setting (one invocation)
// its ranking's candidates are bought with its limit, sold at its stop-loss, take-profit or the close,
// and the day's result is added to its record. The fields of P: T the settings, N the candidate
// slots, K a slot's stride, NT the trades a ranking holds, t0 the limit depths, u0 the tail
// fractions, u1 the SAR weights, u2 1 to sell at the close (no levels). A setting g decodes as
// (ranking, limit, stop tail, take tail, SAR weight), the last fastest.
// cand [rankings][NT]: a ranking's candidates (slot, -1 none). st [N][K] per slot: open, high, low,
// close of the trading day, the previous close, sigma, the half-spread factor, the SAR (0 none), 1
// with 20 days of history, then the falls at the u0 tail fractions and the rises at them (logs from
// the open over the last 100 days). dv [t0] the limit depths (sigmas below the previous close). av
// [u1] the SAR weights. s1, s2, dn [T]: the records' sums of the day's results, of their squares, of
// the losses' squares.
@group(0) @binding(1) var<storage, read> cand : array<f32>;
@group(0) @binding(2) var<storage, read> st : array<f32>;
@group(0) @binding(3) var<storage, read> dv : array<f32>;
@group(0) @binding(4) var<storage, read> av : array<f32>;
@group(0) @binding(5) var<storage, read_write> s1 : array<f32>;
@group(0) @binding(6) var<storage, read_write> s2 : array<f32>;
@group(0) @binding(7) var<storage, read_write> dn : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let g : u32 = wid.x * 64u + lane;
  if (g < P.T) {
    let ca : u32 = g % P.u1;
    var r : u32 = g / P.u1;
    let ct : u32 = r % P.u0;
    r = r / P.u0;
    let cs : u32 = r % P.u0;
    r = r / P.u0;
    let ce : u32 = r % P.t0;
    let k : u32 = r / P.t0;
    let d : f32 = dv[ce];
    let al : f32 = av[ca];
    var y : f32 = 0.0;
    var n : f32 = 0.0;
    for (var j = 0u; j < P.NT; j = j + 1u) {
      let s : f32 = cand[k * P.NT + j];
      if (s < 0.0) { continue; }
      n = n + 1.0;
      let b : u32 = u32(s) * P.K;
      let o : f32 = st[b];
      let h : f32 = st[b + 1u];
      let l : f32 = st[b + 2u];
      let c : f32 = st[b + 3u];
      let pc : f32 = st[b + 4u];
      let sig : f32 = st[b + 5u];
      let half : f32 = st[b + 6u];
      let sar : f32 = st[b + 7u];
      let lim : f32 = pc * exp(-d * sig);
      var entry : f32 = 0.0;
      var intra : bool = false;
      if (o <= lim) { entry = o; } else if (l <= lim / half) { entry = lim; intra = true; } else { continue; }
      var stop : f32 = 0.0;
      var take : f32 = 1.0e30;
      if (P.u2 == 0u && st[b + 8u] > 0.5) {
        var ls : f32 = st[b + 9u + cs];
        if (sar > 0.0) { ls = (1.0 - al) * ls + al * log(min(1.0, sar / entry)); }
        stop = entry * exp(ls);
        take = entry * exp(st[b + 9u + P.u0 + ct]);
      }
      var px : f32 = c;
      if (l <= stop) { px = stop / half; } else if (!intra && h >= take * half) { px = take; }
      y = y + px / entry - 1.0;
    }
    if (n > 0.0) { y = y / n; }
    s1[g] = s1[g] + y;
    s2[g] = s2[g] + y * y;
    let loss : f32 = min(y, 0.0);
    dn[g] = dn[g] + loss * loss;
  }
}
