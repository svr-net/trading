// #include header.wgsl
// The US intraday strategy's trades: every candidate, entry and (stop, take-profit) pair walked
// along the day's 5-minute path. The trade is bought at the close of bar k - 1 (k = P.ent[0][kk])
// and sold at the stop, the take-profit or the last bar's close. The levels are mult[] times the
// volatility left in the day (sig sqrt((bars - k) / bars)) below and above the entry; a multiple
// below 0 is none. ticks [h, l, c] per bar, the candidates' paths one after another; cand [first
// bar, bars, volatility] per candidate. Grid ceil(nC NK G G / 64); out [nC][NK][G G][3]: the net
// return with the stop taken first when one bar touched both, with the take-profit first, and the
// first's exit (0 close, 1 stop, 2 take-profit).
@group(0) @binding(1) var<storage, read> ticks : array<f32>;
@group(0) @binding(2) var<storage, read> cand : array<f32>;
@group(0) @binding(3) var<storage, read> mult : array<f32>;
@group(0) @binding(4) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 64u + lane;
  let nq : u32 = P.G * P.G;
  if (s < P.nC * P.NK * nq) {
    let q : u32 = s % nq;
    let kk : u32 = (s / nq) % P.NK;
    let c : u32 = s / (nq * P.NK);
    let o0 : u32 = u32(cand[c * 3u]);
    let len : u32 = u32(cand[c * 3u + 1u]);
    let k : u32 = P.ent[0][kk];
    let e : f32 = ticks[(o0 + k - 1u) * 3u + 2u];
    let rest : f32 = cand[c * 3u + 2u] * sqrt(f32(len - k) / f32(len));
    let ms : f32 = mult[q / P.G];
    let mt : f32 = mult[q % P.G];
    var stop : f32 = 0.0;
    if (ms >= 0.0) { stop = e * exp(-ms * rest); }
    var take : f32 = 0.0;
    if (mt >= 0.0) { take = e * exp(mt * rest); }
    var p1 : f32 = ticks[(o0 + len - 1u) * 3u + 2u];
    var p2 : f32 = p1;
    var x1 : f32 = 0.0;
    var d2 : bool = false;
    for (var j = k; j < len; j = j + 1u) {
      let hs : bool = ms >= 0.0 && ticks[(o0 + j) * 3u + 1u] <= stop;
      let ht : bool = mt >= 0.0 && ticks[(o0 + j) * 3u] >= take;
      if (x1 == 0.0) {
        if (hs) { p1 = stop; x1 = 1.0; } else if (ht) { p1 = take; x1 = 2.0; }
      }
      if (!d2) {
        if (ht) { p2 = take; d2 = true; } else if (hs) { p2 = stop; d2 = true; }
      }
      if (x1 > 0.0 && d2) { break; }
    }
    let net : f32 = (1.0 - P.sell) / (1.0 + P.buy);
    out[s * 3u] = p1 / e * net - 1.0;
    out[s * 3u + 1u] = p2 / e * net - 1.0;
    out[s * 3u + 2u] = x1;
  }
}
