// #include header.wgsl
// The daily trades' best pair of every candidate: under each level source s, stop limit l and fill
// variant v (its opening-noise floor), the pair q (stop q / G, take-profit q % G) of the highest
// expected outcome whose stop is at least the floor, at most the limit and the widest stop allowed
// (P.buy), and, when P.sell > 0, at most P.sell times its take-profit. lev [S][nC][2 G]: each
// candidate's stops then take-profits (log distances); e [S][nC][G G]: expected outcomes; ok [S][nC]:
// 1 where the candidate has a fitted forecast and levels; lim [L][nC]; flo [V][nC]. P.NK = S,
// P.u1 = L, P.nS = V, P.hzn = the "none" value (+infinity). Grid ceil(V S L nC / 64); out
// [V][S][L][nC][2]: the best expected outcome (-P.hzn where none) and its pair.
@group(0) @binding(1) var<storage, read> lev : array<f32>;
@group(0) @binding(2) var<storage, read> e : array<f32>;
@group(0) @binding(3) var<storage, read> ok : array<f32>;
@group(0) @binding(4) var<storage, read> lim : array<f32>;
@group(0) @binding(5) var<storage, read> flo : array<f32>;
@group(0) @binding(6) var<storage, read_write> out : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let s : u32 = wid.x * 64u + lane;
  let nC : u32 = P.nC;
  if (s < P.nS * P.NK * P.u1 * nC) {
    let c : u32 = s % nC;
    let l : u32 = (s / nC) % P.u1;
    let s0 : u32 = (s / (nC * P.u1)) % P.NK;
    let v : u32 = s / (nC * P.u1 * P.NK);
    let G : u32 = P.G;
    let nq : u32 = G * G;
    let row : u32 = s0 * nC + c;
    var best : f32 = -P.hzn;
    var pair : u32 = 0u;
    if (ok[row] > 0.0) {
      let fl : f32 = flo[v * nC + c];
      let hi : f32 = min(lim[l * nC + c], P.buy);
      for (var q = 0u; q < nq; q = q + 1u) {
        let dist : f32 = lev[row * 2u * G + q / G];
        let take : f32 = lev[row * 2u * G + G + q % G];
        let x : f32 = e[row * nq + q];
        if (dist >= fl && dist <= hi && (!(P.sell > 0.0) || dist <= P.sell * take) && x > best) {
          best = x;
          pair = q;
        }
      }
    }
    out[s * 2u] = best;
    out[s * 2u + 1u] = f32(pair);
  }
}
