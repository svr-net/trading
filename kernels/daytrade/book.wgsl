// #include header.wgsl
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
