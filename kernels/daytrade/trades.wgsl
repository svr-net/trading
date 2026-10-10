// #include header.wgsl
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
