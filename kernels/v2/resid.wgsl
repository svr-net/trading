// #include header.wgsl
// Each stock's residual reversal and tug of war at every close, from the past only. The residual:
// the close-to-close log return less beta times the market's (mret.wgsl), beta the exponentially
// weighted (half-life 64 days) regression of the stock's returns on the market's up to the day
// before (Blitz, Huij, Lansdorp and Verbeek 2013). The tug of war: the share of the last 20 days
// with a positive overnight return (open over the previous close) and a negative daytime one
// (close over open) (Akbas, Boehmer, Jiang and Koch 2022). bars as in features.wgsl; mk [T].
// out [T][N][3]: the residuals summed over 5 and 21 days over sigma sqrt(days), the tug of war.
@group(0) @binding(1) var<storage, read> bars : array<f32>;
@group(0) @binding(2) var<storage, read> mk : array<f32>;
@group(0) @binding(3) var<storage, read_write> out : array<f32>;

var<private> res : array<f32, 21>;
var<private> tug : array<f32, 20>;

fn bar(f : u32, t : u32, a : u32) -> f32 { return bars[(f * P.T + t) * P.N + a]; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let a : u32 = wid.x * 64u + lane;
  if (a < P.N) {
    var n : u32 = 0u;
    var pc : f32 = 0.0;
    var var16 : f32 = 0.0;
    var smm : f32 = 0.0;
    var srm : f32 = 0.0;
    for (var j = 0u; j < 21u; j = j + 1u) { res[j] = 0.0; }
    for (var j = 0u; j < 20u; j = j + 1u) { tug[j] = 0.0; }
    let l16 : f32 = pow(2.0, -1.0 / 16.0);
    let l64 : f32 = pow(2.0, -1.0 / 64.0);
    for (var t = 0u; t < P.T; t = t + 1u) {
      let row : u32 = (t * P.N + a) * 3u;
      out[row] = 0.0;
      out[row + 1u] = 0.0;
      out[row + 2u] = 0.0;
      let o : f32 = bar(0u, t, a);
      let c : f32 = bar(3u, t, a);
      if (!(o > 0.0 && c > 0.0)) { pc = 0.0; n = 0u; continue; }
      var r : f32 = 0.0;
      var g : f32 = 0.0;
      if (pc > 0.0) { r = log(c / pc); g = log(o / pc); }
      let m : f32 = mk[t];
      var beta : f32 = 0.0;
      if (smm > 0.0) { beta = srm / smm; }
      for (var j = 20u; j > 0u; j = j - 1u) { res[j] = res[j - 1u]; }
      for (var j = 19u; j > 0u; j = j - 1u) { tug[j] = tug[j - 1u]; }
      res[0] = r - beta * m;
      tug[0] = 0.0;
      if (g > 0.0 && log(c / o) < 0.0) { tug[0] = 1.0; }
      if (n == 0u) { var16 = r * r; } else { var16 = l16 * var16 + (1.0 - l16) * r * r; }
      let sig : f32 = sqrt(var16);
      if (n >= 64u && sig > 0.0) {
        var s5 : f32 = 0.0;
        var s21 : f32 = 0.0;
        var tw : f32 = 0.0;
        for (var j = 0u; j < 21u; j = j + 1u) {
          if (j < 5u) { s5 = s5 + res[j]; }
          s21 = s21 + res[j];
        }
        for (var j = 0u; j < 20u; j = j + 1u) { tw = tw + tug[j]; }
        out[row] = s5 / (sig * sqrt(5.0));
        out[row + 1u] = s21 / (sig * sqrt(21.0));
        out[row + 2u] = tw / 20.0;
      }
      // The beta's sums take today after today's residual (only the past enters it).
      if (pc > 0.0) { smm = l64 * smm + (1.0 - l64) * m * m; srm = l64 * srm + (1.0 - l64) * r * m; }
      pc = c;
      n = n + 1u;
    }
  }
}
