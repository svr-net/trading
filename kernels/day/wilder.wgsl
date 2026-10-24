// #include header.wgsl
// J. Welles Wilder's indicators (New Concepts in Technical Trading Systems, 1978) of every stock at
// every close, with his own parameters: 14 days, smoothed his way (the first value the mean of 14,
// then avg = (13 avg + x) / 14), and the Parabolic SAR's acceleration factor from 0.02 by 0.02 up
// to 0.20. One invocation per stock runs through its days in order (the indicators are recursive);
// days without a bar are skipped and left 0. Bars: 5 tables [o, h, l, c, v][T][N], 0 where there is
// no bar. Grid ceil(N/64); out [T][N][7]:
//   0  average true range over the close (the true range counts the gap from the previous close)
//   1  RSI, (RSI - 50) / 50
//   2  directional movement, (+DI - -DI) / 100
//   3  ADX / 100 (the trend's strength, either way)
//   4  ADXR / 100, the mean of ADX now and 13 days before (Wilder's rating of a market's trend)
//   5  the Parabolic's side for the next day: 1 long, -1 short
//   6  the close's distance from the next day's SAR, in average true ranges
// Each is 0 until it has its 14 days (28 for ADX, 41 for ADXR).
@group(0) @binding(1) var<storage, read> bars : array<f32>;
@group(0) @binding(2) var<storage, read_write> ind : array<f32>;

fn bar(f : u32, t : u32, a : u32) -> f32 { return bars[(f * P.T + t) * P.N + a]; }

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let a : u32 = wid.x * 64u + lane;
  if (a < P.N) {
    let n : f32 = 14.0;
    var k : f32 = 0.0;
    var pc : f32 = 0.0;
    var ph : f32 = 0.0;
    var pl : f32 = 0.0;
    var atr : f32 = 0.0;
    var pdm : f32 = 0.0;
    var mdm : f32 = 0.0;
    var au : f32 = 0.0;
    var ad : f32 = 0.0;
    var adx : f32 = 0.0;
    var nd : f32 = 0.0;
    var side : f32 = 0.0;
    var sar : f32 = 0.0;
    var ep : f32 = 0.0;
    var af : f32 = 0.02;
    for (var t = 0u; t < P.T; t = t + 1u) {
      let row : u32 = (t * P.N + a) * 7u;
      for (var j = 0u; j < 7u; j = j + 1u) { ind[row + j] = 0.0; }
      let h : f32 = bar(1u, t, a);
      let l : f32 = bar(2u, t, a);
      let c : f32 = bar(3u, t, a);
      if (c > 0.0 && l > 0.0 && h > l) {
        if (k > 0.0) {
          let tr : f32 = max(h, pc) - min(l, pc);
          let upm : f32 = h - ph;
          let dnm : f32 = pl - l;
          var p : f32 = 0.0;
          var m : f32 = 0.0;
          if (upm > dnm && upm > 0.0) { p = upm; }
          if (dnm > upm && dnm > 0.0) { m = dnm; }
          let r : f32 = c - pc;
          var u : f32 = 0.0;
          var d : f32 = 0.0;
          if (r > 0.0) { u = r; } else { d = -r; }
          if (k <= n) {
            atr = atr + tr / n;
            pdm = pdm + p / n;
            mdm = mdm + m / n;
            au = au + u / n;
            ad = ad + d / n;
          } else {
            atr = (atr * (n - 1.0) + tr) / n;
            pdm = (pdm * (n - 1.0) + p) / n;
            mdm = (mdm * (n - 1.0) + m) / n;
            au = (au * (n - 1.0) + u) / n;
            ad = (ad * (n - 1.0) + d) / n;
          }
          if (k >= n && atr > 0.0) {
            ind[row] = atr / c;
            if (au + ad > 0.0) { ind[row + 1u] = (au - ad) / (au + ad); }
            let pdi : f32 = pdm / atr;
            let mdi : f32 = mdm / atr;
            ind[row + 2u] = pdi - mdi;
            var dx : f32 = 0.0;
            if (pdi + mdi > 0.0) { dx = abs(pdi - mdi) / (pdi + mdi); }
            nd = nd + 1.0;
            if (nd <= n) { adx = adx + dx / n; } else { adx = (adx * (n - 1.0) + dx) / n; }
            if (nd >= n) {
              ind[row + 3u] = adx;
              if (nd >= 2.0 * n - 1.0 && t >= 13u) {
                let before : f32 = ind[((t - 13u) * P.N + a) * 7u + 3u];
                if (before > 0.0) { ind[row + 4u] = 0.5 * (adx + before); }
              }
            }
          }
          // The Parabolic: the stop for the next day, from this day's bar.
          if (side == 0.0) {
            if (c >= pc) { side = 1.0; sar = min(l, pl); ep = max(h, ph); } else { side = -1.0; sar = max(h, ph); ep = min(l, pl); }
            af = 0.02;
          } else if (side > 0.0) {
            if (l <= sar) {
              side = -1.0; sar = ep; ep = l; af = 0.02;
              sar = max(sar, max(h, ph));
            } else {
              if (h > ep) { ep = h; af = min(0.2, af + 0.02); }
              sar = min(sar + af * (ep - sar), min(l, pl));
            }
          } else {
            if (h >= sar) {
              side = 1.0; sar = ep; ep = h; af = 0.02;
              sar = min(sar, min(l, pl));
            } else {
              if (l < ep) { ep = l; af = min(0.2, af + 0.02); }
              sar = max(sar + af * (ep - sar), max(h, ph));
            }
          }
          if (k >= n && atr > 0.0) {
            ind[row + 5u] = side;
            ind[row + 6u] = (c - sar) / atr;
          }
        }
        pc = c;
        ph = h;
        pl = l;
        k = k + 1.0;
      }
    }
  }
}
