// #include header.wgsl
// The market's structure at every close (structure.hpp), one workgroup per half-life j running
// through the days in order (its covariance is recursive): the exponentially weighted covariance
// of the stocks' returns (half-life horizon(j)); from the day half its weight is real data, over the
// stocks with variance (at least 3), the effective dimension n / (sum of squared correlations / n)
// and the minimum spanning tree's mean edge (Prim; correlation distance sqrt(2 (1 - rho))).
// Each day, each stock's inverse standard deviation once (the correlations are then products), each
// pair once (the matrix is symmetric), and Prim's keys as squared distances 2 (1 - rho) (the same
// order), a square root only for each edge taken.
// tab [T][N]: each day's returns (0 without a bar today or yesterday). C [H][N][N] (upper
// triangle); work [H][4][N] (Prim's keys, in the tree, active, inverse standard deviation); out
// [3][T][H] (dimension, tree length, 1 where they are defined). P.u7 is 0. Grid H.
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<storage, read_write> work : array<f32>;
@group(0) @binding(4) var<storage, read_write> out : array<f32>;

fn rho(c0 : u32, sw : u32, a : u32, b : u32) -> f32 {
  let x : u32 = min(a, b);
  let y : u32 = max(a, b);
  return C[c0 + x * P.N + y] * work[sw + x] * work[sw + y];
}

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lane : u32) {
  let j : u32 = wid.x + lane;
  if (j < P.H) {
    let N : u32 = P.N;
    let c0 : u32 = j * N * N;
    let kw : u32 = j * 4u * N;
    let iw : u32 = kw + N;
    let aw : u32 = kw + 2u * N;
    let sw : u32 = kw + 3u * N;
    let hl : f32 = f32(horizon(j));
    let l : f32 = pow(2.0, -1.0 / hl);
    let g : f32 = 1.0 - l;
    let zero : f32 = f32(P.u7) * 0.0;
    let inf : f32 = 1.0 / zero;
    var seen : f32 = 0.0;
    for (var t = 0u; t < P.T; t = t + 1u) {
      for (var a = 0u; a < N; a = a + 1u) {
        let ra : f32 = g * tab[t * N + a];
        for (var b = a; b < N; b = b + 1u) { C[c0 + a * N + b] = l * C[c0 + a * N + b] + ra * tab[t * N + b]; }
      }
      seen = seen + 1.0;
      if (seen < hl) { continue; }
      var n : u32 = 0u;
      for (var i = 0u; i < N; i = i + 1u) {
        if (C[c0 + i * N + i] > 0.0) {
          work[aw + i] = 1.0;
          work[sw + i] = 1.0 / sqrt(C[c0 + i * N + i]);
          n = n + 1u;
        } else {
          work[aw + i] = 0.0;
        }
      }
      if (n < 3u) { continue; }
      var off : f32 = 0.0;
      for (var a = 0u; a < N; a = a + 1u) {
        if (work[aw + a] > 0.0) {
          let ia : f32 = work[sw + a];
          for (var b = a + 1u; b < N; b = b + 1u) {
            if (work[aw + b] > 0.0) {
              let p : f32 = C[c0 + a * N + b] * ia * work[sw + b];
              off = off + p * p;
            }
          }
        }
      }
      let sumRho2 : f32 = f32(n) + 2.0 * off;
      out[t * P.H + j] = f32(n) * f32(n) / sumRho2 / f32(n);
      for (var i = 0u; i < N; i = i + 1u) { work[iw + i] = 0.0; work[kw + i] = inf; }
      var start : u32 = 0u;
      loop {
        if (work[aw + start] > 0.0) { break; }
        start = start + 1u;
      }
      work[kw + start] = 0.0;
      var total : f32 = 0.0;
      for (var added = 0u; added < n; added = added + 1u) {
        var u : u32 = N;
        for (var i = 0u; i < N; i = i + 1u) {
          if (work[aw + i] > 0.0 && work[iw + i] == 0.0) {
            if (u == N) { u = i; } else if (work[kw + i] < work[kw + u]) { u = i; }
          }
        }
        work[iw + u] = 1.0;
        total = total + sqrt(work[kw + u]);
        for (var v = 0u; v < N; v = v + 1u) {
          if (work[aw + v] > 0.0 && work[iw + v] == 0.0) {
            let d2 : f32 = max(0.0, 2.0 * (1.0 - rho(c0, sw, u, v)));
            if (d2 < work[kw + v]) { work[kw + v] = d2; }
          }
        }
      }
      out[(P.T + t) * P.H + j] = total / f32(n - 1u);
      out[(2u * P.T + t) * P.H + j] = 1.0;
    }
  }
}
