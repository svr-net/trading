// The differential network's fused kernels in WGSL (include/strategy/diffnet_gpu.hpp compiles their buffers and runs
// their f32 reference; web/tests/anmcgpu.mjs runs them on WebGPU): statement for statement diffnet_gpu.hpp's gradRow,
// predictRow and the two references. Kept here, beside the reference they mirror, as mc_wgsl.hpp is.
//
//   kDiffnetWgsl       three entry points, a fit as one command buffer of (diffnetGrad, diffnetAdam) pairs, one per step
//                      (its record by a dynamic uniform offset), then diffnetPredict
//     diffnetGrad      one invocation per sample of the step's minibatch: the forward pass, the twin network (the input
//                      gradient by backpropagation), the loss on the value and on every derivative, and the exact second-
//                      order adjoint, written to the sample's own row of GRADS (and its loss at the row's end): no atomics
//     diffnetAdam      one invocation per weight: the rows summed in sample order, then the Adam step
//     diffnetPredict   one invocation per scenario: the value and its gradient in the original units
//   kDiffnetFusedWgsl  the whole fit and the predictions as ONE kernel, diffnetFused, in one workgroup of 256 threads
//                      (WebGPU has no barrier across workgroups): every step, thread t adds the gradient rows of samples
//                      t, t + 256, ... into its own row of WORK; storageBarrier; thread t sums the 256 rows in thread order
//                      for weights t, t + 256, ... and steps Adam; storageBarrier. A uniform (first, last, predict) lets a
//                      host split a long fit into several dispatches of the same kernel (a browser's GPU watchdog)
#pragma once

namespace anmc {

inline const char* const kDiffnetWgsl = R"WGSL(
struct Header {
  cfg : array<vec4<u32>, 8>,
  lambda : array<vec4<f32>, 8>,
  xMean : array<vec4<f32>, 8>,
  xScale : array<vec4<f32>, 8>,
};
struct Step { start : u32, m : u32, lr : f32, bc1 : f32, bc2 : f32, pad0 : u32, pad1 : u32, pad2 : u32 };

const MAXW : u32 = 32u;
const SLOTS : u32 = 160u;  // (layers + 1) x MAXW

@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<uniform> S : Step;
@group(0) @binding(2) var<storage, read_write> THETA : array<f32>;
@group(0) @binding(3) var<storage, read> X : array<f32>;
@group(0) @binding(4) var<storage, read> G : array<f32>;
@group(0) @binding(5) var<storage, read> ROWS : array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> ORDER : array<u32>;
@group(0) @binding(7) var<storage, read_write> GRADS : array<f32>;
@group(0) @binding(8) var<storage, read_write> ADAM : array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> OUT : array<f32>;

fn cfg(k : u32) -> u32 { return H.cfg[k / 4u][k % 4u]; }
fn cfgF(k : u32) -> f32 { return bitcast<f32>(H.cfg[k / 4u][k % 4u]); }
fn lambdaAt(j : u32) -> f32 { return H.lambda[j / 4u][j % 4u]; }
fn offW(l : u32) -> u32 { return cfg(7u + l); }
fn offB(l : u32) -> u32 { return cfg(11u + l); }
fn softplus(z : f32) -> f32 { if (z > 30.0) { return z; } return log(1.0 + exp(z)); }
fn sigm(z : f32) -> f32 { return 1.0 / (1.0 + exp(-z)); }

// One sample: forward, twin backpropagation, loss, second-order adjoint; its gradient row.
@compute @workgroup_size(64)
fn diffnetGrad(@builtin(global_invocation_id) gid : vec3<u32>) {
  let k = gid.x;
  if (k >= S.m) { return; }
  let d = cfg(0u);
  let L = cfg(1u);
  let W = cfg(2u);
  let P = cfg(3u);
  let stride = cfg(19u);
  let oo = cfg(16u);
  let r = ORDER[S.start + k];
  var a : array<f32, 160>;
  var z : array<f32, 160>;
  var gb : array<f32, 160>;
  var u : array<f32, 160>;
  var vz : array<f32, 160>;
  var zb : array<f32, 32>;
  var ab : array<f32, 32>;
  var abn : array<f32, 32>;
  for (var j = 0u; j < d; j++) { a[j] = X[r * d + j]; }
  for (var l = 1u; l <= L; l++) {
    let cols = select(W, d, l == 1u);
    let wo = offW(l);
    let bo = offB(l);
    for (var i = 0u; i < W; i++) {
      var s = THETA[bo + i];
      for (var j = 0u; j < cols; j++) { s += THETA[wo + i * cols + j] * a[(l - 1u) * MAXW + j]; }
      z[l * MAXW + i] = s;
      a[l * MAXW + i] = softplus(s);
    }
  }
  var yhat = THETA[oo + W];
  for (var i = 0u; i < W; i++) { yhat += THETA[oo + i] * a[L * MAXW + i]; }
  let row = ROWS[r];
  let alpha = cfgF(17u);
  let beta = cfgF(18u);
  let ey = yhat - row.x;
  var loss = alpha * row.y * ey * ey;
  let dy = 2.0 * alpha * row.y * ey;
  if (row.z > 0.5 && beta > 0.0) {
    for (var i = 0u; i < W; i++) { gb[L * MAXW + i] = THETA[oo + i]; }
    for (var l = L; l >= 1u; l--) {
      let cols = select(W, d, l == 1u);
      let wo = offW(l);
      for (var j = 0u; j < cols; j++) { gb[(l - 1u) * MAXW + j] = 0.0; }
      for (var i = 0u; i < W; i++) {
        let gzi = gb[l * MAXW + i] * sigm(z[l * MAXW + i]);
        for (var j = 0u; j < cols; j++) { gb[(l - 1u) * MAXW + j] += THETA[wo + i * cols + j] * gzi; }
      }
    }
    let dd = f32(d);
    for (var j = 0u; j < d; j++) {
      let e = gb[j] - G[r * d + j];
      let lam = lambdaAt(j);
      loss += beta * row.y * lam * e * e / dd;
      u[j] = 2.0 * beta * row.y * lam * e / dd;
    }
    for (var l = 1u; l <= L; l++) {
      let cols = select(W, d, l == 1u);
      let wo = offW(l);
      for (var i = 0u; i < W; i++) {
        var s = 0.0;
        for (var j = 0u; j < cols; j++) { s += THETA[wo + i * cols + j] * u[(l - 1u) * MAXW + j]; }
        let sg = sigm(z[l * MAXW + i]);
        u[l * MAXW + i] = s * sg;
        vz[l * MAXW + i] = s * gb[l * MAXW + i] * sg * (1.0 - sg);
      }
    }
  }
  let base = k * stride;
  for (var i = 0u; i < W; i++) {
    ab[i] = dy * THETA[oo + i];
    GRADS[base + oo + i] = u[L * MAXW + i] + dy * a[L * MAXW + i];
  }
  GRADS[base + oo + W] = dy;
  for (var l = L; l >= 1u; l--) {
    let cols = select(W, d, l == 1u);
    let wo = offW(l);
    let bo = offB(l);
    for (var i = 0u; i < W; i++) { zb[i] = ab[i] * sigm(z[l * MAXW + i]) + vz[l * MAXW + i]; }
    for (var i = 0u; i < W; i++) {
      let gzi = gb[l * MAXW + i] * sigm(z[l * MAXW + i]);
      for (var j = 0u; j < cols; j++) {
        GRADS[base + wo + i * cols + j] = zb[i] * a[(l - 1u) * MAXW + j] + gzi * u[(l - 1u) * MAXW + j];
      }
      GRADS[base + bo + i] = zb[i];
    }
    if (l > 1u) {
      for (var j = 0u; j < W; j++) { abn[j] = 0.0; }
      for (var i = 0u; i < W; i++) {
        for (var j = 0u; j < W; j++) { abn[j] += THETA[wo + i * W + j] * zb[i]; }
      }
      for (var j = 0u; j < W; j++) { ab[j] = abn[j]; }
    }
  }
  GRADS[base + P] = loss;
}

// One weight: the minibatch's gradient rows summed in sample order, then the Adam step.
@compute @workgroup_size(64)
fn diffnetAdam(@builtin(global_invocation_id) gid : vec3<u32>) {
  let p = gid.x;
  if (p >= cfg(3u)) { return; }
  let stride = cfg(19u);
  var s = 0.0;
  for (var k = 0u; k < S.m; k++) { s += GRADS[k * stride + p]; }
  let gp = s / f32(S.m);
  let b1 = cfgF(20u);
  let b2 = cfgF(21u);
  var mv = ADAM[p];
  mv.x = b1 * mv.x + (1.0 - b1) * gp;
  mv.y = b2 * mv.y + (1.0 - b2) * gp * gp;
  ADAM[p] = mv;
  THETA[p] = THETA[p] - S.lr * (mv.x / S.bc1) / (sqrt(mv.y / S.bc2) + cfgF(22u));
}

// One scenario: value and gradient in the original units.
@compute @workgroup_size(64)
fn diffnetPredict(@builtin(global_invocation_id) gid : vec3<u32>) {
  let r = gid.x;
  if (r >= cfg(7u)) { return; }
  let d = cfg(0u);
  let L = cfg(1u);
  let W = cfg(2u);
  let oo = cfg(16u);
  var a : array<f32, 160>;
  var z : array<f32, 160>;
  var gb : array<f32, 160>;
  for (var j = 0u; j < d; j++) { a[j] = (X[r * d + j] - H.xMean[j / 4u][j % 4u]) / H.xScale[j / 4u][j % 4u]; }
  for (var l = 1u; l <= L; l++) {
    let cols = select(W, d, l == 1u);
    let wo = offW(l);
    let bo = offB(l);
    for (var i = 0u; i < W; i++) {
      var s = THETA[bo + i];
      for (var j = 0u; j < cols; j++) { s += THETA[wo + i * cols + j] * a[(l - 1u) * MAXW + j]; }
      z[l * MAXW + i] = s;
      a[l * MAXW + i] = softplus(s);
    }
  }
  var yhat = THETA[oo + W];
  for (var i = 0u; i < W; i++) {
    yhat += THETA[oo + i] * a[L * MAXW + i];
    gb[L * MAXW + i] = THETA[oo + i];
  }
  for (var l = L; l >= 1u; l--) {
    let cols = select(W, d, l == 1u);
    let wo = offW(l);
    for (var j = 0u; j < cols; j++) { gb[(l - 1u) * MAXW + j] = 0.0; }
    for (var i = 0u; i < W; i++) {
      let gzi = gb[l * MAXW + i] * sigm(z[l * MAXW + i]);
      for (var j = 0u; j < cols; j++) { gb[(l - 1u) * MAXW + j] += THETA[wo + i * cols + j] * gzi; }
    }
  }
  let ys = cfgF(25u);
  let base = r * (d + 1u);
  OUT[base] = cfgF(24u) + ys * yhat;
  for (var j = 0u; j < d; j++) { OUT[base + 1u + j] = gb[j] * ys / H.xScale[j / 4u][j % 4u]; }
}
)WGSL";

inline const char* const kDiffnetFusedWgsl = R"WGSL(
struct Header {
  cfg : array<vec4<u32>, 8>,
  lambda : array<vec4<f32>, 8>,
  xMean : array<vec4<f32>, 8>,
  xScale : array<vec4<f32>, 8>,
};
struct Range { first : u32, last : u32, predict : u32, pad : u32 };

const MAXW : u32 = 32u;
const TG : u32 = 256u;

@group(0) @binding(0) var<uniform> H : Header;
@group(0) @binding(1) var<uniform> R : Range;
@group(0) @binding(2) var<storage, read_write> THETA : array<f32>;
@group(0) @binding(3) var<storage, read> XG : array<f32>;           // training x | training g | prediction x
@group(0) @binding(4) var<storage, read> ROWS : array<vec4<f32>>;   // value, weight, has derivatives, 0
@group(0) @binding(5) var<storage, read> DATAU : array<u32>;        // step records | batch order
@group(0) @binding(6) var<storage, read_write> WORK : array<f32>;   // Adam (m, v) | one gradient row per thread
@group(0) @binding(7) var<storage, read_write> OUT : array<f32>;    // predictions | loss per step

fn cfg(k : u32) -> u32 { return H.cfg[k / 4u][k % 4u]; }
fn cfgF(k : u32) -> f32 { return bitcast<f32>(H.cfg[k / 4u][k % 4u]); }
fn lambdaAt(j : u32) -> f32 { return H.lambda[j / 4u][j % 4u]; }
fn offW(l : u32) -> u32 { return cfg(7u + l); }
fn offB(l : u32) -> u32 { return cfg(11u + l); }
fn softplus(z : f32) -> f32 { if (z > 30.0) { return z; } return log(1.0 + exp(z)); }
fn sigm(z : f32) -> f32 { return 1.0 / (1.0 + exp(-z)); }

// One training row: forward, twin backpropagation, loss, second-order adjoint, added to WORK[row0 ..].
fn sampleGrad(r : u32, row0 : u32) {
  let d = cfg(0u);
  let L = cfg(1u);
  let W = cfg(2u);
  let P = cfg(3u);
  let oo = cfg(16u);
  let gOff = cfg(26u);
  var a : array<f32, 160>;
  var z : array<f32, 160>;
  var gb : array<f32, 160>;
  var u : array<f32, 160>;
  var vz : array<f32, 160>;
  var zb : array<f32, 32>;
  var ab : array<f32, 32>;
  var abn : array<f32, 32>;
  for (var j = 0u; j < d; j++) { a[j] = XG[r * d + j]; }
  for (var l = 1u; l <= L; l++) {
    let cols = select(W, d, l == 1u);
    let wo = offW(l);
    let bo = offB(l);
    for (var i = 0u; i < W; i++) {
      var s = THETA[bo + i];
      for (var j = 0u; j < cols; j++) { s += THETA[wo + i * cols + j] * a[(l - 1u) * MAXW + j]; }
      z[l * MAXW + i] = s;
      a[l * MAXW + i] = softplus(s);
    }
  }
  var yhat = THETA[oo + W];
  for (var i = 0u; i < W; i++) { yhat += THETA[oo + i] * a[L * MAXW + i]; }
  let row = ROWS[r];
  let alpha = cfgF(17u);
  let beta = cfgF(18u);
  let ey = yhat - row.x;
  var loss = alpha * row.y * ey * ey;
  let dy = 2.0 * alpha * row.y * ey;
  if (row.z > 0.5 && beta > 0.0) {
    for (var i = 0u; i < W; i++) { gb[L * MAXW + i] = THETA[oo + i]; }
    for (var l = L; l >= 1u; l--) {
      let cols = select(W, d, l == 1u);
      let wo = offW(l);
      for (var j = 0u; j < cols; j++) { gb[(l - 1u) * MAXW + j] = 0.0; }
      for (var i = 0u; i < W; i++) {
        let gzi = gb[l * MAXW + i] * sigm(z[l * MAXW + i]);
        for (var j = 0u; j < cols; j++) { gb[(l - 1u) * MAXW + j] += THETA[wo + i * cols + j] * gzi; }
      }
    }
    let dd = f32(d);
    for (var j = 0u; j < d; j++) {
      let e = gb[j] - XG[gOff + r * d + j];
      let lam = lambdaAt(j);
      loss += beta * row.y * lam * e * e / dd;
      u[j] = 2.0 * beta * row.y * lam * e / dd;
    }
    for (var l = 1u; l <= L; l++) {
      let cols = select(W, d, l == 1u);
      let wo = offW(l);
      for (var i = 0u; i < W; i++) {
        var s = 0.0;
        for (var j = 0u; j < cols; j++) { s += THETA[wo + i * cols + j] * u[(l - 1u) * MAXW + j]; }
        let sg = sigm(z[l * MAXW + i]);
        u[l * MAXW + i] = s * sg;
        vz[l * MAXW + i] = s * gb[l * MAXW + i] * sg * (1.0 - sg);
      }
    }
  }
  for (var i = 0u; i < W; i++) {
    ab[i] = dy * THETA[oo + i];
    WORK[row0 + oo + i] += u[L * MAXW + i] + dy * a[L * MAXW + i];
  }
  WORK[row0 + oo + W] += dy;
  for (var l = L; l >= 1u; l--) {
    let cols = select(W, d, l == 1u);
    let wo = offW(l);
    let bo = offB(l);
    for (var i = 0u; i < W; i++) { zb[i] = ab[i] * sigm(z[l * MAXW + i]) + vz[l * MAXW + i]; }
    for (var i = 0u; i < W; i++) {
      let gzi = gb[l * MAXW + i] * sigm(z[l * MAXW + i]);
      for (var j = 0u; j < cols; j++) {
        WORK[row0 + wo + i * cols + j] += zb[i] * a[(l - 1u) * MAXW + j] + gzi * u[(l - 1u) * MAXW + j];
      }
      WORK[row0 + bo + i] += zb[i];
    }
    if (l > 1u) {
      for (var j = 0u; j < W; j++) { abn[j] = 0.0; }
      for (var i = 0u; i < W; i++) {
        for (var j = 0u; j < W; j++) { abn[j] += THETA[wo + i * W + j] * zb[i]; }
      }
      for (var j = 0u; j < W; j++) { ab[j] = abn[j]; }
    }
  }
  WORK[row0 + P] += loss;
}

// One prediction row: value and gradient in the original units.
fn predictRow(r : u32) {
  let d = cfg(0u);
  let L = cfg(1u);
  let W = cfg(2u);
  let oo = cfg(16u);
  let pOff = cfg(27u);
  var a : array<f32, 160>;
  var z : array<f32, 160>;
  var gb : array<f32, 160>;
  for (var j = 0u; j < d; j++) { a[j] = (XG[pOff + r * d + j] - H.xMean[j / 4u][j % 4u]) / H.xScale[j / 4u][j % 4u]; }
  for (var l = 1u; l <= L; l++) {
    let cols = select(W, d, l == 1u);
    let wo = offW(l);
    let bo = offB(l);
    for (var i = 0u; i < W; i++) {
      var s = THETA[bo + i];
      for (var j = 0u; j < cols; j++) { s += THETA[wo + i * cols + j] * a[(l - 1u) * MAXW + j]; }
      z[l * MAXW + i] = s;
      a[l * MAXW + i] = softplus(s);
    }
  }
  var yhat = THETA[oo + W];
  for (var i = 0u; i < W; i++) {
    yhat += THETA[oo + i] * a[L * MAXW + i];
    gb[L * MAXW + i] = THETA[oo + i];
  }
  for (var l = L; l >= 1u; l--) {
    let cols = select(W, d, l == 1u);
    let wo = offW(l);
    for (var j = 0u; j < cols; j++) { gb[(l - 1u) * MAXW + j] = 0.0; }
    for (var i = 0u; i < W; i++) {
      let gzi = gb[l * MAXW + i] * sigm(z[l * MAXW + i]);
      for (var j = 0u; j < cols; j++) { gb[(l - 1u) * MAXW + j] += THETA[wo + i * cols + j] * gzi; }
    }
  }
  let ys = cfgF(25u);
  let base = r * (d + 1u);
  OUT[base] = cfgF(24u) + ys * yhat;
  for (var j = 0u; j < d; j++) { OUT[base + 1u + j] = gb[j] * ys / H.xScale[j / 4u][j % 4u]; }
}

// The whole fit, then the predictions, in one workgroup.
@compute @workgroup_size(256)
fn diffnetFused(@builtin(local_invocation_index) t : u32) {
  let P = cfg(3u);
  let stride = P + 1u;
  let orderOff = cfg(28u);
  let scratch = cfg(29u);
  let b1 = cfgF(20u);
  let b2 = cfgF(21u);
  let eps = cfgF(22u);
  for (var s = R.first; s < R.last; s++) {
    let rec = s * 64u;
    let start = DATAU[rec];
    let m = DATAU[rec + 1u];
    let row0 = scratch + t * stride;
    for (var p = 0u; p < stride; p++) { WORK[row0 + p] = 0.0; }
    for (var k = t; k < m; k += TG) { sampleGrad(DATAU[orderOff + start + k], row0); }
    storageBarrier();
    let lr = bitcast<f32>(DATAU[rec + 2u]);
    let bc1 = bitcast<f32>(DATAU[rec + 3u]);
    let bc2 = bitcast<f32>(DATAU[rec + 4u]);
    for (var p = t; p < P; p += TG) {
      var sum = 0.0;
      for (var q = 0u; q < TG; q++) { sum += WORK[scratch + q * stride + p]; }
      let gp = sum / f32(m);
      let mm = b1 * WORK[2u * p] + (1.0 - b1) * gp;
      let vv = b2 * WORK[2u * p + 1u] + (1.0 - b2) * gp * gp;
      WORK[2u * p] = mm;
      WORK[2u * p + 1u] = vv;
      THETA[p] = THETA[p] - lr * (mm / bc1) / (sqrt(vv / bc2) + eps);
    }
    if (t == 0u) {
      var l = 0.0;
      for (var q = 0u; q < TG; q++) { l += WORK[scratch + q * stride + P]; }
      OUT[cfg(30u) + s] = l;
    }
    storageBarrier();
  }
  if (R.predict == 1u) {
    for (var r = t; r < cfg(7u); r += TG) { predictRow(r); }
  }
}
)WGSL";

}  // namespace anmc
