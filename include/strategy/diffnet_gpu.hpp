// The differential network's fit as fused GPU kernels (WGSL in diffnet_wgsl.hpp), with their reference here.
//
//   plan       a DifferentialProblem (differential_proxy.hpp: the standardized rows, the initial weights, the batch order
//              and the learning rate of every step, exactly what DifferentialProxy::train runs) laid out as flat f32/u32
//              buffers: compileDifferentialFit (two kernels), compileDifferentialFused (one kernel),
//              compileDifferentialPredict
//   reference  the kernels in C++ in single precision, statement for statement the shader's and in its order (WebGPU has
//              no doubles): runDifferentialFitReference, runDifferentialGradReference, runDifferentialFusedReference,
//              runDifferentialPredictReference. ofm_anmc --kernels checks them against DifferentialProxy's double
//              precision; web/tests/anmcgpu.mjs runs the WGSL on WebGPU against them
//   offload    a back end that can run the fused kernel sets diffnetOffload(); FusedDifferentialProxy then hands its fit
//              over, and runs the reference itself whenever the offload declines, so results never depend on whether a
//              GPU was there, only on its single precision
//
// Header (kDiffHeaderWords 32-bit words, u32 unless marked f32):
//   0 inputs  1 layers  2 width  3 parameters  4 rows  5 batch  6 steps  7 prediction rows
//   8..11 offset of W_l (l = 1..4)  12..15 offset of b_l  16 output offset  17 alpha (f32)  18 beta = 1 - alpha (f32)
//   19 gradient row stride (parameters + 1)  20 Adam beta1 (f32)  21 beta2 (f32)  22 epsilon (f32)  24 y mean (f32)
//   25 y scale (f32)  26..30 the fused kernel's offsets: derivative labels in XG, prediction states in XG, the batch order
//   in DATAU, the gradient rows in WORK, the step losses in OUT  32..63 lambda per input (f32)  64..95 state means (f32)
//   96..127 state scales (f32)
// Step record (kDiffStepWords words, 5 used): start in the batch order, batch size m, learning rate (f32), Adam's bias
// corrections 1 - beta1^t and 1 - beta2^t (f32).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "strategy/differential_proxy.hpp"
#include "strategy/diffnet_wgsl.hpp"

namespace anmc {

constexpr std::size_t kDiffMaxInputs = 32, kDiffMaxWidth = 32, kDiffMaxLayers = 4;  // the WGSL's fixed arrays
constexpr std::size_t kDiffHeaderWords = 128;
constexpr std::size_t kDiffStepWords = 64;      // a step record: 256 bytes, the dynamic uniform offset alignment
constexpr std::size_t kDiffFusedThreads = 256;  // diffnetFused's one workgroup

// One fit for diffnetGrad + diffnetAdam.
struct DiffnetFitPlan {
  std::size_t inputs = 0, layers = 0, width = 0, parameters = 0, rows = 0, batch = 0, steps = 0;
  std::vector<std::uint32_t> header;     // kDiffHeaderWords
  std::vector<std::uint32_t> stepTable;  // steps x kDiffStepWords
  std::vector<float> theta;              // initial weights
  std::vector<float> x, g;               // rows x inputs: standardized states and derivative labels
  std::vector<float> rowData;            // rows x 4: standardized value, weight, has derivatives, 0
  std::vector<std::uint32_t> order;      // epochs x rows
  std::size_t gradientFloats() const { return batch * (parameters + 1); }  // GRADS: a row a sample, its loss last
};

// diffnetPredict: the value and its gradient at every scenario.
struct DiffnetPredictPlan {
  std::size_t inputs = 0, rows = 0, parameters = 0;
  std::vector<std::uint32_t> header;
  std::vector<float> theta;
  std::vector<float> x;  // rows x inputs, raw states
  std::size_t outputFloats() const { return rows * (inputs + 1); }  // per row: the value, then the gradient
};

// The whole fit (and predictions) for diffnetFused.
struct DiffnetFusedPlan {
  std::size_t inputs = 0, parameters = 0, rows = 0, batch = 0, steps = 0, predictRows = 0;
  std::vector<std::uint32_t> header;  // kDiffHeaderWords, with the offsets 26..30
  std::vector<float> theta;           // initial weights
  std::vector<float> xg;              // training states | training derivative labels | prediction states
  std::vector<float> rowData;         // rows x 4
  std::vector<std::uint32_t> datau;   // step records | batch order
  std::size_t workFloats() const { return 2 * parameters + kDiffFusedThreads * (parameters + 1); }  // Adam (m, v) | 256 rows
  std::size_t outFloats() const { return predictRows * (inputs + 1) + steps; }                       // predictions | step losses
};
struct DiffnetFusedResult {
  std::vector<float> theta;        // trained weights
  std::vector<float> predictions;  // predictRows x (inputs + 1)
  std::vector<float> stepLoss;     // each step's summed loss
};

// A back end's runner of the fused kernel: fills the result; false: not run (the reference runs it).
using DiffnetOffload = bool (*)(const DiffnetFusedPlan&, DiffnetFusedResult&);
inline DiffnetOffload& diffnetOffload() {
  static DiffnetOffload f = nullptr;
  return f;
}

namespace diffnet {

inline std::uint32_t bits(float f) {
  std::uint32_t u = 0;
  std::memcpy(&u, &f, sizeof u);
  return u;
}
inline float asFloat(std::uint32_t u) {
  float f = 0;
  std::memcpy(&f, &u, sizeof f);
  return f;
}
inline std::vector<float> toFloat(const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); }

inline void checkArchitecture(const DifferentialProblem& p) {
  if (p.inputs < 1 || p.inputs > kDiffMaxInputs) throw std::invalid_argument("diffnet kernels: 1 to 32 inputs");
  if (p.width < 1 || p.width > kDiffMaxWidth) throw std::invalid_argument("diffnet kernels: width 1 to 32");
  if (p.hiddenLayers < 1 || p.hiddenLayers > kDiffMaxLayers) throw std::invalid_argument("diffnet kernels: 1 to 4 hidden layers");
  if (p.rows < 1 || p.batch < 1) throw std::invalid_argument("diffnet kernels: no training rows");
}

inline std::vector<std::uint32_t> header(const DifferentialProblem& p) {
  std::vector<std::uint32_t> h(kDiffHeaderWords, 0u);
  h[0] = std::uint32_t(p.inputs), h[1] = std::uint32_t(p.hiddenLayers), h[2] = std::uint32_t(p.width);
  h[3] = std::uint32_t(p.parameters()), h[4] = std::uint32_t(p.rows), h[5] = std::uint32_t(p.batch);
  for (std::size_t l = 1; l <= p.hiddenLayers; ++l) h[7 + l] = std::uint32_t(p.wOffset[l]), h[11 + l] = std::uint32_t(p.bOffset[l]);
  h[16] = std::uint32_t(p.outOffset);
  h[17] = bits(float(p.alpha)), h[18] = bits(float(1 - p.alpha)), h[19] = std::uint32_t(p.parameters() + 1);
  h[20] = bits(float(kAdamBeta1)), h[21] = bits(float(kAdamBeta2)), h[22] = bits(float(kAdamEpsilon));
  h[24] = bits(float(p.yMean)), h[25] = bits(float(p.yScale));
  for (std::size_t j = 0; j < p.inputs; ++j) {
    h[32 + j] = bits(float(p.lambda[j]));
    h[64 + j] = bits(float(p.xMean[j]));
    h[96 + j] = bits(float(p.xScale[j]));
  }
  return h;
}

// The kernels' view of the header.
struct Net32 {
  std::uint32_t d, L, W, P, stride, oo, wo[5] = {}, bo[5] = {};
  float alpha, beta, b1, b2, eps, yMean, yScale;
  const std::uint32_t* h;
  explicit Net32(const std::vector<std::uint32_t>& hdr) : h(hdr.data()) {
    d = h[0], L = h[1], W = h[2], P = h[3], stride = h[19], oo = h[16];
    for (std::uint32_t l = 1; l <= 4; ++l) wo[l] = h[7 + l], bo[l] = h[11 + l];
    alpha = asFloat(h[17]), beta = asFloat(h[18]), b1 = asFloat(h[20]), b2 = asFloat(h[21]), eps = asFloat(h[22]);
    yMean = asFloat(h[24]), yScale = asFloat(h[25]);
  }
  float lambda(std::uint32_t j) const { return asFloat(h[32 + j]); }
  float xMean(std::uint32_t j) const { return asFloat(h[64 + j]); }
  float xScale(std::uint32_t j) const { return asFloat(h[96 + j]); }
};

constexpr std::uint32_t kMaxW = 32, kSlots = 160;  // the shader's MAXW and its (layers + 1) x MAXW arrays
inline float softplusF(float z) { return z > 30.0f ? z : std::log(1.0f + std::exp(z)); }
inline float sigmF(float z) { return 1.0f / (1.0f + std::exp(-z)); }

// diffnetGrad's (and diffnetFused's sampleGrad's) work for training row r: its gradient row into out[0 .. P - 1] and its
// loss into out[P].
inline void gradRow(const Net32& n, const float* xs, const float* gs, const float* rowData, const float* theta, std::uint32_t r, float* out) {
  const std::uint32_t d = n.d, L = n.L, W = n.W;
  float a[kSlots] = {}, z[kSlots] = {}, gb[kSlots] = {}, u[kSlots] = {}, vz[kSlots] = {}, zb[kMaxW] = {}, ab[kMaxW] = {}, abn[kMaxW] = {};
  for (std::uint32_t j = 0; j < d; ++j) a[j] = xs[r * d + j];
  for (std::uint32_t l = 1; l <= L; ++l) {
    const std::uint32_t cols = l == 1 ? d : W;
    for (std::uint32_t i = 0; i < W; ++i) {
      float s = theta[n.bo[l] + i];
      for (std::uint32_t j = 0; j < cols; ++j) s += theta[n.wo[l] + i * cols + j] * a[(l - 1) * kMaxW + j];
      z[l * kMaxW + i] = s;
      a[l * kMaxW + i] = softplusF(s);
    }
  }
  float yhat = theta[n.oo + W];
  for (std::uint32_t i = 0; i < W; ++i) yhat += theta[n.oo + i] * a[L * kMaxW + i];
  const float* row = rowData + 4 * r;
  const float ey = yhat - row[0];
  float loss = n.alpha * row[1] * ey * ey;
  const float dy = 2.0f * n.alpha * row[1] * ey;
  if (row[2] > 0.5f && n.beta > 0.0f) {
    for (std::uint32_t i = 0; i < W; ++i) gb[L * kMaxW + i] = theta[n.oo + i];
    for (std::uint32_t l = L; l >= 1; --l) {
      const std::uint32_t cols = l == 1 ? d : W;
      for (std::uint32_t j = 0; j < cols; ++j) gb[(l - 1) * kMaxW + j] = 0.0f;
      for (std::uint32_t i = 0; i < W; ++i) {
        const float gzi = gb[l * kMaxW + i] * sigmF(z[l * kMaxW + i]);
        for (std::uint32_t j = 0; j < cols; ++j) gb[(l - 1) * kMaxW + j] += theta[n.wo[l] + i * cols + j] * gzi;
      }
    }
    const float dd = float(d);
    for (std::uint32_t j = 0; j < d; ++j) {
      const float e = gb[j] - gs[r * d + j];
      const float lam = n.lambda(j);
      loss += n.beta * row[1] * lam * e * e / dd;
      u[j] = 2.0f * n.beta * row[1] * lam * e / dd;
    }
    for (std::uint32_t l = 1; l <= L; ++l) {
      const std::uint32_t cols = l == 1 ? d : W;
      for (std::uint32_t i = 0; i < W; ++i) {
        float s = 0.0f;
        for (std::uint32_t j = 0; j < cols; ++j) s += theta[n.wo[l] + i * cols + j] * u[(l - 1) * kMaxW + j];
        const float sg = sigmF(z[l * kMaxW + i]);
        u[l * kMaxW + i] = s * sg;
        vz[l * kMaxW + i] = s * gb[l * kMaxW + i] * sg * (1.0f - sg);
      }
    }
  }
  for (std::uint32_t i = 0; i < W; ++i) {
    ab[i] = dy * theta[n.oo + i];
    out[n.oo + i] = u[L * kMaxW + i] + dy * a[L * kMaxW + i];
  }
  out[n.oo + W] = dy;
  for (std::uint32_t l = L; l >= 1; --l) {
    const std::uint32_t cols = l == 1 ? d : W;
    for (std::uint32_t i = 0; i < W; ++i) zb[i] = ab[i] * sigmF(z[l * kMaxW + i]) + vz[l * kMaxW + i];
    for (std::uint32_t i = 0; i < W; ++i) {
      const float gzi = gb[l * kMaxW + i] * sigmF(z[l * kMaxW + i]);
      for (std::uint32_t j = 0; j < cols; ++j) out[n.wo[l] + i * cols + j] = zb[i] * a[(l - 1) * kMaxW + j] + gzi * u[(l - 1) * kMaxW + j];
      out[n.bo[l] + i] = zb[i];
    }
    if (l > 1) {
      for (std::uint32_t j = 0; j < W; ++j) abn[j] = 0.0f;
      for (std::uint32_t i = 0; i < W; ++i)
        for (std::uint32_t j = 0; j < W; ++j) abn[j] += theta[n.wo[l] + i * W + j] * zb[i];
      for (std::uint32_t j = 0; j < W; ++j) ab[j] = abn[j];
    }
  }
  out[n.P] = loss;
}

// diffnetPredict's (and diffnetFused's predictRow's) work for one scenario: the value and its gradient into o[0 .. d].
inline void predictRow(const Net32& n, const float* theta, const float* xRow, float* o) {
  const std::uint32_t d = n.d, L = n.L, W = n.W;
  float a[kSlots] = {}, z[kSlots] = {}, gb[kSlots] = {};
  for (std::uint32_t j = 0; j < d; ++j) a[j] = (xRow[j] - n.xMean(j)) / n.xScale(j);
  for (std::uint32_t l = 1; l <= L; ++l) {
    const std::uint32_t cols = l == 1 ? d : W;
    for (std::uint32_t i = 0; i < W; ++i) {
      float s = theta[n.bo[l] + i];
      for (std::uint32_t j = 0; j < cols; ++j) s += theta[n.wo[l] + i * cols + j] * a[(l - 1) * kMaxW + j];
      z[l * kMaxW + i] = s;
      a[l * kMaxW + i] = softplusF(s);
    }
  }
  float yhat = theta[n.oo + W];
  for (std::uint32_t i = 0; i < W; ++i) {
    yhat += theta[n.oo + i] * a[L * kMaxW + i];
    gb[L * kMaxW + i] = theta[n.oo + i];
  }
  for (std::uint32_t l = L; l >= 1; --l) {
    const std::uint32_t cols = l == 1 ? d : W;
    for (std::uint32_t j = 0; j < cols; ++j) gb[(l - 1) * kMaxW + j] = 0.0f;
    for (std::uint32_t i = 0; i < W; ++i) {
      const float gzi = gb[l * kMaxW + i] * sigmF(z[l * kMaxW + i]);
      for (std::uint32_t j = 0; j < cols; ++j) gb[(l - 1) * kMaxW + j] += theta[n.wo[l] + i * cols + j] * gzi;
    }
  }
  o[0] = n.yMean + n.yScale * yhat;
  for (std::uint32_t j = 0; j < d; ++j) o[1 + j] = gb[j] * n.yScale / n.xScale(j);
}

// Adam on weights [lo, hi) from the step's summed gradient `sum(q)` over `count` samples (the shader's arithmetic).
template <class Sum>
void adamStep(const Net32& n, std::size_t lo, std::size_t hi, Sum&& sum, float count, float lr, float bc1, float bc2, float* m, float* v, float* theta) {
  for (std::size_t q = lo; q < hi; ++q) {
    const float gp = sum(q) / count;
    m[q] = n.b1 * m[q] + (1.0f - n.b1) * gp;
    v[q] = n.b2 * v[q] + (1.0f - n.b2) * gp * gp;
    theta[q] = theta[q] - lr * (m[q] / bc1) / (std::sqrt(v[q] / bc2) + n.eps);
  }
}

}  // namespace diffnet

inline DiffnetFitPlan compileDifferentialFit(const DifferentialProblem& p) {
  diffnet::checkArchitecture(p);
  DiffnetFitPlan plan;
  plan.inputs = p.inputs, plan.layers = p.hiddenLayers, plan.width = p.width, plan.parameters = p.parameters(), plan.rows = p.rows, plan.batch = p.batch;
  plan.header = diffnet::header(p);
  plan.theta = diffnet::toFloat(p.theta);
  plan.x = diffnet::toFloat(p.x);
  plan.g = diffnet::toFloat(p.g);
  plan.rowData.assign(4 * p.rows, 0.0f);
  for (std::size_t r = 0; r < p.rows; ++r) {
    plan.rowData[4 * r] = float(p.y[r]);
    plan.rowData[4 * r + 1] = float(p.w[r]);
    plan.rowData[4 * r + 2] = p.hasGradient[r] ? 1.0f : 0.0f;
  }
  plan.order = p.order;
  const std::size_t per = p.stepsPerEpoch();
  plan.steps = p.steps();
  plan.header[6] = std::uint32_t(plan.steps);
  plan.stepTable.assign(plan.steps * kDiffStepWords, 0u);
  for (std::size_t s = 0; s < plan.steps; ++s) {
    std::uint32_t* rec = plan.stepTable.data() + s * kDiffStepWords;
    const std::size_t begin = (s % per) * p.batch;
    rec[0] = std::uint32_t((s / per) * p.rows + begin);
    rec[1] = std::uint32_t(std::min(p.batch, p.rows - begin));
    rec[2] = diffnet::bits(float(p.learningRate[s]));
    const double t = double(s + 1);
    rec[3] = diffnet::bits(float(1 - std::pow(diffnet::kAdamBeta1, t)));
    rec[4] = diffnet::bits(float(1 - std::pow(diffnet::kAdamBeta2, t)));
  }
  return plan;
}

// The network of problem p at weights theta, at the raw states `states` (rows x inputs).
inline DiffnetPredictPlan compileDifferentialPredict(const DifferentialProblem& p, const std::vector<double>& theta, const std::vector<double>& states) {
  diffnet::checkArchitecture(p);
  if (theta.size() != p.parameters()) throw std::invalid_argument("diffnet kernels: wrong number of weights");
  if (states.size() % p.inputs) throw std::invalid_argument("diffnet kernels: states are not rows of the network's inputs");
  DiffnetPredictPlan plan;
  plan.inputs = p.inputs, plan.rows = states.size() / p.inputs, plan.parameters = p.parameters();
  plan.header = diffnet::header(p);
  plan.header[7] = std::uint32_t(plan.rows);
  plan.theta = diffnet::toFloat(theta);
  plan.x = diffnet::toFloat(states);
  return plan;
}

// One step of diffnetGrad: the per-sample gradient rows of step `step` at weights theta.
inline std::vector<float> runDifferentialGradReference(const DiffnetFitPlan& plan, const std::vector<float>& theta, std::size_t step) {
  const diffnet::Net32 n(plan.header);
  const std::uint32_t* rec = plan.stepTable.data() + step * kDiffStepWords;
  std::vector<float> grads(plan.gradientFloats(), 0.0f);
  detail::parallelFor(rec[1], -1, [&](std::size_t k) {
    diffnet::gradRow(n, plan.x.data(), plan.g.data(), plan.rowData.data(), theta.data(), plan.order[rec[0] + k], grads.data() + k * n.stride);
  });
  return grads;
}

// The (diffnetGrad, diffnetAdam) pairs of every step: the trained weights; each step's summed loss into stepLoss.
inline std::vector<float> runDifferentialFitReference(const DiffnetFitPlan& plan, std::vector<float>* stepLoss = nullptr) {
  const diffnet::Net32 n(plan.header);
  std::vector<float> theta = plan.theta, grads(plan.gradientFloats(), 0.0f), m(plan.parameters, 0.0f), v(plan.parameters, 0.0f);
  if (stepLoss) stepLoss->clear();
  for (std::size_t s = 0; s < plan.steps; ++s) {
    const std::uint32_t* rec = plan.stepTable.data() + s * kDiffStepWords;
    const std::uint32_t start = rec[0], count = rec[1];
    detail::parallelFor(count, -1, [&](std::size_t k) {
      diffnet::gradRow(n, plan.x.data(), plan.g.data(), plan.rowData.data(), theta.data(), plan.order[start + k], grads.data() + k * n.stride);
    });
    if (stepLoss) {
      float l = 0.0f;
      for (std::uint32_t k = 0; k < count; ++k) l += grads[std::size_t(k) * n.stride + n.P];
      stepLoss->push_back(l);
    }
    detail::parallelChunks(plan.parameters, -1, [&](std::size_t lo, std::size_t hi) {
      diffnet::adamStep(n, lo, hi, [&](std::size_t q) {
        float sum = 0.0f;
        for (std::uint32_t k = 0; k < count; ++k) sum += grads[std::size_t(k) * n.stride + q];
        return sum;
      }, float(count), diffnet::asFloat(rec[2]), diffnet::asFloat(rec[3]), diffnet::asFloat(rec[4]), m.data(), v.data(), theta.data());
    });
  }
  return theta;
}

inline std::vector<float> runDifferentialPredictReference(const DiffnetPredictPlan& plan) {
  const diffnet::Net32 n(plan.header);
  std::vector<float> out(plan.outputFloats(), 0.0f);
  detail::parallelChunks(plan.rows, -1, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t r = lo; r < hi; ++r) diffnet::predictRow(n, plan.theta.data(), plan.x.data() + r * n.d, out.data() + r * (n.d + 1));
  });
  return out;
}

// The fit for diffnetFused, with predictions at the raw states `predictStates` (rows x inputs; may be empty).
inline DiffnetFusedPlan compileDifferentialFused(const DifferentialProblem& p, const std::vector<double>& predictStates) {
  const DiffnetFitPlan fit = compileDifferentialFit(p);
  if (predictStates.size() % p.inputs) throw std::invalid_argument("diffnet kernels: prediction states are not rows of the network's inputs");
  DiffnetFusedPlan plan;
  plan.inputs = fit.inputs, plan.parameters = fit.parameters, plan.rows = fit.rows, plan.batch = fit.batch, plan.steps = fit.steps;
  plan.predictRows = predictStates.size() / p.inputs;
  plan.theta = fit.theta;
  plan.rowData = fit.rowData;
  plan.xg = fit.x;
  plan.xg.insert(plan.xg.end(), fit.g.begin(), fit.g.end());
  for (double x : predictStates) plan.xg.push_back(float(x));
  plan.datau = fit.stepTable;
  plan.datau.insert(plan.datau.end(), fit.order.begin(), fit.order.end());
  plan.header = fit.header;
  plan.header[7] = std::uint32_t(plan.predictRows);
  plan.header[26] = std::uint32_t(fit.rows * fit.inputs);
  plan.header[27] = std::uint32_t(2 * fit.rows * fit.inputs);
  plan.header[28] = std::uint32_t(fit.steps * kDiffStepWords);
  plan.header[29] = std::uint32_t(2 * fit.parameters);
  plan.header[30] = std::uint32_t(plan.predictRows * (fit.inputs + 1));
  return plan;
}

// diffnetFused in C++, thread by thread in the kernel's order: with a batch of at most 256 the same sums as the two
// kernels', so the same weights, bit for bit.
inline DiffnetFusedResult runDifferentialFusedReference(const DiffnetFusedPlan& plan) {
  const diffnet::Net32 n(plan.header);
  const std::size_t P = plan.parameters, stride = P + 1, T = kDiffFusedThreads;
  const float* xs = plan.xg.data();
  const float* gs = plan.xg.data() + plan.header[26];
  const std::uint32_t* order = plan.datau.data() + plan.header[28];
  DiffnetFusedResult res;
  res.theta = plan.theta;
  std::vector<float> m(P, 0.0f), v(P, 0.0f), work(T * stride, 0.0f);
  for (std::size_t s = 0; s < plan.steps; ++s) {
    const std::uint32_t* rec = plan.datau.data() + s * kDiffStepWords;
    const std::uint32_t start = rec[0], count = rec[1];
    // phase 1: thread t adds the gradient rows of samples t, t + T, ... into its own row
    detail::parallelFor(T, -1, [&](std::size_t t) {
      thread_local std::vector<float> tmp;
      tmp.resize(stride);
      float* row = work.data() + t * stride;
      std::fill(row, row + stride, 0.0f);
      for (std::size_t k = t; k < count; k += T) {
        diffnet::gradRow(n, xs, gs, plan.rowData.data(), res.theta.data(), order[start + k], tmp.data());
        for (std::size_t q = 0; q < stride; ++q) row[q] += tmp[q];
      }
    });
    // phase 2: each weight sums the T rows in thread order, then Adam; thread 0 the loss
    detail::parallelChunks(P, -1, [&](std::size_t lo, std::size_t hi) {
      diffnet::adamStep(n, lo, hi, [&](std::size_t q) {
        float sum = 0.0f;
        for (std::size_t t = 0; t < T; ++t) sum += work[t * stride + q];
        return sum;
      }, float(count), diffnet::asFloat(rec[2]), diffnet::asFloat(rec[3]), diffnet::asFloat(rec[4]), m.data(), v.data(), res.theta.data());
    });
    float loss = 0.0f;
    for (std::size_t t = 0; t < T; ++t) loss += work[t * stride + P];
    res.stepLoss.push_back(loss);
  }
  res.predictions.assign(plan.predictRows * (n.d + 1), 0.0f);
  const float* px = plan.xg.data() + plan.header[27];
  detail::parallelChunks(plan.predictRows, -1, [&](std::size_t lo, std::size_t hi) {
    for (std::size_t r = lo; r < hi; ++r) diffnet::predictRow(n, res.theta.data(), px + r * n.d, res.predictions.data() + r * (n.d + 1));
  });
  return res;
}

// The differential proxy trained by the fused kernel: its fit compiled into a DiffnetFusedPlan and run by the offload
// (a GPU) or the kernel's reference, its weights adopted; predictions in double precision from those weights.
class FusedDifferentialProxy : public Proxy {
 public:
  explicit FusedDifferentialProxy(DifferentialOptions o = DifferentialOptions()) : net_(o) {}
  void fit(const OuterScenarios& o, const std::vector<std::size_t>& rows) override {
    const DifferentialProblem p = net_.problem(o, rows);
    const DiffnetFusedPlan plan = compileDifferentialFused(p, {});
    DiffnetFusedResult r;
    gpu_ = diffnetOffload() && diffnetOffload()(plan, r);
    if (!gpu_) r = runDifferentialFusedReference(plan);
    stepLoss_ = r.stepLoss;
    net_.adopt(p, std::vector<double>(r.theta.begin(), r.theta.end()));
  }
  void predict(const double* x, double& mu, double& v) const override { net_.predict(x, mu, v); }
  bool outsideTraining(const double* x) const override { return net_.outsideTraining(x); }
  std::string describe() const override { return "fused-kernel " + net_.describe() + (gpu_ ? " (GPU)" : " (f32 reference)"); }
  const DifferentialProxy& network() const { return net_; }
  const std::vector<float>& stepLoss() const { return stepLoss_; }

 private:
  DifferentialProxy net_;
  std::vector<float> stepLoss_;
  bool gpu_ = false;
};

inline ProxyFactory fusedDifferentialProxy(DifferentialOptions o = DifferentialOptions()) {
  return [o](std::size_t) { return std::make_shared<FusedDifferentialProxy>(o); };
}

}  // namespace anmc
