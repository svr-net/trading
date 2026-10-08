// CPU execution of the strategy-search kernels (fused_kernels.cpp), statement for statement
// in single precision, so the GPU algorithm is unit-tested natively.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "sat/gpu/fused_backtest.hpp"

namespace sat::gpu {

namespace {

struct Header {
  std::uint32_t nM, nN, nD, nC, nS, evalFrom;
  float cost;
  std::uint32_t offCand, offSel, offRet, offProb, offRank, nSeries, nEval;
};

Header readHeader(const FusedPlan& p) {
  if (p.header.size() != kHeaderWords) throw std::invalid_argument("plan header has the wrong size");
  Header h{};
  const auto& w = p.header;
  h.nM = w[0];
  h.nN = w[1];
  h.nD = w[2];
  h.nC = w[3];
  h.nS = w[4];
  h.evalFrom = w[5];
  std::memcpy(&h.cost, &w[6], sizeof h.cost);
  h.offCand = w[7];
  h.offSel = w[8];
  h.offRet = w[9];
  h.offProb = w[10];
  h.offRank = w[11];
  h.nSeries = w[12];
  h.nEval = w[13];
  return h;
}

// weights() of the kernels; prob/rank point at the date's row of the model.
void weights(std::uint32_t kind, float param, const float* prob, const float* rank, std::uint32_t n, float* w) {
  for (std::uint32_t i = 0; i < n; ++i) w[i] = 0.0f;
  if (kind == 0u) {
    const auto k = static_cast<std::uint32_t>(std::clamp(std::floor(param + 0.5f), 1.0f, static_cast<float>(n)));
    for (std::uint32_t i = 0; i < n; ++i)
      if (static_cast<std::uint32_t>(rank[i]) < k) w[i] = 1.0f / static_cast<float>(k);
  } else if (kind == 1u) {
    const auto k = static_cast<std::uint32_t>(std::clamp(std::floor(param + 0.5f), 1.0f, std::max(1.0f, static_cast<float>(n / 2u))));
    for (std::uint32_t i = 0; i < n; ++i) {
      const auto r = static_cast<std::uint32_t>(rank[i]);
      if (r < k) w[i] = 1.0f / static_cast<float>(k);
      else if (r >= n - k) w[i] = -1.0f / static_cast<float>(k);
    }
  } else if (kind == 2u) {
    std::uint32_t c = 0;
    for (std::uint32_t i = 0; i < n; ++i)
      if (prob[i] > param) ++c;
    for (std::uint32_t i = 0; i < n; ++i)
      if (prob[i] > param) w[i] = 1.0f / static_cast<float>(c);
  } else {
    float total = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i)
      if (prob[i] > param) total += prob[i] - param;
    if (total > 0.0f)
      for (std::uint32_t i = 0; i < n; ++i)
        if (prob[i] > param) w[i] = (prob[i] - param) / total;
  }
}

float score(std::uint32_t metric, float n, float s1, float s2, float sd) {
  if (n < 2.0f) return -1.0e30f;
  const float mean = s1 / n;
  if (metric == 0u) return mean;
  if (metric == 1u) return mean / std::sqrt(std::max(0.0f, (s2 - s1 * s1 / n) / (n - 1.0f)) + 1.0e-10f);
  return mean / std::sqrt(sd / n + 1.0e-10f);
}

}  // namespace

FusedOutput runFusedReference(const FusedPlan& plan) {
  const Header H = readHeader(plan);
  const float* T = plan.tables.data();
  const std::uint32_t n = H.nN, D = H.nD;
  std::vector<float> book(static_cast<std::size_t>(H.nC) * D * 2), pre(static_cast<std::size_t>(H.nC) * (D + 1) * 4, 0.0f);
  std::vector<float> w(kMaxAssets), nw(kMaxAssets), wN(kMaxAssets), wP(kMaxAssets);

  // 1. candidate-backtest (the workgroup staging only changes where the same values are read).
  for (std::uint32_t c = 0; c < H.nC; ++c) {
    const float* q = T + H.offCand + c * 4u;
    const auto model = static_cast<std::uint32_t>(q[0]), kind = static_cast<std::uint32_t>(q[1]);
    const float param = q[2];
    const std::uint32_t hold = std::max(1u, static_cast<std::uint32_t>(q[3]));
    std::fill(w.begin(), w.end(), 0.0f);
    float s1 = 0.0f, s2 = 0.0f, sd = 0.0f;
    for (std::uint32_t d = 0; d < D; ++d) {
      float turnover = 0.0f;
      if (d % hold == 0u) {
        const std::size_t base = (static_cast<std::size_t>(model) * D + d) * n;
        weights(kind, param, T + H.offProb + base, T + H.offRank + base, n, nw.data());
        for (std::uint32_t i = 0; i < n; ++i) {
          turnover += std::fabs(nw[i] - w[i]);
          w[i] = nw[i];
        }
      }
      float gross = 0.0f;
      for (std::uint32_t i = 0; i < n; ++i) gross += w[i] * T[H.offRet + d * n + i];
      const float r = gross - H.cost * turnover;
      book[(static_cast<std::size_t>(c) * D + d) * 2] = gross;
      book[(static_cast<std::size_t>(c) * D + d) * 2 + 1] = turnover;
      s1 += r;
      s2 += r * r;
      if (r < 0.0f) sd += r * r;
      float* p = pre.data() + (static_cast<std::size_t>(c) * (D + 1) + d + 1) * 4;
      p[0] = s1;
      p[1] = s2;
      p[2] = sd;
    }
  }

  // 2. adaptive-select.
  FusedOutput out;
  out.adapt.assign(std::max<std::size_t>(1, static_cast<std::size_t>(H.nS) * H.nEval) * kAdaptStride, 0.0f);
  auto candidateWeights = [&](std::uint32_t c, std::uint32_t day, float* wt) {
    const float* q = T + H.offCand + c * 4u;
    const std::uint32_t hold = std::max(1u, static_cast<std::uint32_t>(q[3]));
    const std::uint32_t rd = (day / hold) * hold;
    const std::size_t base = (static_cast<std::size_t>(q[0]) * D + rd) * n;
    weights(static_cast<std::uint32_t>(q[1]), q[2], T + H.offProb + base, T + H.offRank + base, n, wt);
  };
  for (std::uint32_t s = 0; s < H.nS; ++s) {
    const float* q = T + H.offSel + s * 8u;
    const auto L = static_cast<std::uint32_t>(q[0]);
    const std::uint32_t A = std::max(1u, static_cast<std::uint32_t>(q[1]));
    const auto metric = static_cast<std::uint32_t>(q[2]);
    const bool allowCash = q[3] > 0.5f;
    const float minScore = q[4];
    int sel = -1, prev = -1;
    for (std::uint32_t d = H.evalFrom; d < D; ++d) {
      if ((d - H.evalFrom) % A == 0u) {
        const std::uint32_t from = d > L ? d - L : 0u;
        const auto len = static_cast<float>(d - from);
        std::uint32_t best = 0;
        float bestScore = -3.0e38f;
        for (std::uint32_t c = 0; c < H.nC; ++c) {
          const float* p0 = pre.data() + (static_cast<std::size_t>(c) * (D + 1) + from) * 4;
          const float* p1 = pre.data() + (static_cast<std::size_t>(c) * (D + 1) + d) * 4;
          const float sc = score(metric, len, p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
          if (sc > bestScore) {
            best = c;
            bestScore = sc;
          }
        }
        sel = static_cast<int>(best);
        if (allowCash && !(bestScore > minScore)) sel = -1;
      }
      float gross = 0.0f, turnover = 0.0f;
      if (sel >= 0) gross = book[(static_cast<std::size_t>(sel) * D + d) * 2];
      if (d > H.evalFrom && sel == prev) {
        if (sel >= 0) turnover = book[(static_cast<std::size_t>(sel) * D + d) * 2 + 1];
      } else {
        std::fill(wN.begin(), wN.begin() + n, 0.0f);
        std::fill(wP.begin(), wP.begin() + n, 0.0f);
        if (sel >= 0) candidateWeights(static_cast<std::uint32_t>(sel), d, wN.data());
        if (d > H.evalFrom && prev >= 0) candidateWeights(static_cast<std::uint32_t>(prev), d - 1u, wP.data());
        for (std::uint32_t i = 0; i < n; ++i) turnover += std::fabs(wN[i] - wP[i]);
      }
      float* a = out.adapt.data() + (static_cast<std::size_t>(s) * H.nEval + d - H.evalFrom) * 4;
      a[0] = gross - H.cost * turnover;
      a[1] = turnover;
      a[2] = static_cast<float>(sel);
      a[3] = gross;
      prev = sel;
    }
  }

  // 3. series-summary.
  out.stats.assign(static_cast<std::size_t>(H.nSeries) * kStats, 0.0f);
  for (std::uint32_t k = 0; k < H.nSeries; ++k) {
    float cnt = 0, s1 = 0, s2 = 0, sd = 0, logW = 0, peak = 0, mdd = 0, wins = 0, turn = 0;
    for (std::uint32_t d = H.evalFrom; d < D; ++d) {
      float r, t;
      if (k < H.nC) {
        const float* b = book.data() + (static_cast<std::size_t>(k) * D + d) * 2;
        r = b[0] - H.cost * b[1];
        t = b[1];
      } else {
        const float* a = out.adapt.data() + (static_cast<std::size_t>(k - H.nC) * H.nEval + d - H.evalFrom) * 4;
        r = a[0];
        t = a[1];
      }
      cnt += 1.0f;
      s1 += r;
      s2 += r * r;
      if (r < 0.0f) sd += r * r;
      if (r > 0.0f) wins += 1.0f;
      turn += t;
      logW += std::log(std::max(1.0f + r, 1.0e-12f));
      peak = std::max(peak, logW);
      mdd = std::max(mdd, 1.0f - std::exp(logW - peak));
    }
    float* o = out.stats.data() + static_cast<std::size_t>(k) * kStats;
    o[0] = cnt;
    o[1] = s1;
    o[2] = s2;
    o[3] = sd;
    o[4] = logW;
    o[5] = mdd;
    o[6] = wins;
    o[7] = turn;
  }
  return out;
}

}  // namespace sat::gpu
