#include "sat/gpu/fused_backtest.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace sat::gpu {

namespace {

std::uint32_t floatBits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  return u;
}

}  // namespace

std::string limitation(const std::vector<ModelPredictions>& models, const std::vector<StrategySpec>& strategies,
                       const std::vector<SelectorSpec>& selectors) {
  if (models.empty() || strategies.empty()) return "no models or strategies";
  if (models.size() > kMaxModels) return "more than " + std::to_string(kMaxModels) + " models";
  if (models[0].probability.assets() > kMaxAssets)
    return "more than " + std::to_string(kMaxAssets) + " stocks (the GPU stages one day of the universe in workgroup memory)";
  for (const auto& s : selectors)
    if (s.topM != 1) return "a selector holding a mix of the top " + std::to_string(s.topM) + " candidates (the GPU holds one)";
  return {};
}

FusedPlan compile(const std::vector<ModelPredictions>& models, const std::vector<StrategySpec>& strategies,
                  const Panel& nextReturns, double costBps, const std::vector<SelectorSpec>& selectors, std::size_t evalFrom) {
  if (const std::string why = limitation(models, strategies, selectors); !why.empty()) throw std::invalid_argument(why);
  auto [start, end] = commonRange(models);
  end = std::min(end, nextReturns.dates());
  FusedPlan p;
  p.numModels = models.size();
  p.numAssets = nextReturns.assets();
  p.numDays = end - start;
  p.numCandidates = models.size() * strategies.size();
  p.numSelectors = selectors.size();
  p.evalFrom = evalFrom;
  p.start = start;
  p.costBps = costBps;
  if (evalFrom >= p.numDays) throw std::invalid_argument("evaluation starts after the last out-of-sample date");
  for (const auto& s : strategies)
    if (s.holding < 1) throw std::invalid_argument("holding period must be at least 1");
  for (const auto& s : selectors) {
    if (s.lookback < 2 || s.adaptEvery < 1) throw std::invalid_argument("selector look-back must be >= 2 and step >= 1");
    if (s.switchBar != 0 || s.switchCost != 0) throw std::invalid_argument("the GPU selector kernel has no switching bar; run it on the CPU");
  }

  const std::size_t M = p.numModels, N = p.numAssets, D = p.numDays, C = p.numCandidates, S = p.numSelectors;
  const std::size_t offCand = 0, offSel = offCand + C * kCandidateStride, offRet = offSel + S * kSelectorStride,
                    offProb = offRet + D * N, offRank = offProb + M * D * N, total = offRank + M * D * N;
  p.tables.assign(total, 0.0f);
  float* t = p.tables.data();
  std::size_t c = 0;
  for (std::size_t m = 0; m < M; ++m)
    for (const auto& s : strategies) {
      float* q = t + offCand + (c++) * kCandidateStride;
      q[0] = static_cast<float>(m);
      q[1] = static_cast<float>(static_cast<int>(s.kind));
      q[2] = static_cast<float>(s.param);
      q[3] = static_cast<float>(s.holding);
    }
  for (std::size_t k = 0; k < S; ++k) {
    float* q = t + offSel + k * kSelectorStride;
    q[0] = static_cast<float>(selectors[k].lookback);
    q[1] = static_cast<float>(selectors[k].adaptEvery);
    q[2] = static_cast<float>(static_cast<int>(selectors[k].metric));
    q[3] = selectors[k].allowCash ? 1.0f : 0.0f;
    q[4] = static_cast<float>(selectors[k].minScore);
  }
  for (std::size_t d = 0; d < D; ++d)
    for (std::size_t i = 0; i < N; ++i) {
      const double r = nextReturns(start + d, i);
      t[offRet + d * N + i] = std::isfinite(r) ? static_cast<float>(r) : 0.0f;
    }
  for (std::size_t m = 0; m < M; ++m) {
    const Panel& prob = models[m].probability;
    if (prob.assets() != N) throw std::invalid_argument("models disagree on the number of stocks");
    for (std::size_t d = 0; d < D; ++d) {
      const auto rank = rankRow(prob.row(start + d), N);
      for (std::size_t i = 0; i < N; ++i) {
        t[offProb + (m * D + d) * N + i] = static_cast<float>(prob(start + d, i));
        t[offRank + (m * D + d) * N + i] = static_cast<float>(rank[i]);
      }
    }
  }
  p.header.assign(kHeaderWords, 0u);
  auto& h = p.header;
  h[0] = static_cast<std::uint32_t>(M);
  h[1] = static_cast<std::uint32_t>(N);
  h[2] = static_cast<std::uint32_t>(D);
  h[3] = static_cast<std::uint32_t>(C);
  h[4] = static_cast<std::uint32_t>(S);
  h[5] = static_cast<std::uint32_t>(evalFrom);
  h[6] = floatBits(static_cast<float>(costBps * 1e-4));
  h[7] = static_cast<std::uint32_t>(offCand);
  h[8] = static_cast<std::uint32_t>(offSel);
  h[9] = static_cast<std::uint32_t>(offRet);
  h[10] = static_cast<std::uint32_t>(offProb);
  h[11] = static_cast<std::uint32_t>(offRank);
  h[12] = static_cast<std::uint32_t>(C + S);
  h[13] = static_cast<std::uint32_t>(D - evalFrom);
  return p;
}

GridResult summarise(const FusedPlan& p, const FusedOutput& o) {
  const std::size_t E = p.numEval();
  if (o.stats.size() != p.numSeries() * kStats) throw std::invalid_argument("GPU statistics do not match the plan");
  if (o.adapt.size() < p.numSelectors * E * kAdaptStride) throw std::invalid_argument("GPU adaptive series do not match the plan");
  GridResult g;
  g.start = p.start;
  g.days = p.numDays;
  g.evalFrom = p.evalFrom;
  auto metrics = [&](std::size_t k) {
    const float* s = o.stats.data() + k * kStats;
    SeriesMoments m;
    m.n = s[0];
    m.sum = s[1];
    m.sumSq = s[2];
    m.sumDownSq = s[3];
    m.sumLog = s[4];
    m.maxDrawdown = s[5];
    m.wins = s[6];
    m.turnover = s[7];
    return fromMoments(m);
  };
  for (std::size_t c = 0; c < p.numCandidates; ++c) g.candidates.push_back(metrics(c));
  for (std::size_t s = 0; s < p.numSelectors; ++s) {
    g.selectors.push_back(metrics(p.numCandidates + s));
    std::vector<double> net(E), turnover(E);
    std::vector<int> sel(E);
    for (std::size_t d = 0; d < E; ++d) {
      const float* a = o.adapt.data() + (s * E + d) * kAdaptStride;
      net[d] = a[0];
      turnover[d] = a[1];
      sel[d] = static_cast<int>(std::lround(a[2]));
    }
    g.adaptiveNet.push_back(std::move(net));
    g.adaptiveTurnover.push_back(std::move(turnover));
    g.selection.push_back(std::move(sel));
  }
  return g;
}

}  // namespace sat::gpu
