#include "sat/algo/ensemble.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "sat/algo/pairs.hpp"
#include "sat/core/stats.hpp"

namespace sat::algo {

PairsBookResult pairsBook(const Panel& close, const PairsBookSpec& s) {
  const std::size_t T = close.dates(), N = close.assets();
  if (N < 2) throw std::invalid_argument("pairs book: at least two instruments");
  if (s.scanWindow < 60 || s.rescanEvery < 1 || s.pairs < 1) throw std::invalid_argument("pairs book: scan window >= 60, rescan >= 1, pairs >= 1");
  if (T <= s.scanWindow + 2) throw std::invalid_argument("pairs book: series shorter than the scan window");
  PairsSpec ps;
  ps.delta = s.delta;
  ps.observationVariance = s.observationVariance;
  ps.entryZ = s.entryZ;
  ps.exitZ = s.exitZ;
  ps.costBps = s.costBps;
  std::vector<std::vector<double>> px(N);
  for (std::size_t i = 0; i < N; ++i) px[i] = close.series(i);
  PairsBookResult out;
  out.start = s.scanWindow;
  for (std::size_t t = s.scanWindow; t + 1 < T; t += s.rescanEvery) {
    const std::size_t from = t - s.scanWindow, until = std::min(T, t + s.rescanEvery + 1);
    const std::size_t days = until - t - 1;  // returns earned from t .. until - 2
    // Scan on the past window only.
    struct Candidate {
      double adf;
      std::size_t a, b;
    };
    std::vector<Candidate> found;
    for (std::size_t a = 0; a < N; ++a)
      for (std::size_t b = a + 1; b < N; ++b) {
        const std::vector<double> ya(px[a].begin() + static_cast<std::ptrdiff_t>(from), px[a].begin() + static_cast<std::ptrdiff_t>(t)),
            xb(px[b].begin() + static_cast<std::ptrdiff_t>(from), px[b].begin() + static_cast<std::ptrdiff_t>(t));
        const auto c = engleGranger(ya, xb);
        if (c.cointegrated() && c.hedgeRatio > 0) found.push_back({c.adf, a, b});
      }
    std::sort(found.begin(), found.end(), [](const Candidate& l, const Candidate& r) { return l.adf < r.adf; });
    found.resize(std::min(found.size(), s.pairs));
    ++out.scans;
    std::vector<double> book(days, 0.0);
    for (const auto& c : found) {
      // The filter runs over the scan window (warm-up) and the trading period; it is causal,
      // so the positions taken from t on use prices up to the day they are set.
      const std::vector<double> y(px[c.a].begin() + static_cast<std::ptrdiff_t>(from), px[c.a].begin() + static_cast<std::ptrdiff_t>(until)),
          x(px[c.b].begin() + static_cast<std::ptrdiff_t>(from), px[c.b].begin() + static_cast<std::ptrdiff_t>(until));
      const auto r = kalmanPairs(y, x, ps);
      for (std::size_t k = 0; k < days; ++k) book[k] += r.returns[s.scanWindow + k] / static_cast<double>(found.size());
    }
    for (std::size_t k = 0; k < days; ++k) {
      out.returns.push_back(book[k]);
      out.activePairs.push_back(static_cast<double>(found.size()));
    }
  }
  return out;
}

AllocationMethod parseAllocationMethod(const std::string& n) {
  for (auto m : allAllocationMethods())
    if (allocationName(m) == n) return m;
  if (n == "equal") return AllocationMethod::Equal;
  if (n == "best") return AllocationMethod::Best;
  if (n == "sharpe") return AllocationMethod::SharpeWeighted;
  if (n == "inversevol") return AllocationMethod::InverseVolatility;
  if (n == "riskadjusted") return AllocationMethod::RiskAdjustedSharpe;
  if (n == "exponential") return AllocationMethod::ExponentialWeights;
  throw std::invalid_argument("unknown allocation method: " + n);
}

std::string allocationName(AllocationMethod m) {
  switch (m) {
    case AllocationMethod::Equal: return "equal weight";
    case AllocationMethod::Best: return "follow the leader";
    case AllocationMethod::SharpeWeighted: return "Sharpe-weighted";
    case AllocationMethod::InverseVolatility: return "inverse volatility";
    case AllocationMethod::RiskAdjustedSharpe: return "Sharpe / volatility";
    case AllocationMethod::ExponentialWeights: return "exponential weights";
  }
  return "?";
}

std::vector<AllocationMethod> allAllocationMethods() {
  return {AllocationMethod::Equal, AllocationMethod::Best, AllocationMethod::SharpeWeighted, AllocationMethod::InverseVolatility,
          AllocationMethod::RiskAdjustedSharpe, AllocationMethod::ExponentialWeights};
}

AllocationResult allocate(const std::vector<std::vector<double>>& sleeves, const AllocationSpec& s) {
  const std::size_t S = sleeves.size();
  if (S == 0) throw std::invalid_argument("allocation: at least one strategy");
  const std::size_t T = sleeves[0].size();
  for (const auto& v : sleeves)
    if (v.size() != T) throw std::invalid_argument("allocation: strategies differ in length");
  if (s.lookback < 5 || s.rebalanceEvery < 1 || s.topN < 1) throw std::invalid_argument("allocation: look-back >= 5, rebalance >= 1, top N >= 1");
  if (T <= s.lookback + 1) throw std::invalid_argument("allocation: series shorter than the look-back");
  AllocationResult out;
  out.weights.assign(S, {});
  std::vector<double> w(S, 0.0);
  double turnoverSum = 0;
  for (std::size_t t = s.lookback; t < T; ++t) {
    double turnover = 0;
    if ((t - s.lookback) % s.rebalanceEvery == 0) {
      std::vector<double> score(S, 0.0), vol(S, 0.0), next(S, 0.0);
      for (std::size_t k = 0; k < S; ++k) {
        const std::vector<double> past(sleeves[k].begin() + static_cast<std::ptrdiff_t>(t - s.lookback), sleeves[k].begin() + static_cast<std::ptrdiff_t>(t));
        vol[k] = stdev(past);
        score[k] = vol[k] > 0 ? mean(past) / vol[k] * std::sqrt(252.0) : 0.0;
      }
      switch (s.method) {
        case AllocationMethod::Equal:
          std::fill(next.begin(), next.end(), 1.0);
          break;
        case AllocationMethod::Best: {
          std::vector<std::size_t> idx(S);
          std::iota(idx.begin(), idx.end(), 0);
          std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return score[a] > score[b]; });
          for (std::size_t k = 0; k < std::min(s.topN, S); ++k)
            if (!s.allowCash || score[idx[k]] > 0) next[idx[k]] = 1.0;
          break;
        }
        case AllocationMethod::SharpeWeighted:
          for (std::size_t k = 0; k < S; ++k) next[k] = std::max(0.0, score[k]);
          break;
        case AllocationMethod::InverseVolatility:
          for (std::size_t k = 0; k < S; ++k) next[k] = vol[k] > 0 ? 1.0 / vol[k] : 0.0;
          break;
        case AllocationMethod::RiskAdjustedSharpe:
          for (std::size_t k = 0; k < S; ++k) next[k] = vol[k] > 0 ? std::max(0.0, score[k]) / vol[k] : 0.0;
          break;
        case AllocationMethod::ExponentialWeights: {
          const double top = *std::max_element(score.begin(), score.end());
          for (std::size_t k = 0; k < S; ++k) next[k] = std::exp(s.eta * (score[k] - top));
          if (s.allowCash && top <= 0) std::fill(next.begin(), next.end(), 0.0);
          break;
        }
      }
      double total = std::accumulate(next.begin(), next.end(), 0.0);
      if (!(total > 0) && !s.allowCash) {
        std::fill(next.begin(), next.end(), 1.0);
        total = static_cast<double>(S);
      }
      for (std::size_t k = 0; k < S; ++k) {
        next[k] = total > 0 ? next[k] / total : 0.0;
        turnover += std::fabs(next[k] - w[k]);
      }
      w = next;
      ++out.rebalances;
      turnoverSum += turnover;
    }
    double r = -s.costBps * 1e-4 * turnover, invested = 0;
    for (std::size_t k = 0; k < S; ++k) {
      r += w[k] * sleeves[k][t];
      invested += w[k];
      out.weights[k].push_back(w[k]);
    }
    out.returns.push_back(r);
    out.cash.push_back(1.0 - invested);
  }
  out.averageTurnover = out.rebalances ? turnoverSum / static_cast<double>(out.rebalances) : 0.0;
  return out;
}

}  // namespace sat::algo
