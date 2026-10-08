#include "sat/strategy/strategy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "sat/afml/bet_sizing.hpp"
#include "sat/core/stats.hpp"

namespace sat {

StrategyKind parseStrategyKind(const std::string& name) {
  if (name == "topk") return StrategyKind::LongTopK;
  if (name == "longshort") return StrategyKind::LongShort;
  if (name == "threshold") return StrategyKind::Threshold;
  if (name == "probweighted") return StrategyKind::ProbabilityWeighted;
  if (name == "betsize") return StrategyKind::BetSized;
  throw std::invalid_argument("unknown strategy '" + name + "' (topk, longshort, threshold, probweighted, betsize)");
}

std::string strategyKindName(StrategyKind kind) {
  switch (kind) {
    case StrategyKind::LongTopK: return "topk";
    case StrategyKind::LongShort: return "longshort";
    case StrategyKind::Threshold: return "threshold";
    case StrategyKind::ProbabilityWeighted: return "probweighted";
    case StrategyKind::BetSized: return "betsize";
  }
  return "?";
}

std::string StrategySpec::label() const {
  char buf[64];
  switch (kind) {
    case StrategyKind::LongTopK: std::snprintf(buf, sizeof buf, "Top-%d", static_cast<int>(std::lround(param))); break;
    case StrategyKind::LongShort: std::snprintf(buf, sizeof buf, "Long-short %d", static_cast<int>(std::lround(param))); break;
    case StrategyKind::Threshold: std::snprintf(buf, sizeof buf, "P>%.2f", param); break;
    case StrategyKind::ProbabilityWeighted: std::snprintf(buf, sizeof buf, "P-weighted>%.2f", param); break;
    case StrategyKind::BetSized: std::snprintf(buf, sizeof buf, "Bet size>%.2f", param); break;
  }
  std::string s = buf;
  if (holding > 1) s += " /" + std::to_string(holding) + "d";
  return s;
}

std::vector<std::uint32_t> rankRow(const double* prob, std::size_t n) {
  const auto order = argsortDescending(prob, n);
  std::vector<std::uint32_t> rank(n);
  for (std::size_t k = 0; k < n; ++k) rank[order[k]] = static_cast<std::uint32_t>(k);
  return rank;
}

std::vector<std::uint32_t> rankTable(const Panel& prob) {
  std::vector<std::uint32_t> out(prob.dates() * prob.assets());
  for (std::size_t t = 0; t < prob.dates(); ++t) {
    const auto r = rankRow(prob.row(t), prob.assets());
    std::copy(r.begin(), r.end(), out.begin() + static_cast<std::ptrdiff_t>(t * prob.assets()));
  }
  return out;
}

void strategyWeights(const StrategySpec& s, const double* prob, const std::uint32_t* rank, std::size_t n, double* w) {
  std::fill(w, w + n, 0.0);
  switch (s.kind) {
    case StrategyKind::LongTopK: {
      const auto k = static_cast<std::uint32_t>(std::clamp<long>(static_cast<long>(std::floor(s.param + 0.5)), 1, static_cast<long>(n)));
      for (std::size_t i = 0; i < n; ++i)
        if (rank[i] < k) w[i] = 1.0 / k;
      break;
    }
    case StrategyKind::LongShort: {
      const auto k = static_cast<std::uint32_t>(std::clamp<long>(static_cast<long>(std::floor(s.param + 0.5)), 1, std::max<long>(1, static_cast<long>(n / 2))));
      for (std::size_t i = 0; i < n; ++i) {
        if (rank[i] < k) w[i] = 1.0 / k;
        else if (rank[i] >= n - k) w[i] = -1.0 / k;
      }
      break;
    }
    case StrategyKind::Threshold: {
      std::size_t c = 0;
      for (std::size_t i = 0; i < n; ++i) c += prob[i] > s.param ? 1 : 0;
      for (std::size_t i = 0; i < n; ++i)
        if (prob[i] > s.param) w[i] = 1.0 / static_cast<double>(c);
      break;
    }
    case StrategyKind::ProbabilityWeighted: {
      double total = 0.0;
      for (std::size_t i = 0; i < n; ++i) total += prob[i] > s.param ? prob[i] - s.param : 0.0;
      if (total > 0.0)
        for (std::size_t i = 0; i < n; ++i)
          if (prob[i] > s.param) w[i] = (prob[i] - s.param) / total;
      break;
    }
    case StrategyKind::BetSized: {
      double gross = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double m = afml::betSize(prob[i]);
        if (std::fabs(m) >= s.param && m != 0.0) {
          w[i] = m;
          gross += std::fabs(m);
        }
      }
      if (gross > 0.0)
        for (std::size_t i = 0; i < n; ++i) w[i] /= gross;
      break;
    }
  }
}

BacktestResult backtest(const StrategySpec& s, const Panel& prob, const std::vector<std::uint32_t>& ranks,
                        const Panel& nextReturns, std::size_t start, std::size_t end, double costBps) {
  const std::size_t N = prob.assets();
  if (end > nextReturns.dates() || start >= end) throw std::invalid_argument("backtest: empty date range");
  if (s.holding < 1) throw std::invalid_argument("backtest: holding period must be at least 1");
  BacktestResult out;
  out.label = s.label();
  DailySeries& d = out.series;
  d.start = start;
  const double cost = costBps * 1e-4;
  std::vector<double> w(N, 0.0), prev(N, 0.0);
  for (std::size_t t = start; t < end; ++t) {
    const std::size_t off = t - start;
    double turnover = 0.0;
    if (rebalanceOffset(off, s.holding) == off) {
      strategyWeights(s, prob.row(t), ranks.data() + t * N, N, w.data());
      for (std::size_t i = 0; i < N; ++i) turnover += std::fabs(w[i] - prev[i]);
      prev = w;
    }
    double gross = 0.0, net = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      if (w[i] != 0.0 && std::isfinite(nextReturns(t, i))) gross += w[i] * nextReturns(t, i);
      net += w[i];
    }
    d.gross.push_back(gross);
    d.turnover.push_back(turnover);
    d.net.push_back(gross - cost * turnover);
    d.netExposure.push_back(net);
  }
  out.metrics = evaluatePerformance(d.net, d.turnover);
  return out;
}

BacktestResult equalWeightBenchmark(const Panel& nextReturns, std::size_t start, std::size_t end, double costBps) {
  const std::size_t N = nextReturns.assets();
  if (end > nextReturns.dates() || start >= end) throw std::invalid_argument("benchmark: empty date range");
  BacktestResult out;
  out.label = "Equal-weight market";
  out.series.start = start;
  for (std::size_t t = start; t < end; ++t) {
    double s = 0.0, n = 0.0;
    for (std::size_t i = 0; i < N; ++i)
      if (std::isfinite(nextReturns(t, i))) {
        s += nextReturns(t, i);
        n += 1.0;
      }
    const double gross = n > 0 ? s / n : 0.0, turnover = t == start ? 1.0 : 0.0;
    out.series.gross.push_back(gross);
    out.series.turnover.push_back(turnover);
    out.series.net.push_back(gross - costBps * 1e-4 * turnover);
    out.series.netExposure.push_back(1.0);
  }
  out.metrics = evaluatePerformance(out.series.net, out.series.turnover);
  return out;
}

}  // namespace sat
