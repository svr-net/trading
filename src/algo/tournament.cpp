#include "sat/algo/tournament.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sat/adaptive/self_adaptive.hpp"
#include "sat/afml/backtest_stats.hpp"
#include "sat/core/stats.hpp"
#include "sat/hedge/hedging.hpp"

namespace sat::algo {

namespace {

double rowMean(const Panel& p, std::size_t t) {
  double s = 0;
  std::size_t n = 0;
  for (std::size_t i = 0; i < p.assets(); ++i)
    if (std::isfinite(p(t, i))) {
      s += p(t, i);
      ++n;
    }
  return n ? s / static_cast<double>(n) : 0.0;
}

}  // namespace

std::vector<double> equalWeightMarket(const MarketData& d) {
  const Panel r = d.returns();
  std::vector<double> out;
  for (std::size_t t = 1; t < d.numDates(); ++t) out.push_back(rowMean(r, t));
  return out;
}

TournamentResult runTournament(const MarketData& d, const PredictionSet& p, const ExperimentSpec& exp, const TournamentSpec& spec,
                               const CandidateBook* precomputed) {
  // Each candidate series: value k is earned from date first + k to first + k + 1.
  struct Series {
    std::string name, family;
    std::size_t first;
    std::vector<double> r;
  };
  std::vector<Series> all;
  const std::size_t T = d.numDates();
  const auto strategies = exp.strategies.empty() ? ExperimentSpec::defaultStrategies() : exp.strategies;
  const CandidateBook own = precomputed ? CandidateBook() : CandidateBook(p.models, strategies, p.nextReturns, exp.costBps);
  const CandidateBook& book = precomputed ? *precomputed : own;
  if (precomputed && precomputed->size() != p.models.size() * strategies.size()) throw std::invalid_argument("tournament: the candidate book does not match the experiment");
  const std::size_t S = strategies.size();
  for (std::size_t m = 0; m < p.models.size(); ++m) {
    std::vector<double> avg(book.days(), 0.0);
    for (std::size_t s = 0; s < S; ++s) {
      const auto net = book.netSeries(m * S + s, 0);
      for (std::size_t k = 0; k < net.size(); ++k) avg[k] += net[k] / static_cast<double>(S);
    }
    all.push_back({p.models[m].name, "ML model (all rules)", book.start(), avg});
  }
  const std::size_t evalFrom = std::min(exp.selector.lookback, book.days() - 1);
  const AdaptiveResult a = runSelector(book, exp.selector, evalFrom);
  const std::size_t aFirst = book.start() + evalFrom;
  all.push_back({"self-adaptive selector (paper)", "ML self-adaptive", aFirst, a.net});
  all.push_back({"self-adaptive, vol target", "hedged", aFirst, hedge::volatilityTarget(a.net, spec.hedgeTargetVol, 36, spec.hedgeMaxLeverage)});
  {
    std::vector<double> m(a.net.size());
    for (std::size_t k = 0; k < m.size(); ++k) m[k] = rowMean(p.nextReturns, aFirst + k);
    const auto kal = hedge::kalmanRegression(a.net, m, 1e-4, std::max(1e-10, std::pow(stdev(a.net), 2)));
    all.push_back({"self-adaptive, Kalman beta hedge", "hedged", aFirst, hedge::applyHedge(a.net, m, kal.beta, spec.hedgeCostBps)});
  }
  if (T > spec.trend.warmup + 30) all.push_back({"trend following (EWMAC)", "trend", spec.trend.warmup, trendFollowing(d.close, spec.trend).returns});
  if (T > spec.pairs.scanWindow + 30) {
    const auto pb = pairsBook(d.close, spec.pairs);
    all.push_back({"cointegrated pairs book", "statistical arbitrage", pb.start, pb.returns});
  }
  const auto mr = equalWeightMarket(d);
  if (mr.size() > spec.regimes.window + 30) {
    const auto sw = regimeSwitch(mr, spec.regimes);
    all.push_back({"HMM regime-switched market", "regimes", sw.start, sw.returns});
  }
  all.push_back({"equal-weight market", "benchmark", 0, mr});

  std::size_t from = 0, to = T - 1;
  for (const auto& s : all) {
    from = std::max(from, s.first);
    to = std::min(to, s.first + s.r.size());
  }
  const std::size_t L = spec.allocation.lookback;
  if (to <= from + L + 20) throw std::invalid_argument("tournament: too few common days after the warm-ups and the look-back");
  std::vector<std::vector<double>> R;
  for (const auto& s : all)
    R.emplace_back(s.r.begin() + static_cast<std::ptrdiff_t>(from - s.first), s.r.begin() + static_cast<std::ptrdiff_t>(to - s.first));
  const std::vector<std::vector<double>> pool(R.begin(), R.end() - 1);  // never allocate to the benchmark

  TournamentResult out;
  out.start = from + L;
  for (std::size_t k = 0; k < all.size(); ++k)
    out.sleeves.push_back({all[k].name, all[k].family, std::vector<double>(R[k].begin() + static_cast<std::ptrdiff_t>(L), R[k].end())});
  std::vector<double> sharpes;
  for (auto m : allAllocationMethods()) {
    AllocationSpec x = spec.allocation;
    x.method = m;
    out.allocators.push_back({m, allocate(pool, x), 0.0});
    sharpes.push_back(afml::periodSharpe(out.allocators.back().result.returns));
  }
  const double trialVar = std::max(1e-12, std::pow(stdev(sharpes), 2));
  for (std::size_t k = 0; k < out.allocators.size(); ++k) {
    const auto& r = out.allocators[k].result.returns;
    out.allocators[k].deflatedSharpe = afml::deflatedSharpe(sharpes[k], static_cast<double>(r.size()), skewness(r), kurtosis(r),
                                                            static_cast<double>(sharpes.size()), trialVar);
  }
  return out;
}

}  // namespace sat::algo
