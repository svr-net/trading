// Backtests every approach of the library (machine-learning models, the paper's
// self-adaptive selector, hedged variants, trend following, cointegrated pairs, HMM regime
// switching) on the same days, then searches for the best self-adaptive allocation across
// them without fooling itself: allocation methods and their settings are ranked on one set
// of independently generated markets (selection) and the winner is re-tested on a second,
// unseen set (validation).
//   strategy_tournament [markets per set=6] [days=2520]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "sat/sat.hpp"

using namespace sat;

namespace {

struct Market {
  std::vector<algo::TournamentSleeve> sleeves;  // the last is the benchmark
};

std::vector<Market> collect(std::uint64_t seedBase, std::size_t markets, std::size_t days) {
  std::vector<Market> out;
  for (std::size_t m = 0; m < markets; ++m) {
    SyntheticMarketSpec ms;
    ms.seed = seedBase + m;
    ms.numDates = days;
    const MarketData data = generateSyntheticMarket(ms);
    ExperimentSpec exp;
    const PredictionSet p = runPredictions(data, exp);
    algo::TournamentSpec ts;
    ts.allocation.lookback = 5;  // keep the sleeves from the first common day; allocators are re-run below
    out.push_back({algo::runTournament(data, p, exp, ts).sleeves});
    std::fprintf(stderr, "  market seed %llu done\n", static_cast<unsigned long long>(ms.seed));
  }
  return out;
}

constexpr std::size_t kSkip = 252;  // every configuration is scored after the longest look-back

std::vector<double> scored(const std::vector<double>& r, std::size_t lookback) {
  // allocate() returns days lookback.. ; drop the rest of the burn-in so all are aligned.
  return std::vector<double>(r.begin() + static_cast<std::ptrdiff_t>(kSkip - lookback), r.end());
}

struct Score {
  double sharpe = 0, worst = 1e9, drawdown = 0, annual = 0;
};

Score evaluate(const std::vector<Market>& set, const algo::AllocationSpec& spec) {
  Score s;
  for (const auto& m : set) {
    std::vector<std::vector<double>> pool;
    for (std::size_t k = 0; k + 1 < m.sleeves.size(); ++k) pool.push_back(m.sleeves[k].returns);
    const auto pm = evaluatePerformance(scored(algo::allocate(pool, spec).returns, spec.lookback));
    s.sharpe += pm.sharpe / static_cast<double>(set.size());
    s.drawdown += pm.maxDrawdown / static_cast<double>(set.size());
    s.annual += pm.annualReturn / static_cast<double>(set.size());
    s.worst = std::min(s.worst, pm.sharpe);
  }
  return s;
}

Score sleeveScore(const std::vector<Market>& set, std::size_t k) {
  Score s;
  for (const auto& m : set) {
    const auto pm = evaluatePerformance(std::vector<double>(m.sleeves[k].returns.begin() + kSkip, m.sleeves[k].returns.end()));
    s.sharpe += pm.sharpe / static_cast<double>(set.size());
    s.drawdown += pm.maxDrawdown / static_cast<double>(set.size());
    s.annual += pm.annualReturn / static_cast<double>(set.size());
    s.worst = std::min(s.worst, pm.sharpe);
  }
  return s;
}

std::string label(const algo::AllocationSpec& s) {
  std::string l = algo::allocationName(s.method) + ", look-back " + std::to_string(s.lookback) + ", every " + std::to_string(s.rebalanceEvery);
  if (s.method == algo::AllocationMethod::Best) l += ", top " + std::to_string(s.topN);
  if (s.method == algo::AllocationMethod::ExponentialWeights) l += ", eta " + std::to_string(static_cast<int>(s.eta));
  return l;
}

void row(const std::string& name, const Score& s) {
  std::printf("  %-58s %7.1f%% %7.2f %8.2f %7.1f%%\n", name.c_str(), 100 * s.annual, s.sharpe, s.worst, 100 * s.drawdown);
}

void header(const char* title) { std::printf("\n%s\n  %-58s %8s %7s %8s %8s\n", title, "approach", "ann.ret", "Sharpe", "worst SR", "max DD"); }

}  // namespace

int main(int argc, char** argv) {
  const std::size_t markets = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 6;
  const std::size_t days = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 2520;
  std::fprintf(stderr, "selection markets\n");
  const auto selection = collect(101, markets, days);
  std::fprintf(stderr, "validation markets\n");
  const auto validation = collect(201, markets, days);
  const auto& names = selection[0].sleeves;

  header("Every strategy on the selection markets (average)");
  for (std::size_t k = 0; k < names.size(); ++k) row(names[k].name, sleeveScore(selection, k));

  // The search space of the meta-allocator.
  std::vector<algo::AllocationSpec> grid;
  for (auto m : algo::allAllocationMethods())
    for (std::size_t lb : {63, 126, 252})
      for (std::size_t every : {5, 21, 63}) {
        algo::AllocationSpec s;
        s.method = m;
        s.lookback = lb;
        s.rebalanceEvery = every;
        if (m == algo::AllocationMethod::Best)
          for (std::size_t n : {1, 2, 3}) {
            s.topN = n;
            grid.push_back(s);
          }
        else if (m == algo::AllocationMethod::ExponentialWeights)
          for (double eta : {1.0, 2.0, 4.0}) {
            s.eta = eta;
            grid.push_back(s);
          }
        else
          grid.push_back(s);
      }
  std::vector<std::pair<Score, std::size_t>> ranked;
  for (std::size_t g = 0; g < grid.size(); ++g) ranked.push_back({evaluate(selection, grid[g]), g});
  std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first.sharpe > b.first.sharpe; });
  header(("Top allocation settings of " + std::to_string(grid.size()) + " on the selection markets").c_str());
  for (std::size_t k = 0; k < std::min<std::size_t>(8, ranked.size()); ++k) row(label(grid[ranked[k].second]), ranked[k].first);
  // The best setting of each method, to see which family is robust.
  header("Best setting per method: selection -> validation (unseen markets)");
  for (auto m : algo::allAllocationMethods()) {
    const auto it = std::find_if(ranked.begin(), ranked.end(), [&](const auto& r) { return grid[r.second].method == m; });
    const auto& spec = grid[it->second];
    row(label(spec) + " [sel]", it->first);
    row(label(spec) + " [val]", evaluate(validation, spec));
  }
  header("Reference sleeves on the validation markets");
  for (std::size_t k = 0; k < names.size(); ++k)
    if (names[k].family != "ML model (all rules)") row(names[k].name, sleeveScore(validation, k));
  const auto& winner = grid[ranked[0].second];
  std::printf("\nChosen on the selection markets: %s\n", label(winner).c_str());
  row("validation", evaluate(validation, winner));
  return 0;
}
