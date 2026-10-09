#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "sat/adaptive/experiment.hpp"
#include "sat/adaptive/self_adaptive.hpp"
#include "sat/algo/ensemble.hpp"
#include "sat/algo/regimes.hpp"
#include "sat/algo/trend.hpp"
#include "sat/data/market_data.hpp"

namespace sat::algo {

/// Every approach of the library backtested on the same days, and self-adaptive allocation
/// across them. The sleeves are:
///  - each machine-learning model, with equal capital in every trading rule on its forecasts
///    (no hindsight choice of rule);
///  - the paper's self-adaptive selector over all model x rule candidates, plain, with a
///    volatility target, and with a Kalman-filter beta hedge against the market;
///  - trend following (EWMAC), the adaptive cointegrated-pairs book, and the HMM
///    regime-switched market;
///  - the equal-weight market (benchmark, never allocated to).
/// Every allocation method is run on the same sleeves; their deflated Sharpe ratios account
/// for having tried all of them.
struct TournamentSpec {
  AllocationSpec allocation;
  double hedgeTargetVol = 0.10, hedgeMaxLeverage = 2.0, hedgeCostBps = 2.0;
  TrendSpec trend;
  PairsBookSpec pairs;
  RegimeSwitchSpec regimes;
};

struct TournamentSleeve {
  std::string name, family;
  std::vector<double> returns;  ///< over the evaluation days
};

struct TournamentAllocator {
  AllocationMethod method;
  AllocationResult result;       ///< returns over the evaluation days
  double deflatedSharpe = 0.0;
};

struct TournamentResult {
  std::vector<TournamentSleeve> sleeves;  ///< the last one is the benchmark
  std::vector<TournamentAllocator> allocators;
  std::size_t start = 0;  ///< date index of the first evaluated return (earned to start + 1)
};

/// `book`, when given, is the candidate book to use (for example built from the GPU kernels'
/// read-back); otherwise it is backtested here.
TournamentResult runTournament(const MarketData& data, const PredictionSet& predictions, const ExperimentSpec& experiment,
                               const TournamentSpec& spec, const CandidateBook* book = nullptr);

/// Equal-weight market return from date t to t + 1, for t = 0 .. T - 2.
std::vector<double> equalWeightMarket(const MarketData& data);

}  // namespace sat::algo
