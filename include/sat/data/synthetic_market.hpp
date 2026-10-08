#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sat/data/market_data.hpp"

namespace sat {

/// One state of the hidden Markov chain that drives the synthetic market.
struct MarketRegime {
  std::string name;
  double drift = 0.0;          ///< daily expected market return
  double volatility = 0.01;    ///< daily market volatility
  double momentum = 0.0;       ///< autocorrelation of the idiosyncratic returns (> 0 trend, < 0 reversal)
  double volumeReversal = 0.0; ///< extra reversal of returns made on abnormal volume
};

/// A regime-switching factor model of a stock universe.
///
/// The market return follows the drift and volatility of the current regime; each stock
/// loads on it with its own beta and adds an idiosyncratic AR(1) component whose
/// autocorrelation also depends on the regime (trends in bull markets, reversals in
/// range-bound ones). Abnormal volume makes the next day's idiosyncratic move revert.
/// Intraday open/high/low and the VWAP come from a Brownian bridge between the previous
/// close and the close. The predictable part of returns therefore changes with the regime,
/// which is the situation a single fixed trading strategy cannot keep up with.
struct SyntheticMarketSpec {
  std::size_t numAssets = 30;
  std::size_t numDates = 1260;
  std::uint64_t seed = 7;
  std::vector<MarketRegime> regimes = defaultRegimes();
  /// Row-stochastic transition matrix; empty means `persistence` on the diagonal and the
  /// rest spread evenly.
  std::vector<std::vector<double>> transition;
  double persistence = 0.985;
  double idiosyncraticVol = 0.016;  ///< average daily idiosyncratic volatility
  double betaDispersion = 0.3;      ///< standard deviation of the betas around 1
  double startPrice = 50.0;
  double meanVolume = 1e6;

  static std::vector<MarketRegime> defaultRegimes();
};

MarketData generateSyntheticMarket(const SyntheticMarketSpec& spec);

}  // namespace sat
