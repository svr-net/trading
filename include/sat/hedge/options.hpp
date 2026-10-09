#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sat::hedge {

/// Black-Scholes price and delta of a European option (no dividends).
double blackScholes(double spot, double strike, double years, double rate, double vol, bool call);
double blackScholesDelta(double spot, double strike, double years, double rate, double vol, bool call);

/// Delta hedging of a short call: sell the call at the implied-volatility price, hold delta
/// shares rebalanced every `rebalanceEvery` simulation steps, financed at the risk-free rate,
/// with `costBps` per unit of traded notional. The stock follows geometric Brownian motion
/// with drift `mu` and the true volatility `vol`. With continuous rebalancing and implied =
/// true volatility the hedge is perfect; discrete rebalancing leaves an error that shrinks
/// like 1/sqrt(number of rebalances), and a gap between implied and true volatility leaves
/// a P&L of the option's gamma times their difference.
struct DeltaHedgeSpec {
  double spot = 100, strike = 100, years = 0.25, rate = 0.02;
  double vol = 0.2;         ///< true volatility of the stock
  double impliedVol = 0.2;  ///< volatility at which the option is sold and hedged
  double mu = 0.05;
  std::size_t stepsPerYear = 252 * 8;  ///< simulation grid (several steps a day)
  std::size_t paths = 2000;
  double costBps = 0.0;
  std::uint64_t seed = 13;
};

struct DeltaHedgeResult {
  std::size_t rebalanceEvery = 1, rebalances = 0;
  double premium = 0.0;
  std::vector<double> pnl;  ///< final P&L per path, in units of the premium
  double mean = 0.0, sd = 0.0;
};

DeltaHedgeResult simulateDeltaHedge(const DeltaHedgeSpec& spec, std::size_t rebalanceEvery);

/// Option overlays on a holding of an index (one unit at each roll), rolled every
/// `tenorDays` trading days and priced at Black-Scholes with the trailing `volWindow`-day
/// realised volatility plus `volPremium` as implied volatility:
/// - protective put: buy a put struck at `putMoneyness` x spot;
/// - collar: the same put, financed by selling a call struck at `callMoneyness` x spot.
/// Returns daily returns of the unhedged holding and of both hedged portfolios (marked to
/// market every day at the implied volatility fixed at purchase).
struct OptionOverlaySpec {
  std::size_t tenorDays = 21;
  double putMoneyness = 0.95;
  double callMoneyness = 1.05;
  double volPremium = 0.02;
  std::size_t volWindow = 21;
  double rate = 0.0;
};

struct OptionOverlayResult {
  std::vector<double> unhedged, protectivePut, collar;
  double averagePutCost = 0.0;    ///< premium per roll, fraction of spot
  double averageCallIncome = 0.0;
  std::size_t rolls = 0;
};

OptionOverlayResult optionOverlay(const std::vector<double>& price, const OptionOverlaySpec& spec);

}  // namespace sat::hedge
