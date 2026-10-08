#pragma once

#include <cstddef>

#include "sat/core/panel.hpp"

namespace sat::afml {

/// Microstructure features from daily bars, each over a rolling window of `window` days
/// (NaN until the window is full):
/// - Roll's effective spread, 2 sqrt(max(0, -cov(dp_t, dp_{t-1}))) relative to price: bid-ask
///   bounce makes consecutive price changes negatively correlated;
/// - Corwin-Schultz spread from two-day high-low ranges, which mix volatility (grows with the
///   horizon) and the spread (does not);
/// - Amihud illiquidity, mean |return| per million of dollar volume;
/// - Kyle's lambda, the regression slope of returns on signed dollar volume (signed by the
///   day's return, a tick-rule proxy), the price impact of order flow.
Panel rollSpread(const Panel& close, std::size_t window);
Panel corwinSchultzSpread(const Panel& high, const Panel& low, std::size_t window);
Panel amihudIlliquidity(const Panel& close, const Panel& volume, std::size_t window);
Panel kyleLambda(const Panel& close, const Panel& volume, std::size_t window);

}  // namespace sat::afml
