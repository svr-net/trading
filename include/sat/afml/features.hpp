#pragma once

#include <string>
#include <vector>

#include "sat/core/panel.hpp"
#include "sat/data/market_data.hpp"
#include "sat/features/dataset.hpp"

namespace sat::afml {

/// Names of the additional features: "ffd" (fractionally differentiated log price),
/// "vol" (exponentially weighted volatility), "roll", "corwin", "amihud", "kyle"
/// (microstructure, 20-day windows).
const std::vector<std::string>& extraFeatureNames();

/// Appends the named features to a feature set, normalised like the alphas, and raises its
/// warm-up to cover their windows. `ffdD` is the order of differentiation of "ffd"
/// (weights truncated at 1e-3).
void appendExtraFeatures(FeatureSet& fs, const MarketData& data, const std::vector<std::string>& names, Normalisation norm,
                         double ffdD = 0.4);

/// CUSUM event mask: 1 on the dates where the symmetric CUSUM filter of each stock's log
/// price fires with a threshold of `multiple` times the stock's median daily volatility,
/// 0 elsewhere. Used to train on events instead of every day.
Panel cusumEventMask(const MarketData& data, double multiple);

}  // namespace sat::afml
