#pragma once

#include <map>
#include <string>
#include <vector>

#include "ofm/data.hpp"
#include "ofm/model.hpp"
#include "ofm/trade.hpp"

namespace ofm {

/// Results as JSON (no prices): run facts, expected returns at the last close, factor
/// statistics, cumulative curves, metrics for the whole period, both halves and each year, the
/// pass mark, and trade statistics.
std::string reportJson(const Market& m, const Forecast& f, const Backtest& bt, const Costs& costs);

/// The same as plain text.
std::string reportText(const Market& m, const Forecast& f, const Backtest& bt, const Costs& costs);

}  // namespace ofm
