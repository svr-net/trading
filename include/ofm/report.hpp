#pragma once

#include <map>
#include <string>
#include <vector>

#include "ofm/data.hpp"
#include "ofm/daybook.hpp"
#include "ofm/model.hpp"
#include "ofm/structure.hpp"
#include "ofm/trade.hpp"

namespace ofm {

/// Results as JSON (no prices): run facts, expected returns at the last close, factor
/// statistics, cumulative curves, metrics for the whole period, both halves and each year, the
/// pass mark, and trade statistics; with a daily book, its curve, metrics and orders for the next open.
std::string reportJson(const Market& m, const Forecast& f, const Backtest& bt, const Costs& costs, const MarketStructure* ms = nullptr,
                       const DayBook* day = nullptr);

/// The same as plain text.
std::string reportText(const Market& m, const Forecast& f, const Backtest& bt, const Costs& costs, const MarketStructure* ms = nullptr,
                       const DayBook* day = nullptr);

}  // namespace ofm
