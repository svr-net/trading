#pragma once

// The intraday strategy (US, where 5-minute bars exist): each day, shortly after the open, buy the
// strongest of the stocks then most traded and sell the same day at a stop-loss, a take-profit or
// the close, all on the real 5-minute path. Everything is learnt walk-forward, from the days before:
//  - Entry: at the close of the k-th 5-minute bar, k in 1, 2, 4, 8 (5 to 40 minutes after the
//    open); each k keeps its own book, and the one whose record is best so far (ADWIN, shrunk by
//    its t-statistic) trades; cash while none beats cash.
//  - Forecast: the move from the entry to the close (in the stock's volatility), by least squares
//    on the opening move so far (the stock's and the market's), the overnight gap, the stock's last
//    day (open-to-close, close in the range) and the model's forecast, fitted on every stock and
//    day so far, one fit per k.
//  - Stop and take-profit: multiples of the stock's volatility for the rest of the day (from the
//    model's horizons, as in the daily trades); each pair's net outcome on the 5-minute path fitted
//    on the forecast; each stock takes the pair with the best expected outcome.
//  - Selection: the 10 strongest expected to gain after costs, a tenth of the capital each.
// Fills: on the 5-minute bars after the entry; within one bar touching both levels the stop is
// taken (the fill test takes the take-profit instead). Every trade pays the purchase and the sale
// cost. Long only.

#include <cstddef>
#include <string>
#include <vector>

#include "ofm/daybook.hpp"
#include "ofm/model.hpp"

namespace ofm {

struct IntraBook {
  std::vector<std::size_t> days;     ///< the days traded (with paths), in order
  std::vector<double> ret;           ///< each day's net return (stop first within a bar)
  std::vector<double> optimistic;    ///< the same trades with the take-profit first within a bar
  std::vector<int> entryBars;        ///< the entry (k) of the book in use each day (0: cash)
  std::vector<int> forecastUsed;     ///< the forecast in use each day: 0 every input, 1 the strongest single input, -1 cash
  std::size_t trades = 0, stops = 0, takes = 0;
  std::size_t candidates = 0;        ///< candidate stock-days with a path
  DayBook::Skill skill[4];           ///< out-of-sample correlation of each k's forecast with the outcome
  std::vector<std::string> featureNames;
  std::vector<DayBook::Skill> featureSkill[4];  ///< each input's correlation with the outcome, per k
};

/// buy, sell: costs per side (fractions); universe: each day only the stocks then most traded.
IntraBook intraBook(const Market& m, const Forecast& f, const day::Paths& paths, double buy, double sell, std::size_t universe);

/// The strategy's report: the backtest over the days with paths (whole, each half, each year), the
/// fill test, the entries used and each input's correlation with the outcome. Results only.
std::string intraText(const Market& m, const IntraBook& b, double costPerSide);

}  // namespace ofm
