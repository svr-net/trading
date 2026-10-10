#pragma once

// Rolling portfolio of the K most valuable stocks each day, with stop-loss and take-profit exits.
//
// Two meanings of "most valuable" are run side by side:
//  - "expected return": the K stocks with the highest expected return from the model (E);
//  - "largest": the K stocks with the highest traded value (close times volume, averaged over the
//    model's longest horizon), the nearest measure of size the bars carry.
// Decisions at each close, fills at the next open, K equal shares to start. A held stock leaves
// when it drops out of the K (for the expected-return ranking: only when the best outsider's
// expected return beats it by a round trip over the forecast's life). Exits free a slot, which
// the best stock outside the portfolio fills at the next open with the cash.
//
// Stops and take-profits: a position bought at price P has a stop at P exp(-a w) and a take-profit
// at P exp(b w), w = sigma sqrt(L): the stock's volatility over the forecast's life L (sigma the
// exponentially weighted daily volatility with half-life L, both at the entry decision). Each day
// the high and low are checked against them; a gap through a level fills at the open, and when
// both are touched the stop is assumed first. The multipliers a and b are not set by hand: each
// pair of a grid of powers of two (and "none") runs as a shadow portfolio, and each day the live
// portfolio uses the pair whose shadow has grown most so far (no stops at all until one has).

#include <string>
#include <vector>

#include "ofm/data.hpp"
#include "ofm/model.hpp"
#include "ofm/trade.hpp"

namespace ofm {

enum class Exit : char { Stop = 's', Take = 't', Close = 'c', Rotate = 'r', Open = 'o' };

struct TopKTrade {
  std::size_t asset = 0, entryDay = 0, exitDay = 0;
  double ret = 0;  ///< net of costs
  Exit reason = Exit::Open;
};

struct TopKPosition {
  std::size_t asset = 0, entryDay = 0;
  double ret = 0;       ///< net of the purchase cost, to the last close
  double stopDist = 0;  ///< stop level / last close - 1 (NaN: no stop)
  double takeDist = 0;  ///< take-profit level / last close - 1 (NaN: none)
  double entryRel = 1;  ///< entry price / last close
  double width = 0;     ///< sigma sqrt(life) at entry
};

/// One line of the plan for the next day, relative to the last close (fractions; NaN: none).
/// The rule's levels of a held stock are fixed at its entry; a purchase's are set from the next
/// open, taken as the last close. Where the rule has no stop (or no take-profit), the best finite
/// level so far is shown as advice, set from the last close. Probabilities: of touching the stop / the take-profit first, during the next
/// day and over the forecast's life, for a log price with the model's expected daily return as
/// drift and the stock's daily volatility (simulated, 2 x 10^4 paths, 48 steps a day).
struct TopKOrder {
  std::size_t asset = 0;
  int action = 0;  ///< +1 buy at the open, 0 hold, -1 sell at the open
  std::size_t rank = 0;  ///< rank by the portfolio's score at the last close (1 best; past the end: not ranked)
  bool advisoryStop = false, advisoryTake = false;  ///< the rule has none: best finite level so far
  double open = 0, stop = 0, take = 0, close = 0, sigma = 0;
  double pStopDay = 0, pTakeDay = 0, pStopLife = 0, pTakeLife = 0;
};

/// The daily trades judged against the day's actual bars: what was expected at the close before
/// (levels, probabilities, expected return) against what the open, high, low, close and volume did.
struct TopKEvaluation {
  std::size_t trades = 0, days = 0;
  double predStop = 0, realStop = 0, predTake = 0, realTake = 0;  ///< mean probability vs frequency
  double expected = 0, realised = 0;  ///< mean expected excess return vs realised (open to close, over all ranked)
  double ic = 0, signHit = 0;         ///< correlation of the two over all trades; share with the same sign
  double grossTop = 0, grossAll = 0;  ///< mean open-to-close return before costs: the 10, all ranked
};

/// One trade of the last completed day: the plan made at the close before against the bars.
struct TopKCheck {
  std::size_t asset = 0;
  double stop = 0, take = 0, expected = 0, pStop = 0, pTake = 0;  ///< planned, relative to the open
  double high = 0, low = 0, close = 0, volume = 0;  ///< actual, relative to the open (volume: to its usual)
  double ret = 0;  ///< net of costs
  Exit reason = Exit::Close;
};

struct TopKResult {
  std::string name;
  std::size_t K = 0, start = 0;
  std::vector<double> daily;      ///< net returns from `start`, adaptive stops
  std::vector<double> noStops;    ///< the same portfolio without stops or take-profits
  std::vector<double> turnover;
  std::vector<double> kStop, kTake;  ///< multipliers used each day (inf: none)
  std::vector<TopKTrade> trades;
  std::vector<double> grid;       ///< Sharpe of each fixed (stop, take) pair over the whole period, row-major
  std::vector<double> gridK;      ///< the multipliers of the grid (inf: none)
  std::vector<TopKPosition> positions;  ///< at the last close
  std::vector<int> orders;        ///< at the last close: +1 buy, -1 sell at the next open
  std::vector<TopKOrder> plan;    ///< the next day: the 10 held after the open, then the sales
  double life = 1;                ///< the forecast's life at the last close (days)
  bool oneDay = false;            ///< positions held one day: levels reset from each open, one day's volatility
  std::vector<double> grossTop, grossAll;  ///< before costs, open to open: the day's 10 best and all ranked stocks
  bool dayTrades = false;         ///< bought at the open, sold at the stop, the take-profit or the close
  TopKEvaluation eval;
  std::vector<TopKCheck> lastDay; ///< the last completed day's trades against the bars
  std::string lastDate;
};

/// Both rankings, from the same start as the backtest (bt.start).
std::vector<TopKResult> topKBacktests(const Market& m, const Forecast& f, const Costs& costs, const Backtest& bt, std::size_t K = 10);

std::string topKJson(const Market& m, const std::vector<TopKResult>& r, const Backtest& bt);
std::string topKText(const Market& m, const std::vector<TopKResult>& r, const Backtest& bt);

}  // namespace ofm
