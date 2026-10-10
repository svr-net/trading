#pragma once

// Daily trading of the forecasts, with only the costs given.
//
// Decisions at each close, fills at the next open:
//  - the first portfolio is the market: every stock with a forecast, equal weight;
//  - a held stock is sold when its expected excess return over the market, over the expected
//    life of a forecast, is below minus a round trip (it is expected to lag the market by more
//    than the trades cost); a stock is bought when it is expected to beat the market by more
//    than that. The life is 1 / (1 - rho), rho the average day-to-day rank correlation of the
//    forecasts so far (capped at 1 - 1/n for n observations). A stock that falls or runs up is
//    sold when its updated forecast says so: these are the adaptive stop-loss and take-profit.
//  - a bought stock gets an equal share of the portfolio, funded from cash and then pro rata
//    from the other holdings; sale proceeds with nothing to buy go back pro rata.
// Index futures hedge (optional): the index future's own expected next-day return decides a short
// hedge sized by the book's expanding beta to the future, when the expected fall over the
// forecast's life exceeds the futures round trip. The forecast is the future's average return
// (shrunk by its own t-statistic) plus a conditional part from its trends at the model's horizons
// and the market's topology and geometry (whitened by their expanding covariance; premia shrunk
// jointly by positive-part James-Stein, so features that together carry no more evidence than
// chance add nothing). The hedge is also volatility managed (Moreira and Muir 2017): when the
// book's predicted variance exceeds its long-run variance, exposure is cut to their ratio. The
// variance forecast is the exponentially weighted variance whose half-life (among the model's
// horizons) has predicted next-day variance best so far. The hedge is the larger of the two; it
// trades the most market-correlated series that has bars, so it continues when a series ends. Days the continuation
// series jumps at a roll (its move against the stock market's is an outlier by Chauvenet's
// criterion) count as the stock market's move.

#include <map>
#include <string>
#include <vector>

#include "ofm/data.hpp"
#include "ofm/model.hpp"
#include "ofm/structure.hpp"

namespace ofm {

/// Market-standard transaction costs, in percent of the value traded.
constexpr double kDealingPct = 0.10;  ///< per side: commission and half the spread
constexpr double kStampPct = 0.50;    ///< UK stamp duty on purchases (0 for CFDs, spread bets, AIM shares)

struct Costs {
  double buyBps = 100 * (kDealingPct + kStampPct);  ///< dealing plus stamp duty on purchases
  double sellBps = 100 * kDealingPct;
  double futuresBps = 1;  ///< per unit of futures notional traded
  /// From percentages: dealing per side, stamp duty on purchases.
  static Costs fromPercent(double dealingPct, double stampPct, double futuresBps = 1) {
    Costs c;
    c.buyBps = 100 * (dealingPct + stampPct), c.sellBps = 100 * dealingPct, c.futuresBps = futuresBps;
    return c;
  }
};

struct Metrics {
  double annualReturn = 0, volatility = 0, sharpe = 0, maxDrawdown = 0, days = 0;
};
Metrics metrics(const std::vector<double>& daily);

struct Trade {
  std::size_t asset = 0, entryDay = 0, exitDay = 0;
  double ret = 0;  ///< net of costs
};

struct Backtest {
  std::size_t start = 0;           ///< first day with a return (the day after the first decision)
  std::vector<double> model, hedged, market;  ///< daily net returns from `start`
  std::vector<double> holdings, hedge, turnover;
  std::vector<Trade> trades;
  std::vector<double> life;        ///< expected life of a forecast at each day (days), from `start`
  std::vector<int> held;           ///< at the last close: 1 held
  std::vector<int> orders;         ///< at the last close: +1 buy, -1 sell, 0 none
  std::string hedgeSeries;
};

/// ms: the market's topology and geometry (structure.hpp), added to the hedge forecast's features.
Backtest backtest(const Market& m, const Forecast& f, const Costs& costs, const std::map<std::string, std::vector<double>>& series,
                  const MarketStructure* ms = nullptr);

/// The series used for the hedge: the one whose daily moves were most correlated with the stock
/// market's before the first trading day (no later data is looked at); empty if none overlaps.
std::string pickHedgeSeries(const Market& m, const std::map<std::string, std::vector<double>>& series, std::size_t before);
/// All series positively correlated with the market before `before`, most correlated first.
std::vector<std::string> rankHedgeSeries(const Market& m, const std::map<std::string, std::vector<double>>& series, std::size_t before);

}  // namespace ofm
