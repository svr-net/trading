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
// Index futures hedge (optional): the index future's own expected next-day return, from its
// trends at the model's horizons (whitened by their expanding covariance, premia shrunk as for
// the stocks), decides a short hedge sized by the book's expanding beta to the future, when the
// expected fall over the signal's life exceeds the futures round trip. Days the continuation
// series jumps at a roll (its move against the stock market's is an outlier by Chauvenet's
// criterion) count as the stock market's move.

#include <map>
#include <string>
#include <vector>

#include "ofm/data.hpp"
#include "ofm/model.hpp"

namespace ofm {

struct Costs {
  double buyBps = 60;     ///< dealing plus stamp duty on purchases (UK: 10 + 50)
  double sellBps = 10;
  double futuresBps = 1;  ///< per unit of futures notional traded
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

Backtest backtest(const Market& m, const Forecast& f, const Costs& costs, const std::map<std::string, std::vector<double>>& series);

/// The series used for the hedge: the one whose daily moves were most correlated with the stock
/// market's before the first trading day (no later data is looked at); empty if none overlaps.
std::string pickHedgeSeries(const Market& m, const std::map<std::string, std::vector<double>>& series, std::size_t before);

}  // namespace ofm
