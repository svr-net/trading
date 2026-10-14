#pragma once

// The daily book: day trades on the model's forecasts, a second book beside the model portfolio
// for accounts that deal cheaply (CFDs, spread bets, futures; about 0.05% per side or less).
//
// Each close t, the stocks are taken down the ranking of expected return; a stock is booked only
// if the trade is expected to make money: its forecast beats the market, the network expects the
// trade to gain after costs, and its own record of day trades beats cash (self-adaptive gate);
// at most 10. Empty slots stay in cash. Each booked stock is
// bought at the next open with a tenth of the capital and sold the same day at its stop, its
// take-profit or the close. The levels are multiples of the stock's own open-to-close volatility,
// picked per stock by a small network that predicts the net log return of each of the 36 stop /
// take pairs from the stock's features at the close. Everything is learnt walk-forward:
//  - the network (shared by all stocks; two hidden layers of 16, tanh, Adam) learns after each
//    day from every eligible stock's 36 actual outcomes on that day's bar;
//  - each stock's record is an ADWIN window over the net log returns of its own day trade (taken
//    every day, booked or not), scored by its mean shrunk by its t-statistic, m * max(0, 1 - 1/t^2);
//    a record no better than noise scores 0, and the slot stays in cash.
// Fills: the stop, the take-profit or the close from the day's real high, low and close; when a
// bar touched both levels the stop is taken (worst case). Every trade pays the purchase and the
// sale cost.

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "ofm/data.hpp"
#include "ofm/model.hpp"
#include "ofm/trade.hpp"

namespace ofm {

namespace day {

constexpr double kNone = std::numeric_limits<double>::infinity();
/// Stop and take-profit multiples of the volatility; kNone: no level.
const std::vector<double>& grid();

struct Bar {
  double o, h, l, c, v;
};

/// One trade's net return: bought at the open, sold at the stop, the take-profit or the close.
/// worstCase: when both levels were touched, the stop is taken to come first. exit: 0 close,
/// 1 stop, 2 take-profit.
double tradeReturn(const Bar& b, double sig, double ks, double kt, double buy, double sell, bool worstCase, int* exit);

/// Running sums of a series.
struct Moments {
  double s = 0, s2 = 0, n = 0;
  void add(double x) { s += x, s2 += x * x, n += 1; }
  double mean() const { return n > 0 ? s / n : 0; }
  double sd() const;  ///< population
};

/// A two-hidden-layer perceptron (tanh), trained online by Adam on squared error.
struct Mlp {
  std::size_t nIn, nH, nOut;
  std::vector<double> w1, b1, w2, b2, w3, b3;
  std::vector<std::vector<double>> m, v;
  double step = 0;
  Mlp(std::size_t in, std::size_t h, std::size_t out);
  std::vector<double> predict(const std::vector<double>& x) const;
  /// One Adam step on a batch (mean squared error over all outputs).
  void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y);

 private:
  void forward(const std::vector<double>& x, std::vector<double>& h1, std::vector<double>& h2, std::vector<double>& y) const;
  std::vector<double>* param(std::size_t q);
};

/// Running standardisation of features (mean and variance of everything seen so far).
struct Scaler {
  std::vector<Moments> s;
  explicit Scaler(std::size_t n) : s(n) {}
  void add(const std::vector<double>& x);
  std::vector<double> apply(std::vector<double> x) const;
};

/// The self-adaptive selector's memory and score: an ADWIN window (Bifet and Gavalda 2007;
/// scale-free bound, confidence 1/horizon, each side at least the square root of the window) over
/// a record, which drops its older part when the record changes; the score is the window's mean
/// shrunk by its t-statistic, m * max(0, 1 - 1/t^2).
struct Adwin {
  std::vector<double> p1{0.0}, p2{0.0};
  std::size_t from = 0;
  double horizon = 1;  ///< one false cut expected over this many observations
  void push(double x);
  double score() const;
  std::size_t window() const { return p1.size() - 1 - from; }
};

}  // namespace day

struct DayOrder {
  std::size_t asset = 0;
  double expected = 0;      ///< the model's expected excess return over the market for the next day
  double tradeReturn = 0;   ///< the network's expected net return of the trade (after costs)
  double sigma = 0;         ///< the stock's open-to-close volatility (log)
  double stop = day::kNone, take = day::kNone;  ///< multiples of sigma (kNone: no level)
  double record = 0;        ///< the stock's record score (net log return a day, shrunk)
  std::size_t window = 0;   ///< days in its record's window
};

struct DayBook {
  double buy = 0, sell = 0;          ///< costs per side (fractions)
  std::size_t start = 0;             ///< first day with a return (as the backtest's)
  std::vector<double> ret;           ///< daily net returns from start
  std::vector<int> booked;           ///< stocks booked each day
  std::size_t trades = 0, stops = 0, takes = 0;
  std::vector<DayOrder> orders;      ///< for the next open, best expected first
  /// The screen at the last close: stocks with a bar and a forecast, and how many pass each test.
  struct Screen {
    std::size_t stocks = 0, bars = 0, flatBars = 0, candidates = 0, forecastUp = 0, gainExpected = 0, recordUp = 0, all = 0;
  } screen;
};

/// The daily book from day `start` to the last day, and its orders for the next open.
DayBook dayBook(const Market& m, const Forecast& f, std::size_t start, double buy, double sell);

}  // namespace ofm
