#pragma once

// The daily trades: up to 10 day trades at the next open, each bought at the open and sold the
// same day at its stop-loss, its take-profit or the close. Everything is learnt walk-forward from
// what is known at each close; nothing is fixed by hand.
//
//  - Volatility: each stock's open-to-close volatility is an exponentially weighted average of its
//    squared open-to-close log returns. The half-life (among the model's dyadic horizons) is the
//    one whose variance forecasts have scored best so far (QLIKE loss, pooled over the stocks).
//  - Intraday forecast: the next day's open-to-close return of each stock, in its volatility, by
//    least squares on its features at the close (the model's forecast, the day's move, the gap,
//    the range, the 5-day move, volume, the market's move), fitted on every stock and day so far.
//  - Stop and take-profit: every trade has both, from its predicted low and high at several
//    quantiles. Four sources predict them: the stock's own history of moves (their quantiles); the
//    same moves in the stock's average true range, times its true range at the close (Wilder's
//    volatility unit, New Concepts in Technical Trading Systems, 1978); a neural network (two hidden
//    layers of 32 rectified linear units, Adam) that learns from the actual lows and highs each
//    trade's fall and rise from the open (pinball loss), from its features and technical
//    indicators; and the same network also given Wilder's indicators (ATR, RSI, directional
//    movement, ADX, ADXR, Parabolic SAR). For each (stop, take-profit) pair of each source the net
//    outcome (after costs, worst-case fills) is fitted on the intraday forecast, or on it and the
//    stock's ADXR (Wilder's trend rating), whichever has forecast better out of sample; each stock
//    takes the pair with the best expected outcome.
//  - Opening noise: a stop must be further from the open than the stock's usual drop from the open
//    to the day's low (exponentially weighted, the volatility's half-life); closer stops cannot be
//    tested honestly on daily bars, which do not show the order of the day's moves.
//  - Level source and stop limit: which source sets the levels, and how far a stop may be from the
//    open, are learnt together. Each source and candidate limit (the same multiples of the day's
//    median volatility) keeps its own book; the book used is the one whose record is best so far
//    (ADWIN, shrunk by its t-statistic), and cash while none beats cash. The sources' forecast
//    losses are reported but do not choose: a closer forecast of the low and high need not trade
//    better.
//  - Selection: down the ranking of expected return, a stock is booked only if its trade is
//    expected to gain after costs and its own record of these trades (an ADWIN window, its mean
//    shrunk by its t-statistic, m * max(0, 1 - 1/t^2)) beats cash; the 10 strongest are taken.
//  - Capital: shared equally among the trades booked, scaled down when needed so the book's
//    expected daily volatility is no more than the market's (the equal-weight index's
//    open-to-close volatility); the rest stays in cash.
//  - Risk limits (DayRisk): a stop at most 5% below the open (a wider pair is not traded), and the
//    weights scaled down so that a day when every stop is hit loses at most 2% of the capital.
// Fills: the stop, the take-profit or the close from the day's real high, low and close; when a
// bar touched both levels, the stock's intraday path (5-minute bars, where fetched: the strongest
// candidates of recent days) says which came first; otherwise the stop is taken (worst case).
// Every trade pays the purchase and the sale cost.

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ofm/data.hpp"
#include "ofm/model.hpp"
#include "ofm/trade.hpp"

namespace ofm {

namespace day {

constexpr double kNone = std::numeric_limits<double>::infinity();

struct Bar {
  double o, h, l, c, v;
};

/// One bar of a stock's regular session (5 minutes), in time order: its high and low.
struct Tick {
  double h, l;
};
/// Intraday paths of some stocks on some days, keyed day * N + stock.
using Paths = std::unordered_map<std::size_t, std::vector<Tick>>;

/// One trade's net return: bought at the open, sold at the stop, the take-profit or the close.
/// When the day's bar touched both levels, the intraday path (if any) says which came first;
/// without one, or when both fell in the same tick, worstCase takes the stop first. exit: 0 close,
/// 1 stop, 2 take-profit. resolved: set when the path decided the order.
double tradeReturn(const Bar& b, double sig, double ks, double kt, double buy, double sell, bool worstCase, int* exit,
                   const std::vector<Tick>* path = nullptr, bool* resolved = nullptr);

/// Running sums of a series.
struct Moments {
  double s = 0, s2 = 0, n = 0;
  void add(double x) { s += x, s2 += x * x, n += 1; }
  double mean() const { return n > 0 ? s / n : 0; }
  double sd() const;  ///< population
};

/// Running standardisation of features (mean and variance of everything seen so far).
struct Scaler {
  std::vector<Moments> s;
  mutable double boundN = -1, bound = 0;  ///< Chauvenet's bound, cached for the count it was computed at
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
  double intraday = 0;      ///< forecast open-to-close return (log)
  double tradeReturn = 0;   ///< expected net return of the trade (after costs)
  double weight = 0;        ///< share of the capital
  double sigma = 0;         ///< the stock's open-to-close volatility (log)
  double stop = day::kNone, take = day::kNone;  ///< multiples of sigma (kNone: no level)
  double record = 0;        ///< the stock's record score (net log return a day, shrunk)
  std::size_t window = 0;   ///< days in its record's window
};

struct DayBook {
  double buy = 0, sell = 0;          ///< costs per side (fractions)
  std::size_t start = 0;             ///< first day with a return (as the backtest's)
  std::vector<double> ret;           ///< daily net returns from start (the stop first when a bar touched both levels)
  std::vector<double> optimistic;    ///< the same trades with the take-profit first (fill test)
  std::vector<int> booked;           ///< stocks booked each day
  std::vector<double> exposure;      ///< share of the capital invested each day
  std::uint32_t halfLife = 0;        ///< the volatility's half-life chosen at the last close
  double stopLimit = 0;              ///< the learnt stop limit at the last close (fraction of the open)
  double limitScore = 0;             ///< its book's record score
  std::size_t bookSize = 0;          ///< the book size: the n strongest trades
  bool backtestLoses = false;        ///< the daily trades lost over the backtest: no orders
  std::string levelsFrom;            ///< what sets the stop and take-profit levels
  /// Mean pinball loss a stock-day of the level predictors (0: not scored): history, Wilder's true
  /// range, the network, the network with Wilder's indicators.
  double levelLoss[4] = {0, 0, 0, 0};
  std::size_t sourceDays[4] = {0, 0, 0, 0};  ///< days each level predictor set the levels (as levelLoss)
  bool trendFits = false;            ///< the pairs' outcomes are fitted with ADXR (Wilder's trend rating)
  double pairErr[2] = {0, 0};        ///< out-of-sample mean squared error of the pair fits (in volatilities): plain, with ADXR
  double maxTarget = 0;              ///< largest |open-to-close return / volatility| learnt from (data check)
  std::size_t bigTargets = 0;        ///< how many were outliers by Chauvenet's criterion, cut to its bound
  /// Out-of-sample correlation of the intraday forecast with the outcome (in volatilities).
  struct Skill {
    double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    void add(double x, double y) { n += 1, sx += x, sy += y, sxx += x * x, syy += y * y, sxy += x * y; }
    double corr() const {
      const double vx = sxx - sx * sx / n, vy = syy - sy * sy / n;
      return n > 2 && vx > 0 && vy > 0 ? (sxy - sx * sy / n) / std::sqrt(vx * vy) : 0.0;
    }
  } skill;
  std::size_t trades = 0, stops = 0, takes = 0;
  std::size_t pathTrades = 0, pathResolved = 0;  ///< trades with an intraday path; of them, both levels touched and the path decided
  std::size_t pathDays = 0;                      ///< stock-days with an intraday path
  /// Each day's strongest candidates (day, stock): every stock any of the books could trade.
  std::vector<std::pair<std::size_t, std::size_t>> candidates;
  std::vector<DayOrder> orders;      ///< for the next open, best expected first
  /// The screen at the last close: stocks with a bar and a forecast, and how many pass each test.
  struct Screen {
    std::size_t stocks = 0, bars = 0, flatBars = 0, candidates = 0, forecastUp = 0, gainExpected = 0, recordUp = 0, all = 0;
  } screen;
};

/// Risk limits of the daily trades, applied to every book (backtest and orders alike).
struct DayRisk {
  double maxStop = 0.05;        ///< a stop at most this far below the open (fraction)
  double maxRiskReward = 0;     ///< a stop at most this many times as far from the open as the take-profit (0: no limit; tight
                                ///< ratios put stops in the day's noise: on the sample 1.5 cut the Sharpe ratio from 8 to 1.2)
  double maxDayLoss = 0.02;     ///< the book's loss if every stop is hit, at most this share of the capital (weights scaled down)
};

/// The daily trades from day `start` to the last day, and the orders for the next open.
/// paths: intraday paths (exact fills; on those stock-days a stop inside the opening noise can be
/// tested, so the floor does not apply). candidatesFrom: list each day's strongest candidates from
/// this day on (in book.candidates), for fetching their intraday paths.
DayBook dayBook(const Market& m, const Forecast& f, std::size_t start, double buy, double sell, const day::Paths* paths = nullptr,
                std::size_t candidatesFrom = std::numeric_limits<std::size_t>::max(), const DayRisk& risk = DayRisk{});

}  // namespace ofm
