#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sat::afml {

/// One trade: time in days since the start (fractional), price, size and aggressor side
/// (+1 buyer-initiated, -1 seller-initiated).
struct Trade {
  double time = 0.0;
  double price = 0.0;
  double volume = 0.0;
  int side = 0;
};

/// A synthetic trade stream in which activity varies a lot from day to day.
///
/// Each day draws a log-normal activity level. The number of trades scales with it, and
/// every trade moves the price by a draw of the same size, so the variance of a day's
/// return is proportional to its activity. Sampling by clock time then mixes calm and busy
/// periods (fat tails, unstable variance), while sampling every so many trades, shares or
/// dollars is closer to i.i.d. normal: the motivation for information-driven bars.
/// Order flow is persistent (the next aggressor repeats the last with probability
/// `persistence`) and moves the price, which is what imbalance bars detect.
struct TradeStreamSpec {
  std::size_t days = 60;
  double tradesPerDay = 1500.0;
  double activityDispersion = 0.6;  ///< standard deviation of log activity
  double dailyVolatility = 0.02;    ///< at average activity
  double startPrice = 50.0;
  double meanSize = 300.0;
  double persistence = 0.6;
  double impact = 0.25;             ///< share of a trade's move in the aggressor's direction
  std::uint64_t seed = 5;
};

std::vector<Trade> generateTrades(const TradeStreamSpec& spec);

struct Bar {
  double timeOpen = 0, timeClose = 0;
  double open = 0, high = 0, low = 0, close = 0;
  double volume = 0, dollarVolume = 0, vwap = 0;
  std::size_t ticks = 0;
  double imbalance = 0;  ///< sum of trade signs
};

/// Bars closing every `interval` days of clock time (empty intervals produce no bar).
std::vector<Bar> timeBars(const std::vector<Trade>& trades, double interval);
/// Bars of `count` trades each.
std::vector<Bar> tickBars(const std::vector<Trade>& trades, std::size_t count);
/// Bars closing once `threshold` shares have traded.
std::vector<Bar> volumeBars(const std::vector<Trade>& trades, double threshold);
/// Bars closing once `threshold` of notional (price x size) has traded.
std::vector<Bar> dollarBars(const std::vector<Trade>& trades, double threshold);
/// Tick imbalance bars: a bar closes when the absolute running sum of trade signs exceeds
/// its expected value, E[T] |2 P[b = +1] - 1|, where the expected bar length E[T] and the
/// expected sign are exponentially weighted averages over past bars (`span` bars).
/// The first expectation uses `initialTicks`. Two safeguards keep the bar length from
/// running away: the threshold is at least sqrt(E[T] v), with v the observed variance of the
/// imbalance per tick (what an unremarkable flow reaches in about E[T] ticks, persistent or
/// not), and E[T] is clamped to [initialTicks / 4, initialTicks * 4].
std::vector<Bar> tickImbalanceBars(const std::vector<Trade>& trades, std::size_t initialTicks, double span = 20.0);

/// How close a bar series' log returns are to i.i.d. normal.
struct BarStatistics {
  std::size_t count = 0;
  double barsPerDayMean = 0, barsPerDaySd = 0;
  double returnSd = 0;
  double skewness = 0, kurtosis = 0;
  double jarqueBera = 0;          ///< n/6 (S^2 + (K - 3)^2 / 4): 0 for a normal sample
  double serialCorrelation = 0;   ///< first-order autocorrelation of returns
  double varianceOfVariance = 0;  ///< dispersion of the variance across 10 equal subsamples, relative to the mean variance
};

BarStatistics barStatistics(const std::vector<Bar>& bars, std::size_t days);

/// Log returns between consecutive bar closes.
std::vector<double> barReturns(const std::vector<Bar>& bars);

}  // namespace sat::afml
