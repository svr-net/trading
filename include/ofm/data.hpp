#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace ofm {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// Dates x assets, row-major.
struct Panel {
  std::size_t T = 0, N = 0;
  std::vector<double> v;
  Panel() = default;
  Panel(std::size_t t, std::size_t n, double fill = kNaN) : T(t), N(n), v(t * n, fill) {}
  double& operator()(std::size_t t, std::size_t i) { return v[t * N + i]; }
  double operator()(std::size_t t, std::size_t i) const { return v[t * N + i]; }
};

/// Daily bars of a stock universe.
struct Market {
  std::vector<std::string> dates, tickers;
  Panel open, high, low, close, volume;
  std::size_t T() const { return close.T; }
  std::size_t N() const { return close.N; }
};

/// Parses the long CSV format `date,ticker,open,high,low,close,volume` (header row, any column
/// order). Dates sort as strings (ISO). Missing bars are NaN.
Market parseMarketCsv(const std::string& text);

/// Close prices of named series (futures, indices, currencies) from the same long format,
/// carried forward onto the market's dates (never backward); NaN before a series starts and
/// after its last bar.
std::map<std::string, std::vector<double>> parseSeriesCsv(const std::string& text, const std::vector<std::string>& dates);

/// A synthetic market for the demo and the tests: a market factor, a slow momentum effect, a
/// short-term reversal and a low-volatility premium are built in, with intraday ranges and
/// volumes; and an index future that follows the market. The model is not told any of this.
struct Synthetic {
  Market market;
  std::map<std::string, std::vector<double>> series;
};
Synthetic syntheticMarket(std::uint64_t seed = 7, std::size_t assets = 120, std::size_t days = 2268);

}  // namespace ofm
