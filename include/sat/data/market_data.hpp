#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "sat/core/panel.hpp"

namespace sat {

/// Daily open/high/low/close/volume/VWAP panels of a stock universe.
///
/// Rows are trading dates (oldest first) and columns are stocks. `regime` holds the
/// hidden market regime of each date when the data come from the synthetic generator,
/// and -1 for real data.
struct MarketData {
  std::vector<std::string> tickers;
  std::vector<std::string> dates;
  Panel open, high, low, close, volume, vwap;
  std::vector<int> regime;
  std::vector<std::string> regimeNames;

  std::size_t numDates() const { return close.dates(); }
  std::size_t numAssets() const { return close.assets(); }

  /// Close-to-close simple returns; the first row is NaN.
  Panel returns() const;

  /// Simple return from the close of t to the close of t + horizon; NaN where it runs past the end.
  Panel forwardReturns(std::size_t horizon = 1) const;

  /// Throws std::invalid_argument unless every panel has the same shape and prices are positive.
  void validate() const;

  /// The dates [from, to) of every panel.
  MarketData slice(std::size_t from, std::size_t to) const;
};

/// Parses daily bars in long CSV format with a header row:
///   date,ticker,open,high,low,close,volume[,vwap]
/// Column order is taken from the header (case-insensitive). Dates are sorted as strings
/// (ISO yyyy-mm-dd sorts correctly). Without a vwap column the typical price (H + L + C) / 3
/// is used. Gaps of a stock are filled forward with zero volume; stocks missing on the
/// first date are back-filled from their first quote.
MarketData parseCsv(const std::string& text);

/// The inverse of parseCsv.
std::string toCsv(const MarketData& data);

}  // namespace sat
