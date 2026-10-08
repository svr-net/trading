#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace sat {

/// A date x asset table of doubles (row-major by date), the basic object of the library.
///
/// Prices, volumes, factor values, labels and predicted probabilities are all panels:
/// value(t, i) is the quantity of asset i on trading date t. Missing values are NaN.
class Panel {
 public:
  static constexpr double kMissing = std::numeric_limits<double>::quiet_NaN();

  Panel() = default;
  Panel(std::size_t dates, std::size_t assets, double value = kMissing)
      : dates_(dates), assets_(assets), data_(dates * assets, value) {}

  std::size_t dates() const { return dates_; }
  std::size_t assets() const { return assets_; }
  bool empty() const { return data_.empty(); }

  double& operator()(std::size_t t, std::size_t i) { return data_[t * assets_ + i]; }
  double operator()(std::size_t t, std::size_t i) const { return data_[t * assets_ + i]; }

  double* row(std::size_t t) { return data_.data() + t * assets_; }
  const double* row(std::size_t t) const { return data_.data() + t * assets_; }

  std::vector<double>& data() { return data_; }
  const std::vector<double>& data() const { return data_; }

  /// Copy of the time series of asset i.
  std::vector<double> series(std::size_t i) const;

  /// Number of finite (non-NaN, non-infinite) entries.
  std::size_t countFinite() const;

  /// Same shape, every entry `value`.
  Panel like(double value = kMissing) const { return Panel(dates_, assets_, value); }

 private:
  std::size_t dates_ = 0;
  std::size_t assets_ = 0;
  std::vector<double> data_;
};

}  // namespace sat
