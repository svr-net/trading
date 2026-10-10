#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace sat {

// ADWIN (Bifet and Gavalda, 2007) over a growing series: the window [from, size) keeps every
// new value and drops its older part whenever some split shows the two parts' means differ by
// more than chance allows at confidence delta. The paper's bound assumes values in [0, 1]; its
// range term would dwarf daily market moves or information coefficients, so the scale-free
// form is used: a split is a change when |m0 - m1| > sqrt(2 ln(2 ln(n) / delta) (v0/n0 + v1/n1)),
// with each part at least 30 values long. At delta = 1e-4 this caught an 8x jump in volatility
// within about 19 days and a doubling within about 70, with no false cut in 25,000 simulated
// days of stationary noise.
class Adwin {
 public:
  explicit Adwin(double delta) : delta_(delta) { p1_.push_back(0.0), p2_.push_back(0.0); }

  void push(double v) {
    p1_.push_back(p1_.back() + v), p2_.push_back(p2_.back() + v * v);
    cut();
  }
  std::size_t size() const { return p1_.size() - 1; }
  std::size_t from() const { return from_; }
  double sum(std::size_t a, std::size_t b) const { return p1_[b] - p1_[a]; }
  double sumSquares(std::size_t a, std::size_t b) const { return p2_[b] - p2_[a]; }

 private:
  void cut() {
    constexpr std::size_t kMinSide = 30;
    for (bool changed = true; changed;) {
      changed = false;
      const std::size_t end = size(), n = end - from_;
      if (n < 2 * kMinSide) return;
      const double logTerm = std::log(2.0 * std::log(static_cast<double>(n)) / delta_);
      const std::size_t step = std::max<std::size_t>(1, n / 64);
      for (std::size_t k = from_ + kMinSide; k + kMinSide <= end; k += step) {
        const double n0 = static_cast<double>(k - from_), n1 = static_cast<double>(end - k);
        const double m0 = sum(from_, k) / n0, m1 = sum(k, end) / n1;
        const double v0 = std::max(0.0, sumSquares(from_, k) / n0 - m0 * m0);
        const double v1 = std::max(0.0, sumSquares(k, end) / n1 - m1 * m1);
        const double eps = std::sqrt(2.0 * logTerm * (v0 / n0 + v1 / n1));
        if (std::fabs(m0 - m1) > eps) {
          from_ = k;
          changed = true;
          break;
        }
      }
    }
  }

  double delta_;
  std::vector<double> p1_, p2_;
  std::size_t from_ = 0;
};

}  // namespace sat
