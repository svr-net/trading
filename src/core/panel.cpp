#include "sat/core/panel.hpp"

#include <cmath>

namespace sat {

std::vector<double> Panel::series(std::size_t i) const {
  std::vector<double> out(dates_);
  for (std::size_t t = 0; t < dates_; ++t) out[t] = (*this)(t, i);
  return out;
}

std::size_t Panel::countFinite() const {
  std::size_t n = 0;
  for (double v : data_) n += std::isfinite(v) ? 1 : 0;
  return n;
}

}  // namespace sat
