#include "sat/afml/bet_sizing.hpp"

#include <algorithm>
#include <cmath>

#include "sat/core/stats.hpp"

namespace sat::afml {

double betSize(double p, std::size_t numClasses) {
  if (!std::isfinite(p)) return 0.0;
  p = std::clamp(p, 1e-9, 1.0 - 1e-9);
  const double z = (p - 1.0 / static_cast<double>(std::max<std::size_t>(2, numClasses))) / std::sqrt(p * (1.0 - p));
  return 2.0 * normalCdf(z) - 1.0;
}

double discretizeBet(double size, double step) {
  if (!(step > 0)) return size;
  return std::clamp(std::round(size / step) * step, -1.0, 1.0);
}

}  // namespace sat::afml
