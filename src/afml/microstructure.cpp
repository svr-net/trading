#include "sat/afml/microstructure.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace sat::afml {

namespace {

void checkWindow(std::size_t window) {
  if (window < 3) throw std::invalid_argument("microstructure features need a window of at least 3 days");
}

}  // namespace

Panel rollSpread(const Panel& close, std::size_t window) {
  checkWindow(window);
  Panel out = close.like();
  for (std::size_t i = 0; i < close.assets(); ++i)
    for (std::size_t t = window + 1; t < close.dates(); ++t) {
      // Covariance of consecutive price changes over the window, relative to the price.
      std::vector<double> a, b;
      for (std::size_t k = t - window + 1; k <= t; ++k) {
        a.push_back((close(k, i) - close(k - 1, i)) / close(t, i));
        b.push_back((close(k - 1, i) - close(k - 2, i)) / close(t, i));
      }
      double ma = 0, mb = 0, cov = 0;
      for (std::size_t k = 0; k < a.size(); ++k) {
        ma += a[k] / a.size();
        mb += b[k] / b.size();
      }
      for (std::size_t k = 0; k < a.size(); ++k) cov += (a[k] - ma) * (b[k] - mb) / (a.size() - 1);
      out(t, i) = 2.0 * std::sqrt(std::max(0.0, -cov));
    }
  return out;
}

Panel corwinSchultzSpread(const Panel& high, const Panel& low, std::size_t window) {
  checkWindow(window);
  Panel daily = high.like(), out = high.like();
  const double k = 3.0 - 2.0 * std::sqrt(2.0);
  for (std::size_t i = 0; i < high.assets(); ++i) {
    for (std::size_t t = 1; t < high.dates(); ++t) {
      const double l1 = std::log(high(t, i) / low(t, i)), l0 = std::log(high(t - 1, i) / low(t - 1, i));
      const double beta = l1 * l1 + l0 * l0;
      const double hl2 = std::log(std::max(high(t, i), high(t - 1, i)) / std::min(low(t, i), low(t - 1, i)));
      const double gamma = hl2 * hl2;
      const double alpha = (std::sqrt(2.0 * beta) - std::sqrt(beta)) / k - std::sqrt(gamma / k);
      daily(t, i) = std::max(0.0, 2.0 * (std::exp(alpha) - 1.0) / (1.0 + std::exp(alpha)));
    }
    for (std::size_t t = window; t < high.dates(); ++t) {
      double s = 0;
      for (std::size_t q = t - window + 1; q <= t; ++q) s += daily(q, i);
      out(t, i) = s / static_cast<double>(window);
    }
  }
  return out;
}

Panel amihudIlliquidity(const Panel& close, const Panel& volume, std::size_t window) {
  checkWindow(window);
  Panel out = close.like();
  for (std::size_t i = 0; i < close.assets(); ++i)
    for (std::size_t t = window; t < close.dates(); ++t) {
      double s = 0;
      std::size_t n = 0;
      for (std::size_t q = t - window + 1; q <= t; ++q) {
        const double dv = close(q, i) * volume(q, i);
        if (dv > 0) {
          s += std::fabs(close(q, i) / close(q - 1, i) - 1.0) / (dv / 1e6);
          ++n;
        }
      }
      if (n) out(t, i) = s / static_cast<double>(n);
    }
  return out;
}

Panel kyleLambda(const Panel& close, const Panel& volume, std::size_t window) {
  checkWindow(window);
  Panel out = close.like();
  for (std::size_t i = 0; i < close.assets(); ++i)
    for (std::size_t t = window; t < close.dates(); ++t) {
      double sx = 0, sy = 0, sxx = 0, sxy = 0;
      const double n = static_cast<double>(window);
      for (std::size_t q = t - window + 1; q <= t; ++q) {
        const double r = close(q, i) / close(q - 1, i) - 1.0;
        const double x = (r > 0 ? 1.0 : (r < 0 ? -1.0 : 0.0)) * close(q, i) * volume(q, i) / 1e6;
        sx += x;
        sy += r;
        sxx += x * x;
        sxy += x * r;
      }
      const double den = sxx - sx * sx / n;
      if (den > 0) out(t, i) = (sxy - sx * sy / n) / den;
    }
  return out;
}

}  // namespace sat::afml
