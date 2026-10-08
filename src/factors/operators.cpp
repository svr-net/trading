#include "sat/factors/operators.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "sat/core/stats.hpp"

namespace sat::ops {

namespace {

constexpr double kNaN = Panel::kMissing;

void sameShape(const Panel& a, const Panel& b) {
  if (a.dates() != b.dates() || a.assets() != b.assets()) throw std::invalid_argument("panel shapes differ");
}

template <class F>
Panel map(const Panel& a, F f) {
  Panel out = a.like();
  const auto& x = a.data();
  auto& y = out.data();
  for (std::size_t k = 0; k < x.size(); ++k) y[k] = f(x[k]);
  return out;
}

template <class F>
Panel zip(const Panel& a, const Panel& b, F f) {
  sameShape(a, b);
  Panel out = a.like();
  const auto& x = a.data();
  const auto& z = b.data();
  auto& y = out.data();
  for (std::size_t k = 0; k < x.size(); ++k) y[k] = f(x[k], z[k]);
  return out;
}

// Applies f(window pointer-free accessor) to each full window of each asset's series.
template <class F>
Panel rolling(const Panel& x, std::size_t d, F f) {
  if (d == 0) throw std::invalid_argument("window length must be positive");
  Panel out = x.like();
  std::vector<double> w(d);
  for (std::size_t i = 0; i < x.assets(); ++i)
    for (std::size_t t = d - 1; t < x.dates(); ++t) {
      bool ok = true;
      for (std::size_t k = 0; k < d; ++k) {
        w[k] = x(t + 1 - d + k, i);
        if (!std::isfinite(w[k])) {
          ok = false;
          break;
        }
      }
      out(t, i) = ok ? f(w) : kNaN;
    }
  return out;
}

}  // namespace

Panel operator+(const Panel& a, const Panel& b) { return zip(a, b, [](double x, double y) { return x + y; }); }
Panel operator-(const Panel& a, const Panel& b) { return zip(a, b, [](double x, double y) { return x - y; }); }
Panel operator*(const Panel& a, const Panel& b) { return zip(a, b, [](double x, double y) { return x * y; }); }
Panel operator/(const Panel& a, const Panel& b) {
  return zip(a, b, [](double x, double y) { return y == 0.0 ? kNaN : x / y; });
}
Panel operator+(const Panel& a, double b) { return map(a, [b](double x) { return x + b; }); }
Panel operator-(const Panel& a, double b) { return map(a, [b](double x) { return x - b; }); }
Panel operator*(const Panel& a, double b) { return map(a, [b](double x) { return x * b; }); }
Panel operator*(double a, const Panel& b) { return b * a; }
Panel operator/(const Panel& a, double b) { return map(a, [b](double x) { return b == 0.0 ? kNaN : x / b; }); }
Panel operator-(double a, const Panel& b) { return map(b, [a](double x) { return a - x; }); }
Panel operator-(const Panel& a) { return map(a, [](double x) { return -x; }); }

Panel lessThan(const Panel& a, const Panel& b) {
  return zip(a, b, [](double x, double y) { return std::isnan(x) || std::isnan(y) ? kNaN : (x < y ? 1.0 : 0.0); });
}
Panel lessThan(const Panel& a, double b) {
  return map(a, [b](double x) { return std::isnan(x) ? kNaN : (x < b ? 1.0 : 0.0); });
}
Panel greaterThan(double a, const Panel& b) {
  return map(b, [a](double x) { return std::isnan(x) ? kNaN : (a > x ? 1.0 : 0.0); });
}
Panel where(const Panel& cond, const Panel& a, const Panel& b) {
  sameShape(cond, a);
  sameShape(cond, b);
  Panel out = cond.like();
  for (std::size_t k = 0; k < out.data().size(); ++k) {
    const double c = cond.data()[k];
    out.data()[k] = std::isnan(c) ? kNaN : (c != 0.0 ? a.data()[k] : b.data()[k]);
  }
  return out;
}
Panel where(const Panel& cond, const Panel& a, double b) { return where(cond, a, cond.like(b)); }

Panel abs(const Panel& x) { return map(x, [](double v) { return std::fabs(v); }); }
Panel sign(const Panel& x) {
  return map(x, [](double v) { return std::isnan(v) ? kNaN : (v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0)); });
}
Panel log(const Panel& x) { return map(x, [](double v) { return v > 0.0 ? std::log(v) : kNaN; }); }
Panel signedPower(const Panel& x, double a) {
  return map(x, [a](double v) { return std::isnan(v) ? kNaN : (v < 0.0 ? -1.0 : 1.0) * std::pow(std::fabs(v), a); });
}
Panel power(const Panel& x, double a) {
  return map(x, [a](double v) {
    const double r = std::pow(v, a);
    return std::isfinite(r) ? r : kNaN;
  });
}

Panel rank(const Panel& x) {
  Panel out = x.like();
  std::vector<double> row(x.assets());
  for (std::size_t t = 0; t < x.dates(); ++t) {
    std::copy(x.row(t), x.row(t) + x.assets(), row.begin());
    const auto r = averageRanks(row);
    std::size_t n = 0;
    for (double v : r) n += std::isfinite(v) ? 1 : 0;
    for (std::size_t i = 0; i < x.assets(); ++i) out(t, i) = n ? r[i] / static_cast<double>(n) : kNaN;
  }
  return out;
}

Panel scale(const Panel& x, double a) {
  Panel out = x.like();
  for (std::size_t t = 0; t < x.dates(); ++t) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.assets(); ++i)
      if (std::isfinite(x(t, i))) s += std::fabs(x(t, i));
    for (std::size_t i = 0; i < x.assets(); ++i) out(t, i) = s > 0.0 ? a * x(t, i) / s : kNaN;
  }
  return out;
}

Panel delay(const Panel& x, std::size_t d) {
  Panel out = x.like();
  for (std::size_t t = d; t < x.dates(); ++t)
    for (std::size_t i = 0; i < x.assets(); ++i) out(t, i) = x(t - d, i);
  return out;
}

Panel delta(const Panel& x, std::size_t d) { return x - delay(x, d); }

Panel tsSum(const Panel& x, std::size_t d) {
  return rolling(x, d, [](const std::vector<double>& w) {
    double s = 0.0;
    for (double v : w) s += v;
    return s;
  });
}

Panel tsMean(const Panel& x, std::size_t d) { return tsSum(x, d) / static_cast<double>(d); }

Panel tsProduct(const Panel& x, std::size_t d) {
  return rolling(x, d, [](const std::vector<double>& w) {
    double p = 1.0;
    for (double v : w) p *= v;
    return p;
  });
}

Panel tsStddev(const Panel& x, std::size_t d) {
  return rolling(x, d, [](const std::vector<double>& w) {
    if (w.size() < 2) return kNaN;
    double m = 0.0;
    for (double v : w) m += v;
    m /= static_cast<double>(w.size());
    double s = 0.0;
    for (double v : w) s += (v - m) * (v - m);
    return std::sqrt(s / static_cast<double>(w.size() - 1));
  });
}

Panel tsMin(const Panel& x, std::size_t d) {
  return rolling(x, d, [](const std::vector<double>& w) { return *std::min_element(w.begin(), w.end()); });
}

Panel tsMax(const Panel& x, std::size_t d) {
  return rolling(x, d, [](const std::vector<double>& w) { return *std::max_element(w.begin(), w.end()); });
}

Panel tsArgMax(const Panel& x, std::size_t d) {
  return rolling(x, d, [](const std::vector<double>& w) {
    return static_cast<double>(std::max_element(w.begin(), w.end()) - w.begin() + 1);
  });
}

Panel tsRank(const Panel& x, std::size_t d) {
  return rolling(x, d, [](const std::vector<double>& w) {
    const double last = w.back();
    double below = 0.0, equal = 0.0;
    for (double v : w) {
      if (v < last) below += 1.0;
      else if (v == last) equal += 1.0;
    }
    return (below + 0.5 * (equal + 1.0)) / static_cast<double>(w.size());
  });
}

namespace {

Panel rollingPair(const Panel& x, const Panel& y, std::size_t d, bool corr) {
  sameShape(x, y);
  if (d < 2) throw std::invalid_argument("correlation window must be at least 2");
  Panel out = x.like();
  for (std::size_t i = 0; i < x.assets(); ++i)
    for (std::size_t t = d - 1; t < x.dates(); ++t) {
      double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
      bool ok = true;
      for (std::size_t k = t + 1 - d; k <= t; ++k) {
        const double a = x(k, i), b = y(k, i);
        if (!std::isfinite(a) || !std::isfinite(b)) {
          ok = false;
          break;
        }
        sx += a;
        sy += b;
      }
      if (!ok) continue;
      const double n = static_cast<double>(d), mx = sx / n, my = sy / n;
      for (std::size_t k = t + 1 - d; k <= t; ++k) {
        const double a = x(k, i) - mx, b = y(k, i) - my;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
      }
      if (!corr) out(t, i) = sxy / (n - 1.0);
      else {
        const double scaleXy = std::sqrt(sxx * syy);
        // Constant windows (e.g. ranks that do not move) have no correlation.
        out(t, i) = scaleXy > 1e-14 * (1.0 + sxx + syy) ? std::clamp(sxy / scaleXy, -1.0, 1.0) : kNaN;
      }
    }
  return out;
}

}  // namespace

Panel tsCorr(const Panel& x, const Panel& y, std::size_t d) { return rollingPair(x, y, d, true); }
Panel tsCov(const Panel& x, const Panel& y, std::size_t d) { return rollingPair(x, y, d, false); }

}  // namespace sat::ops
