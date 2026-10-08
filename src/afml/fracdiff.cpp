#include "sat/afml/fracdiff.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "sat/core/stats.hpp"

namespace sat::afml {

std::vector<double> fracDiffWeights(double d, double threshold, std::size_t maxWidth) {
  std::vector<double> w = {1.0};
  for (std::size_t k = 1; k < maxWidth; ++k) {
    const double next = -w.back() * (d - static_cast<double>(k) + 1.0) / static_cast<double>(k);
    if (std::fabs(next) < threshold) break;
    w.push_back(next);
  }
  return w;
}

std::vector<double> fracDiff(const std::vector<double>& x, double d, double threshold) {
  const auto w = fracDiffWeights(d, threshold);
  std::vector<double> out(x.size(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t t = w.size() - 1; t < x.size(); ++t) {
    double s = 0.0;
    bool ok = true;
    for (std::size_t k = 0; k < w.size(); ++k) {
      const double v = x[t - k];
      if (!std::isfinite(v)) {
        ok = false;
        break;
      }
      s += w[k] * v;
    }
    if (ok) out[t] = s;
  }
  return out;
}

AdfResult adfTest(const std::vector<double>& series, std::size_t lags) {
  std::vector<double> x;
  for (double v : series)
    if (std::isfinite(v)) x.push_back(v);
  if (x.size() < lags + 20) throw std::invalid_argument("ADF test needs at least 20 observations beyond the lags");
  std::vector<std::vector<double>> X;
  std::vector<double> y;
  for (std::size_t t = lags + 1; t < x.size(); ++t) {
    std::vector<double> row = {1.0, x[t - 1]};
    for (std::size_t i = 1; i <= lags; ++i) row.push_back(x[t - i] - x[t - i - 1]);
    X.push_back(row);
    y.push_back(x[t] - x[t - 1]);
  }
  std::vector<double> se;
  const auto beta = ols(X, y, &se);
  AdfResult r;
  r.lags = lags;
  r.observations = y.size();
  r.statistic = se[1] > 0 ? beta[1] / se[1] : 0.0;
  return r;
}

FracDiffScan scanFracDiff(const std::vector<double>& x, double step, double threshold, std::size_t lags) {
  if (!(step > 0 && step <= 1)) throw std::invalid_argument("fractional differentiation step must be in (0, 1]");
  FracDiffScan s;
  bool found = false;
  for (int k = 0;; ++k) {
    const double d = std::min(1.0, k * step);
    const auto y = fracDiff(x, d, threshold);
    std::vector<double> a, b;
    for (std::size_t t = 0; t < y.size(); ++t)
      if (std::isfinite(y[t])) {
        a.push_back(x[t]);
        b.push_back(y[t]);
      }
    s.d.push_back(d);
    s.width.push_back(fracDiffWeights(d, threshold).size());
    s.adf.push_back(b.size() >= lags + 20 ? adfTest(b, lags).statistic : std::numeric_limits<double>::quiet_NaN());
    s.correlation.push_back(correlation(a, b));
    if (!found && std::isfinite(s.adf.back()) && s.adf.back() < AdfResult::critical5) {
      s.minimumD = d;
      found = true;
    }
    if (d >= 1.0) break;
  }
  return s;
}

}  // namespace sat::afml
