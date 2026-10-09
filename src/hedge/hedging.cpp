#include "sat/hedge/hedging.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "sat/core/stats.hpp"

namespace sat::hedge {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}

std::vector<double> rollingBeta(const std::vector<double>& y, const std::vector<double>& x, std::size_t window) {
  if (y.size() != x.size()) throw std::invalid_argument("rolling beta: series differ in length");
  if (window < 5) throw std::invalid_argument("rolling beta needs a window of at least 5");
  std::vector<double> b(y.size(), kNaN);
  for (std::size_t t = window; t < y.size(); ++t) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0, n = 0;
    for (std::size_t k = t - window; k < t; ++k) {
      if (!std::isfinite(x[k]) || !std::isfinite(y[k])) continue;
      sx += x[k];
      sy += y[k];
      sxx += x[k] * x[k];
      sxy += x[k] * y[k];
      n += 1;
    }
    const double den = sxx - sx * sx / n;
    if (n >= 5 && den > 0) b[t] = (sxy - sx * sy / n) / den;
  }
  return b;
}

double minimumVarianceHedgeRatio(const std::vector<double>& y, const std::vector<double>& x) {
  const double rho = correlation(y, x), sy = stdev(y), sx = stdev(x);
  return std::isfinite(rho) && sx > 0 ? rho * sy / sx : 0.0;
}

KalmanRegression kalmanRegression(const std::vector<double>& y, const std::vector<double>& x, double delta, double ve) {
  if (y.size() != x.size()) throw std::invalid_argument("Kalman regression: series differ in length");
  if (!(delta > 0 && delta < 1) || !(ve > 0)) throw std::invalid_argument("Kalman regression: 0 < delta < 1 and a positive observation variance");
  const std::size_t n = y.size();
  KalmanRegression k;
  k.alpha.assign(n, kNaN);
  k.beta.assign(n, kNaN);
  k.forecastError.assign(n, kNaN);
  k.forecastVariance.assign(n, kNaN);
  // State theta = (beta, alpha); covariance P (2x2), state noise W = delta / (1 - delta) I.
  double b = 0.0, a = 0.0;
  double p00 = 0, p01 = 0, p11 = 0;
  const double w = delta / (1.0 - delta);
  bool started = false;
  for (std::size_t t = 0; t < n; ++t) {
    if (!std::isfinite(x[t]) || !std::isfinite(y[t])) continue;
    if (!started) {
      // Diffuse start: large prior variance.
      p00 = p11 = 1.0;
      started = true;
    }
    // Predict: theta stays, P grows by W.
    const double r00 = p00 + w, r01 = p01, r11 = p11 + w;
    k.beta[t] = b;
    k.alpha[t] = a;
    // Observation y = beta x + alpha: F = (x, 1).
    const double yhat = b * x[t] + a;
    const double q = x[t] * x[t] * r00 + 2 * x[t] * r01 + r11 + ve;
    const double e = y[t] - yhat;
    k.forecastError[t] = e;
    k.forecastVariance[t] = q;
    // Update.
    const double k0 = (r00 * x[t] + r01) / q, k1 = (r01 * x[t] + r11) / q;
    b += k0 * e;
    a += k1 * e;
    // P = R - K F R
    const double fr0 = x[t] * r00 + r01, fr1 = x[t] * r01 + r11;
    p00 = r00 - k0 * fr0;
    p01 = r01 - k0 * fr1;
    p11 = r11 - k1 * fr1;
  }
  return k;
}

std::vector<double> applyHedge(const std::vector<double>& y, const std::vector<double>& x, const std::vector<double>& beta, double costBps) {
  if (y.size() != x.size() || y.size() != beta.size()) throw std::invalid_argument("hedge: series differ in length");
  std::vector<double> out(y.size());
  double prev = 0.0;
  for (std::size_t t = 0; t < y.size(); ++t) {
    const double b = std::isfinite(beta[t]) ? beta[t] : 0.0;
    out[t] = y[t] - b * (std::isfinite(x[t]) ? x[t] : 0.0) - costBps * 1e-4 * std::fabs(b - prev);
    prev = b;
  }
  return out;
}

std::vector<double> volatilityTarget(const std::vector<double>& r, double target, double span, double maxLeverage, std::vector<double>* leverage) {
  if (!(target > 0) || !(span >= 2) || !(maxLeverage > 0)) throw std::invalid_argument("volatility target: positive target, span >= 2, positive cap");
  const double alpha = 2.0 / (span + 1.0);
  std::vector<double> out(r.size()), lev(r.size());
  double var = 0.0, weight = 0.0;
  for (std::size_t t = 0; t < r.size(); ++t) {
    // Forecast from returns before t; until there are 5 of them, no position.
    const double vol = weight > 0 ? std::sqrt(var / weight * 252.0) : 0.0;
    lev[t] = t >= 5 && vol > 0 ? std::min(maxLeverage, target / vol) : 0.0;
    out[t] = lev[t] * r[t];
    if (std::isfinite(r[t])) {
      var = (1 - alpha) * var + alpha * r[t] * r[t];
      weight = (1 - alpha) * weight + alpha;
    }
  }
  if (leverage) *leverage = lev;
  return out;
}

std::vector<double> kellyScale(const std::vector<double>& r, std::size_t window, double fraction, double maxLeverage, std::vector<double>* leverage) {
  if (window < 10) throw std::invalid_argument("Kelly sizing needs a window of at least 10 days");
  std::vector<double> out(r.size(), 0.0), lev(r.size(), 0.0);
  for (std::size_t t = window; t < r.size(); ++t) {
    const std::vector<double> past(r.begin() + static_cast<std::ptrdiff_t>(t - window), r.begin() + static_cast<std::ptrdiff_t>(t));
    const double m = mean(past), s = stdev(past);
    lev[t] = s > 0 ? std::clamp(fraction * m / (s * s), 0.0, maxLeverage) : 0.0;
    out[t] = lev[t] * r[t];
  }
  if (leverage) *leverage = lev;
  return out;
}

}  // namespace sat::hedge
