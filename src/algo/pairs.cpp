#include "sat/algo/pairs.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "sat/afml/fracdiff.hpp"
#include "sat/core/random.hpp"
#include "sat/core/stats.hpp"
#include "sat/hedge/hedging.hpp"

namespace sat::algo {

double halfLife(const std::vector<double>& s) {
  if (s.size() < 10) throw std::invalid_argument("half-life needs at least 10 observations");
  std::vector<std::vector<double>> X;
  std::vector<double> y;
  for (std::size_t t = 1; t < s.size(); ++t) {
    X.push_back({1.0, s[t - 1]});
    y.push_back(s[t] - s[t - 1]);
  }
  const auto b = ols(X, y);
  return b[1] < 0 ? -std::log(2.0) / b[1] : std::numeric_limits<double>::infinity();
}

Cointegration engleGranger(const std::vector<double>& y, const std::vector<double>& x) {
  if (y.size() != x.size() || y.size() < 30) throw std::invalid_argument("cointegration needs two series of the same length, at least 30");
  std::vector<std::vector<double>> X;
  for (double v : x) X.push_back({1.0, v});
  const auto b = ols(X, y);
  Cointegration c;
  c.intercept = b[0];
  c.hedgeRatio = b[1];
  std::vector<double> spread(y.size());
  for (std::size_t t = 0; t < y.size(); ++t) spread[t] = y[t] - b[1] * x[t] - b[0];
  c.adf = afml::adfTest(spread, 1).statistic;
  c.halfLife = halfLife(spread);
  return c;
}

void generatePair(const PairSpec& s, std::vector<double>& y, std::vector<double>& x) {
  if (s.days < 30 || !(s.halfLifeDays > 0)) throw std::invalid_argument("pair: at least 30 days and a positive half-life");
  Rng rng(s.seed);
  const double phi = std::exp(-std::log(2.0) / s.halfLifeDays);
  x.assign(s.days, 0.0);
  y.assign(s.days, 0.0);
  double px = 50.0, spread = 0.0;
  for (std::size_t t = 0; t < s.days; ++t) {
    px *= std::exp(s.priceVol * rng.normal());
    spread = phi * spread + s.spreadVol * px * rng.normal();
    x[t] = px;
    y[t] = s.beta * px + 5.0 + spread;
  }
}

PairsResult kalmanPairs(const std::vector<double>& y, const std::vector<double>& x, const PairsSpec& s) {
  if (y.size() != x.size()) throw std::invalid_argument("pairs: series differ in length");
  const auto k = hedge::kalmanRegression(y, x, s.delta, s.observationVariance);
  const std::size_t n = y.size();
  PairsResult out;
  out.beta = k.beta;
  out.zscore.assign(n, std::numeric_limits<double>::quiet_NaN());
  out.position.assign(n, 0);
  out.returns.assign(n, 0.0);
  int pos = 0;
  double held = 0.0;  // hedge ratio of the open position
  for (std::size_t t = 0; t + 1 < n; ++t) {
    if (std::isfinite(k.forecastError[t]) && k.forecastVariance[t] > 0) out.zscore[t] = k.forecastError[t] / std::sqrt(k.forecastVariance[t]);
    const double z = out.zscore[t];
    int next = pos;
    if (t >= s.warmup && std::isfinite(z)) {
      if (pos == 0) {
        if (z < -s.entryZ) next = 1;
        else if (z > s.entryZ) next = -1;
      } else if ((pos == 1 && z > -s.exitZ) || (pos == -1 && z < s.exitZ)) {
        next = 0;
      }
    }
    const double b = std::isfinite(k.beta[t]) ? k.beta[t] : 0.0;
    double turnover = 0.0;
    if (next != pos) {
      turnover = (pos != 0 ? 1.0 : 0.0) + (next != 0 ? 1.0 : 0.0);  // close and/or open, per unit gross
      if (next != 0) held = b;
      ++out.trades;
    }
    pos = next;
    out.position[t] = pos;
    const double gross = std::fabs(y[t]) + std::fabs(held * x[t]);
    const double pnl = pos == 0 || gross <= 0 ? 0.0 : pos * ((y[t + 1] - y[t]) - held * (x[t + 1] - x[t])) / gross;
    out.returns[t] = pnl - s.costBps * 1e-4 * turnover;
  }
  out.returns.pop_back();  // the last date has no next-day move
  return out;
}

}  // namespace sat::algo
