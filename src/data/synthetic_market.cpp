#include "sat/data/synthetic_market.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "sat/core/random.hpp"

namespace sat {

std::vector<MarketRegime> SyntheticMarketSpec::defaultRegimes() {
  return {
      {"bull", 0.0008, 0.009, 0.12, 0.15},
      {"bear", -0.0010, 0.018, -0.06, 0.22},
      {"range", 0.0, 0.008, -0.16, 0.30},
  };
}

namespace {

// Business-day labels from 2018-01-02, skipping weekends (no holiday calendar).
std::vector<std::string> businessDays(std::size_t n) {
  static const int daysIn[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  int y = 2018, m = 1, d = 2, dow = 2;  // 2018-01-02 was a Tuesday (0 = Sunday)
  std::vector<std::string> out;
  while (out.size() < n) {
    if (dow != 0 && dow != 6) {
      char buf[40];
      std::snprintf(buf, sizeof buf, "%04d-%02d-%02d", y, m, d);
      out.emplace_back(buf);
    }
    dow = (dow + 1) % 7;
    const bool leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    if (++d > daysIn[m - 1] + (m == 2 && leap ? 1 : 0)) {
      d = 1;
      if (++m > 12) {
        m = 1;
        ++y;
      }
    }
  }
  return out;
}

}  // namespace

MarketData generateSyntheticMarket(const SyntheticMarketSpec& spec) {
  const std::size_t T = spec.numDates, N = spec.numAssets, R = spec.regimes.size();
  if (T < 2 || N < 1 || R < 1) throw std::invalid_argument("synthetic market needs dates, stocks and regimes");
  std::vector<std::vector<double>> P = spec.transition;
  if (P.empty()) {
    P.assign(R, std::vector<double>(R, R > 1 ? (1.0 - spec.persistence) / static_cast<double>(R - 1) : 0.0));
    for (std::size_t r = 0; r < R; ++r) P[r][r] = R > 1 ? spec.persistence : 1.0;
  }
  if (P.size() != R) throw std::invalid_argument("transition matrix must be regimes x regimes");
  for (const auto& row : P) {
    double s = 0.0;
    for (double p : row) {
      if (p < 0.0) throw std::invalid_argument("transition probabilities must be non-negative");
      s += p;
    }
    if (row.size() != R || std::fabs(s - 1.0) > 1e-6) throw std::invalid_argument("transition rows must sum to 1");
  }

  Rng rng(spec.seed);
  MarketData d;
  for (std::size_t i = 0; i < N; ++i) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%04zu.HK", i + 1);
    d.tickers.emplace_back(buf);
  }
  d.dates = businessDays(T);
  for (const auto& r : spec.regimes) d.regimeNames.push_back(r.name);
  d.open = d.high = d.low = d.close = d.volume = d.vwap = Panel(T, N);
  d.regime.assign(T, 0);

  std::vector<double> beta(N), idio(N), price(N), eps(N, 0.0), volShock(N, 0.0), baseVol(N);
  for (std::size_t i = 0; i < N; ++i) {
    beta[i] = std::max(0.2, 1.0 + spec.betaDispersion * rng.normal());
    idio[i] = spec.idiosyncraticVol * std::exp(0.25 * rng.normal());
    price[i] = spec.startPrice * std::exp(0.5 * rng.normal());
    baseVol[i] = spec.meanVolume * std::exp(0.6 * rng.normal());
  }
  std::size_t state = 0;
  for (std::size_t t = 0; t < T; ++t) {
    if (t > 0) {
      const double u = rng.uniform();
      double c = 0.0;
      std::size_t next = R - 1;
      for (std::size_t r = 0; r < R; ++r) {
        c += P[state][r];
        if (u < c) {
          next = r;
          break;
        }
      }
      state = next;
    }
    d.regime[t] = static_cast<int>(state);
    const MarketRegime& g = spec.regimes[state];
    const double market = t == 0 ? 0.0 : g.drift + g.volatility * rng.normal();
    for (std::size_t i = 0; i < N; ++i) {
      double ret = 0.0, e = 0.0, shock = 0.0;
      if (t > 0) {
        // Idiosyncratic AR(1) with regime-dependent autocorrelation; moves made on abnormal
        // volume partly revert the next day.
        const double reversal = g.volumeReversal * std::min(volShock[i], 1.5) / 1.5;
        e = (g.momentum - reversal) * eps[i] + idio[i] * rng.normal();
        ret = beta[i] * market + e;
        ret = std::clamp(ret, -0.35, 0.35);
        shock = std::max(0.0, 0.8 * rng.normal() + 25.0 * std::fabs(e) - 0.4);
      }
      eps[i] = e;
      volShock[i] = shock;
      const double prev = price[i];
      const double close = prev * (1.0 + ret);
      // Overnight gap takes part of the move, the session the rest (Brownian bridge range).
      const double gap = t == 0 ? 0.0 : 0.3 * ret + 0.002 * rng.normal();
      const double open = prev * std::exp(gap);
      const double sessionVol = (beta[i] * g.volatility + idio[i]) * 0.6;
      const double hi = std::max(open, close) * std::exp(std::fabs(sessionVol * rng.normal()) * 0.8);
      const double lo = std::min(open, close) * std::exp(-std::fabs(sessionVol * rng.normal()) * 0.8);
      const double w = std::clamp(0.25 * (open + hi + lo + close) * std::exp(0.001 * rng.normal()), lo, hi);
      d.open(t, i) = open;
      d.high(t, i) = hi;
      d.low(t, i) = lo;
      d.close(t, i) = close;
      d.vwap(t, i) = w;
      d.volume(t, i) = std::round(baseVol[i] * std::exp(0.35 * rng.normal() + 0.6 * shock));
      price[i] = close;
    }
  }
  d.validate();
  return d;
}

}  // namespace sat
