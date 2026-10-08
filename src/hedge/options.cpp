#include "sat/hedge/options.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sat/core/random.hpp"
#include "sat/core/stats.hpp"

namespace sat::hedge {

double blackScholes(double S, double K, double T, double r, double v, bool call) {
  if (T <= 0 || v <= 0) return std::max(call ? S - K : K - S, 0.0);
  const double sd = v * std::sqrt(T), d1 = (std::log(S / K) + (r + 0.5 * v * v) * T) / sd, d2 = d1 - sd;
  const double df = std::exp(-r * T);
  return call ? S * normalCdf(d1) - K * df * normalCdf(d2) : K * df * normalCdf(-d2) - S * normalCdf(-d1);
}

double blackScholesDelta(double S, double K, double T, double r, double v, bool call) {
  if (T <= 0 || v <= 0) return call ? (S > K ? 1.0 : 0.0) : (S < K ? -1.0 : 0.0);
  const double d1 = (std::log(S / K) + (r + 0.5 * v * v) * T) / (v * std::sqrt(T));
  return call ? normalCdf(d1) : normalCdf(d1) - 1.0;
}

DeltaHedgeResult simulateDeltaHedge(const DeltaHedgeSpec& s, std::size_t every) {
  if (every < 1 || s.paths < 2 || s.stepsPerYear < 1) throw std::invalid_argument("delta hedge: positive rebalancing interval, paths and steps");
  const auto steps = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(s.years * static_cast<double>(s.stepsPerYear))));
  const double dt = s.years / static_cast<double>(steps);
  Rng rng(s.seed);
  DeltaHedgeResult out;
  out.rebalanceEvery = every;
  out.premium = blackScholes(s.spot, s.strike, s.years, s.rate, s.impliedVol, true);
  for (std::size_t p = 0; p < s.paths; ++p) {
    double S = s.spot;
    double delta = blackScholesDelta(S, s.strike, s.years, s.rate, s.impliedVol, true);
    double cash = out.premium - delta * S - s.costBps * 1e-4 * delta * S;
    std::size_t rebalances = 1;
    for (std::size_t k = 1; k <= steps; ++k) {
      S *= std::exp((s.mu - 0.5 * s.vol * s.vol) * dt + s.vol * std::sqrt(dt) * rng.normal());
      cash *= std::exp(s.rate * dt);
      if (k % every == 0 && k < steps) {
        const double tau = s.years - static_cast<double>(k) * dt;
        const double d = blackScholesDelta(S, s.strike, tau, s.rate, s.impliedVol, true);
        cash -= (d - delta) * S + s.costBps * 1e-4 * std::fabs(d - delta) * S;
        delta = d;
        ++rebalances;
      }
    }
    // Close: deliver the call payoff, sell the shares.
    const double pnl = cash + delta * S - std::max(S - s.strike, 0.0);
    out.pnl.push_back(pnl / out.premium);
    out.rebalances = rebalances;
  }
  out.mean = mean(out.pnl);
  out.sd = stdev(out.pnl);
  return out;
}

OptionOverlayResult optionOverlay(const std::vector<double>& price, const OptionOverlaySpec& s) {
  if (s.tenorDays < 2 || s.volWindow < 5) throw std::invalid_argument("option overlay: tenor >= 2 and volatility window >= 5 days");
  const std::size_t T = price.size();
  OptionOverlayResult out;
  if (T < s.volWindow + s.tenorDays + 2) throw std::invalid_argument("option overlay: series too short");
  const double year = 252.0;
  double put = 0, call = 0, vol = 0, Kp = 0, Kc = 0;
  std::size_t rollDay = 0;
  auto marks = [&](std::size_t t, double& p, double& c) {
    const double tau = static_cast<double>(s.tenorDays - (t - rollDay)) / year;
    p = blackScholes(price[t], Kp, tau, s.rate, vol, false);
    c = blackScholes(price[t], Kc, tau, s.rate, vol, true);
  };
  double costSum = 0, incomeSum = 0;
  for (std::size_t t = s.volWindow; t + 1 < T; ++t) {
    if ((t - s.volWindow) % s.tenorDays == 0) {
      // Roll: buy the put (and sell the call) at today's close.
      std::vector<double> r;
      for (std::size_t k = t - s.volWindow + 1; k <= t; ++k) r.push_back(std::log(price[k] / price[k - 1]));
      vol = std::max(0.01, stdev(r) * std::sqrt(year) + s.volPremium);
      Kp = s.putMoneyness * price[t];
      Kc = s.callMoneyness * price[t];
      rollDay = t;
      marks(t, put, call);
      costSum += put / price[t];
      incomeSum += call / price[t];
      ++out.rolls;
    }
    double p1, c1;
    // Tomorrow's marks; at expiry the options pay their intrinsic value.
    if (t + 1 - rollDay >= s.tenorDays) {
      p1 = std::max(Kp - price[t + 1], 0.0);
      c1 = std::max(price[t + 1] - Kc, 0.0);
    } else {
      marks(t + 1, p1, c1);
    }
    const double s0 = price[t], s1 = price[t + 1];
    out.unhedged.push_back(s1 / s0 - 1.0);
    out.protectivePut.push_back((s1 + p1) / (s0 + put) - 1.0);
    out.collar.push_back((s1 + p1 - c1) / (s0 + put - call) - 1.0);
    put = p1;
    call = c1;
  }
  out.averagePutCost = out.rolls ? costSum / static_cast<double>(out.rolls) : 0.0;
  out.averageCallIncome = out.rolls ? incomeSum / static_cast<double>(out.rolls) : 0.0;
  return out;
}

}  // namespace sat::hedge
