#include "sat/algo/execution.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "sat/core/random.hpp"

namespace sat::algo {

namespace {

void validate(const ExecutionSpec& s) {
  if (s.periods == 0 || s.horizonDays <= 0 || s.shares <= 0) throw std::invalid_argument("execution needs shares, periods and a horizon");
  const double tau = s.horizonDays / static_cast<double>(s.periods);
  if (s.eta - 0.5 * s.gamma * tau <= 0) throw std::invalid_argument("temporary impact must exceed half the permanent impact per period");
}

}  // namespace

void scheduleCostVariance(const ExecutionSpec& s, const std::vector<double>& x, double& cost, double& var) {
  validate(s);
  if (x.size() != s.periods + 1) throw std::invalid_argument("schedule needs periods + 1 holdings");
  const double tau = s.horizonDays / static_cast<double>(s.periods);
  const double etaTilde = s.eta - 0.5 * s.gamma * tau;
  cost = 0.5 * s.gamma * x.front() * x.front();
  var = 0.0;
  for (std::size_t k = 1; k < x.size(); ++k) {
    const double n = x[k - 1] - x[k];
    cost += s.epsilon * std::fabs(n) + etaTilde / tau * n * n;
    var += s.sigma * s.sigma * tau * x[k] * x[k];
  }
}

ExecutionPlan almgrenChriss(const ExecutionSpec& s) {
  validate(s);
  if (s.riskAversion < 0) throw std::invalid_argument("risk aversion must be non-negative");
  const std::size_t N = s.periods;
  const double tau = s.horizonDays / static_cast<double>(N), T = s.horizonDays;
  const double etaTilde = s.eta - 0.5 * s.gamma * tau;
  ExecutionPlan p;
  p.kappa = std::acosh(1.0 + 0.5 * tau * tau * s.riskAversion * s.sigma * s.sigma / etaTilde) / tau;
  p.holdings.resize(N + 1);
  for (std::size_t j = 0; j <= N; ++j) {
    const double t = tau * static_cast<double>(j);
    // kappa T beyond ~700 overflows sinh; the ratio is then exp(-kappa t) to double precision.
    if (p.kappa * T < 1e-10)
      p.holdings[j] = s.shares * (1.0 - t / T);
    else if (p.kappa * T > 600)
      p.holdings[j] = j == N ? 0.0 : s.shares * std::exp(-p.kappa * t);
    else
      p.holdings[j] = s.shares * std::sinh(p.kappa * (T - t)) / std::sinh(p.kappa * T);
  }
  p.holdings[N] = 0.0;
  for (std::size_t k = 1; k <= N; ++k) p.trades.push_back(p.holdings[k - 1] - p.holdings[k]);
  scheduleCostVariance(s, p.holdings, p.expectedCost, p.variance);
  p.halfLifeDays = p.kappa > 0 ? 1.0 / p.kappa : std::numeric_limits<double>::infinity();
  return p;
}

std::vector<FrontierPoint> efficientFrontier(const ExecutionSpec& spec, double lo, double hi, std::size_t points) {
  if (!(lo > 0) || !(hi > lo) || points < 2) throw std::invalid_argument("frontier needs 0 < lambdaMin < lambdaMax and 2+ points");
  std::vector<FrontierPoint> out;
  for (std::size_t i = 0; i < points; ++i) {
    ExecutionSpec s = spec;
    s.riskAversion = lo * std::pow(hi / lo, static_cast<double>(i) / static_cast<double>(points - 1));
    const auto p = almgrenChriss(s);
    out.push_back({s.riskAversion, p.expectedCost, std::sqrt(p.variance)});
  }
  return out;
}

std::vector<double> simulateShortfall(const ExecutionSpec& s, const std::vector<double>& x, std::size_t paths, std::uint64_t seed) {
  validate(s);
  if (x.size() != s.periods + 1) throw std::invalid_argument("schedule needs periods + 1 holdings");
  const double tau = s.horizonDays / static_cast<double>(s.periods);
  Rng rng(seed);
  std::vector<double> out(paths);
  for (std::size_t p = 0; p < paths; ++p) {
    double S = s.price, proceeds = 0.0;
    for (std::size_t k = 1; k < x.size(); ++k) {
      const double n = x[k - 1] - x[k], v = n / tau;
      const double sign = n > 0 ? 1.0 : (n < 0 ? -1.0 : 0.0);
      // The k-th sale fills at the previous price less temporary impact; then the price
      // diffuses and absorbs the permanent impact of the trade.
      proceeds += n * (S - (s.epsilon * sign + s.eta * v));
      S += s.sigma * std::sqrt(tau) * rng.normal() - tau * s.gamma * v;
    }
    out[p] = x.front() * s.price - proceeds;
  }
  return out;
}

}  // namespace sat::algo
