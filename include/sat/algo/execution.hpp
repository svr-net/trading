#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sat::algo {

/// Optimal liquidation of a block (Almgren-Chriss): sell `shares` over `periods` intervals of
/// `horizonDays / periods` days. Arithmetic Brownian price with volatility `sigma` (currency per
/// share per sqrt(day)), linear permanent impact gamma, and temporary impact
/// h(v) = epsilon + eta * v per share for a trading rate v (shares per day).
struct ExecutionSpec {
  double shares = 1e6, price = 50.0, horizonDays = 5.0;
  std::size_t periods = 5;
  double sigma = 0.95, epsilon = 0.0625, eta = 2.5e-6, gamma = 2.5e-7;
  double riskAversion = 1e-6;  ///< lambda in E + lambda * Var
};

struct ExecutionPlan {
  double kappa = 0.0;                ///< urgency (1/day); 0 means a straight line (TWAP)
  std::vector<double> holdings;      ///< x_0 = shares ... x_N = 0
  std::vector<double> trades;        ///< n_k = x_{k-1} - x_k, k = 1..N
  double expectedCost = 0.0, variance = 0.0;  ///< of the implementation shortfall
  double halfLifeDays = 0.0;         ///< 1 / kappa, the trade's characteristic time
};

/// Optimal trajectory x_j = X sinh(kappa (T - t_j)) / sinh(kappa T), with kappa from the
/// discrete-time relation cosh(kappa tau) - 1 = tau^2 lambda sigma^2 / (2 eta~), and its
/// expected cost and variance. lambda = 0 gives the straight-line (TWAP) schedule.
ExecutionPlan almgrenChriss(const ExecutionSpec& spec);

/// Expected cost and variance of an arbitrary schedule (holdings x_0..x_N).
void scheduleCostVariance(const ExecutionSpec& spec, const std::vector<double>& holdings, double& expectedCost, double& variance);

struct FrontierPoint {
  double riskAversion, expectedCost, sd;
};

/// The efficient frontier traced by sweeping lambda over [lambdaMin, lambdaMax] (log-spaced).
std::vector<FrontierPoint> efficientFrontier(const ExecutionSpec& spec, double lambdaMin, double lambdaMax, std::size_t points);

/// Monte Carlo implementation shortfall (arrival value minus proceeds) of a schedule under
/// the model's own dynamics: one value per path.
std::vector<double> simulateShortfall(const ExecutionSpec& spec, const std::vector<double>& holdings, std::size_t paths,
                                      std::uint64_t seed);

}  // namespace sat::algo
