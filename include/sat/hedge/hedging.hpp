#pragma once

#include <cstddef>
#include <vector>

namespace sat::hedge {

/// Beta of y on x over the `window` observations before each date (no look-ahead): the
/// hedge ratio known when the hedge for date t is put on. NaN until the window is full.
std::vector<double> rollingBeta(const std::vector<double>& y, const std::vector<double>& x, std::size_t window);

/// Minimum-variance hedge ratio rho sigma_y / sigma_x of a position y hedged with an
/// instrument x (Hull's optimal hedge ratio; equal to the regression slope).
double minimumVarianceHedgeRatio(const std::vector<double>& y, const std::vector<double>& x);

/// Time-varying regression y_t = alpha_t + beta_t x_t + e_t estimated by a Kalman filter in
/// which (alpha, beta) follow a random walk. `delta` sets how fast they may move (state noise
/// delta / (1 - delta) times the identity), `observationVariance` the noise of e_t.
/// beta[t] and alpha[t] are the one-step-ahead predictions from data before t, and
/// `forecastError` / `forecastVariance` the innovation of y_t and its variance.
struct KalmanRegression {
  std::vector<double> alpha, beta, forecastError, forecastVariance;
};

KalmanRegression kalmanRegression(const std::vector<double>& y, const std::vector<double>& x, double delta = 1e-4,
                                  double observationVariance = 1e-3);

/// Return of y with a short hedge of beta_t in x: y_t - beta_t x_t, minus `costBps` per unit
/// of change in the hedge ratio. Dates with an unknown beta are left unhedged.
std::vector<double> applyHedge(const std::vector<double>& y, const std::vector<double>& x, const std::vector<double>& beta,
                               double costBps = 0.0);

/// Volatility targeting: scales each day's return by target / forecast volatility, the
/// forecast being the exponentially weighted volatility of the returns before that day
/// (annualised, `span` days). Leverage is capped at `maxLeverage`; `leverage` receives it.
std::vector<double> volatilityTarget(const std::vector<double>& r, double targetAnnualVol, double span, double maxLeverage,
                                     std::vector<double>* leverage = nullptr);

/// Fractional Kelly sizing: leverage f = fraction x mean / variance of the returns over the
/// previous `window` days (0 until the window is full), clipped to [0, maxLeverage].
std::vector<double> kellyScale(const std::vector<double>& r, std::size_t window, double fraction, double maxLeverage,
                               std::vector<double>* leverage = nullptr);

}  // namespace sat::hedge
