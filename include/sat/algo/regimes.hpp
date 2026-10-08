#pragma once

#include <cstddef>
#include <vector>

namespace sat::algo {

/// Gaussian hidden Markov model of daily returns (Hamilton's regime-switching model):
/// K states, each with its own mean and volatility, switching by a Markov chain.
struct HiddenMarkovModel {
  std::vector<double> mean, sd;               ///< per state
  std::vector<std::vector<double>> transition;  ///< row-stochastic
  std::vector<double> initial;
  double logLikelihood = 0.0;
  std::size_t iterations = 0;

  std::size_t states() const { return mean.size(); }
};

/// Fits the model by the Baum-Welch (EM) algorithm, starting from states split by
/// volatility quantiles. States are returned ordered by volatility (state 0 calmest).
HiddenMarkovModel fitHmm(const std::vector<double>& returns, std::size_t states = 2, std::size_t maxIterations = 200,
                         double tolerance = 1e-7);

/// Filtered state probabilities P(state at t | returns up to t): what is known in real time.
/// Row t holds the K probabilities.
std::vector<std::vector<double>> filterHmm(const HiddenMarkovModel& model, const std::vector<double>& returns);

/// Online regime detection and regime-dependent exposure: the model is re-fitted every
/// `refitEvery` days on the previous `window` days, and each day's exposure is set from the
/// filtered probability of the most volatile state at the previous close:
/// exposure = 1 when that probability is below `threshold`, `riskOffExposure` otherwise.
struct RegimeSwitchSpec {
  std::size_t states = 2, window = 504, refitEvery = 63;
  double threshold = 0.5, riskOffExposure = 0.0, costBps = 5.0;
};

struct RegimeSwitchResult {
  std::vector<double> returns, exposure, highVolProbability;
  std::size_t start = 0;  ///< input index of returns[0] (the first day after the estimation window)
  HiddenMarkovModel lastModel;
};

RegimeSwitchResult regimeSwitch(const std::vector<double>& returns, const RegimeSwitchSpec& spec);

}  // namespace sat::algo
