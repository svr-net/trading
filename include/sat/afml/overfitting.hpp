#pragma once

#include <cstddef>
#include <vector>

#include "sat/adaptive/self_adaptive.hpp"
#include "sat/afml/backtest_stats.hpp"

namespace sat::afml {

/// How much of a strategy's backtest performance survives the search that found it.
struct StrategyAssessment {
  double sharpe = 0.0;            ///< per period
  double annualSharpe = 0.0;
  double skew = 0.0, kurt = 3.0;
  double psr = 0.0;               ///< P(true Sharpe > 0)
  double dsr = 0.0;               ///< P(true Sharpe > expected maximum of the trials)
  DrawdownStats drawdown;
  double concentration = 0.0;     ///< of positive returns
};

StrategyAssessment assessStrategy(const std::vector<double>& returns, double trials, double trialVariance);

/// The candidate pool as a multiple-testing problem: every candidate (model x rule) is a
/// trial. Reports the expected maximum Sharpe ratio of that many unskilled trials, the
/// deflated Sharpe ratios of the best fixed candidate and of the self-adaptive strategy,
/// and the probability of backtest overfitting of picking the best candidate in sample.
struct OverfittingReport {
  std::size_t trials = 0, days = 0;
  double trialVariance = 0.0;       ///< variance of the candidates' per-period Sharpe ratios
  double expectedMaxSharpe = 0.0;   ///< per period
  std::vector<double> candidateSharpe;  ///< per period
  std::size_t bestFixed = 0;
  StrategyAssessment best, adaptive;
  PboResult pbo;
};

OverfittingReport assessOverfitting(const CandidateBook& book, const std::vector<double>& adaptiveNet, std::size_t evalFrom,
                                    std::size_t blocks = 16);

}  // namespace sat::afml
