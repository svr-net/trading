#pragma once

#include <cstddef>
#include <vector>

namespace sat {

constexpr double kTradingDaysPerYear = 252.0;

/// Sufficient statistics of a daily return series, accumulated in one pass. The GPU
/// summary kernel accumulates exactly these, so both engines share fromMoments().
struct SeriesMoments {
  double n = 0, sum = 0, sumSq = 0, sumDownSq = 0, sumLog = 0, maxDrawdown = 0, wins = 0, turnover = 0;
};

SeriesMoments accumulateMoments(const std::vector<double>& daily, const std::vector<double>& turnover = {});

/// Risk and return of a daily strategy return series (net of costs).
struct PerformanceMetrics {
  std::size_t days = 0;
  double totalReturn = 0.0;
  double annualReturn = 0.0;      ///< geometric, 252 days a year
  double annualVolatility = 0.0;
  double sharpe = 0.0;            ///< mean / sd x sqrt(252), zero risk-free rate
  double sortino = 0.0;           ///< mean / downside deviation x sqrt(252)
  double maxDrawdown = 0.0;       ///< largest peak-to-trough loss of wealth, as a positive fraction
  double calmar = 0.0;            ///< annual return / max drawdown
  double winRate = 0.0;           ///< share of days with a positive return
  double averageTurnover = 0.0;   ///< mean daily turnover (sum of |weight changes|)
};

PerformanceMetrics fromMoments(const SeriesMoments& m);

inline PerformanceMetrics evaluatePerformance(const std::vector<double>& daily, const std::vector<double>& turnover = {}) {
  return fromMoments(accumulateMoments(daily, turnover));
}

/// Wealth path W_t = prod (1 + r), starting from 1 (one more entry than `daily`).
std::vector<double> equityCurve(const std::vector<double>& daily);

/// Drawdown path 1 - W_t / max W_s<=t (same length as equityCurve).
std::vector<double> drawdownCurve(const std::vector<double>& daily);

}  // namespace sat
