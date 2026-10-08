#include "sat/strategy/performance.hpp"

#include <algorithm>
#include <cmath>

namespace sat {

SeriesMoments accumulateMoments(const std::vector<double>& daily, const std::vector<double>& turnover) {
  SeriesMoments m;
  double logW = 0.0, peak = 0.0;
  for (double r : daily) {
    m.n += 1;
    m.sum += r;
    m.sumSq += r * r;
    if (r < 0) m.sumDownSq += r * r;
    if (r > 0) m.wins += 1;
    logW += std::log(std::max(1.0 + r, 1e-12));
    peak = std::max(peak, logW);
    m.maxDrawdown = std::max(m.maxDrawdown, 1.0 - std::exp(logW - peak));
  }
  m.sumLog = logW;
  for (double t : turnover) m.turnover += t;
  return m;
}

PerformanceMetrics fromMoments(const SeriesMoments& m) {
  PerformanceMetrics p;
  p.days = static_cast<std::size_t>(m.n);
  if (m.n < 1) return p;
  const double n = m.n, mean = m.sum / n;
  const double var = n > 1 ? std::max(0.0, (m.sumSq - m.sum * m.sum / n) / (n - 1)) : 0.0;
  const double sd = std::sqrt(var), down = std::sqrt(m.sumDownSq / n), ann = std::sqrt(kTradingDaysPerYear);
  p.totalReturn = std::exp(m.sumLog) - 1.0;
  p.annualReturn = std::exp(m.sumLog * kTradingDaysPerYear / n) - 1.0;
  p.annualVolatility = sd * ann;
  p.sharpe = sd > 0 ? mean / sd * ann : 0.0;
  p.sortino = down > 0 ? mean / down * ann : 0.0;
  p.maxDrawdown = m.maxDrawdown;
  p.calmar = m.maxDrawdown > 0 ? p.annualReturn / m.maxDrawdown : 0.0;
  p.winRate = m.wins / n;
  p.averageTurnover = m.turnover / n;
  return p;
}

std::vector<double> equityCurve(const std::vector<double>& daily) {
  std::vector<double> w(daily.size() + 1, 1.0);
  for (std::size_t k = 0; k < daily.size(); ++k) w[k + 1] = w[k] * (1.0 + daily[k]);
  return w;
}

std::vector<double> drawdownCurve(const std::vector<double>& daily) {
  const auto w = equityCurve(daily);
  std::vector<double> dd(w.size());
  double peak = 0.0;
  for (std::size_t k = 0; k < w.size(); ++k) {
    peak = std::max(peak, w[k]);
    dd[k] = 1.0 - w[k] / peak;
  }
  return dd;
}

}  // namespace sat
