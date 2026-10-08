#include "sat/afml/overfitting.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sat/core/stats.hpp"
#include "sat/strategy/performance.hpp"

namespace sat::afml {

StrategyAssessment assessStrategy(const std::vector<double>& r, double trials, double trialVariance) {
  StrategyAssessment a;
  a.sharpe = periodSharpe(r);
  a.annualSharpe = a.sharpe * std::sqrt(kTradingDaysPerYear);
  a.skew = skewness(r);
  a.kurt = kurtosis(r);
  if (!std::isfinite(a.skew)) a.skew = 0.0;
  if (!std::isfinite(a.kurt)) a.kurt = 3.0;
  const double n = static_cast<double>(r.size());
  a.psr = probabilisticSharpe(a.sharpe, 0.0, n, a.skew, a.kurt);
  a.dsr = deflatedSharpe(a.sharpe, n, a.skew, a.kurt, trials, trialVariance);
  a.drawdown = drawdownStats(r);
  a.concentration = returnConcentration(r, true);
  return a;
}

OverfittingReport assessOverfitting(const CandidateBook& book, const std::vector<double>& adaptiveNet, std::size_t evalFrom,
                                    std::size_t blocks) {
  const std::size_t C = book.size(), D = book.days();
  if (evalFrom >= D) throw std::invalid_argument("evaluation starts after the last out-of-sample date");
  OverfittingReport rep;
  rep.trials = C;
  rep.days = D - evalFrom;
  Matrix R(rep.days, C);
  for (std::size_t c = 0; c < C; ++c) {
    const auto net = book.netSeries(c, evalFrom);
    for (std::size_t d = 0; d < net.size(); ++d) R(d, c) = net[d];
    rep.candidateSharpe.push_back(periodSharpe(net));
  }
  const double sd = stdev(rep.candidateSharpe);
  rep.trialVariance = std::isfinite(sd) ? sd * sd : 0.0;
  rep.expectedMaxSharpe = expectedMaxSharpe(static_cast<double>(C), rep.trialVariance);
  rep.bestFixed = static_cast<std::size_t>(std::max_element(rep.candidateSharpe.begin(), rep.candidateSharpe.end()) - rep.candidateSharpe.begin());
  rep.best = assessStrategy(book.netSeries(rep.bestFixed, evalFrom), static_cast<double>(C), rep.trialVariance);
  // The selector is one strategy, but its rules (look-back, step, score) were also chosen;
  // deflating it by the same trials is conservative.
  rep.adaptive = assessStrategy(adaptiveNet, static_cast<double>(C), rep.trialVariance);
  if (C >= 2 && rep.days >= 2 * blocks) rep.pbo = probabilityOfBacktestOverfitting(R, blocks);
  return rep;
}

}  // namespace sat::afml
