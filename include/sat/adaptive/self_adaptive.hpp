#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "sat/core/matrix.hpp"
#include "sat/core/panel.hpp"
#include "sat/ml/walk_forward.hpp"
#include "sat/strategy/strategy.hpp"

namespace sat {

/// A fixed strategy driven by one model's predictions: one member of the pool the
/// self-adaptive strategy chooses from.
struct Candidate {
  std::size_t model = 0;
  StrategySpec strategy;
  std::string label;
};

/// Out-of-sample backtests of every candidate over the common prediction range of the models.
///
/// Day d is date start + d. Each candidate's gross return and turnover per day are stored,
/// and its weights on any day can be recomputed from the stored predictions (they only
/// depend on the predictions of its last rebalance date), which is what a switch between
/// candidates needs to price its turnover.
class CandidateBook {
 public:
  CandidateBook() = default;
  /// `stampBps`: charged on purchases on top of `costBps` (UK stamp duty).
  CandidateBook(const std::vector<ModelPredictions>& models, const std::vector<StrategySpec>& strategies,
                const Panel& nextReturns, double costBps, double stampBps = 0.0);

  /// The same book with the candidate backtests taken from the candidate-backtest kernel
  /// (WebGPU or its CPU emulation): `kernelBook` is its read-back, [candidate][day][2] of
  /// gross return and turnover in f32, for a plan compiled from the same models, strategies,
  /// returns and cost. Everything downstream (net series, selectors) is unchanged.
  CandidateBook(const std::vector<ModelPredictions>& models, const std::vector<StrategySpec>& strategies,
                const Panel& nextReturns, double costBps, const std::vector<float>& kernelBook);

  std::size_t size() const { return candidates_.size(); }
  std::size_t days() const { return end_ - start_; }
  std::size_t start() const { return start_; }
  std::size_t end() const { return end_; }
  std::size_t assets() const { return nextReturns_.assets(); }
  double costBps() const { return costBps_; }
  double stampBps() const { return stampBps_; }
  const std::vector<Candidate>& candidates() const { return candidates_; }
  const std::vector<std::string>& modelNames() const { return modelNames_; }
  const std::vector<Panel>& probabilities() const { return probs_; }
  const std::vector<std::vector<std::uint32_t>>& ranks() const { return ranks_; }
  const Panel& nextReturns() const { return nextReturns_; }

  double gross(std::size_t c, std::size_t d) const { return gross_(c, d); }
  double turnover(std::size_t c, std::size_t d) const { return turnover_(c, d); }
  /// Purchases (sum of positive weight changes) of candidate c on day d.
  double buys(std::size_t c, std::size_t d) const {
    return std::max(0.0, 0.5 * (turnover_(c, d) + exposure_(c, d) - (d > 0 ? exposure_(c, d - 1) : 0.0)));
  }
  double net(std::size_t c, std::size_t d) const {
    return gross_(c, d) - costBps_ * 1e-4 * turnover_(c, d) - (stampBps_ > 0 ? stampBps_ * 1e-4 * buys(c, d) : 0.0);
  }
  std::vector<double> netSeries(std::size_t c, std::size_t from = 0) const;
  std::vector<double> turnoverSeries(std::size_t c, std::size_t from = 0) const;

  /// Weights of candidate c held over day d.
  void weights(std::size_t c, std::size_t d, double* w) const;

 private:
  std::vector<Candidate> candidates_;
  std::vector<std::string> modelNames_;
  std::vector<Panel> probs_;
  std::vector<std::vector<std::uint32_t>> ranks_;
  Panel nextReturns_;
  std::size_t start_ = 0, end_ = 0;
  double costBps_ = 0.0, stampBps_ = 0.0;
  Matrix gross_, turnover_, exposure_;  // candidates x days

  void setUp(const std::vector<ModelPredictions>& models, const std::vector<StrategySpec>& strategies);
};

/// How a candidate's recent record is scored.
enum class ScoreMetric : int { Return = 0, Sharpe = 1, Sortino = 2 };
ScoreMetric parseScoreMetric(const std::string& name);
std::string scoreMetricName(ScoreMetric m);

/// The self-adaptive rule.
///
/// Every `adaptEvery` days the selector scores each candidate on its net returns over the
/// previous `lookback` days and trades the best one (or an equal mix of the best `topM`)
/// until the next adaptation date. With `allowCash`, it stays out of the market when no
/// candidate scores above `minScore`, which is the risk control. Switching between
/// candidates pays the cost of the turnover between their portfolios.
struct SelectorSpec {
  std::size_t lookback = 63;
  std::size_t adaptEvery = 21;
  ScoreMetric metric = ScoreMetric::Sharpe;
  std::size_t topM = 1;
  bool allowCash = true;
  double minScore = 0.0;
  std::string label() const;
};

/// Score of a window with n returns, sum s1, sum of squares s2 and sum of squared losses sd.
/// The GPU kernel evaluates the same expression in single precision.
double windowScore(ScoreMetric metric, double n, double s1, double s2, double sd);

struct AdaptiveResult {
  SelectorSpec spec;
  std::size_t evalFrom = 0;           ///< first day traded (offset into the book)
  std::vector<double> gross, turnover, net;  ///< days evalFrom .. end
  std::vector<int> selection;         ///< candidate held each day (-1 = cash; first of topM)
  std::vector<std::size_t> adaptations;  ///< offsets of the adaptation dates
  std::size_t switches = 0;
  std::vector<double> share;          ///< fraction of days each candidate was held; last entry = cash
  PerformanceMetrics metrics;
};

/// Runs the selector over the book from day `evalFrom` (default: the first full look-back).
AdaptiveResult runSelector(const CandidateBook& book, const SelectorSpec& spec, std::size_t evalFrom = SIZE_MAX);

/// Common out-of-sample dates [first, second) of a set of models.
std::pair<std::size_t, std::size_t> commonRange(const std::vector<ModelPredictions>& models);

/// Every candidate and every selector of a grid, evaluated over the same days.
/// This is what the GPU kernels compute (gpu::compile / gpu::summarise).
struct GridResult {
  std::size_t start = 0, days = 0, evalFrom = 0;
  std::vector<PerformanceMetrics> candidates;  ///< fixed strategies over days evalFrom .. days
  std::vector<PerformanceMetrics> selectors;
  std::vector<std::vector<double>> adaptiveNet;       ///< per selector, days evalFrom .. days
  std::vector<std::vector<double>> adaptiveTurnover;
  std::vector<std::vector<int>> selection;            ///< per selector and day (-1 = cash)
  double elapsedMs = 0.0;

  std::size_t switches(std::size_t s) const;
};

GridResult evaluateGrid(const CandidateBook& book, const std::vector<SelectorSpec>& selectors, std::size_t evalFrom);

}  // namespace sat
