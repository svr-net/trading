#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sat/adaptive/self_adaptive.hpp"
#include "sat/ml/walk_forward.hpp"
#include "sat/strategy/strategy.hpp"

namespace sat::gpu {

/// The strategy-search pipeline for GPUs (WebGPU / WGSL).
///
/// The self-adaptive strategy needs the out-of-sample record of every candidate (model x
/// trading rule) and, for a robustness study, the run of every selector setting over that
/// record. Both are embarrassingly parallel:
///  1. candidate-backtest: one invocation per candidate walks through the days, sets its
///     weights on its rebalance dates from the day's predictions (staged once per
///     workgroup in shared memory), books gross return and turnover, and keeps prefix sums
///     of its net return, squared return and squared loss;
///  2. adaptive-select: one invocation per selector setting scores every candidate on each
///     adaptation date from the prefix sums in O(1), holds the best and prices the
///     turnover of every switch from the two candidates' weights;
///  3. series-summary: one invocation per series (candidates, then selectors) accumulates the
///     SeriesMoments of its evaluation days.
/// compile() packs everything the kernels read into one table; the host only uploads it,
/// dispatches the three kernels and reads back two arrays (FusedOutput), which summarise()
/// turns into a GridResult, the same structure evaluateGrid() computes on the CPU.
/// runFusedReference() executes the kernels on the CPU in single precision.

constexpr std::size_t kWorkgroupSize = 64;
constexpr std::size_t kMaxAssets = 128;     ///< stocks per day staged in workgroup memory
constexpr std::size_t kMaxModels = 8;
constexpr std::size_t kHeaderWords = 16;
constexpr std::size_t kCandidateStride = 4; ///< model, kind, param, holding
constexpr std::size_t kSelectorStride = 8;  ///< look-back, step, metric, allow cash, min score, padding
constexpr std::size_t kStats = 8;           ///< SeriesMoments: n, sum, sum sq, sum loss sq, sum log, max DD, wins, turnover
constexpr std::size_t kAdaptStride = 4;     ///< per selector and day: net, turnover, selection, gross

struct FusedPlan {
  std::size_t numModels = 0, numAssets = 0, numDays = 0, numCandidates = 0, numSelectors = 0, evalFrom = 0;
  std::size_t start = 0;  ///< first date of day 0
  double costBps = 0.0;

  std::vector<std::uint32_t> header;  ///< uniform block (cost as f32 bits)
  /// candidates [C][4] | selectors [S][8] | next returns [D][N] | probabilities [M][D][N] | ranks [M][D][N]
  std::vector<float> tables;

  std::size_t numEval() const { return numDays - evalFrom; }
  std::size_t numSeries() const { return numCandidates + numSelectors; }
  std::size_t candidateDispatch() const { return (numCandidates + kWorkgroupSize - 1) / kWorkgroupSize; }
  std::size_t selectorDispatch() const { return (numSelectors + kWorkgroupSize - 1) / kWorkgroupSize; }
  std::size_t summaryDispatch() const { return (numSeries() + kWorkgroupSize - 1) / kWorkgroupSize; }
  /// Buffer sizes in bytes.
  std::size_t bookBytes() const { return numCandidates * numDays * 8; }
  std::size_t prefixBytes() const { return numCandidates * (numDays + 1) * 16; }
  std::size_t adaptBytes() const { return std::max<std::size_t>(1, numSelectors * numEval()) * kAdaptStride * 4; }
  std::size_t statsBytes() const { return numSeries() * kStats * 4; }
};

/// What the GPU returns.
struct FusedOutput {
  std::vector<float> stats;  ///< [series][kStats]
  std::vector<float> adapt;  ///< [selector][day][kAdaptStride]
};

/// Why the kernels cannot run this grid, or an empty string if they can.
std::string limitation(const std::vector<ModelPredictions>& models, const std::vector<StrategySpec>& strategies,
                       const std::vector<SelectorSpec>& selectors);

/// Compiles the candidate pool (every model x strategy) and the selector grid.
/// Throws std::invalid_argument if limitation() is not empty.
FusedPlan compile(const std::vector<ModelPredictions>& models, const std::vector<StrategySpec>& strategies,
                  const Panel& nextReturns, double costBps, const std::vector<SelectorSpec>& selectors, std::size_t evalFrom);

/// Turns the read-back into the result evaluateGrid() computes.
GridResult summarise(const FusedPlan& plan, const FusedOutput& output);

/// The three kernels executed on the CPU in single precision, as the GPU runs them.
FusedOutput runFusedReference(const FusedPlan& plan);

/// WGSL sources of the kernels.
const std::string& candidateBacktestKernel();
const std::string& adaptiveSelectKernel();
const std::string& seriesSummaryKernel();

}  // namespace sat::gpu
