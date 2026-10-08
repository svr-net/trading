#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "sat/core/random.hpp"

namespace sat::afml {

/// A label's span: it is set at t0 and depends on prices up to t1 (inclusive).
using Span = std::pair<std::size_t, std::size_t>;

/// Number of labels whose span covers each of the dates 0 .. numDates - 1.
std::vector<double> concurrency(const std::vector<Span>& spans, std::size_t numDates);

/// Average uniqueness of each label: the mean of 1 / concurrency over its span. Labels that
/// share their returns with many others carry less independent information.
std::vector<double> averageUniqueness(const std::vector<Span>& spans, const std::vector<double>& concurrency);

/// Weights proportional to the absolute return a label can claim for itself, sum of
/// r_t / c_t over its span (r: log returns of the date), scaled to sum to the label count.
std::vector<double> returnAttributionWeights(const std::vector<Span>& spans, const std::vector<double>& concurrency,
                                             const std::vector<double>& logReturns);

/// Time decay of weights along cumulative uniqueness (labels in chronological order):
/// the newest label keeps weight 1 and the oldest gets `oldest` (in [-1, 1]; negative
/// values give the oldest part of the sample zero weight).
std::vector<double> timeDecay(const std::vector<double>& uniqueness, double oldest);

/// Sequential bootstrap: draws labels one at a time with probability proportional to the
/// average uniqueness each would have given the labels already drawn, so that the sample
/// overlaps less than a standard bootstrap.
std::vector<std::size_t> sequentialBootstrap(const std::vector<Span>& spans, std::size_t numDates, std::size_t draws, Rng& rng);

/// Mean average uniqueness of a sample (labels may repeat) taken on its own.
double sampleUniqueness(const std::vector<Span>& spans, const std::vector<std::size_t>& sample, std::size_t numDates);

/// Train/test split of sample indices.
struct Split {
  std::vector<std::size_t> train, test;
};

/// k-fold cross-validation over contiguous blocks of dates, for samples with spans
/// [t0, t1]. With `purge`, training samples whose span overlaps the test block's time range
/// are removed, and so are those starting within `embargo` dates after it (serial
/// correlation leaks forward). Without it, this is ordinary k-fold on blocks.
std::vector<Split> purgedKFold(const std::vector<Span>& spans, std::size_t numDates, std::size_t k, std::size_t embargo,
                               bool purge = true);

/// Combinatorial purged cross-validation: the dates are cut into N groups and every choice
/// of k of them is a test set (C(N, k) splits), purged and embargoed as above. Each group
/// is tested C(N-1, k-1) times, which assembles phi = k C(N, k) / N complete backtest paths.
struct CombinatorialSplits {
  std::size_t groups = 0, testGroups = 0, paths = 0;
  std::vector<Split> splits;
  std::vector<std::vector<std::size_t>> testGroupsOf;  ///< per split
  std::vector<std::size_t> groupOfDate;               ///< date -> group
  /// paths[p][g]: the split whose prediction path p uses for group g.
  std::vector<std::vector<std::size_t>> pathSplit;
};

CombinatorialSplits combinatorialPurgedSplits(const std::vector<Span>& spans, std::size_t numDates, std::size_t groups,
                                              std::size_t testGroups, std::size_t embargo);

}  // namespace sat::afml
