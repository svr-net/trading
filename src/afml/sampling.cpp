#include "sat/afml/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sat::afml {

std::vector<double> concurrency(const std::vector<Span>& spans, std::size_t numDates) {
  std::vector<double> diff(numDates + 1, 0.0), c(numDates, 0.0);
  for (const auto& [t0, t1] : spans) {
    if (t0 >= numDates) continue;
    diff[t0] += 1.0;
    diff[std::min(t1, numDates - 1) + 1] -= 1.0;
  }
  double run = 0.0;
  for (std::size_t t = 0; t < numDates; ++t) c[t] = run += diff[t];
  return c;
}

std::vector<double> averageUniqueness(const std::vector<Span>& spans, const std::vector<double>& conc) {
  std::vector<double> u(spans.size(), 0.0);
  for (std::size_t k = 0; k < spans.size(); ++k) {
    const auto [t0, t1] = spans[k];
    double s = 0.0;
    std::size_t n = 0;
    for (std::size_t t = t0; t <= t1 && t < conc.size(); ++t, ++n) s += conc[t] > 0 ? 1.0 / conc[t] : 0.0;
    u[k] = n ? s / static_cast<double>(n) : 0.0;
  }
  return u;
}

std::vector<double> returnAttributionWeights(const std::vector<Span>& spans, const std::vector<double>& conc,
                                             const std::vector<double>& logReturns) {
  std::vector<double> w(spans.size(), 0.0);
  for (std::size_t k = 0; k < spans.size(); ++k) {
    double s = 0.0;
    // The label is exposed to the returns after t0 up to t1.
    for (std::size_t t = spans[k].first + 1; t <= spans[k].second && t < conc.size(); ++t)
      if (conc[t] > 0 && std::isfinite(logReturns[t])) s += logReturns[t] / conc[t];
    w[k] = std::fabs(s);
  }
  const double total = std::accumulate(w.begin(), w.end(), 0.0);
  if (total > 0)
    for (double& v : w) v *= static_cast<double>(w.size()) / total;
  return w;
}

std::vector<double> timeDecay(const std::vector<double>& uniqueness, double oldest) {
  oldest = std::clamp(oldest, -1.0, 1.0);
  std::vector<double> cum(uniqueness.size());
  std::partial_sum(uniqueness.begin(), uniqueness.end(), cum.begin());
  const double total = cum.empty() ? 0.0 : cum.back();
  std::vector<double> w(uniqueness.size(), 1.0);
  if (total <= 0) return w;
  // Linear in cumulative uniqueness through (total, 1); at x = 0 the weight is `oldest`
  // (or 0 from the point where a negative intercept crosses zero).
  const double slope = oldest >= 0 ? (1.0 - oldest) / total : 1.0 / ((oldest + 1.0) * total);
  const double intercept = 1.0 - slope * total;
  for (std::size_t k = 0; k < w.size(); ++k) w[k] = std::max(0.0, intercept + slope * cum[k]);
  return w;
}

std::vector<std::size_t> sequentialBootstrap(const std::vector<Span>& spans, std::size_t numDates, std::size_t draws, Rng& rng) {
  const std::size_t n = spans.size();
  if (n == 0) return {};
  std::vector<double> conc(numDates, 0.0), prob(n);
  std::vector<std::size_t> out;
  for (std::size_t d = 0; d < draws; ++d) {
    // Uniqueness each label would have if it were added now.
    double total = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
      double s = 0.0;
      std::size_t len = 0;
      for (std::size_t t = spans[k].first; t <= spans[k].second && t < numDates; ++t, ++len) s += 1.0 / (conc[t] + 1.0);
      prob[k] = len ? s / static_cast<double>(len) : 0.0;
      total += prob[k];
    }
    double u = rng.uniform() * total, acc = 0.0;
    std::size_t pick = n - 1;
    for (std::size_t k = 0; k < n; ++k) {
      acc += prob[k];
      if (u < acc) {
        pick = k;
        break;
      }
    }
    out.push_back(pick);
    for (std::size_t t = spans[pick].first; t <= spans[pick].second && t < numDates; ++t) conc[t] += 1.0;
  }
  return out;
}

double sampleUniqueness(const std::vector<Span>& spans, const std::vector<std::size_t>& sample, std::size_t numDates) {
  std::vector<Span> chosen;
  for (std::size_t k : sample) chosen.push_back(spans.at(k));
  const auto u = averageUniqueness(chosen, concurrency(chosen, numDates));
  return u.empty() ? 0.0 : std::accumulate(u.begin(), u.end(), 0.0) / static_cast<double>(u.size());
}

namespace {

// Training indices: not in the test set, not overlapping the test ranges, not in the
// embargo after each test range. Ranges are inclusive [from, to] in dates.
std::vector<std::size_t> purgedTrain(const std::vector<Span>& spans, const std::vector<std::pair<std::size_t, std::size_t>>& testRanges,
                                     const std::vector<char>& isTest, std::size_t embargo, bool purge) {
  std::vector<std::size_t> train;
  for (std::size_t k = 0; k < spans.size(); ++k) {
    if (isTest[k]) continue;
    bool keep = true;
    if (purge)
      for (const auto& [from, to] : testRanges) {
        const bool overlaps = spans[k].first <= to && spans[k].second >= from;
        const bool embargoed = spans[k].first > to && spans[k].first <= to + embargo;
        if (overlaps || embargoed) {
          keep = false;
          break;
        }
      }
    if (keep) train.push_back(k);
  }
  return train;
}

std::vector<std::size_t> groupBounds(std::size_t numDates, std::size_t groups) {
  std::vector<std::size_t> b(groups + 1);
  for (std::size_t g = 0; g <= groups; ++g) b[g] = g * numDates / groups;
  return b;
}

}  // namespace

std::vector<Split> purgedKFold(const std::vector<Span>& spans, std::size_t numDates, std::size_t k, std::size_t embargo, bool purge) {
  if (k < 2 || k > numDates) throw std::invalid_argument("k-fold needs 2 <= k <= number of dates");
  const auto b = groupBounds(numDates, k);
  std::vector<Split> out;
  for (std::size_t f = 0; f < k; ++f) {
    Split s;
    std::vector<char> isTest(spans.size(), 0);
    std::size_t to = b[f];
    for (std::size_t i = 0; i < spans.size(); ++i)
      if (spans[i].first >= b[f] && spans[i].first < b[f + 1]) {
        isTest[i] = 1;
        s.test.push_back(i);
        to = std::max(to, spans[i].second);  // the test labels reach this far
      }
    s.train = purgedTrain(spans, {{b[f], to}}, isTest, embargo, purge);
    out.push_back(std::move(s));
  }
  return out;
}

CombinatorialSplits combinatorialPurgedSplits(const std::vector<Span>& spans, std::size_t numDates, std::size_t groups,
                                              std::size_t testGroups, std::size_t embargo) {
  if (groups < 2 || testGroups < 1 || testGroups >= groups) throw std::invalid_argument("CPCV needs 1 <= k < N groups");
  if (groups > 16) throw std::invalid_argument("CPCV: at most 16 groups");
  CombinatorialSplits c;
  c.groups = groups;
  c.testGroups = testGroups;
  const auto b = groupBounds(numDates, groups);
  c.groupOfDate.resize(numDates);
  for (std::size_t g = 0; g < groups; ++g)
    for (std::size_t t = b[g]; t < b[g + 1]; ++t) c.groupOfDate[t] = g;
  // Every subset of `testGroups` groups, in lexicographic order.
  std::vector<std::size_t> pick(testGroups);
  std::iota(pick.begin(), pick.end(), std::size_t{0});
  while (true) {
    Split s;
    std::vector<char> isTest(spans.size(), 0);
    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    for (std::size_t g : pick) {
      std::size_t to = b[g];
      for (std::size_t i = 0; i < spans.size(); ++i)
        if (spans[i].first < numDates && c.groupOfDate[spans[i].first] == g) {
          isTest[i] = 1;
          s.test.push_back(i);
          to = std::max(to, spans[i].second);
        }
      ranges.push_back({b[g], to});
    }
    std::sort(s.test.begin(), s.test.end());
    s.train = purgedTrain(spans, ranges, isTest, embargo, true);
    c.splits.push_back(std::move(s));
    c.testGroupsOf.push_back(pick);
    // Next combination.
    std::size_t i = testGroups;
    while (i > 0 && pick[i - 1] == groups - testGroups + i - 1) --i;
    if (i == 0) break;
    ++pick[i - 1];
    for (std::size_t j = i; j < testGroups; ++j) pick[j] = pick[j - 1] + 1;
  }
  // Paths: the m-th split (in order) that tests group g feeds path m for that group.
  c.paths = testGroups * c.splits.size() / groups;
  c.pathSplit.assign(c.paths, std::vector<std::size_t>(groups, 0));
  std::vector<std::size_t> used(groups, 0);
  for (std::size_t s = 0; s < c.splits.size(); ++s)
    for (std::size_t g : c.testGroupsOf[s]) c.pathSplit[used[g]++][g] = s;
  return c;
}

}  // namespace sat::afml
