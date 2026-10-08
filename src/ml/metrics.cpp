#include "sat/ml/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sat {

namespace {

void finitePairs(const std::vector<double>& y, const std::vector<double>& p, std::vector<double>& ys, std::vector<double>& ps) {
  const std::size_t n = std::min(y.size(), p.size());
  for (std::size_t k = 0; k < n; ++k)
    if (std::isfinite(y[k]) && std::isfinite(p[k])) {
      ys.push_back(y[k] > 0.5 ? 1.0 : 0.0);
      ps.push_back(p[k]);
    }
}

}  // namespace

ClassificationMetrics classificationMetrics(const std::vector<double>& y, const std::vector<double>& p) {
  std::vector<double> ys, ps;
  finitePairs(y, p, ys, ps);
  ClassificationMetrics m;
  m.count = ys.size();
  if (ys.empty()) return m;
  double tp = 0, fp = 0, tn = 0, fn = 0, ll = 0, pos = 0;
  for (std::size_t k = 0; k < ys.size(); ++k) {
    const bool up = ps[k] > 0.5, lab = ys[k] > 0.5;
    tp += up && lab;
    fp += up && !lab;
    tn += !up && !lab;
    fn += !up && lab;
    pos += lab;
    const double q = std::clamp(ps[k], 1e-12, 1.0 - 1e-12);
    ll -= lab ? std::log(q) : std::log(1.0 - q);
  }
  const double n = static_cast<double>(ys.size());
  m.accuracy = (tp + tn) / n;
  m.precision = tp + fp > 0 ? tp / (tp + fp) : 0.0;
  m.recall = tp + fn > 0 ? tp / (tp + fn) : 0.0;
  m.f1 = m.precision + m.recall > 0 ? 2 * m.precision * m.recall / (m.precision + m.recall) : 0.0;
  m.logLoss = ll / n;
  m.baseRate = pos / n;
  // AUC by the rank-sum (Mann-Whitney) statistic with averaged ties.
  std::vector<std::size_t> idx(ps.size());
  std::iota(idx.begin(), idx.end(), std::size_t{0});
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return ps[a] < ps[b]; });
  double rankSum = 0.0;
  for (std::size_t k = 0; k < idx.size();) {
    std::size_t e = k;
    while (e + 1 < idx.size() && ps[idx[e + 1]] == ps[idx[k]]) ++e;
    const double avg = 0.5 * static_cast<double>(k + e) + 1.0;
    for (std::size_t j = k; j <= e; ++j)
      if (ys[idx[j]] > 0.5) rankSum += avg;
    k = e + 1;
  }
  const double neg = n - pos;
  m.auc = pos > 0 && neg > 0 ? (rankSum - pos * (pos + 1) / 2.0) / (pos * neg) : 0.5;
  return m;
}

std::vector<std::vector<double>> rocCurve(const std::vector<double>& y, const std::vector<double>& p, std::size_t points) {
  std::vector<double> ys, ps;
  finitePairs(y, p, ys, ps);
  std::vector<std::vector<double>> out;
  const double pos = std::accumulate(ys.begin(), ys.end(), 0.0), neg = static_cast<double>(ys.size()) - pos;
  if (pos == 0 || neg == 0) return {{0, 0}, {1, 1}};
  std::vector<double> sorted = ps;
  std::sort(sorted.begin(), sorted.end());
  points = std::max<std::size_t>(points, 2);
  out.push_back({1.0, 1.0});
  for (std::size_t k = 1; k < points; ++k) {
    const double thr = sorted[std::min(sorted.size() - 1, k * sorted.size() / points)];
    double tp = 0, fp = 0;
    for (std::size_t j = 0; j < ys.size(); ++j)
      if (ps[j] > thr) (ys[j] > 0.5 ? tp : fp) += 1.0;
    out.push_back({fp / neg, tp / pos});
  }
  out.push_back({0.0, 0.0});
  std::reverse(out.begin(), out.end());
  return out;
}

}  // namespace sat
