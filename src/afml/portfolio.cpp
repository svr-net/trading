#include "sat/afml/portfolio.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "sat/core/random.hpp"

namespace sat::afml {

Matrix covarianceMatrix(const Matrix& R) {
  const std::size_t T = R.rows(), N = R.cols();
  if (T < 2) throw std::invalid_argument("covariance needs at least two observations");
  std::vector<double> m(N, 0.0);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t i = 0; i < N; ++i) m[i] += R(t, i) / static_cast<double>(T);
  Matrix c(N, N, 0.0);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j <= i; ++j) c(i, j) += (R(t, i) - m[i]) * (R(t, j) - m[j]) / static_cast<double>(T - 1);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < i; ++j) c(j, i) = c(i, j);
  return c;
}

Matrix correlationFromCovariance(const Matrix& cov) {
  const std::size_t N = cov.rows();
  Matrix r(N, N, 0.0);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      const double d = std::sqrt(cov(i, i) * cov(j, j));
      r(i, j) = d > 0 ? std::clamp(cov(i, j) / d, -1.0, 1.0) : (i == j ? 1.0 : 0.0);
    }
  return r;
}

Linkage clusterAssets(const Matrix& corr) {
  const std::size_t N = corr.rows();
  if (N == 0) throw std::invalid_argument("clustering needs at least one asset");
  Matrix d(N, N);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j) d(i, j) = std::sqrt(std::max(0.0, (1.0 - corr(i, j)) / 2.0));
  // Distance of distances: assets are close when their correlation profiles are alike.
  Matrix e(N, N, 0.0);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < i; ++j) {
      double s = 0;
      for (std::size_t k = 0; k < N; ++k) s += (d(k, i) - d(k, j)) * (d(k, i) - d(k, j));
      e(i, j) = e(j, i) = std::sqrt(s);
    }
  // Single linkage over active clusters; cluster distance = min over members.
  std::vector<std::size_t> id(N);
  std::iota(id.begin(), id.end(), std::size_t{0});
  std::vector<std::vector<std::size_t>> members(N);
  for (std::size_t i = 0; i < N; ++i) members[i] = {i};
  std::vector<std::vector<std::size_t>> clusterLeaves;  // by cluster id, for merged ids
  for (std::size_t i = 0; i < N; ++i) clusterLeaves.push_back({i});
  Linkage L;
  std::vector<std::size_t> active(N);
  std::iota(active.begin(), active.end(), std::size_t{0});  // indices into members/id
  while (active.size() > 1) {
    double best = std::numeric_limits<double>::infinity();
    std::size_t a = 0, b = 1;
    for (std::size_t x = 0; x < active.size(); ++x)
      for (std::size_t y = x + 1; y < active.size(); ++y) {
        double dist = std::numeric_limits<double>::infinity();
        for (std::size_t p : members[active[x]])
          for (std::size_t q : members[active[y]]) dist = std::min(dist, e(p, q));
        if (dist < best) {
          best = dist;
          a = x;
          b = y;
        }
      }
    const std::size_t ca = active[a], cb = active[b];
    L.left.push_back(id[ca]);
    L.right.push_back(id[cb]);
    L.height.push_back(best);
    // Leaves in order: left cluster then right cluster.
    std::vector<std::size_t> leaves = clusterLeaves[id[ca]];
    leaves.insert(leaves.end(), clusterLeaves[id[cb]].begin(), clusterLeaves[id[cb]].end());
    clusterLeaves.push_back(leaves);
    members[ca].insert(members[ca].end(), members[cb].begin(), members[cb].end());
    id[ca] = N + L.height.size() - 1;
    active.erase(active.begin() + static_cast<std::ptrdiff_t>(b));
  }
  L.order = clusterLeaves.back();
  return L;
}

std::vector<double> inverseVarianceWeights(const Matrix& cov) {
  std::vector<double> w(cov.rows());
  double s = 0;
  for (std::size_t i = 0; i < w.size(); ++i) s += w[i] = cov(i, i) > 0 ? 1.0 / cov(i, i) : 0.0;
  for (double& v : w) v /= s;
  return w;
}

std::vector<double> minimumVarianceWeights(const Matrix& cov) {
  const std::size_t N = cov.rows();
  Matrix a = cov;
  double trace = 0;
  for (std::size_t i = 0; i < N; ++i) trace += cov(i, i);
  for (double ridge = 0.0;; ridge = ridge == 0.0 ? 1e-8 * trace / N : ridge * 10) {
    for (std::size_t i = 0; i < N; ++i) a(i, i) = cov(i, i) + ridge;
    try {
      auto w = solveSpd(a, std::vector<double>(N, 1.0));
      const double s = std::accumulate(w.begin(), w.end(), 0.0);
      for (double& v : w) v /= s;
      return w;
    } catch (const std::invalid_argument&) {
      if (ridge > trace) throw;
    }
  }
}

std::vector<double> longOnlyMinimumVarianceWeights(const Matrix& cov) {
  const std::size_t N = cov.rows();
  std::vector<std::size_t> active(N);
  std::iota(active.begin(), active.end(), std::size_t{0});
  std::vector<double> w(N, 0.0);
  while (!active.empty()) {
    Matrix sub(active.size(), active.size());
    for (std::size_t i = 0; i < active.size(); ++i)
      for (std::size_t j = 0; j < active.size(); ++j) sub(i, j) = cov(active[i], active[j]);
    const auto ws = minimumVarianceWeights(sub);
    std::vector<std::size_t> keep;
    for (std::size_t i = 0; i < active.size(); ++i)
      if (ws[i] > 0) keep.push_back(active[i]);
    if (keep.size() == active.size()) {
      std::fill(w.begin(), w.end(), 0.0);
      for (std::size_t i = 0; i < active.size(); ++i) w[active[i]] = ws[i];
      return w;
    }
    if (keep.empty()) {  // keep the single least volatile asset
      std::size_t best = active[0];
      for (std::size_t i : active)
        if (cov(i, i) < cov(best, best)) best = i;
      keep = {best};
    }
    active = keep;
  }
  return w;
}

namespace {

double clusterVariance(const Matrix& cov, const std::vector<std::size_t>& items) {
  Matrix sub(items.size(), items.size());
  for (std::size_t i = 0; i < items.size(); ++i)
    for (std::size_t j = 0; j < items.size(); ++j) sub(i, j) = cov(items[i], items[j]);
  return portfolioVariance(sub, inverseVarianceWeights(sub));
}

}  // namespace

std::vector<double> hierarchicalRiskParity(const Matrix& cov, const std::vector<std::size_t>& order) {
  std::vector<double> w(cov.rows(), 1.0);
  std::vector<std::vector<std::size_t>> queue = {order};
  while (!queue.empty()) {
    std::vector<std::vector<std::size_t>> next;
    for (const auto& c : queue) {
      if (c.size() < 2) continue;
      const std::vector<std::size_t> l(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(c.size() / 2)), r(c.begin() + static_cast<std::ptrdiff_t>(c.size() / 2), c.end());
      const double vl = clusterVariance(cov, l), vr = clusterVariance(cov, r);
      const double alpha = 1.0 - vl / (vl + vr);
      for (std::size_t i : l) w[i] *= alpha;
      for (std::size_t i : r) w[i] *= 1.0 - alpha;
      next.push_back(l);
      next.push_back(r);
    }
    queue = std::move(next);
  }
  return w;
}

double portfolioVariance(const Matrix& cov, const std::vector<double>& w) {
  double v = 0;
  for (std::size_t i = 0; i < w.size(); ++i)
    for (std::size_t j = 0; j < w.size(); ++j) v += w[i] * cov(i, j) * w[j];
  return v;
}

AllocationComparison compareAllocations(const AllocationTrial& s) {
  if (s.assets < 2 || s.clusters < 1 || s.inSample < s.assets + 2) throw std::invalid_argument("allocation trial: too few assets or observations");
  Rng rng(s.seed);
  AllocationComparison out;
  const std::size_t N = s.assets, T = s.inSample + s.outOfSample;
  for (std::size_t trial = 0; trial < s.trials; ++trial) {
    // Asset i belongs to cluster i % clusters; returns = cluster factor + own noise, with
    // asset volatilities from 5% to 25% a year and occasional common and idiosyncratic jumps.
    std::vector<double> vol(N);
    for (auto& v : vol) v = (0.05 + 0.2 * rng.uniform()) / std::sqrt(252.0);
    const double load = std::sqrt(s.clusterCorrelation), own = std::sqrt(1.0 - s.clusterCorrelation);
    Matrix R(T, N);
    for (std::size_t t = 0; t < T; ++t) {
      std::vector<double> f(s.clusters);
      for (auto& x : f) x = rng.normal();
      const bool shock = rng.uniform() < 0.02;
      for (std::size_t i = 0; i < N; ++i) {
        double z = load * f[i % s.clusters] + own * rng.normal();
        if (shock) z += -3.0 * rng.uniform();
        R(t, i) = vol[i] * z;
      }
    }
    Matrix in(s.inSample, N), oos(s.outOfSample, N);
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t i = 0; i < N; ++i) (t < s.inSample ? in(t, i) : oos(t - s.inSample, i)) = R(t, i);
    const Matrix cov = covarianceMatrix(in), covOut = covarianceMatrix(oos);
    const auto L = clusterAssets(correlationFromCovariance(cov));
    const auto h = hierarchicalRiskParity(cov, L.order), iv = inverseVarianceWeights(cov), mv = minimumVarianceWeights(cov),
               lo = longOnlyMinimumVarianceWeights(cov);
    out.hrpVariance.push_back(252.0 * portfolioVariance(covOut, h));
    out.ivpVariance.push_back(252.0 * portfolioVariance(covOut, iv));
    out.minVarVariance.push_back(252.0 * portfolioVariance(covOut, mv));
    out.longOnlyMinVarVariance.push_back(252.0 * portfolioVariance(covOut, lo));
    const double n = static_cast<double>(s.trials);
    out.hrpMaxWeight += *std::max_element(h.begin(), h.end()) / n;
    out.ivpMaxWeight += *std::max_element(iv.begin(), iv.end()) / n;
    out.minVarMaxWeight += *std::max_element(mv.begin(), mv.end()) / n;
    out.longOnlyMaxWeight += *std::max_element(lo.begin(), lo.end()) / n;
    double gross = 0;
    for (double v : mv) gross += std::fabs(v);
    out.minVarGross += gross / n;
  }
  return out;
}

}  // namespace sat::afml
