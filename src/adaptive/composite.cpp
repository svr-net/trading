#include "sat/adaptive/composite.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "sat/adaptive/self_adaptive.hpp"
#include "sat/ml/metrics.hpp"

namespace sat {

CompositeMethod parseCompositeMethod(const std::string& name) {
  if (name == "none" || name.empty()) return CompositeMethod::None;
  if (name == "average") return CompositeMethod::Average;
  if (name == "stacked") return CompositeMethod::Stacked;
  if (name == "online" || name == "boa") return CompositeMethod::Online;
  throw std::invalid_argument("unknown composite method '" + name + "' (none, average, stacked, online)");
}

std::string compositeMethodName(CompositeMethod m) {
  switch (m) {
    case CompositeMethod::None: return "none";
    case CompositeMethod::Average: return "average";
    case CompositeMethod::Stacked: return "stacked";
    case CompositeMethod::Online: return "online";
  }
  return "none";
}

namespace {

constexpr double kClip = 1e-4;

double clipP(double p) { return std::min(1.0 - kClip, std::max(kClip, p)); }
double logit(double p) {
  p = clipP(p);
  return std::log(p / (1.0 - p));
}
double logistic(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// Solves A x = b (n x n, row-major) by Gaussian elimination with partial pivoting.
bool solve(std::vector<double> A, std::vector<double> b, std::size_t n, std::vector<double>& x) {
  for (std::size_t c = 0; c < n; ++c) {
    std::size_t piv = c;
    for (std::size_t r = c + 1; r < n; ++r)
      if (std::fabs(A[r * n + c]) > std::fabs(A[piv * n + c])) piv = r;
    if (std::fabs(A[piv * n + c]) < 1e-12) return false;
    if (piv != c) {
      for (std::size_t k = 0; k < n; ++k) std::swap(A[c * n + k], A[piv * n + k]);
      std::swap(b[c], b[piv]);
    }
    for (std::size_t r = c + 1; r < n; ++r) {
      const double f = A[r * n + c] / A[c * n + c];
      if (f == 0) continue;
      for (std::size_t k = c; k < n; ++k) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  x.assign(n, 0.0);
  for (std::size_t c = n; c-- > 0;) {
    double s = b[c];
    for (std::size_t k = c + 1; k < n; ++k) s -= A[c * n + k] * x[k];
    x[c] = s / A[c * n + c];
  }
  return true;
}

// Ridge-penalised logistic regression of y on X (rows of M log-odds) with an intercept:
// minimises log loss + lambda/2 |w - prior|^2 with w >= 0 (active set: a weight that turns
// negative is fixed at zero and the rest re-fitted). Returns {w..., intercept}.
std::vector<double> fitStack(const std::vector<std::vector<double>>& X, const std::vector<double>& y, std::size_t M, double lambda,
                             const std::vector<double>& start) {
  const double prior = 1.0 / static_cast<double>(M);
  std::vector<bool> free(M, true);
  std::vector<double> beta = start;  // M weights, then the intercept
  for (std::size_t pass = 0; pass <= M; ++pass) {
    for (std::size_t m = 0; m < M; ++m)
      if (!free[m]) beta[m] = 0.0;
    std::vector<std::size_t> idx;
    for (std::size_t m = 0; m < M; ++m)
      if (free[m]) idx.push_back(m);
    idx.push_back(M);
    const std::size_t n = idx.size();
    for (int it = 0; it < 25; ++it) {
      std::vector<double> H(n * n, 0.0), g(n, 0.0);
      for (std::size_t r = 0; r < X.size(); ++r) {
        double z = beta[M];
        for (std::size_t m = 0; m < M; ++m) z += beta[m] * X[r][m];
        const double p = logistic(z), e = p - y[r], v = std::max(1e-6, p * (1 - p));
        for (std::size_t a = 0; a < n; ++a) {
          const double xa = idx[a] == M ? 1.0 : X[r][idx[a]];
          g[a] += e * xa;
          for (std::size_t b = 0; b <= a; ++b) H[a * n + b] += v * xa * (idx[b] == M ? 1.0 : X[r][idx[b]]);
        }
      }
      for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = 0; b < a; ++b) H[b * n + a] = H[a * n + b];
        if (idx[a] != M) {
          g[a] += lambda * (beta[idx[a]] - prior);
          H[a * n + a] += lambda;
        } else {
          H[a * n + a] += 1e-6;
        }
      }
      std::vector<double> step;
      if (!solve(H, g, n, step)) break;
      double size = 0;
      for (std::size_t a = 0; a < n; ++a) {
        beta[idx[a]] -= step[a];
        size = std::max(size, std::fabs(step[a]));
      }
      if (size < 1e-8) break;
    }
    bool clipped = false;
    for (std::size_t m = 0; m < M; ++m)
      if (free[m] && beta[m] < 0) free[m] = false, clipped = true;
    if (!clipped) break;
  }
  for (std::size_t m = 0; m < M; ++m)
    if (!free[m]) beta[m] = 0.0;
  return beta;
}

// Row t is resolved at date `now` when every label of t ended before `now`.
bool resolved(const Panel& labels, const Panel& labelEnds, std::size_t t, std::size_t now) {
  for (std::size_t i = 0; i < labels.assets(); ++i) {
    if (!std::isfinite(labels(t, i))) continue;
    const double end = labelEnds.empty() ? static_cast<double>(t + 1) : labelEnds(t, i);
    if (!(end < static_cast<double>(now))) return false;
  }
  return true;
}

}  // namespace

CompositePredictions compositePredictions(const std::vector<ModelPredictions>& members, const Panel& labels,
                                          const Panel& labelEnds, const CompositeSpec& spec) {
  if (members.empty()) throw std::invalid_argument("composite model: no member models");
  if (spec.method == CompositeMethod::None) throw std::invalid_argument("composite model: no method");
  const auto t0 = std::chrono::steady_clock::now();
  const std::size_t M = members.size();
  const std::size_t T = members[0].probability.dates(), N = members[0].probability.assets();
  for (const auto& m : members)
    if (m.probability.dates() != T || m.probability.assets() != N) throw std::invalid_argument("composite model: members differ in shape");
  if (labels.dates() != T || labels.assets() != N) throw std::invalid_argument("composite model: labels differ in shape");
  if (!labelEnds.empty() && (labelEnds.dates() != T || labelEnds.assets() != N))
    throw std::invalid_argument("composite model: label ends differ in shape");
  const auto [s, e] = commonRange(members);
  if (e <= s) throw std::invalid_argument("composite model: members have no common out-of-sample dates");

  CompositePredictions out;
  for (const auto& m : members) out.members.push_back(m.name);
  ModelPredictions& mp = out.model;
  mp.spec.type = "composite";
  mp.name = "Composite (" + compositeMethodName(spec.method) + ")";
  mp.spec.name = mp.name;
  mp.probability = Panel(T, N);
  mp.start = s;
  mp.end = e;
  const double equal = 1.0 / static_cast<double>(M);

  auto memberRow = [&](std::size_t t, std::size_t i, std::vector<double>& p) {
    for (std::size_t m = 0; m < M; ++m) {
      p[m] = members[m].probability(t, i);
      if (!std::isfinite(p[m])) return false;
    }
    return true;
  };
  std::vector<double> p(M);

  if (spec.method == CompositeMethod::Average) {
    for (std::size_t t = s; t < e; ++t) {
      for (std::size_t i = 0; i < N; ++i)
        if (memberRow(t, i, p)) {
          double sum = 0;
          for (double v : p) sum += v;
          mp.probability(t, i) = sum * equal;
        }
      out.weights.push_back(std::vector<double>(M, equal));
    }
  } else if (spec.method == CompositeMethod::Stacked) {
    if (spec.window < 5 || spec.refitEvery < 1) throw std::invalid_argument("stacked composite: window >= 5 and refit >= 1");
    std::vector<double> beta(M + 1, 0.0);
    for (std::size_t m = 0; m < M; ++m) beta[m] = equal;
    for (std::size_t t = s; t < e; ++t) {
      if ((t - s) % spec.refitEvery == 0) {
        // Rows of the last `window` resolved dates within the members' range.
        std::vector<std::size_t> dates;
        for (std::size_t u = t; u-- > s && dates.size() < spec.window;)
          if (resolved(labels, labelEnds, u, t)) dates.push_back(u);
        std::size_t rows = 0;
        for (auto u : dates)
          for (std::size_t i = 0; i < N; ++i) rows += std::isfinite(labels(u, i)) ? 1 : 0;
        const std::size_t stride = spec.maxRows > 0 ? std::max<std::size_t>(1, (rows + spec.maxRows - 1) / spec.maxRows) : 1;
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        std::size_t k = 0;
        for (auto u : dates)
          for (std::size_t i = 0; i < N; ++i) {
            if (!std::isfinite(labels(u, i))) continue;
            if (k++ % stride != 0 || !memberRow(u, i, p)) continue;
            std::vector<double> x(M);
            for (std::size_t m = 0; m < M; ++m) x[m] = logit(p[m]);
            X.push_back(std::move(x));
            y.push_back(labels(u, i) > 0.5 ? 1.0 : 0.0);
          }
        if (X.size() >= 200) beta = fitStack(X, y, M, spec.ridge * static_cast<double>(X.size()) / 1000.0, beta);
      }
      for (std::size_t i = 0; i < N; ++i)
        if (memberRow(t, i, p)) {
          double z = beta[M];
          for (std::size_t m = 0; m < M; ++m) z += beta[m] * logit(p[m]);
          mp.probability(t, i) = logistic(z);
        }
      out.weights.push_back(std::vector<double>(beta.begin(), beta.begin() + static_cast<std::ptrdiff_t>(M)));
    }
  } else {  // Bernstein Online Aggregation with the gradient trick on the log loss
    std::vector<double> w(M, equal), R(M, 0.0), V(M, 0.0), eta(M, 0.0);
    double B = 0.0;
    std::size_t next = s;  // first row not yet learned from
    for (std::size_t t = s; t < e; ++t) {
      while (next < t && resolved(labels, labelEnds, next, t)) {
        // Linearised regret of each member against the mixture on row `next`.
        std::vector<double> r(M, 0.0);
        std::size_t n = 0;
        for (std::size_t i = 0; i < N; ++i) {
          const double y = labels(next, i), q = mp.probability(next, i);
          if (!std::isfinite(y) || !std::isfinite(q) || !memberRow(next, i, p)) continue;
          const double qc = clipP(q);
          const double g = (qc - (y > 0.5 ? 1.0 : 0.0)) / (qc * (1 - qc));
          for (std::size_t m = 0; m < M; ++m) r[m] += g * (q - p[m]);
          ++n;
        }
        ++next;
        if (n == 0 || M == 1) continue;
        for (std::size_t m = 0; m < M; ++m) {
          r[m] /= static_cast<double>(n);
          V[m] += r[m] * r[m];
          B = std::max(B, std::fabs(r[m]));
        }
        for (std::size_t m = 0; m < M; ++m) {
          eta[m] = V[m] > 0 && B > 0 ? std::min(1.0 / (2.0 * B), std::sqrt(std::log(static_cast<double>(M)) / V[m])) : 0.0;
          R[m] += r[m] - eta[m] * r[m] * r[m];
        }
        double top = -1e300;
        for (std::size_t m = 0; m < M; ++m) top = std::max(top, eta[m] * R[m]);
        double sum = 0;
        for (std::size_t m = 0; m < M; ++m) sum += (w[m] = eta[m] * std::exp(eta[m] * R[m] - top));
        if (!(sum > 0) || !std::isfinite(sum)) std::fill(w.begin(), w.end(), equal);
        else
          for (auto& v : w) v /= sum;
      }
      for (std::size_t i = 0; i < N; ++i)
        if (memberRow(t, i, p)) {
          double q = 0;
          for (std::size_t m = 0; m < M; ++m) q += w[m] * p[m];
          mp.probability(t, i) = q;
        }
      out.weights.push_back(w);
    }
  }

  std::vector<double> y, q;
  for (std::size_t t = s; t < e; ++t)
    for (std::size_t i = 0; i < N; ++i) {
      y.push_back(labels(t, i));
      q.push_back(mp.probability(t, i));
    }
  mp.oos = classificationMetrics(y, q);
  mp.elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

}  // namespace sat
