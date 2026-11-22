// Adaptive nested Monte Carlo (docs/adaptive-nested-mc.md): risk statistics that need a conditional value at a horizon
// (VaR, ES, EPE) from one ordinary run of an existing engine plus inner simulations only where the statistic needs
// them. Header-only, C++17, no dependencies.
//
//   1 split      the outer scenarios shuffled (SplitMix64, derive(seed, -1, 0)) into a training and an evaluation half
//   2 proxy      the training payoffs Y regressed on Hermite polynomials of the standardized horizon state (OLS, tiny
//                ridge, Cholesky); each prediction with its estimation variance (HC0 sandwich)
//   3 bins       the evaluation scenarios binned by proxy value (14 bins, finer in both tails); the payoff noise r_b^2
//                per bin from the training residuals
//   4 scenarios  each evaluation scenario: proxy mu, its variance v, its bin, two sample folds (fold 0: the outer
//                payoff Y and the odd inner batches; fold 1: the even ones); the blend mu + w (mbar - mu) by precision,
//                its standard error and the refinement gain G
//   5 audit      16 random scenarios per bin get 64 inner paths; the proxy's squared bias tau_b^2 per bin, measured
//                again after every round
//   6 statistic  VaR, ES or EPE: its value and an urgency per scenario (above 1: another batch)
//   7 loop       the most urgent scenarios (at most 2,000 a round) get a batch of 64 inner paths each, in parallel,
//                seeded by (scenario, batch number); until nothing is urgent, the budget or the rounds run out
//
// The engine enters at two points: its ordinary paths (OuterScenarios: the horizon state and the payoff discounted to
// the horizon) and a restart from a horizon state (InnerSimulator: fills payoffs from a state and a seed; thread-safe).
// The same inputs and seed give the same result on any number of threads.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace anmc {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ------------------------------------------------------------------------------------------------ seeds
struct SplitMix64 {
  std::uint64_t s;
  explicit SplitMix64(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }  // [0, 1)
  std::uint64_t below(std::uint64_t n) { return n ? next() % n : 0; }
};

inline std::uint64_t mix(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// seed = mix(mix(mix(S + gamma) ^ stream) ^ substream * c): a stream per purpose, a substream per draw.
inline std::uint64_t derive(std::uint64_t seed, std::int64_t stream, std::uint64_t substream) {
  const std::uint64_t gamma = 0x9E3779B97F4A7C15ULL, c = 0xD1B54A32D192ED03ULL;
  return mix(mix(mix(seed + gamma) ^ static_cast<std::uint64_t>(stream)) ^ (substream * c));
}

inline double normCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

// ------------------------------------------------------------------------------------------------ settings and inputs
struct Options {
  std::size_t totalInnerBudget = 2000000;   // inner paths for the whole run, audit included
  std::size_t innerBatchSize = 64;          // paths per refinement step
  std::size_t minInnerForVariance = 32;     // paths before a scenario uses its own sample variance
  std::size_t maxInnerPerScenario = 4096;   // cap per scenario; also the best precision in the gain
  std::size_t maxScenariosPerRound = 2000;
  std::size_t maxRounds = 50;
  std::size_t auditScenariosPerBin = 16;    // 0: no audit (not recommended)
  std::size_t minScenariosPerBinForMisspecification = 8;
  std::vector<double> binProbabilities = {0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995};
  double trainingFraction = 0.5;
  bool refineOutsideTrainingRange = true;
  std::uint64_t seed = 20261122;
  int maxDegreeOfParallelism = -1;          // -1: every core
  int hermiteDegree = 3;
  double relativeRidge = 1e-10;

  void validate() const {
    if (innerBatchSize == 0) throw std::invalid_argument("innerBatchSize must be positive");
    if (maxInnerPerScenario < innerBatchSize) throw std::invalid_argument("maxInnerPerScenario must be at least innerBatchSize");
    if (!(trainingFraction > 0 && trainingFraction < 1)) throw std::invalid_argument("trainingFraction must be in (0, 1)");
    if (hermiteDegree < 0) throw std::invalid_argument("hermiteDegree must be at least 0");
    for (std::size_t i = 0; i < binProbabilities.size(); ++i) {
      if (!(binProbabilities[i] > 0 && binProbabilities[i] < 1)) throw std::invalid_argument("binProbabilities must be in (0, 1)");
      if (i && binProbabilities[i] <= binProbabilities[i - 1]) throw std::invalid_argument("binProbabilities must increase");
    }
  }
};

// The engine's ordinary run: for each path its horizon state (d numbers, Markov: they determine the value) and its
// payoff discounted to the horizon.
struct OuterScenarios {
  std::size_t n = 0, d = 0;
  std::vector<double> state;   // n * d, row by row
  std::vector<double> payoff;  // n
  const double* at(std::size_t i) const { return state.data() + i * d; }
};

// The engine restarted at a horizon state: fills `count` payoffs discounted to the horizon, drawing all its
// randomness from `seed`; called concurrently for different scenarios.
using InnerSimulator = std::function<void(const double* state, std::size_t d, double* payoffs, std::size_t count, std::uint64_t seed)>;

// ------------------------------------------------------------------------------------------------ 2 · the proxy
// Probabilists' Hermite polynomials per coordinate, all products of total degree at most D (capped at 2,000).
class HermiteBasis {
 public:
  HermiteBasis(std::size_t d, int degree) : d_(d), degree_(degree) {
    std::vector<int> a(d, 0);
    build(a, 0, degree);
    if (alphas_.empty()) alphas_.push_back(std::vector<int>(d, 0));
  }
  std::size_t size() const { return alphas_.size(); }
  void eval(const double* z, double* out) const {
    std::vector<double> he(static_cast<std::size_t>(degree_ + 1) * d_);
    for (std::size_t j = 0; j < d_; ++j) {
      double* h = he.data() + j * static_cast<std::size_t>(degree_ + 1);
      h[0] = 1;
      if (degree_ >= 1) h[1] = z[j];
      for (int k = 1; k < degree_; ++k) h[k + 1] = z[j] * h[k] - k * h[k - 1];
    }
    for (std::size_t f = 0; f < alphas_.size(); ++f) {
      double p = 1;
      for (std::size_t j = 0; j < d_; ++j) p *= he[j * static_cast<std::size_t>(degree_ + 1) + static_cast<std::size_t>(alphas_[f][j])];
      out[f] = p;
    }
  }

 private:
  void build(std::vector<int>& a, std::size_t j, int left) {
    if (alphas_.size() >= 2000) return;
    if (j == d_) {
      alphas_.push_back(a);
      return;
    }
    for (int k = 0; k <= left && alphas_.size() < 2000; ++k) {
      a[j] = k;
      build(a, j + 1, left - k);
    }
    a[j] = 0;
  }
  std::size_t d_;
  int degree_;
  std::vector<std::vector<int>> alphas_;
};

namespace detail {
// Cholesky of a symmetric positive definite p x p matrix (row-major, in place: the lower factor); false if not.
inline bool cholesky(std::vector<double>& a, std::size_t p) {
  for (std::size_t j = 0; j < p; ++j) {
    double s = a[j * p + j];
    for (std::size_t k = 0; k < j; ++k) s -= a[j * p + k] * a[j * p + k];
    if (!(s > 0)) return false;
    const double l = std::sqrt(s);
    a[j * p + j] = l;
    for (std::size_t i = j + 1; i < p; ++i) {
      double t = a[i * p + j];
      for (std::size_t k = 0; k < j; ++k) t -= a[i * p + k] * a[j * p + k];
      a[i * p + j] = t / l;
    }
  }
  return true;
}
inline void cholSolve(const std::vector<double>& l, std::size_t p, double* b) {
  for (std::size_t i = 0; i < p; ++i) {
    double s = b[i];
    for (std::size_t k = 0; k < i; ++k) s -= l[i * p + k] * b[k];
    b[i] = s / l[i * p + i];
  }
  for (std::size_t i = p; i-- > 0;) {
    double s = b[i];
    for (std::size_t k = i + 1; k < p; ++k) s -= l[k * p + i] * b[k];
    b[i] = s / l[i * p + i];
  }
}

// Runs job(i) for i in [0, n) on up to `threads` threads; each i on exactly one thread.
inline void parallelFor(std::size_t n, int threads, const std::function<void(std::size_t)>& job) {
  std::size_t t = threads > 0 ? static_cast<std::size_t>(threads) : std::max(1u, std::thread::hardware_concurrency());
  t = std::min(t, n);
  if (t <= 1) {
    for (std::size_t i = 0; i < n; ++i) job(i);
    return;
  }
  std::vector<std::thread> pool;
  std::vector<std::exception_ptr> errors(t);
  for (std::size_t w = 0; w < t; ++w) {
    pool.emplace_back([&, w] {
      try {
        for (std::size_t i = w; i < n; i += t) job(i);
      } catch (...) {
        errors[w] = std::current_exception();
      }
    });
  }
  for (auto& th : pool) th.join();
  for (auto& e : errors) if (e) std::rethrow_exception(e);
}
}  // namespace detail

// A fitted proxy: mu(x) and its estimation variance v(x) (finite training data, not misspecification). Immutable.
class FittedProxy {
 public:
  FittedProxy(std::size_t d, int degree) : basis_(d, degree), d_(d) {}
  const std::vector<double>& coefficients() const { return beta_; }
  std::size_t d() const { return d_; }

  void predict(const double* x, double& mu, double& v) const {
    const std::size_t p = basis_.size();
    std::vector<double> z(d_), phi(p);
    for (std::size_t j = 0; j < d_; ++j) z[j] = (x[j] - mean_[j]) / scale_[j];
    basis_.eval(z.data(), phi.data());
    mu = 0;
    for (std::size_t a = 0; a < p; ++a) mu += phi[a] * beta_[a];
    v = 0;
    for (std::size_t a = 0; a < p; ++a) {
      double s = 0;
      for (std::size_t b = 0; b < p; ++b) s += cov_[a * p + b] * phi[b];
      v += phi[a] * s;
    }
    if (!(v > 0)) v = 0;
  }

  // OLS with a tiny relative ridge on the rows `rows` of the scenarios, then the HC0 sandwich.
  void fit(const OuterScenarios& o, const std::vector<std::size_t>& rows, double relativeRidge) {
    const std::size_t n = rows.size(), p = basis_.size();
    mean_.assign(d_, 0);
    scale_.assign(d_, 1);
    lo_.assign(d_, kInf);
    hi_.assign(d_, -kInf);
    for (std::size_t r : rows)
      for (std::size_t j = 0; j < d_; ++j) {
        const double x = o.at(r)[j];
        mean_[j] += x;
        lo_[j] = std::min(lo_[j], x);
        hi_[j] = std::max(hi_[j], x);
      }
    for (std::size_t j = 0; j < d_; ++j) mean_[j] /= static_cast<double>(n);
    if (n > 1) {
      for (std::size_t j = 0; j < d_; ++j) {
        double ss = 0;
        for (std::size_t r : rows) ss += (o.at(r)[j] - mean_[j]) * (o.at(r)[j] - mean_[j]);
        const double s = std::sqrt(ss / static_cast<double>(n - 1));
        scale_[j] = s > 0 ? s : 1;
      }
    }
    // Both passes in fixed chunks added in a fixed order: the same sums on any number of threads.
    std::vector<double> g(p * p, 0), rhs(p, 0), z(d_), phi(p);
    for (std::size_t r : rows) {
      for (std::size_t j = 0; j < d_; ++j) z[j] = (o.at(r)[j] - mean_[j]) / scale_[j];
      basis_.eval(z.data(), phi.data());
      for (std::size_t a = 0; a < p; ++a) {
        rhs[a] += phi[a] * o.payoff[r];
        for (std::size_t b = 0; b <= a; ++b) g[a * p + b] += phi[a] * phi[b];
      }
    }
    double tr = 0;
    for (std::size_t a = 0; a < p; ++a) tr += g[a * p + a];
    const double lambda = relativeRidge * tr / static_cast<double>(p);
    for (std::size_t a = 0; a < p; ++a) {
      g[a * p + a] += lambda;
      for (std::size_t b = 0; b < a; ++b) g[b * p + a] = g[a * p + b];
    }
    std::vector<double> l = g;
    if (!detail::cholesky(l, p)) throw std::runtime_error("the proxy's normal equations are singular: lower the Hermite degree");
    beta_ = rhs;
    detail::cholSolve(l, p, beta_.data());
    // HC0: Cov = G^-1 (sum e_i^2 phi_i phi_i^T) G^-1
    std::vector<double> meat(p * p, 0);
    for (std::size_t r : rows) {
      for (std::size_t j = 0; j < d_; ++j) z[j] = (o.at(r)[j] - mean_[j]) / scale_[j];
      basis_.eval(z.data(), phi.data());
      double f = 0;
      for (std::size_t a = 0; a < p; ++a) f += phi[a] * beta_[a];
      const double e2 = (o.payoff[r] - f) * (o.payoff[r] - f);
      for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b <= a; ++b) meat[a * p + b] += e2 * phi[a] * phi[b];
    }
    for (std::size_t a = 0; a < p; ++a)
      for (std::size_t b = 0; b < a; ++b) meat[b * p + a] = meat[a * p + b];
    // G^-1 M: solve column by column, then (G^-1 M) G^-1 = (G^-1 (G^-1 M)^T)^T
    std::vector<double> gm(p * p), col(p);
    for (std::size_t c = 0; c < p; ++c) {
      for (std::size_t a = 0; a < p; ++a) col[a] = meat[a * p + c];
      detail::cholSolve(l, p, col.data());
      for (std::size_t a = 0; a < p; ++a) gm[a * p + c] = col[a];
    }
    cov_.assign(p * p, 0);
    for (std::size_t c = 0; c < p; ++c) {
      for (std::size_t a = 0; a < p; ++a) col[a] = gm[c * p + a];  // row c of G^-1 M
      detail::cholSolve(l, p, col.data());
      for (std::size_t a = 0; a < p; ++a) cov_[c * p + a] = col[a];
    }
    for (std::size_t a = 0; a < p; ++a)  // symmetric up to rounding
      for (std::size_t b = 0; b < a; ++b) cov_[a * p + b] = cov_[b * p + a] = 0.5 * (cov_[a * p + b] + cov_[b * p + a]);
  }

  bool outsideTraining(const double* x) const {
    for (std::size_t j = 0; j < d_; ++j)
      if (x[j] < lo_[j] || x[j] > hi_[j]) return true;
    return false;
  }

 private:
  HermiteBasis basis_;
  std::size_t d_;
  std::vector<double> mean_, scale_, lo_, hi_, beta_, cov_;
};

// ------------------------------------------------------------------------------------------------ 4 · scenario state
struct Fold {
  double n = 0, mean = 0, m2 = 0;
  // Chan et al.: exact and stable
  void merge(double nb, double mb, double m2b) {
    if (nb <= 0) return;
    const double np = n + nb, delta = mb - mean;
    mean += delta * nb / np;
    m2 += m2b + delta * delta * n * nb / np;
    n = np;
  }
};

class ScenarioSet {
 public:
  std::vector<double> mu, v;          // the proxy and its estimation variance
  std::vector<int> bin;
  std::vector<std::size_t> original;  // index in the injected outer scenarios
  std::vector<Fold> fold0, fold1;     // fold 0: Y and odd batches; fold 1: even batches
  std::vector<std::size_t> batches, inner;
  std::vector<bool> audited;
  std::vector<double> r2, tau2;       // per bin: payoff noise, squared proxy bias
  std::size_t minInnerForVariance = 32, maxInnerPerScenario = 4096;

  std::size_t size() const { return mu.size(); }

  Fold pooled(std::size_t i) const {
    Fold f = fold0[i];
    f.merge(fold1[i].n, fold1[i].mean, fold1[i].m2);
    return f;
  }
  // The noise of one payoff in the scenario: its own sample variance once it has 1 + MinInnerForVariance samples.
  double s2(std::size_t i) const {
    const Fold f = pooled(i);
    if (f.n >= 1.0 + static_cast<double>(minInnerForVariance)) return f.m2 / (f.n - 1);
    return r2[static_cast<std::size_t>(bin[i])];
  }
  double proxyVariance(std::size_t i) const { return v[i] + tau2[static_cast<std::size_t>(bin[i])]; }

  // The blend of the proxy and the pooled mean by their precision.
  double estimate(std::size_t i) const {
    const Fold f = pooled(i);
    return blend(mu[i], proxyVariance(i), f.mean, f.n, s2(i));
  }
  double estimateVariance(std::size_t i) const {
    const Fold f = pooled(i);
    return blendVariance(proxyVariance(i), f.n, s2(i));
  }
  double stdError(std::size_t i) const { return std::sqrt(estimateVariance(i)); }

  // How much of the remaining error more inner paths could remove (0: none, 1: all).
  double refinementGain(std::size_t i) const {
    const Fold f = pooled(i);
    const double s = s2(i), sp = proxyVariance(i);
    const double now = blendVariance(sp, f.n, s);
    if (!(now > 0)) return 0;
    const double left = static_cast<double>(maxInnerPerScenario > inner[i] ? maxInnerPerScenario - inner[i] : 0);
    const double later = blendVariance(sp, f.n + left, s);
    return std::clamp(1 - later / now, 0.0, 1.0);
  }

  // Selection (which side of a cutoff) and value come from independent samples.
  double selectionEstimate(std::size_t i) const {
    if (fold1[i].n > 0) return blend(mu[i], proxyVariance(i), fold0[i].mean, fold0[i].n, s2(i));
    return mu[i];
  }
  double valueEstimate(std::size_t i) const { return fold1[i].n > 0 ? fold1[i].mean : fold0[i].mean; }
  double valueVariance(std::size_t i) const {
    const double n = fold1[i].n > 0 ? fold1[i].n : fold0[i].n;
    return n > 0 ? s2(i) / n : kInf;
  }

  void addBatch(std::size_t i, const double* y, std::size_t count) {
    double m = 0;
    for (std::size_t k = 0; k < count; ++k) m += y[k];
    m /= static_cast<double>(count);
    double ss = 0;
    for (std::size_t k = 0; k < count; ++k) ss += (y[k] - m) * (y[k] - m);
    (batches[i] % 2 == 0 ? fold1[i] : fold0[i]).merge(static_cast<double>(count), m, ss);
    ++batches[i];
    inner[i] += count;
  }

  static double blend(double mu0, double sp, double mean, double n, double s) {
    if (!(n > 0)) return mu0;
    const double u = s / n;
    if (!(sp > 0)) return mu0;
    if (!(u > 0)) return mean;
    return mu0 + sp / (sp + u) * (mean - mu0);
  }
  static double blendVariance(double sp, double n, double s) {
    if (!(n > 0)) return sp;
    const double u = s / n;
    if (!(sp > 0) || !(u > 0)) return 0;
    return sp * u / (sp + u);
  }
};

// ------------------------------------------------------------------------------------------------ 6 · statistics
// e(g, se) = g Phi(-g / se): the expected error of a scenario at distance g from a cutoff.
inline double misclassification(double g, double se) {
  if (!(se > 0)) return 0;
  return g * normCdf(-g / se);
}

// Ranks by reducible error e_i G_i (largest first, ties by index), skips G < 0.05, marks urgency 2 until the
// reducible errors add up to the excess.
inline void markLargestContributors(const ScenarioSet& s, const std::vector<double>& e, double excess, std::vector<double>& urgency) {
  std::vector<std::size_t> idx;
  std::vector<double> red(e.size());
  for (std::size_t i = 0; i < e.size(); ++i) {
    const double g = s.refinementGain(i);
    red[i] = e[i] * g;
    if (g >= 0.05 && red[i] > 0) idx.push_back(i);
  }
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return red[a] > red[b] || (red[a] == red[b] && a < b); });
  double acc = 0;
  for (std::size_t i : idx) {
    if (acc >= excess) break;
    urgency[i] = std::max(urgency[i], 2.0);
    acc += red[i];
  }
}

struct RiskStatistic {
  virtual ~RiskStatistic() = default;
  virtual std::string name() const = 0;
  virtual double evaluate(const ScenarioSet& s) const = 0;
  virtual double standardError(const ScenarioSet& s) const = 0;
  virtual void urgency(const ScenarioSet& s, std::vector<double>& u) const = 0;
};

inline std::size_t ceilCount(double x) { return static_cast<std::size_t>(std::ceil(x - 1e-12)); }

// VaR at confidence alpha: the k-th smallest loss L_i = base - V~_i, k = ceil(n alpha).
struct ValueAtRisk : RiskStatistic {
  double alpha, base, z;
  ValueAtRisk(double confidence, double baseValue, double zCritical = 2) : alpha(confidence), base(baseValue), z(zCritical) {}
  std::string name() const override { return "VaR " + std::to_string(alpha); }
  double evaluate(const ScenarioSet& s) const override {
    std::vector<double> l(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) l[i] = base - s.estimate(i);
    const std::size_t k = std::clamp<std::size_t>(ceilCount(static_cast<double>(l.size()) * alpha), 1, l.size());
    std::nth_element(l.begin(), l.begin() + static_cast<long>(k - 1), l.end());
    return l[k - 1];
  }
  double standardError(const ScenarioSet&) const override { return kNaN; }
  void urgency(const ScenarioSet& s, std::vector<double>& u) const override {
    const double var = evaluate(s), denom = normCdf(-z);
    for (std::size_t i = 0; i < s.size(); ++i) {
      const double se = s.stdError(i);
      if (!(se > 0)) continue;
      const double li = base - s.estimate(i);
      u[i] = std::max(u[i], s.refinementGain(i) * normCdf(-std::fabs(li - var) / se) / denom);
    }
  }
};

// ES at confidence alpha: the mean of base - Value_i over the k = ceil(n (1 - alpha)) scenarios with the largest
// base - Selection_i.
struct ExpectedShortfall : RiskStatistic {
  double alpha, base, rel, abs;
  ExpectedShortfall(double confidence, double baseValue, double relativeTolerance = 0.01, double absoluteTolerance = 0)
      : alpha(confidence), base(baseValue), rel(relativeTolerance), abs(absoluteTolerance) {}
  std::string name() const override { return "ES " + std::to_string(alpha); }
  std::vector<std::size_t> tail(const ScenarioSet& s) const {
    std::vector<std::size_t> idx(s.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<double> loss(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) loss[i] = base - s.selectionEstimate(i);
    const std::size_t k = std::clamp<std::size_t>(ceilCount(static_cast<double>(s.size()) * (1 - alpha)), 1, s.size());
    std::partial_sort(idx.begin(), idx.begin() + static_cast<long>(k), idx.end(),
                      [&](std::size_t a, std::size_t b) { return loss[a] > loss[b] || (loss[a] == loss[b] && a < b); });
    idx.resize(k);
    return idx;
  }
  double evaluate(const ScenarioSet& s) const override {
    const auto t = tail(s);
    double sum = 0;
    for (std::size_t i : t) sum += base - s.valueEstimate(i);
    return sum / static_cast<double>(t.size());
  }
  double standardError(const ScenarioSet& s) const override {
    const auto t = tail(s);
    const double k = static_cast<double>(t.size());
    if (t.size() < 2) return kNaN;
    double m = 0, ss = 0;
    for (std::size_t i : t) m += base - s.valueEstimate(i);
    m /= k;
    for (std::size_t i : t) ss += (base - s.valueEstimate(i) - m) * (base - s.valueEstimate(i) - m);
    return std::sqrt(ss / (k - 1)) / std::sqrt(k);
  }
  void urgency(const ScenarioSet& s, std::vector<double>& u) const override {
    const auto t = tail(s);
    const double k = static_cast<double>(t.size());
    const double es = evaluate(s), tol = std::max(abs, rel * std::fabs(es)), h = tol / 2;
    if (!(h > 0)) return;
    double vv = 0;
    for (std::size_t i : t) vv += s.valueVariance(i);
    if (std::sqrt(vv) / k > h)  // tail noise: each tail value against an equal share of the budget
      for (std::size_t i : t) u[i] = std::max(u[i], s.valueVariance(i) / (h * h * k));
    // cutoff misclassification: the k-th largest base - V~_i
    std::vector<double> loss(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) loss[i] = base - s.estimate(i);
    std::vector<double> sorted = loss;
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(t.size() - 1), sorted.end(), std::greater<double>());
    const double c = sorted[t.size() - 1];
    std::vector<double> e(s.size());
    double sum = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
      e[i] = misclassification(std::fabs(loss[i] - c), s.stdError(i)) / k;
      sum += e[i];
    }
    if (sum > h) markLargestContributors(s, e, sum - h, u);
  }
};

// EPE: the mean of 1{Selection_i > 0} Value_i.
struct PositiveExposure : RiskStatistic {
  double rel, abs;
  explicit PositiveExposure(double relativeTolerance = 0.01, double absoluteTolerance = 0) : rel(relativeTolerance), abs(absoluteTolerance) {}
  std::string name() const override { return "EPE"; }
  double evaluate(const ScenarioSet& s) const override {
    double sum = 0;
    for (std::size_t i = 0; i < s.size(); ++i)
      if (s.selectionEstimate(i) > 0) sum += s.valueEstimate(i);
    return sum / static_cast<double>(s.size());
  }
  double standardError(const ScenarioSet& s) const override {
    const double n = static_cast<double>(s.size());
    if (s.size() < 2) return kNaN;
    double m = 0, ss = 0;
    for (std::size_t i = 0; i < s.size(); ++i) m += s.selectionEstimate(i) > 0 ? s.valueEstimate(i) : 0;
    m /= n;
    for (std::size_t i = 0; i < s.size(); ++i) {
      const double c = (s.selectionEstimate(i) > 0 ? s.valueEstimate(i) : 0) - m;
      ss += c * c;
    }
    return std::sqrt(ss / (n - 1)) / std::sqrt(n);
  }
  void urgency(const ScenarioSet& s, std::vector<double>& u) const override {
    const double n = static_cast<double>(s.size());
    const double tol = std::max(abs, rel * std::fabs(evaluate(s)));
    std::vector<double> e(s.size());
    double sum = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
      e[i] = misclassification(std::fabs(s.estimate(i)), s.stdError(i)) / n;
      sum += e[i];
    }
    if (sum > tol) markLargestContributors(s, e, sum - tol, u);
  }
};

// ------------------------------------------------------------------------------------------------ the result
enum class StopReason { Converged, ScenarioCapsReached, BudgetExhausted, MaxRoundsReached };
inline const char* toString(StopReason r) {
  switch (r) {
    case StopReason::Converged: return "Converged";
    case StopReason::ScenarioCapsReached: return "ScenarioCapsReached";
    case StopReason::BudgetExhausted: return "BudgetExhausted";
    case StopReason::MaxRoundsReached: return "MaxRoundsReached";
  }
  return "?";
}

struct BinDiagnostics {
  double lower = 0, upper = 1;
  std::size_t scenarios = 0, refined = 0;
  double residSd = 0, biasRms = 0;
};

struct Result {
  double estimate = kNaN, baselineEstimate = kNaN, standardError = kNaN;
  std::size_t innerPathsUsed = 0, rounds = 0, refinedScenarios = 0, auditedScenarios = 0, unresolvedScenarios = 0;
  StopReason stopReason = StopReason::Converged;
  std::vector<BinDiagnostics> bins;
  std::shared_ptr<FittedProxy> proxy;
  ScenarioSet scenarios;

  std::string toString() const {
    std::ostringstream o;
    o << "estimate " << estimate << " (regression only " << baselineEstimate << "), standard error " << standardError << "\n"
      << "inner paths " << innerPathsUsed << " in " << rounds << " rounds; refined " << refinedScenarios << ", audited " << auditedScenarios
      << "; stop: " << anmc::toString(stopReason) << ", unresolved " << unresolvedScenarios << "\n"
      << "bin,lower,upper,scenarios,refined,resid sd,bias rms\n";
    for (std::size_t b = 0; b < bins.size(); ++b)
      o << b << "," << bins[b].lower << "," << bins[b].upper << "," << bins[b].scenarios << "," << bins[b].refined << "," << bins[b].residSd << ","
        << bins[b].biasRms << "\n";
    return o.str();
  }
};

// ------------------------------------------------------------------------------------------------ the estimator
class AdaptiveNestedEstimator {
 public:
  explicit AdaptiveNestedEstimator(Options o = Options()) : opt_(std::move(o)) { opt_.validate(); }

  Result estimate(const OuterScenarios& outer, const InnerSimulator& sim, const RiskStatistic& stat) const {
    if (outer.payoff.size() != outer.n || outer.state.size() != outer.n * outer.d) throw std::invalid_argument("outer scenarios: sizes do not match");
    for (std::size_t i = 0; i < outer.n; ++i)
      if (!std::isfinite(outer.payoff[i])) throw std::invalid_argument("outer payoff " + std::to_string(i) + " is not finite");
    Result res;
    // 1 · split
    std::vector<std::size_t> perm(outer.n);
    std::iota(perm.begin(), perm.end(), 0);
    SplitMix64 rng(derive(opt_.seed, -1, 0));
    for (std::size_t i = outer.n; i > 1; --i) std::swap(perm[i - 1], perm[rng.below(i)]);
    const std::size_t nTrain = static_cast<std::size_t>(std::llround(static_cast<double>(outer.n) * opt_.trainingFraction));
    if (nTrain < 2 || outer.n - nTrain < 1) throw std::invalid_argument("at least 2 training and 1 evaluation scenario are needed");
    std::vector<std::size_t> train(perm.begin(), perm.begin() + static_cast<long>(nTrain)), eval(perm.begin() + static_cast<long>(nTrain), perm.end());
    std::sort(eval.begin(), eval.end());  // evaluation index order: by original index (stable, reproducible)

    // 2 · proxy
    auto proxy = std::make_shared<FittedProxy>(outer.d, opt_.hermiteDegree);
    proxy->fit(outer, train, opt_.relativeRidge);
    res.proxy = proxy;

    // 3 · bins
    ScenarioSet& s = res.scenarios;
    const std::size_t n = eval.size(), nb = opt_.binProbabilities.size() + 1;
    s.minInnerForVariance = opt_.minInnerForVariance;
    s.maxInnerPerScenario = opt_.maxInnerPerScenario;
    s.mu.resize(n);
    s.v.resize(n);
    s.original = eval;
    detail::parallelFor(n, opt_.maxDegreeOfParallelism, [&](std::size_t i) { proxy->predict(outer.at(eval[i]), s.mu[i], s.v[i]); });
    std::vector<double> sortedMu = s.mu;
    std::sort(sortedMu.begin(), sortedMu.end());
    std::vector<double> cuts;
    for (double p : opt_.binProbabilities) cuts.push_back(quantile(sortedMu, p));
    auto binOf = [&](double m) { return static_cast<int>(std::count_if(cuts.begin(), cuts.end(), [&](double c) { return c < m; })); };
    s.bin.resize(n);
    for (std::size_t i = 0; i < n; ++i) s.bin[i] = binOf(s.mu[i]);
    std::vector<double> rsum(nb, 0);
    std::vector<std::size_t> rcount(nb, 0);
    double gsum = 0;
    for (std::size_t r : train) {
      double m, vv;
      proxy->predict(outer.at(r), m, vv);
      const double e2 = (outer.payoff[r] - m) * (outer.payoff[r] - m);
      const auto b = static_cast<std::size_t>(binOf(m));
      rsum[b] += e2;
      ++rcount[b];
      gsum += e2;
    }
    s.r2.assign(nb, gsum / static_cast<double>(train.size()));
    for (std::size_t b = 0; b < nb; ++b)
      if (rcount[b] >= 2) s.r2[b] = rsum[b] / static_cast<double>(rcount[b]);
    s.tau2.assign(nb, 0);

    // 4 · per-scenario state: fold 0 starts with the outer payoff
    s.fold0.assign(n, Fold());
    s.fold1.assign(n, Fold());
    s.batches.assign(n, 0);
    s.inner.assign(n, 0);
    s.audited.assign(n, false);
    for (std::size_t i = 0; i < n; ++i) s.fold0[i].merge(1, outer.payoff[eval[i]], 0);
    res.baselineEstimate = stat.evaluate(s);

    std::size_t budget = opt_.totalInnerBudget;
    // 5 · audit, once: random scenarios per bin, from the same generator as the split
    if (opt_.auditScenariosPerBin > 0) {
      std::vector<std::vector<std::size_t>> byBin(nb);
      for (std::size_t i = 0; i < n; ++i) byBin[static_cast<std::size_t>(s.bin[i])].push_back(i);
      std::vector<std::pair<std::size_t, std::size_t>> jobs;  // (scenario, paths)
      const std::size_t paths = std::max(opt_.innerBatchSize, opt_.minInnerForVariance);
      for (auto& members : byBin) {
        for (std::size_t i = members.size(); i > 1; --i) std::swap(members[i - 1], members[rng.below(i)]);
        for (std::size_t k = 0; k < std::min(opt_.auditScenariosPerBin, members.size()); ++k) {
          if (budget < paths) break;
          jobs.emplace_back(members[k], paths);
          budget -= paths;
        }
      }
      run(outer, sim, s, jobs);
      for (auto& j : jobs) s.audited[j.first] = true;
      res.auditedScenarios = jobs.size();
      res.innerPathsUsed += jobs.size() * paths;
    }
    measureBias(s, nb);

    // 7 · the adaptive loop
    std::vector<double> u(n);
    res.stopReason = StopReason::MaxRoundsReached;
    std::size_t lastUrgent = 0;
    for (std::size_t round = 0; round < opt_.maxRounds; ++round) {
      std::fill(u.begin(), u.end(), 0.0);
      stat.urgency(s, u);
      if (opt_.refineOutsideTrainingRange)
        for (std::size_t i = 0; i < n; ++i)
          if (s.inner[i] < opt_.minInnerForVariance && proxy->outsideTraining(outer.at(eval[i]))) u[i] = kInf;
      std::vector<std::size_t> urgent;
      std::size_t capped = 0;
      for (std::size_t i = 0; i < n; ++i) {
        if (!(u[i] > 1)) continue;
        if (s.inner[i] >= opt_.maxInnerPerScenario) {
          ++capped;
          continue;
        }
        urgent.push_back(i);
      }
      lastUrgent = urgent.size() + capped;
      if (urgent.empty()) {
        res.stopReason = capped ? StopReason::ScenarioCapsReached : StopReason::Converged;
        break;
      }
      std::stable_sort(urgent.begin(), urgent.end(), [&](std::size_t a, std::size_t b) { return u[a] > u[b] || (u[a] == u[b] && a < b); });
      if (urgent.size() > opt_.maxScenariosPerRound) urgent.resize(opt_.maxScenariosPerRound);
      std::vector<std::pair<std::size_t, std::size_t>> jobs;
      bool out = false;
      for (std::size_t i : urgent) {
        const std::size_t paths = std::min(opt_.innerBatchSize, opt_.maxInnerPerScenario - s.inner[i]);
        if (paths > budget) {
          out = true;
          break;
        }
        jobs.emplace_back(i, paths);
        budget -= paths;
      }
      if (jobs.empty()) {
        res.stopReason = StopReason::BudgetExhausted;
        break;
      }
      run(outer, sim, s, jobs);
      for (auto& j : jobs) res.innerPathsUsed += j.second;
      res.rounds = round + 1;
      measureBias(s, nb);
      if (out) {
        res.stopReason = StopReason::BudgetExhausted;
        // what still asks after this round
        std::fill(u.begin(), u.end(), 0.0);
        stat.urgency(s, u);
        lastUrgent = static_cast<std::size_t>(std::count_if(u.begin(), u.end(), [](double x) { return x > 1; }));
        break;
      }
    }
    if (res.stopReason == StopReason::MaxRoundsReached) {
      std::fill(u.begin(), u.end(), 0.0);
      stat.urgency(s, u);
      lastUrgent = static_cast<std::size_t>(std::count_if(u.begin(), u.end(), [](double x) { return x > 1; }));
    }
    res.unresolvedScenarios = res.stopReason == StopReason::Converged ? 0 : lastUrgent;
    res.estimate = stat.evaluate(s);
    res.standardError = stat.standardError(s);
    res.refinedScenarios = static_cast<std::size_t>(std::count_if(s.inner.begin(), s.inner.end(), [](std::size_t x) { return x > 0; }));
    res.bins.resize(nb);
    for (std::size_t b = 0; b < nb; ++b) {
      res.bins[b].lower = b == 0 ? 0 : opt_.binProbabilities[b - 1];
      res.bins[b].upper = b + 1 == nb ? 1 : opt_.binProbabilities[b];
      res.bins[b].residSd = std::sqrt(s.r2[b]);
      res.bins[b].biasRms = std::sqrt(s.tau2[b]);
    }
    for (std::size_t i = 0; i < n; ++i) {
      auto& bd = res.bins[static_cast<std::size_t>(s.bin[i])];
      ++bd.scenarios;
      if (s.inner[i] > 0) ++bd.refined;
    }
    return res;
  }

  const Options& options() const { return opt_; }

 private:
  static double quantile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return kNaN;
    const double h = p * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (h - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
  }

  // One batch for each job (scenario, paths), in parallel; seeds fixed before the run: (original index, batch number).
  void run(const OuterScenarios& outer, const InnerSimulator& sim, ScenarioSet& s, const std::vector<std::pair<std::size_t, std::size_t>>& jobs) const {
    std::vector<std::vector<double>> pay(jobs.size());
    std::vector<std::uint64_t> seeds(jobs.size());
    for (std::size_t j = 0; j < jobs.size(); ++j)
      seeds[j] = derive(opt_.seed, static_cast<std::int64_t>(s.original[jobs[j].first]), s.batches[jobs[j].first]);
    detail::parallelFor(jobs.size(), opt_.maxDegreeOfParallelism, [&](std::size_t j) {
      const std::size_t i = jobs[j].first;
      pay[j].assign(jobs[j].second, 0.0);
      sim(outer.at(s.original[i]), outer.d, pay[j].data(), jobs[j].second, seeds[j]);
      for (double y : pay[j])
        if (!std::isfinite(y)) throw std::runtime_error("inner payoff not finite for scenario " + std::to_string(s.original[i]));
    });
    for (std::size_t j = 0; j < jobs.size(); ++j) s.addBatch(jobs[j].first, pay[j].data(), jobs[j].second);
  }

  // 5 · the proxy's squared bias per bin from every scenario with enough samples
  void measureBias(ScenarioSet& s, std::size_t nb) const {
    std::vector<double> sum(nb, 0);
    std::vector<std::size_t> cnt(nb, 0);
    double gsum = 0;
    std::size_t gcnt = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
      const Fold f = s.pooled(i);
      if (f.n < 1.0 + static_cast<double>(opt_.minInnerForVariance)) continue;
      const double x = (f.mean - s.mu[i]) * (f.mean - s.mu[i]) - s.s2(i) / f.n - s.v[i];
      const auto b = static_cast<std::size_t>(s.bin[i]);
      sum[b] += x;
      ++cnt[b];
      gsum += x;
      ++gcnt;
    }
    const double global = gcnt ? std::max(0.0, gsum / static_cast<double>(gcnt)) : 0.0;
    for (std::size_t b = 0; b < nb; ++b)
      s.tau2[b] = cnt[b] >= opt_.minScenariosPerBinForMisspecification ? std::max(0.0, sum[b] / static_cast<double>(cnt[b])) : global;
  }

  Options opt_;
};

}  // namespace anmc
