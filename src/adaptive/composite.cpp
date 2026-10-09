#include "sat/adaptive/composite.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "sat/adaptive/self_adaptive.hpp"
#include "sat/core/stats.hpp"
#include "sat/ml/metrics.hpp"

namespace sat {

CompositeMethod parseCompositeMethod(const std::string& name) {
  if (name == "none" || name.empty()) return CompositeMethod::None;
  if (name == "average") return CompositeMethod::Average;
  if (name == "adaptive" || name == "self-adaptive") return CompositeMethod::Adaptive;
  throw std::invalid_argument("unknown composite method '" + name + "' (none, average, adaptive)");
}

ScoringWindow parseScoringWindow(const std::string& name) {
  if (name == "fixed") return ScoringWindow::Fixed;
  if (name == "exponential") return ScoringWindow::Exponential;
  if (name == "adwin") return ScoringWindow::Adwin;
  throw std::invalid_argument("unknown scoring window '" + name + "' (fixed, exponential, adwin)");
}

std::string scoringWindowName(ScoringWindow w) {
  switch (w) {
    case ScoringWindow::Fixed: return "fixed";
    case ScoringWindow::Exponential: return "exponential";
    case ScoringWindow::Adwin: return "adwin";
  }
  return "fixed";
}

std::string compositeMethodName(CompositeMethod m) {
  switch (m) {
    case CompositeMethod::None: return "none";
    case CompositeMethod::Average: return "average";
    case CompositeMethod::Adaptive: return "self-adaptive";
  }
  return "none";
}

namespace {

// Row t is resolved at date `now` when every label of t ended before `now`.
bool resolved(const Panel& labels, const Panel& labelEnds, std::size_t t, std::size_t now) {
  for (std::size_t i = 0; i < labels.assets(); ++i) {
    if (!std::isfinite(labels(t, i))) continue;
    const double end = labelEnds.empty() ? static_cast<double>(t + 1) : labelEnds(t, i);
    if (!(end < static_cast<double>(now))) return false;
  }
  return true;
}

// The record of one candidate: a series of daily values (information coefficients, or their
// differences from the average's), pushed as their labels resolve, and the statistics of the
// part the scoring window remembers.
class Scorer {
 public:
  struct Stats {
    double mean = Panel::kMissing, sd = 0.0, n = 0.0;
  };

  explicit Scorer(const CompositeSpec& spec) : spec_(spec), lambda_(spec.halfLife > 0 ? std::pow(0.5, 1.0 / spec.halfLife) : 1.0) {
    p1_.push_back(0.0), p2_.push_back(0.0);
  }

  void push(double v) {
    x_.push_back(v);
    p1_.push_back(p1_.back() + v), p2_.push_back(p2_.back() + v * v);
    if (spec_.window == ScoringWindow::Exponential) {
      s0_ = lambda_ * s0_ + 1.0, s1_ = lambda_ * s1_ + v, s2_ = lambda_ * s2_ + v * v, w2_ = lambda_ * lambda_ * w2_ + 1.0;
    } else if (spec_.window == ScoringWindow::Adwin) {
      cut();
    }
  }

  Stats stats() const {
    Stats st;
    if (spec_.window == ScoringWindow::Exponential) {
      if (s0_ <= 0) return st;
      st.mean = s1_ / s0_;
      st.sd = std::sqrt(std::max(0.0, s2_ / s0_ - st.mean * st.mean));
      st.n = s0_ * s0_ / w2_;
      return st;
    }
    const std::size_t end = x_.size();
    const std::size_t from = spec_.window == ScoringWindow::Fixed ? (end > spec_.lookback ? end - spec_.lookback : 0) : from_;
    const double n = static_cast<double>(end - from);
    if (n < 1) return st;
    st.mean = (p1_[end] - p1_[from]) / n;
    st.sd = std::sqrt(std::max(0.0, (p2_[end] - p2_[from]) / n - st.mean * st.mean));
    st.n = n;
    return st;
  }

 private:
  // ADWIN: drop the older part of the window while some split shows a significant change.
  void cut() {
    constexpr std::size_t kMinSide = 10;
    for (bool changed = true; changed;) {
      changed = false;
      const std::size_t end = x_.size(), n = end - from_;
      if (n < 2 * kMinSide) return;
      const double nn = static_cast<double>(n);
      const double mean = (p1_[end] - p1_[from_]) / nn;
      const double var = std::max(0.0, (p2_[end] - p2_[from_]) / nn - mean * mean);
      const double logTerm = std::log(2.0 * std::log(nn) / spec_.adwinDelta);
      const std::size_t step = std::max<std::size_t>(1, n / 64);
      for (std::size_t k = from_ + kMinSide; k + kMinSide <= end; k += step) {
        const double n0 = static_cast<double>(k - from_), n1 = static_cast<double>(end - k);
        const double m0 = (p1_[k] - p1_[from_]) / n0, m1 = (p1_[end] - p1_[k]) / n1;
        const double m = 1.0 / (1.0 / n0 + 1.0 / n1);
        const double eps = std::sqrt(2.0 / m * var * logTerm) + 2.0 / (3.0 * m) * logTerm;
        if (std::fabs(m0 - m1) > eps) {
          from_ = k;
          changed = true;
          break;
        }
      }
    }
  }

  const CompositeSpec& spec_;
  double lambda_;
  std::vector<double> x_, p1_, p2_;
  std::size_t from_ = 0;
  double s0_ = 0, s1_ = 0, s2_ = 0, w2_ = 0;
};

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
  } else {  // the self-adaptive forecast
    if (spec.lookback < 5 || spec.adaptEvery < 1) throw std::invalid_argument("self-adaptive forecast: look-back >= 5 and step >= 1");
    // Candidate c < M is member c; c == M is the equal-weight average.
    auto candidate = [&](std::size_t c, std::size_t t, std::size_t i) {
      if (!memberRow(t, i, p)) return Panel::kMissing;
      if (c < M) return p[c];
      double sum = 0;
      for (double v : p) sum += v;
      return sum * equal;
    };
    // Daily information coefficient of a candidate.
    auto dailyIc = [&](std::size_t c, std::size_t u) {
      std::vector<double> f, y;
      for (std::size_t i = 0; i < N; ++i) {
        const double q = candidate(c, u, i);
        if (std::isfinite(q) && std::isfinite(labels(u, i))) f.push_back(q), y.push_back(labels(u, i));
      }
      if (f.size() < 4) return Panel::kMissing;
      const double r = correlation(averageRanks(f), averageRanks(y));
      return std::isfinite(r) ? r : Panel::kMissing;
    };
    if (spec.window == ScoringWindow::Exponential && !(spec.halfLife >= 0)) throw std::invalid_argument("self-adaptive forecast: half-life >= 0");
    if (spec.window == ScoringWindow::Adwin && !(spec.adwinDelta > 0 && spec.adwinDelta < 1))
      throw std::invalid_argument("self-adaptive forecast: ADWIN confidence in (0, 1)");
    // One record per candidate: Best scores every candidate's information coefficient; Evidence
    // scores each member's daily lead over the average.
    const bool evidence = spec.decision == ScoringDecision::Evidence;
    const std::size_t K = evidence ? M : M + 1;
    std::vector<Scorer> records;
    for (std::size_t c = 0; c < K; ++c) records.emplace_back(spec);
    // Fixed windows re-score on the selector's schedule; the others every day.
    const std::size_t step = spec.window == ScoringWindow::Fixed ? spec.adaptEvery : 1;
    std::size_t next = s;  // first date not yet added to the records
    int chosen = -1;
    for (std::size_t t = s; t < e; ++t) {
      // Incremental: add each date whose labels have all resolved before t.
      for (; next < t && resolved(labels, labelEnds, next, t); ++next) {
        std::vector<double> v(M + 1);
        bool ok = true;
        for (std::size_t c = 0; c <= M && ok; ++c) ok = std::isfinite(v[c] = dailyIc(c, next));
        if (!ok) continue;
        for (std::size_t c = 0; c < K; ++c) records[c].push(evidence ? v[c] - v[M] : v[c]);
      }
      if ((t - s) % step == 0) {
        chosen = -1;
        std::vector<Scorer::Stats> st;
        for (const auto& r : records) st.push_back(r.stats());
        if (evidence) {
          double bestT = spec.minT;
          for (std::size_t c = 0; c < M; ++c) {
            if (!(st[c].n >= 5) || !std::isfinite(st[c].mean)) continue;
            const double tstat = st[c].mean / std::max(1e-12, st[c].sd / std::sqrt(st[c].n));
            if (tstat > bestT) bestT = tstat, chosen = static_cast<int>(c);
          }
        } else {
          // The average, unless a member beats it with a positive record; the average stands
          // in when nothing scores above zero.
          double best = 0.0;
          if (st[M].n >= 5 && std::isfinite(st[M].mean) && st[M].mean > best) best = st[M].mean;
          for (std::size_t c = 0; c < M; ++c)
            if (st[c].n >= 5 && std::isfinite(st[c].mean) && st[c].mean > best) best = st[c].mean, chosen = static_cast<int>(c);
        }
      }
      out.memory.push_back(records[0].stats().n);
      std::vector<double> w(M, chosen < 0 ? equal : 0.0);
      if (chosen >= 0) w[static_cast<std::size_t>(chosen)] = 1.0;
      for (std::size_t i = 0; i < N; ++i)
        if (memberRow(t, i, p)) {
          double q = 0;
          for (std::size_t m = 0; m < M; ++m) q += w[m] * p[m];
          mp.probability(t, i) = q;
        }
      out.weights.push_back(w);
      out.choice.push_back(chosen);
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
