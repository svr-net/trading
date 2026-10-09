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
    // Daily information coefficient of each candidate, computed once per resolved date.
    std::vector<std::vector<double>> ic(M + 1, std::vector<double>(T, Panel::kMissing));
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
    int chosen = -1;
    for (std::size_t t = s; t < e; ++t) {
      if ((t - s) % spec.adaptEvery == 0) {
        double best = 0.0;
        chosen = -1;
        std::vector<double> score(M + 1, Panel::kMissing);
        for (std::size_t c = 0; c <= M; ++c) {
          double sum = 0;
          std::size_t n = 0;
          for (std::size_t u = t; u-- > s && n < spec.lookback;) {
            if (!resolved(labels, labelEnds, u, t)) continue;
            if (!std::isfinite(ic[c][u])) ic[c][u] = dailyIc(c, u);
            if (std::isfinite(ic[c][u])) sum += ic[c][u], ++n;
          }
          if (n >= 5) score[c] = sum / static_cast<double>(n);
        }
        // The average, unless a member beats it with a positive record; the average
        // stands in when nothing scores above zero.
        if (std::isfinite(score[M]) && score[M] > best) best = score[M];
        for (std::size_t c = 0; c < M; ++c)
          if (std::isfinite(score[c]) && score[c] > best) best = score[c], chosen = static_cast<int>(c);
      }
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
