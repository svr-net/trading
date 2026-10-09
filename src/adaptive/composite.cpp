#include "sat/adaptive/composite.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
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
  if (name == "market-adwin") return ScoringWindow::MarketAdwin;
  if (name == "similar-state") return ScoringWindow::SimilarState;
  throw std::invalid_argument("unknown scoring window '" + name + "' (fixed, exponential, adwin, market-adwin, similar-state)");
}

std::string scoringWindowName(ScoringWindow w) {
  switch (w) {
    case ScoringWindow::Fixed: return "fixed";
    case ScoringWindow::Exponential: return "exponential";
    case ScoringWindow::Adwin: return "adwin";
    case ScoringWindow::MarketAdwin: return "market-adwin";
    case ScoringWindow::SimilarState: return "similar-state";
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

// ADWIN (Bifet and Gavalda, 2007) over a growing series: the window [from, size) keeps every
// new value and drops its older part whenever some split shows the two parts' means differ by
// more than chance allows at confidence delta. The paper's bound assumes values in [0, 1]; its
// range term would dwarf daily market moves or information coefficients, so the scale-free
// form is used: a split is a change when |m0 - m1| > sqrt(2 ln(2 ln(n) / delta) (v0/n0 + v1/n1)),
// with each part at least 30 values long. At delta = 1e-4 this caught an 8x jump in volatility
// within about 19 days and a doubling within about 70, with no false cut in 25,000 simulated
// days of stationary noise.
class Adwin {
 public:
  explicit Adwin(double delta) : delta_(delta) { p1_.push_back(0.0), p2_.push_back(0.0); }

  void push(double v) {
    p1_.push_back(p1_.back() + v), p2_.push_back(p2_.back() + v * v);
    cut();
  }
  std::size_t size() const { return p1_.size() - 1; }
  std::size_t from() const { return from_; }
  double sum(std::size_t a, std::size_t b) const { return p1_[b] - p1_[a]; }
  double sumSquares(std::size_t a, std::size_t b) const { return p2_[b] - p2_[a]; }

 private:
  void cut() {
    constexpr std::size_t kMinSide = 30;
    for (bool changed = true; changed;) {
      changed = false;
      const std::size_t end = size(), n = end - from_;
      if (n < 2 * kMinSide) return;
      const double logTerm = std::log(2.0 * std::log(static_cast<double>(n)) / delta_);
      const std::size_t step = std::max<std::size_t>(1, n / 64);
      for (std::size_t k = from_ + kMinSide; k + kMinSide <= end; k += step) {
        const double n0 = static_cast<double>(k - from_), n1 = static_cast<double>(end - k);
        const double m0 = sum(from_, k) / n0, m1 = sum(k, end) / n1;
        const double v0 = std::max(0.0, sumSquares(from_, k) / n0 - m0 * m0);
        const double v1 = std::max(0.0, sumSquares(k, end) / n1 - m1 * m1);
        const double eps = std::sqrt(2.0 * logTerm * (v0 / n0 + v1 / n1));
        if (std::fabs(m0 - m1) > eps) {
          from_ = k;
          changed = true;
          break;
        }
      }
    }
  }

  double delta_;
  std::vector<double> p1_, p2_;
  std::size_t from_ = 0;
};

// The record of one candidate: daily values (information coefficients, or their differences
// from the average's) with their dates, pushed as their labels resolve, and the statistics of
// the part the scoring window remembers.
class Scorer {
 public:
  struct Stats {
    double mean = Panel::kMissing, sd = 0.0, n = 0.0;
  };

  explicit Scorer(const CompositeSpec& spec)
      : spec_(spec), lambda_(spec.halfLife > 0 ? std::pow(0.5, 1.0 / spec.halfLife) : 1.0), adwin_(spec.adwinDelta) {}

  void push(double v, std::size_t date) {
    x_.push_back(v);
    dates_.push_back(date);
    adwin_.push(v);
    s0_ = lambda_ * s0_ + 1.0, s1_ = lambda_ * s1_ + v, s2_ = lambda_ * s2_ + v * v, w2_ = lambda_ * lambda_ * w2_ + 1.0;
  }

  // Market-state ADWIN: forget the record before the date the market's current state began.
  void forgetBefore(std::size_t date) {
    while (marketFrom_ < dates_.size() && dates_[marketFrom_] < date) ++marketFrom_;
  }

  // Similar-state memory: weights from the market state of each recorded date.
  Stats weighted(const std::function<double(std::size_t)>& weightOfDate) const {
    Stats st;
    double w0 = 0, w1 = 0, w2 = 0, ww = 0;
    for (std::size_t k = 0; k < x_.size(); ++k) {
      const double w = weightOfDate(dates_[k]);
      w0 += w, w1 += w * x_[k], w2 += w * x_[k] * x_[k], ww += w * w;
    }
    if (!(w0 > 0)) return st;
    st.mean = w1 / w0;
    st.sd = std::sqrt(std::max(0.0, w2 / w0 - st.mean * st.mean));
    st.n = w0 * w0 / ww;
    return st;
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
    std::size_t from = 0;
    if (spec_.window == ScoringWindow::Fixed) from = end > spec_.lookback ? end - spec_.lookback : 0;
    else if (spec_.window == ScoringWindow::Adwin) from = adwin_.from();
    else if (spec_.window == ScoringWindow::MarketAdwin) from = std::max(adwin_.from(), marketFrom_);
    const double n = static_cast<double>(end - from);
    if (n < 1) return st;
    st.mean = adwin_.sum(from, end) / n;
    st.sd = std::sqrt(std::max(0.0, adwin_.sumSquares(from, end) / n - st.mean * st.mean));
    st.n = n;
    return st;
  }

 private:
  const CompositeSpec& spec_;
  double lambda_;
  Adwin adwin_;
  std::vector<double> x_;
  std::vector<std::size_t> dates_;
  std::size_t marketFrom_ = 0;
  double s0_ = 0, s1_ = 0, s2_ = 0, w2_ = 0;
};

}  // namespace

MarketState marketState(const Panel& nextReturns) {
  const std::size_t T = nextReturns.dates(), N = nextReturns.assets();
  std::vector<double> market(T, 0.0);
  for (std::size_t d = 1; d < T; ++d) {
    double sum = 0, n = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (std::isfinite(nextReturns(d - 1, i))) sum += nextReturns(d - 1, i), ++n;
    market[d] = n > 0 ? sum / n : 0.0;
  }
  MarketState st{std::vector<double>(T, Panel::kMissing), std::vector<double>(T, Panel::kMissing)};
  for (std::size_t d = 63; d < T; ++d) {
    double s1 = 0, s2 = 0, l1 = 0, l2 = 0;
    for (std::size_t k = d - 20; k <= d; ++k) s1 += market[k], s2 += market[k] * market[k];
    for (std::size_t k = d - 62; k <= d; ++k) l1 += market[k], l2 += market[k] * market[k];
    const double sv = std::sqrt(std::max(1e-12, s2 / 21 - (s1 / 21) * (s1 / 21)));
    const double lv = std::sqrt(std::max(1e-12, l2 / 63 - (l1 / 63) * (l1 / 63)));
    st.volatility[d] = std::log(sv);
    st.trend[d] = l1 / (lv * std::sqrt(63.0));
  }
  return st;
}

std::string stateSideName(StateSide s) {
  switch (s) {
    case StateSide::Calm: return "calm";
    case StateSide::Turbulent: return "turbulent";
    case StateSide::Rising: return "rising";
    case StateSide::Falling: return "falling";
  }
  return "calm";
}

Panel marketStateMask(const Panel& nextReturns, StateSide side) {
  const auto st = marketState(nextReturns);
  const std::size_t T = nextReturns.dates();
  Panel mask(T, nextReturns.assets(), 0.0);
  std::vector<double> seen;  // volatilities so far, for the running median
  for (std::size_t d = 0; d < T; ++d) {
    if (!std::isfinite(st.volatility[d])) continue;
    seen.insert(std::upper_bound(seen.begin(), seen.end(), st.volatility[d]), st.volatility[d]);
    const double median = seen[seen.size() / 2];
    bool in = false;
    switch (side) {
      case StateSide::Calm: in = st.volatility[d] <= median; break;
      case StateSide::Turbulent: in = st.volatility[d] > median; break;
      case StateSide::Rising: in = st.trend[d] > 0; break;
      case StateSide::Falling: in = st.trend[d] <= 0; break;
    }
    if (in)
      for (std::size_t i = 0; i < mask.assets(); ++i) mask(d, i) = 1.0;
  }
  return mask;
}

CompositePredictions compositePredictions(const std::vector<ModelPredictions>& members, const Panel& labels,
                                          const Panel& labelEnds, const CompositeSpec& spec, const Panel* nextReturns) {
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
    // The market state, for the market-driven windows: the equal-weight market's return of
    // each day (known at that day's close), its 21-day volatility and 63-day trend.
    const bool marketDriven = spec.window == ScoringWindow::MarketAdwin || spec.window == ScoringWindow::SimilarState;
    if (marketDriven && (!nextReturns || nextReturns->dates() != T || nextReturns->assets() != N))
      throw std::invalid_argument("self-adaptive forecast: the market-state windows need the next-day returns of the members' shape");
    std::vector<double> market(T, 0.0), vol(T, Panel::kMissing), trend(T, Panel::kMissing);
    if (marketDriven) {
      for (std::size_t d = 1; d < T; ++d) {
        double sum = 0, n = 0;
        for (std::size_t i = 0; i < N; ++i)
          if (std::isfinite((*nextReturns)(d - 1, i))) sum += (*nextReturns)(d - 1, i), ++n;
        market[d] = n > 0 ? sum / n : 0.0;
      }
      const auto state = sat::marketState(*nextReturns);
      vol = state.volatility;
      trend = state.trend;
    }
    Adwin marketMoves(spec.adwinDelta);  // on the size of the market's daily moves
    std::size_t marketFed = 1;
    // Fixed windows re-score on the selector's schedule; the others every day.
    const std::size_t step = spec.window == ScoringWindow::Fixed ? spec.adaptEvery : 1;
    std::size_t next = s;  // first date not yet added to the records
    int chosen = -1;
    double memoryNow = 0;
    for (std::size_t t = s; t < e; ++t) {
      // Incremental: add each date whose labels have all resolved before t.
      for (; next < t && resolved(labels, labelEnds, next, t); ++next) {
        std::vector<double> v(M + 1);
        bool ok = true;
        for (std::size_t c = 0; c <= M && ok; ++c) ok = std::isfinite(v[c] = dailyIc(c, next));
        if (!ok) continue;
        for (std::size_t c = 0; c < K; ++c) records[c].push(evidence ? v[c] - v[M] : v[c], next);
      }
      // Market-state ADWIN: feed the market's moves up to today's close; when its state
      // changes, every record recedes to the start of the new state.
      if (spec.window == ScoringWindow::MarketAdwin) {
        for (; marketFed <= t; ++marketFed) marketMoves.push(std::fabs(market[marketFed]));
        const std::size_t since = 1 + marketMoves.from();  // date of the state's first day
        for (auto& r : records) r.forgetBefore(since);
      }
      if ((t - s) % step == 0) {
        chosen = -1;
        memoryNow = 0;
        std::vector<Scorer::Stats> st;
        if (spec.window == ScoringWindow::SimilarState && std::isfinite(vol[t])) {
          // Weights from how close each recorded date's market state is to today's, in units of
          // each feature's spread over the dates seen so far.
          double v1 = 0, v2 = 0, g1 = 0, g2 = 0, n = 0;
          for (std::size_t d = 63; d <= t; ++d) v1 += vol[d], v2 += vol[d] * vol[d], g1 += trend[d], g2 += trend[d] * trend[d], ++n;
          const double sv = std::sqrt(std::max(1e-12, v2 / n - (v1 / n) * (v1 / n)));
          const double sg = std::sqrt(std::max(1e-12, g2 / n - (g1 / n) * (g1 / n)));
          const double h2 = spec.stateBandwidth * spec.stateBandwidth;
          auto weight = [&](std::size_t d) {
            if (!std::isfinite(vol[d])) return 0.0;
            const double a = (vol[d] - vol[t]) / sv, b = (trend[d] - trend[t]) / sg;
            return std::exp(-0.5 * (a * a + b * b) / h2);
          };
          for (const auto& r : records) st.push_back(r.weighted(weight));
        } else {
          for (const auto& r : records) st.push_back(spec.window == ScoringWindow::SimilarState ? Scorer::Stats{} : r.stats());
        }
        memoryNow = st.empty() ? 0.0 : st[0].n;
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
      out.memory.push_back(memoryNow);
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
