#include "ofm/daybook.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace ofm {
namespace day {

// Chauvenet's bound: the |z| beyond which fewer than half an observation of n is expected under a
// normal law, n * P(|Z| > z) = 1/2. Values further out are outliers and are cut to it.
double chauvenet(double n) {
  if (!(n > 1)) return kNone;
  double lo = 0, hi = 40;
  for (int k = 0; k < 100; ++k) {
    const double z = 0.5 * (lo + hi);
    (std::erfc(z / std::sqrt(2.0)) * n > 0.5 ? lo : hi) = z;
  }
  return hi;
}

const std::vector<double>& grid() {
  static const std::vector<double> g = {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, kNone};
  return g;
}

double tradeReturn(const Bar& b, double sig, double ks, double kt, double buy, double sell, bool worstCase, int* exit,
                   const std::vector<Tick>* path, bool* resolved) {
  const double stop = std::isfinite(ks) ? b.o * std::exp(-ks * sig) : 0.0;
  const double take = std::isfinite(kt) ? b.o * std::exp(kt * sig) : kNone;
  const bool hitStop = b.l <= stop, hitTake = b.h >= take;
  bool stopFirst = worstCase;
  if (resolved) *resolved = false;
  if (hitStop && hitTake && path) {
    std::size_t ts = path->size(), tt = path->size();
    for (std::size_t k = 0; k < path->size(); ++k) {
      if (ts == path->size() && (*path)[k].l <= stop) ts = k;
      if (tt == path->size() && (*path)[k].h >= take) tt = k;
    }
    if (ts != tt && ts < path->size() && tt < path->size()) {
      stopFirst = ts < tt;
      if (resolved) *resolved = true;
    }
  }
  double px = b.c;
  *exit = 0;
  if (hitStop && (stopFirst || !hitTake)) px = stop, *exit = 1;
  else if (hitTake) px = take, *exit = 2;
  return (px / b.o) * (1 - sell) / (1 + buy) - 1;
}

double Moments::sd() const {
  if (n <= 0) return 0;
  const double mu = s / n;
  return std::sqrt(std::max(0.0, s2 / n - mu * mu));
}

void Scaler::add(const std::vector<double>& x) {
  for (std::size_t i = 0; i < x.size(); ++i) s[i].add(x[i]);
}

std::vector<double> Scaler::apply(std::vector<double> x) const {
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (s[i].n < 2) {
      x[i] = 0;
      continue;
    }
    const double mu = s[i].s / s[i].n, sd = std::sqrt(std::max(1e-18, s[i].s2 / s[i].n - mu * mu));
    const double z = chauvenet(s[i].n);
    x[i] = std::clamp((x[i] - mu) / sd, -z, z);
  }
  return x;
}

void Adwin::push(double x) {
  p1.push_back(p1.back() + x), p2.push_back(p2.back() + x * x);
  for (bool moved = true; moved;) {
    moved = false;
    const std::size_t end = p1.size() - 1, n = end - from;
    const std::size_t side = std::max<std::size_t>(2, std::size_t(std::ceil(std::sqrt(double(n)))));
    if (n < 2 * side) return;
    const double lt = std::log(2.0 * std::log(double(n)) * horizon);
    for (std::size_t k = from + side; k + side <= end; ++k) {
      const double n0 = double(k - from), n1 = double(end - k);
      const double m0 = (p1[k] - p1[from]) / n0, m1 = (p1[end] - p1[k]) / n1;
      const double v0 = std::max(0.0, (p2[k] - p2[from]) / n0 - m0 * m0), v1 = std::max(0.0, (p2[end] - p2[k]) / n1 - m1 * m1);
      if (std::fabs(m0 - m1) > std::sqrt(2.0 * lt * (v0 / n0 + v1 / n1))) {
        from = k, moved = true;
        break;
      }
    }
  }
}

double Adwin::score() const {
  const std::size_t end = p1.size() - 1;
  const double n = double(end - from);
  if (n < 2) return 0.0;
  const double m = (p1[end] - p1[from]) / n, v = std::max(0.0, ((p2[end] - p2[from]) - n * m * m) / (n - 1));
  if (!(v > 0)) return 0.0;
  return m * std::max(0.0, 1.0 - v / n / (m * m));
}

}  // namespace day

namespace {

// Least squares by normal equations, accumulated over every observation so far.
struct LeastSquares {
  std::size_t k;
  std::vector<double> A, b;
  double n = 0;
  explicit LeastSquares(std::size_t dim) : k(dim), A(dim * dim, 0.0), b(dim, 0.0) {}
  void add(const std::vector<double>& x, double y) {
    for (std::size_t i = 0; i < k; ++i) {
      b[i] += x[i] * y;
      for (std::size_t j = 0; j < k; ++j) A[i * k + j] += x[i] * x[j];
    }
    n += 1;
  }
  // The coefficients (empty until there are more observations than unknowns or if singular).
  std::vector<double> solve() const {
    if (n <= double(k)) return {};
    std::vector<double> M = A, v = b;
    double tr = 0;
    for (std::size_t i = 0; i < k; ++i) tr += M[i * k + i];
    for (std::size_t i = 0; i < k; ++i) M[i * k + i] += 1e-12 * tr;  // numerical floor only
    for (std::size_t c = 0; c < k; ++c) {
      std::size_t piv = c;
      for (std::size_t r = c + 1; r < k; ++r)
        if (std::fabs(M[r * k + c]) > std::fabs(M[piv * k + c])) piv = r;
      if (!(std::fabs(M[piv * k + c]) > 0)) return {};
      if (piv != c) {
        for (std::size_t j = 0; j < k; ++j) std::swap(M[c * k + j], M[piv * k + j]);
        std::swap(v[c], v[piv]);
      }
      for (std::size_t r = c + 1; r < k; ++r) {
        const double q = M[r * k + c] / M[c * k + c];
        for (std::size_t j = c; j < k; ++j) M[r * k + j] -= q * M[c * k + j];
        v[r] -= q * v[c];
      }
    }
    std::vector<double> x(k);
    for (std::size_t c = k; c-- > 0;) {
      double s = v[c];
      for (std::size_t j = c + 1; j < k; ++j) s -= M[c * k + j] * x[j];
      x[c] = s / M[c * k + c];
    }
    return x;
  }
};

// A small neural network (two hidden layers of rectified linear units, linear outputs), trained
// online by Adam on the squared error, with a mask per output (an output without an honest target
// is not trained). Deterministic initialisation (He), so runs are reproducible.
struct Mlp {
  std::vector<std::size_t> size;
  std::vector<std::vector<double>> W, b, mW, vW, mb, vb;  // per layer: weights (out x in), biases, Adam moments
  double lr = 1e-3, b1 = 0.9, b2 = 0.999, steps = 0;
  explicit Mlp(std::vector<std::size_t> sz) : size(std::move(sz)) {
    std::uint64_t r = 0x9E3779B97F4A7C15ull;
    auto unif = [&]() { r ^= r << 13, r ^= r >> 7, r ^= r << 17; return double(r >> 11) / double(1ull << 53); };
    for (std::size_t l = 0; l + 1 < size.size(); ++l) {
      const std::size_t in = size[l], out = size[l + 1];
      W.emplace_back(in * out), b.emplace_back(out, 0.0);
      const double sd = std::sqrt(2.0 / double(in));
      for (auto& w : W.back()) {
        const double u1 = std::max(1e-12, unif()), u2 = unif();
        w = sd * std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2);
      }
      mW.emplace_back(in * out, 0.0), vW.emplace_back(in * out, 0.0), mb.emplace_back(out, 0.0), vb.emplace_back(out, 0.0);
    }
  }
  // Activations of every layer (a[0] the input).
  void forward(const std::vector<double>& x, std::vector<std::vector<double>>* a) const {
    a->assign(size.size(), {});
    (*a)[0] = x;
    for (std::size_t l = 0; l + 1 < size.size(); ++l) {
      const std::size_t in = size[l], out = size[l + 1];
      auto& y = (*a)[l + 1];
      y.assign(out, 0.0);
      for (std::size_t o = 0; o < out; ++o) {
        double z = b[l][o];
        const double* w = &W[l][o * in];
        for (std::size_t k = 0; k < in; ++k) z += w[k] * (*a)[l][k];
        y[o] = l + 2 < size.size() ? std::max(0.0, z) : z;
      }
    }
  }
  std::vector<double> predict(const std::vector<double>& x) const {
    std::vector<std::vector<double>> a;
    forward(x, &a);
    return a.back();
  }
  // One Adam step on a mini-batch of (input, target, mask).
  void step(const std::vector<const std::vector<double>*>& xs, const std::vector<const std::vector<double>*>& ys,
            const std::vector<const std::vector<char>*>& ms) {
    const std::size_t L = size.size() - 1;
    std::vector<std::vector<double>> gW(L), gb(L);
    for (std::size_t l = 0; l < L; ++l) gW[l].assign(W[l].size(), 0.0), gb[l].assign(b[l].size(), 0.0);
    std::vector<std::vector<double>> a;
    for (std::size_t s = 0; s < xs.size(); ++s) {
      forward(*xs[s], &a);
      std::vector<double> d(size[L]);
      for (std::size_t o = 0; o < size[L]; ++o) d[o] = (*ms[s])[o] ? a[L][o] - (*ys[s])[o] : 0.0;
      for (std::size_t l = L; l-- > 0;) {
        const std::size_t in = size[l], out = size[l + 1];
        std::vector<double> dIn(in, 0.0);
        for (std::size_t o = 0; o < out; ++o) {
          if (d[o] == 0) continue;
          gb[l][o] += d[o];
          double* g = &gW[l][o * in];
          const double* w = &W[l][o * in];
          for (std::size_t k = 0; k < in; ++k) g[k] += d[o] * a[l][k], dIn[k] += d[o] * w[k];
        }
        if (l > 0)
          for (std::size_t k = 0; k < in; ++k) dIn[k] = a[l][k] > 0 ? dIn[k] : 0.0;
        d.swap(dIn);
      }
    }
    steps += 1;
    const double n = double(xs.size()), c1 = 1 - std::pow(b1, steps), c2 = 1 - std::pow(b2, steps);
    auto adam = [&](std::vector<double>& w, std::vector<double>& m, std::vector<double>& v, const std::vector<double>& g) {
      for (std::size_t k = 0; k < w.size(); ++k) {
        const double gk = g[k] / n;
        m[k] = b1 * m[k] + (1 - b1) * gk, v[k] = b2 * v[k] + (1 - b2) * gk * gk;
        w[k] -= lr * (m[k] / c1) / (std::sqrt(v[k] / c2) + 1e-8);
      }
    };
    for (std::size_t l = 0; l < L; ++l) adam(W[l], mW[l], vW[l], gW[l]), adam(b[l], mb[l], vb[l], gb[l]);
  }
};

// Exponentially weighted variances at several half-lives; the half-life used is the one whose
// one-step forecasts have the lowest QLIKE loss so far, pooled over every series.
struct AdaptiveVol {
  std::vector<double> lambda, loss;
  std::vector<std::vector<double>> v;  // per series, per half-life
  std::vector<std::size_t> n;
  AdaptiveVol(std::size_t series, const std::vector<std::uint32_t>& halfLives) : loss(halfLives.size(), 0.0), v(series), n(series, 0) {
    for (auto h : halfLives) lambda.push_back(std::exp(-std::log(2.0) / double(h)));
    for (auto& x : v) x.assign(halfLives.size(), 0.0);
  }
  void add(std::size_t i, double r) {
    const double r2 = r * r;
    for (std::size_t h = 0; h < lambda.size(); ++h) {
      double& x = v[i][h];
      if (n[i] > 0 && x > 0) loss[h] += std::log(x) + r2 / x;
      x = n[i] == 0 ? r2 : lambda[h] * x + (1 - lambda[h]) * r2;
    }
    ++n[i];
  }
  std::size_t best() const { return std::size_t(std::min_element(loss.begin(), loss.end()) - loss.begin()); }
  double sigma(std::size_t i) const { return n[i] >= 2 ? std::sqrt(v[i][best()]) : 0.0; }
};

}  // namespace

DayBook dayBook(const Market& m, const Forecast& f, std::size_t start, double buy, double sell, const day::Paths* paths, std::size_t candidatesFrom) {
  using namespace day;
  DayBook book;
  book.buy = buy, book.sell = sell, book.start = start;
  const std::size_t T = m.T(), N = m.N(), K = 10, F = 10;
  if (start == 0 || start >= T || f.E.T != T) return book;
  std::vector<std::uint32_t> halfLives = f.horizons;
  if (halfLives.empty()) halfLives = {2, 4, 8, 16, 32, 64, 128, 256};
  // Multiples of the volatility, from the model's horizons (dyadic, set by the length of the
  // history): h / h_m for each horizon h, h_m the middle one. They give the stops and take-profits
  // (every pair, both set) and the candidate stop limits.
  std::vector<double> mult;
  for (auto h : halfLives) mult.push_back(double(h) / double(halfLives[halfLives.size() / 2]));
  const std::size_t G = mult.size(), P = G * G;

  auto barAt = [&](std::size_t d, std::size_t i, Bar* b) {
    *b = {m.open(d, i), m.high(d, i), m.low(d, i), m.close(d, i), m.volume(d, i)};
    // A flat bar (high = low) is a stale quote, not a trading day.
    return b->o > 0 && b->h > b->l && b->l > 0 && b->c > 0 && b->l <= std::min(b->o, b->c) && b->h >= std::max(b->o, b->c) && std::isfinite(b->v) &&
           b->v > 0;
  };
  // Volatility of each stock's open-to-close return, and of the equal-weight market's (series N).
  AdaptiveVol vol(N + 1, halfLives);
  // Each stock's opening noise: the exponentially weighted mean of its drop from the open to the
  // day's low, log(open / low), at each half-life (the volatility's chosen one is used). A stop
  // inside it cannot be tested honestly on daily bars (the order of the day's moves is unknown).
  std::vector<std::vector<double>> noise(N, std::vector<double>(halfLives.size(), 0.0));
  std::vector<std::size_t> noiseN(N, 0);
  std::vector<double> lam;
  for (auto h : halfLives) lam.push_back(std::exp(-std::log(2.0) / double(h)));
  std::vector<Moments> logVolume(N);
  // Each stock's own daily moves from the open: its falls to the low, log(open / low), and its rises
  // to the high, log(high / open), over its last W days (W the model's longest horizon). The stop
  // and take-profit candidates are their quantiles at (k + 1) / G, k < G, up to the largest: every
  // level lies inside the moves the stock really makes in a day.
  const std::size_t W = halfLives.back();
  std::vector<std::vector<double>> falls(N), rises(N), stopAt(N), takeAt(N);
  auto levels = [&](std::size_t i) {
    stopAt[i].clear(), takeAt[i].clear();
    if (falls[i].size() < G + 1) return;
    for (int side = 0; side < 2; ++side) {
      std::vector<double> v = side == 0 ? falls[i] : rises[i];
      std::sort(v.begin(), v.end());
      auto& out = side == 0 ? stopAt[i] : takeAt[i];
      for (std::size_t k = 0; k < G; ++k) {
        const double pos = double(k + 1) / double(G) * double(v.size() - 1);
        const std::size_t lo = std::size_t(pos);
        const double fr = pos - double(lo);
        out.push_back(lo + 1 < v.size() ? v[lo] * (1 - fr) + v[lo + 1] * fr : v[lo]);
      }
    }
  };
  auto addDay = [&](std::size_t t) {
    Bar b;
    double s = 0, n = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (barAt(t, i, &b)) {
        const double r = std::log(b.c / b.o);
        vol.add(i, r), s += r, n += 1;
        const double x = std::log(b.o / b.l);
        for (std::size_t h = 0; h < lam.size(); ++h) noise[i][h] = noiseN[i] == 0 ? x : lam[h] * noise[i][h] + (1 - lam[h]) * x;
        ++noiseN[i];
        falls[i].push_back(x), rises[i].push_back(std::log(b.h / b.o));
        if (falls[i].size() > W) falls[i].erase(falls[i].begin()), rises[i].erase(rises[i].begin());
        levels(i);
      }
    if (n > 0) vol.add(N, s / n);
  };
  for (std::size_t t = 0; t + 1 < start; ++t) addDay(t);
  for (std::size_t t = 0; t + 1 < start; ++t) {
    Bar b;
    for (std::size_t i = 0; i < N; ++i)
      if (barAt(t, i, &b)) logVolume[i].add(std::log(b.v));
  }
  auto sigmaOf = [&](std::size_t i) { return vol.sigma(i); };
  // Pair q of stock i: its stop and take-profit distances from the open (log), from its own moves.
  auto stopOf = [&](std::size_t i, std::size_t q) { return stopAt[i].empty() ? kNone : stopAt[i][q / G]; };
  auto takeOf = [&](std::size_t i, std::size_t q) { return takeAt[i].empty() ? kNone : takeAt[i][q % G]; };
  auto pathOf = [&](std::size_t d, std::size_t i) -> const std::vector<Tick>* {
    if (!paths) return nullptr;
    const auto it = paths->find(d * N + i);
    return it == paths->end() ? nullptr : &it->second;
  };
  // Features of stock i at the close t (empty if its recent bars are missing).
  auto features = [&](std::size_t t, std::size_t i, double medSig, double mkt) {
    Bar b0, b1, b5;
    const double sig = sigmaOf(i);
    if (t < 5 || !(sig > 0) || !std::isfinite(f.E(t, i)) || !barAt(t, i, &b0) || !barAt(t - 1, i, &b1) || !barAt(t - 5, i, &b5) || logVolume[i].n < 2)
      return std::vector<double>{};
    const double lvMu = logVolume[i].mean(), lvSd = std::sqrt(std::max(1e-12, logVolume[i].s2 / logVolume[i].n - lvMu * lvMu));
    return std::vector<double>{f.E(t, i) / sig, std::log(sig / medSig), std::log(b0.c / b0.o) / sig, std::log(b0.c / b1.c) / sig,
                               std::log(b0.o / b1.c) / sig, b0.h > b0.l ? (b0.c - b0.l) / (b0.h - b0.l) : 0.5, std::log(b0.h / b0.l) / sig,
                               std::log(b0.c / b5.c) / sig, (std::log(b0.v) - lvMu) / lvSd, mkt / medSig};
  };
  // Technical indicators of stock i at the close t, at each of the model's horizons up to 64 days:
  // momentum (in its volatility), the close's place in the range, RSI (net share of up moves),
  // volume against its mean, and the day's range against its mean.
  std::vector<std::uint32_t> indH;
  for (auto h : halfLives)
    if (h <= 64) indH.push_back(h);
  const std::size_t FI = 5 * indH.size();
  auto indicators = [&](std::size_t t, std::size_t i) {
    std::vector<double> x(FI, 0.0);
    Bar bt;
    if (!barAt(t, i, &bt)) return x;
    const double sig = std::max(1e-9, sigmaOf(i));
    for (std::size_t k = 0; k < indH.size(); ++k) {
      const std::size_t h = indH[k];
      if (t < h) continue;
      double hi = 0, lo = kNone, up = 0, dn = 0, lv = 0, lr = 0, n = 0, prev = 0, first = 0;
      Bar b;
      for (std::size_t u = t - h; u <= t; ++u) {
        if (!barAt(u, i, &b)) continue;
        if (n > 0) (b.c > prev ? up : dn) += std::fabs(std::log(b.c / prev));
        else first = b.c;
        prev = b.c, hi = std::max(hi, b.h), lo = std::min(lo, b.l), lv += std::log(b.v), lr += std::log(b.h / b.l), n += 1;
      }
      if (n < 2 || !(hi > lo)) continue;
      x[5 * k + 0] = std::log(bt.c / first) / (sig * std::sqrt(double(h)));
      x[5 * k + 1] = (bt.c - lo) / (hi - lo) - 0.5;
      x[5 * k + 2] = up + dn > 0 ? (up - dn) / (up + dn) : 0.0;
      x[5 * k + 3] = std::log(bt.v) - lv / n;
      x[5 * k + 4] = lr > 0 ? std::log(std::log(bt.h / bt.l) / (lr / n)) : 0.0;
    }
    return x;
  };
  Scaler scaler(F), indScaler(FI);
  LeastSquares intraday(F + 1);                            // open-to-close return / sigma on [1, features]
  std::vector<LeastSquares> pairFit(P, LeastSquares(2));    // pair outcome / sigma on [1, intraday forecast]
  // Each stock's record; each book's record: a stop limit (G) and a book size, the n strongest
  // trades (1 to 10), so the trades can keep to the strongest signals when that has paid.
  // Two learners of each trade's stop/take-profit outcomes: 0 linear (least squares on the intraday
  // forecast), 1 a neural network on the trade's features (every pair's outcome and the intraday
  // move at once). Each (learner, stop limit) keeps its own book; the best record trades.
  const std::size_t M = 2;
  Mlp net({F + 1 + FI, 32, 32, 1 + P});
  std::vector<Adwin> record(N), limitRecord(G * M);
  for (auto& a : record) a.horizon = double(T - start);
  for (auto& a : limitRecord) a.horizon = double(T - start);
  // The day's median volatility over the candidates and the market's close-to-close move at t.
  auto dayContext = [&](std::size_t t, const std::vector<std::size_t>& cand, double* medSig, double* mkt) {
    std::vector<double> sigs;
    for (std::size_t i : cand) sigs.push_back(sigmaOf(i));
    std::sort(sigs.begin(), sigs.end());
    *medSig = sigs.empty() ? 1.0 : std::max(1e-9, sigs[sigs.size() / 2]);
    double s = 0, n = 0;
    Bar b0, b1;
    for (std::size_t i = 0; i < N; ++i)
      if (t > 0 && barAt(t, i, &b0) && barAt(t - 1, i, &b1)) s += std::log(b0.c / b1.c), n += 1;
    *mkt = n > 0 ? s / n : 0.0;
  };
  // Each candidate at the close t: its scaled features, its intraday forecast mu (in its
  // volatility) and every pair's expected net log return (in its volatility); empty if unknown.
  struct TradePlan {
    std::vector<double> xs, e, ind, indRaw;  // scaled features, expected pair outcomes, indicators (scaled, raw)
    double mu = 0;
  };
  auto plan = [&](std::size_t t, const std::vector<std::size_t>& cand, double medSig, double mkt) {
    const auto beta = intraday.solve();
    std::vector<std::vector<double>> ab(P);
    for (std::size_t q = 0; q < P; ++q) ab[q] = pairFit[q].solve();
    std::vector<TradePlan> out(N);
    for (std::size_t i : cand) {
      const auto x = features(t, i, medSig, mkt);
      if (x.empty()) continue;
      TradePlan& pl = out[i];
      pl.xs = scaler.apply(x);
      pl.indRaw = indicators(t, i);
      pl.ind = indScaler.apply(pl.indRaw);
      if (beta.empty()) continue;
      pl.mu = beta[0];
      for (std::size_t j = 0; j < F; ++j) pl.mu += beta[j + 1] * pl.xs[j];
      pl.e.assign(P, -kNone);
      for (std::size_t q = 0; q < P; ++q)
        if (!ab[q].empty()) pl.e[q] = ab[q][0] + ab[q][1] * pl.mu;
    }
    return out;
  };
  // The network's plan: its own intraday forecast and every pair's expected outcome, from the
  // trade's features and the linear forecast.
  auto netInput = [&](const TradePlan& p) {
    std::vector<double> x = p.xs;
    x.push_back(p.mu);
    x.insert(x.end(), p.ind.begin(), p.ind.end());
    return x;
  };
  auto netPlan = [&](const std::vector<std::size_t>& cand, const std::vector<TradePlan>& lin) {
    std::vector<TradePlan> out(N);
    if (net.steps == 0) return out;
    for (std::size_t i : cand) {
      if (lin[i].xs.empty() || lin[i].e.empty()) continue;
      const auto y = net.predict(netInput(lin[i]));
      out[i].xs = lin[i].xs, out[i].mu = y[0];
      out[i].e.assign(y.begin() + 1, y.end());
    }
    return out;
  };
  // The opening-noise floor of stock i: no floor where its intraday path gives the exact fills.
  auto floorOf = [&](std::size_t i, bool exact) { return exact ? 0.0 : noiseN[i] >= 2 ? noise[i][vol.best()] : kNone; };
  // Stock i's best pair whose stop is within the limit (log distance), and its expected return.
  auto bestPair = [&](std::size_t i, const TradePlan& pl, double limit, std::size_t* pair, bool exact) {
    double best = -kNone;
    if (pl.e.empty() || stopAt[i].empty()) return best;
    const double floor = floorOf(i, exact);
    for (std::size_t q = 0; q < P; ++q) {
      const double dist = stopOf(i, q);
      if (dist >= floor && dist <= limit && pl.e[q] > best) best = pl.e[q], *pair = q;
    }
    return best;
  };
  // A book under a stop limit: down the ranking of expected net return, up to 10 trades expected to
  // gain whose stock's record beats cash; equal shares, scaled so the book's expected volatility is
  // at most the market's.
  struct Book {
    std::vector<std::size_t> pick, pair;
    std::vector<double> expect;
    double w = 0;
  };
  // exact(i): whether stock i's fills are exact (an intraday path) on the day traded.
  auto ranked = [&](const std::vector<std::size_t>& cand, const std::vector<TradePlan>& pl, double limit, const auto& exact) {
    std::vector<std::tuple<double, std::size_t, std::size_t, double>> v;  // (expected return, stock, pair, expect)
    for (std::size_t i : cand) {
      std::size_t q = 0;
      const double e = bestPair(i, pl[i], limit, &q, exact(i));
      if (e > 0 && record[i].score() > 0) v.emplace_back(e * sigmaOf(i), i, q, e);
    }
    std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& c) { return std::get<0>(a) > std::get<0>(c); });
    if (v.size() > K) v.resize(K);
    return v;
  };
  // The book of the n strongest: equal shares, scaled so its expected volatility is at most the market's.
  auto makeBook = [&](const std::vector<std::tuple<double, std::size_t, std::size_t, double>>& v, std::size_t n) {
    Book bk;
    for (std::size_t j = 0; j < std::min(n, v.size()); ++j) {
      const auto& [r, i, q, e] = v[j];
      bk.pick.push_back(i), bk.pair.push_back(q), bk.expect.push_back(e);
    }
    if (bk.pick.empty()) return bk;
    const double m0 = double(bk.pick.size());
    double var = 0;
    for (std::size_t i : bk.pick) var += sigmaOf(i) * sigmaOf(i) / (m0 * m0);
    const double mk = vol.sigma(N);
    bk.w = (1 / m0) * ((var > 0 && mk > 0) ? std::min(1.0, mk / std::sqrt(var)) : 1.0);
    return bk;
  };
  // The stop limit in use: the one whose book's record is best so far; cash while none beats cash.
  auto bestLimit = [&]() {
    std::size_t k = 0;
    for (std::size_t j = 1; j < G * M; ++j)
      if (limitRecord[j].score() > limitRecord[k].score()) k = j;
    return k;
  };
  auto dayReturn = [&](const Book& bk, std::size_t d, bool count, bool worstCase = true) {
    double r = 0;
    Bar b;
    for (std::size_t j = 0; j < bk.pick.size(); ++j) {
      barAt(d, bk.pick[j], &b);
      int e;
      bool res;
      const auto* path = pathOf(d, bk.pick[j]);
      r += bk.w * tradeReturn(b, 1.0, stopOf(bk.pick[j], bk.pair[j]), takeOf(bk.pick[j], bk.pair[j]), buy, sell, worstCase, &e, path, &res);
      if (count) ++book.trades, book.stops += e == 1, book.takes += e == 2, book.pathTrades += path != nullptr, book.pathResolved += res;
    }
    return r;
  };

  Bar b;
  std::vector<double> zp(P);
  for (std::size_t d = start; d < T; ++d) {
    const std::size_t t = d - 1;
    if (t > 0) addDay(t);
    // Candidates: a forecast at the close t, a volatility, and a bar on day d.
    std::vector<std::size_t> cand;
    for (std::size_t i = 0; i < N; ++i)
      if (std::isfinite(f.E(t, i)) && sigmaOf(i) > 0 && barAt(d, i, &b)) cand.push_back(i);
    double medSig, mkt;
    dayContext(t, cand, &medSig, &mkt);
    const auto pl = plan(t, cand, medSig, mkt);
    const std::vector<TradePlan> pls[M] = {pl, netPlan(cand, pl)};
    // Every (learner, stop limit) book; the one in use trades.
    const std::size_t use = bestLimit();
    const bool trade = limitRecord[use].score() > 0;
    std::vector<Book> books(G * M);
    auto exactOn = [&](std::size_t i) { return pathOf(d, i) != nullptr; };
    for (std::size_t mo = 0; mo < M; ++mo)
      for (std::size_t k = 0; k < G; ++k) books[mo * G + k] = makeBook(ranked(cand, pls[mo], mult[k] * medSig, exactOn), K);
    // The strongest candidates: every stock a book could trade, with and without exact fills.
    if (d >= candidatesFrom) {
      std::vector<char> seen(N, 0);
      for (std::size_t mo = 0; mo < M; ++mo)
        for (int ex = 0; ex < 2; ++ex)
          for (std::size_t k = 0; k < G; ++k)
            for (const auto& c : ranked(cand, pls[mo], mult[k] * medSig, [&](std::size_t) { return ex == 1; }))
              if (!seen[std::get<1>(c)]) seen[std::get<1>(c)] = 1, book.candidates.emplace_back(d, std::get<1>(c));
    }
    const double r = trade ? dayReturn(books[use], d, true) : 0.0;
    // Fill test: the same trades with the opposite guess when a bar touched both levels.
    book.optimistic.push_back(trade ? dayReturn(books[use], d, false, false) : 0.0);
    book.ret.push_back(r), book.booked.push_back(trade ? int(books[use].pick.size()) : 0);
    book.exposure.push_back(trade ? books[use].w * double(books[use].pick.size()) : 0.0);
    for (std::size_t k = 0; k < G * M; ++k) limitRecord[k].push(std::log1p(dayReturn(books[k], d, false)));
    // After the close d: the day's outcomes, for the stocks' records and the fits.
    for (std::size_t q = 0; q < P; ++q) zp[q] = chauvenet(pairFit[q].n + 1);
    const double zb = chauvenet(intraday.n + 1);
    std::vector<std::vector<double>> sx, sy;
    std::vector<std::vector<char>> sm;
    for (std::size_t i : cand) {
      const TradePlan& p = pl[i];
      if (p.xs.empty()) continue;
      barAt(d, i, &b);
      const double sig = sigmaOf(i);
      const auto* path = pathOf(d, i);
      book.pathDays += path != nullptr;
      int e;
      std::size_t q0 = 0;
      if (bestPair(i, p, kNone, &q0, path != nullptr) > 0)  // the stock's own best trade, whenever a gain is expected: its record
        record[i].push(std::log1p(tradeReturn(b, 1.0, stopOf(i, q0), takeOf(i, q0), buy, sell, true, &e, path)));
      std::vector<double> x1{1.0};
      x1.insert(x1.end(), p.xs.begin(), p.xs.end());
      const double raw = std::log(b.c / b.o) / sig, z = std::clamp(raw, -zb, zb);
      book.maxTarget = std::max(book.maxTarget, std::fabs(raw));
      book.bigTargets += std::fabs(raw) > zb;
      // Out-of-sample skill of the intraday forecast: its correlation with what happened.
      if (!p.e.empty()) book.skill.add(p.mu, z);
      intraday.add(x1, z);
      if (!p.e.empty()) {  // pair fits on genuine forecasts only
        // With intraday paths (where tight stops get traded), only honest outcomes: a stop outside
        // the opening noise, or exact fills from the path.
        const std::vector<double> mu1{1.0, p.mu};
        const double floor = paths && !paths->empty() ? floorOf(i, path != nullptr) : 0.0;
        std::vector<double> y(1 + P, 0.0);
        std::vector<char> mask(1 + P, 0);
        y[0] = z, mask[0] = 1;
        for (std::size_t q = 0; q < P; ++q)
          if (!stopAt[i].empty() && stopOf(i, q) >= floor) {
            y[1 + q] = std::clamp(std::log1p(tradeReturn(b, 1.0, stopOf(i, q), takeOf(i, q), buy, sell, true, &e, path)) / sig, -zp[q], zp[q]);
            mask[1 + q] = 1;
            pairFit[q].add(mu1, y[1 + q]);
          }
        sx.push_back(netInput(p)), sy.push_back(std::move(y)), sm.push_back(std::move(mask));
      }
    }
    // The network learns from the day's trades (one pass, mini-batches in a fixed shuffled order).
    if (!sx.empty()) {
      std::vector<std::size_t> perm(sx.size());
      for (std::size_t k = 0; k < perm.size(); ++k) perm[k] = k;
      std::uint64_t rng = 0x2545F4914F6CDD1Dull ^ d;
      for (std::size_t k = perm.size(); k-- > 1;) {
        rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
        std::swap(perm[k], perm[rng % (k + 1)]);
      }
      const std::size_t B = 32;
      for (std::size_t s0 = 0; s0 < perm.size(); s0 += B) {
        std::vector<const std::vector<double>*> bx, by;
        std::vector<const std::vector<char>*> bm;
        for (std::size_t k = s0; k < std::min(perm.size(), s0 + B); ++k) bx.push_back(&sx[perm[k]]), by.push_back(&sy[perm[k]]), bm.push_back(&sm[perm[k]]);
        net.step(bx, by, bm);
      }
    }
    for (std::size_t i : cand) {
      const auto x = features(t, i, medSig, mkt);
      if (!x.empty()) scaler.add(x), indScaler.add(pl[i].indRaw);
    }
    for (std::size_t i = 0; i < N; ++i)
      if (barAt(d, i, &b)) logVolume[i].add(std::log(b.v));
  }

  // Orders for the next open, from the last close.
  const std::size_t t = T - 1;
  addDay(t);
  book.halfLife = halfLives[vol.best()];
  std::vector<std::size_t> cand;
  for (std::size_t i = 0; i < N; ++i)
    if (std::isfinite(f.E(t, i)) && sigmaOf(i) > 0 && barAt(t, i, &b)) cand.push_back(i);
  double medSig, mkt;
  dayContext(t, cand, &medSig, &mkt);
  const auto pl0 = plan(t, cand, medSig, mkt);
  const std::size_t use = bestLimit();
  const auto pl = use / G == 1 ? netPlan(cand, pl0) : pl0;
  const double limit = mult[use % G] * medSig;
  book.bookSize = K, book.learner = use / G == 1 ? "neural network" : "linear";
  book.stopLimit = 1 - std::exp(-limit), book.limitScore = limitRecord[use].score();
  auto& sc = book.screen;
  sc.stocks = N, sc.candidates = cand.size();
  for (std::size_t i = 0; i < N; ++i)
    if (barAt(t, i, &b)) ++sc.bars, sc.flatBars += b.h == b.l;
  // Next open: fills are exact where intraday paths are kept (they will be fetched for these days).
  const bool exactNext = paths && !paths->empty();
  auto exactNextOf = [&](std::size_t) { return exactNext; };
  for (std::size_t i : cand) {
    std::size_t q = 0;
    const double e = bestPair(i, pl[i], limit, &q, exactNext);
    sc.forecastUp += !pl[i].e.empty() && pl[i].mu > 0, sc.gainExpected += e > 0, sc.recordUp += record[i].score() > 0;
    sc.all += e > 0 && record[i].score() > 0;
  }
  if (limitRecord[use].score() > 0) {
    const Book bk = makeBook(ranked(cand, pl, limit, exactNextOf), book.bookSize);
    for (std::size_t j = 0; j < bk.pick.size(); ++j) {
      const std::size_t i = bk.pick[j], q = bk.pair[j];
      DayOrder o;
      o.asset = i, o.expected = f.E(t, i), o.sigma = sigmaOf(i), o.stop = stopOf(i, q) / o.sigma, o.take = takeOf(i, q) / o.sigma;
      o.intraday = pl[i].mu * o.sigma, o.tradeReturn = std::expm1(bk.expect[j] * o.sigma), o.weight = bk.w;
      o.record = record[i].score(), o.window = record[i].window();
      book.orders.push_back(o);
    }
  }
  return book;
}

}  // namespace ofm
