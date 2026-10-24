#include "ofm/daybook.hpp"

#include "daykernels.hpp"

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

double tradeReturn(const Bar& b, double sig, double ks, double kt, double buy, double sell, bool worstCase, int* exit,
                   const std::vector<Tick>* path, bool* resolved, double half) {
  const double stop = std::isfinite(ks) ? b.o * std::exp(-ks * sig) : 0.0;
  const double take = std::isfinite(kt) ? b.o * std::exp(kt * sig) : kNone;
  // A take-profit (a limit order) fills only when the price trades through it by half the spread.
  const bool hitStop = b.l <= stop, hitTake = b.h >= take * std::exp(half);
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
  // Half the spread paid on the purchase (above the open) and on the sale (below the exit price).
  return (px / b.o) * std::exp(-2 * half) * (1 - sell) / (1 + buy) - 1;
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
    if (s[i].n != boundN) boundN = s[i].n, bound = chauvenet(s[i].n);
    const double z = bound;
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

DayBook dayBook(const Market& m, const Forecast& f, std::size_t start, double buy, double sell, const day::Paths* paths, std::size_t candidatesFrom,
                const DayRisk& risk) {
  using namespace day;
  DayBook book;
  book.buy = buy, book.sell = sell, book.start = start, book.spreadCosts = risk.spreadCosts, book.universe = risk.universe;
  const std::size_t T = m.T(), N = m.N(), K = 10, F = 10;
  if (start == 0 || start >= T || f.E.T != T || f.horizons.empty()) return book;
  const std::vector<std::uint32_t>& halfLives = f.horizons;
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
  // candidates are levels of the fall at the quantiles 1 - 2^-(k+1) (from the median out to the
  // tail), the take-profit candidates levels of the rise at (k + 1) / (G + 1), k < G. Predictors
  // of those levels: the stock's own history (its quantiles), Wilder's true range (below) and
  // neural networks on the trade's features, trained on the actual highs and lows by the quantile
  // (pinball) loss; each source's books' records choose the one that sets them.
  const std::size_t W = halfLives.back();
  std::vector<double> tauFall(G), tauRise(G);
  for (std::size_t k = 0; k < G; ++k) tauFall[k] = 1 - std::ldexp(1.0, -int(k + 1)), tauRise[k] = double(k + 1) / double(G + 1);
  // A third predictor is Wilder's: the same moves measured in the stock's average true range at the
  // previous close (his volatility unit), their quantiles times the true range at the close. Wilder
  // set a stop a fixed multiple of the true range from the price (the Volatility System's 3 ATRs);
  // here the multiples are the quantiles of the stock's own moves in true ranges, so they follow
  // the true range as it changes, which the history's quantiles (fixed distances) do not.
  std::vector<std::vector<double>> falls(N), rises(N), histStop(N), histTake(N);
  std::vector<std::vector<double>> fallsA(N), risesA(N), atrStopX(N), atrTakeX(N), atrStop(N), atrTake(N);
  // A fourth: the moves in the stock's Bollinger band unit (the standard deviation of its last h
  // closes over the close), h the horizon nearest the volatility's chosen half-life (it adapts as
  // that choice moves), their quantiles times the unit at the close.
  std::vector<std::vector<double>> fallsB(N), risesB(N), bolStopX(N), bolTakeX(N), bolStop(N), bolTake(N);
  auto quantiles = [&](const std::vector<double>& falls1, const std::vector<double>& rises1, std::vector<double>& outStop, std::vector<double>& outTake) {
    outStop.clear(), outTake.clear();
    if (falls1.size() < G + 1) return;
    for (int side = 0; side < 2; ++side) {
      std::vector<double> v = side == 0 ? falls1 : rises1;
      std::sort(v.begin(), v.end());
      auto& out = side == 0 ? outStop : outTake;
      for (std::size_t k = 0; k < G; ++k) {
        const double pos = (side == 0 ? tauFall[k] : tauRise[k]) * double(v.size() - 1);
        const std::size_t lo = std::size_t(pos);
        const double fr = pos - double(lo);
        out.push_back(lo + 1 < v.size() ? v[lo] * (1 - fr) + v[lo + 1] * fr : v[lo]);
      }
    }
  };
  // Wilder's indicators of every stock and day (kernels/day/wilder.wgsl), from the bars table below.
  std::vector<double> barTab(5 * T * N, 0.0), wil;
  {
    Bar b;
    for (std::size_t d = 0; d < T; ++d)
      for (std::size_t i = 0; i < N; ++i)
        if (barAt(d, i, &b)) {
          const double v5[5] = {b.o, b.h, b.l, b.c, b.v};
          for (std::size_t c = 0; c < 5; ++c) barTab[(c * T + d) * N + i] = v5[c];
        }
  }
  kernels::wilder(barTab, T, N, wil);
  constexpr std::size_t WI = kernels::kWilder;
  auto wilder = [&](std::size_t t, std::size_t i, std::size_t c) { return wil[(t * N + i) * WI + c]; };
  // Bollinger bands of every stock and day at the horizons up to 64 days (kernels/day/bollinger.wgsl).
  std::vector<std::uint32_t> bolH;
  for (auto h : halfLives)
    if (h <= 64) bolH.push_back(h);
  std::vector<double> bol;
  kernels::bollinger(barTab, T, N, bolH, bol);
  const std::size_t BW = 3 * bolH.size(), BI = 2 * bolH.size();
  auto bollinger = [&](std::size_t t, std::size_t i, std::size_t k, std::size_t c) { return bol[(t * N + i) * BW + 3 * k + c]; };
  // The band horizon in use: the one nearest the volatility's chosen half-life.
  auto bolUnitH = [&]() {
    const double hl = double(halfLives[vol.best()]);
    std::size_t k = 0;
    for (std::size_t j = 1; j < bolH.size(); ++j)
      if (std::fabs(std::log(double(bolH[j]) / hl)) < std::fabs(std::log(double(bolH[k]) / hl))) k = j;
    return k;
  };
  // Pinball losses so far (in volatilities) of the level predictors (history, Wilder's true range,
  // the network, the network also given Wilder's indicators), each comparison on the same
  // stock-days, out of sample: history against true range wherever both predict, all four wherever
  // the networks do too.
  enum { kHist, kAtr, kBol, kNet, kWnet, kSources };
  // The day's levels of every source (log distances; a source without levels for a stock takes the
  // history's).
  std::vector<std::vector<double>> levStop[kSources], levTake[kSources];
  for (int s0 = 0; s0 < kSources; ++s0) levStop[s0].resize(N), levTake[s0].resize(N);
  double lossHA[2] = {0, 0}, lossAll[kSources] = {0, 0, 0, 0, 0};
  std::size_t nHA = 0, nAll = 0;
  // Honest-backtest options (DayRisk): each stock's bid-ask spread by Abdi and Ranaldo (2017) from
  // its daily close, high and low, s^2 = 4 E[(c_t - eta_t)(c_t - eta_t+1)] (c the log close, eta
  // the log mid-range), exponentially weighted at each half-life (the volatility's chosen one is
  // used); and its traded value (log close x volume), weighted the same way, for the point-in-time
  // universe.
  std::vector<std::vector<double>> spread2(N, std::vector<double>(halfLives.size(), 0.0)), tradedValue(N, std::vector<double>(halfLives.size(), 0.0));
  std::vector<std::size_t> spreadN(N, 0), valueN(N, 0);
  std::vector<double> prevC(N, kNaN), prevEta(N, kNaN);
  auto halfSpread = [&](std::size_t i) {
    return risk.spreadCosts && spreadN[i] >= 2 ? 0.5 * std::sqrt(std::max(0.0, spread2[i][vol.best()])) : 0.0;
  };
  auto addCosts = [&](std::size_t t) {
    Bar b;
    for (std::size_t i = 0; i < N; ++i) {
      if (!barAt(t, i, &b)) {
        prevC[i] = kNaN;
        continue;
      }
      const double eta = 0.5 * (std::log(b.h) + std::log(b.l)), tv = std::log(b.c * b.v);
      if (std::isfinite(prevC[i])) {
        const double x = 4 * (prevC[i] - prevEta[i]) * (prevC[i] - eta);
        for (std::size_t h = 0; h < lam.size(); ++h) spread2[i][h] = spreadN[i] == 0 ? x : lam[h] * spread2[i][h] + (1 - lam[h]) * x;
        ++spreadN[i];
      }
      prevC[i] = std::log(b.c), prevEta[i] = eta;
      for (std::size_t h = 0; h < lam.size(); ++h) tradedValue[i][h] = valueN[i] == 0 ? tv : lam[h] * tradedValue[i][h] + (1 - lam[h]) * tv;
      ++valueN[i];
    }
  };
  auto addDay = [&](std::size_t t) {
    addCosts(t);
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
        quantiles(falls[i], rises[i], histStop[i], histTake[i]);
        const double atr = t > 0 ? wilder(t - 1, i, 0) : 0.0;
        if (atr > 0) {
          fallsA[i].push_back(x / atr), risesA[i].push_back(std::log(b.h / b.o) / atr);
          if (fallsA[i].size() > W) fallsA[i].erase(fallsA[i].begin()), risesA[i].erase(risesA[i].begin());
          quantiles(fallsA[i], risesA[i], atrStopX[i], atrTakeX[i]);
        }
        const double unit = t > 0 && !bolH.empty() ? bollinger(t - 1, i, bolUnitH(), 2) : 0.0;
        if (unit > 0) {
          fallsB[i].push_back(x / unit), risesB[i].push_back(std::log(b.h / b.o) / unit);
          if (fallsB[i].size() > W) fallsB[i].erase(fallsB[i].begin()), risesB[i].erase(risesB[i].begin());
          quantiles(fallsB[i], risesB[i], bolStopX[i], bolTakeX[i]);
        }
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
  // The day's universe: the risk.universe most traded stocks at the close t (traded value so far);
  // every stock when it is 0.
  auto inUniverse = [&](std::size_t t) {
    std::vector<char> in(N, 1);
    if (risk.universe == 0) return in;
    std::vector<std::pair<double, std::size_t>> v;
    Bar b;
    for (std::size_t i = 0; i < N; ++i)
      if (valueN[i] > 0 && barAt(t, i, &b)) v.emplace_back(tradedValue[i][vol.best()], i);
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& c) { return a.first > c.first; });
    std::fill(in.begin(), in.end(), 0);
    for (std::size_t k = 0; k < std::min(risk.universe, v.size()); ++k) in[v[k].second] = 1;
    return in;
  };
  // Pair q of stock i: its stop and take-profit distances from the open (log), from its own moves.
  auto stopOf = [&](int s0, std::size_t i, std::size_t q) { return levStop[s0][i].empty() ? kNone : levStop[s0][i][q / G]; };
  auto takeOf = [&](int s0, std::size_t i, std::size_t q) { return levTake[s0][i].empty() ? kNone : levTake[s0][i][q % G]; };
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
  // Computed by the indicators kernel a block of days at a time (each block once).
  const std::size_t blockDays = halfLives.back();
  std::size_t block0 = T;
  std::vector<double> indBlock;
  auto indicators = [&](std::size_t t, std::size_t i) {
    if (t < block0 || t >= block0 + blockDays) {
      block0 = t, kernels::indicators(barTab, T, N, indH, block0, std::min(blockDays, T - block0), indBlock);
    }
    const auto* r = &indBlock[((t - block0) * N + i) * FI];
    return std::vector<double>(r, r + FI);
  };
  // Wilder's indicators of stock i at the close t, for the network: the true range against the
  // volatility (log), RSI, directional movement, ADX, ADXR, the Parabolic's side and the close's
  // distance from it.
  auto wilderFeatures = [&](std::size_t t, std::size_t i) {
    std::vector<double> x(WI);
    for (std::size_t c = 0; c < WI; ++c) x[c] = wilder(t, i, c);
    const double sig = sigmaOf(i);
    x[0] = x[0] > 0 && sig > 0 ? std::log(x[0] / sig) : 0.0;
    return x;
  };
  // Bollinger features of stock i at the close t: the close's place in the bands and their width
  // (log) at each horizon, for the second network.
  auto bollingerFeatures = [&](std::size_t t, std::size_t i) {
    std::vector<double> x(BI);
    for (std::size_t k = 0; k < bolH.size(); ++k) x[2 * k] = bollinger(t, i, k, 0), x[2 * k + 1] = bollinger(t, i, k, 1);
    return x;
  };
  Scaler scaler(F), indScaler(FI), wilScaler(WI), bolScaler(BI);
  LeastSquares intraday(F + 1);  // open-to-close return / sigma on [1, features]
  // Pair outcome / sigma on [1, mu]: mu the intraday forecast. And on [1, mu, r, r mu], r the
  // stock's ADXR at the close: Wilder traded his trend systems only when ADXR rated the market
  // trending and took quick profits otherwise; here how each pair's outcome depends on the trend's
  // strength is learnt. The fits used are the ones whose forecasts of the outcomes have had the
  // lower squared error so far (out of sample, over every pair). Each level source has its own.
  std::vector<LeastSquares> pairFit[kSources], trendFit[kSources];
  for (int s0 = 0; s0 < kSources; ++s0) pairFit[s0].assign(P, LeastSquares(2)), trendFit[s0].assign(P, LeastSquares(4));
  double errPlain = 0, errTrend = 0, nErr = 0;
  auto useTrend = [&]() { return errTrend < errPlain; };
  // Each stock's record; each stop limit's book's record (the best trades).
  // The networks that predict each trade's low and high (see the levels above): on the trade's
  // features and indicators, and on those and Wilder's indicators too.
  std::vector<double> taus = tauFall;
  taus.insert(taus.end(), tauRise.begin(), tauRise.end());
  const std::size_t NI = F + 1 + FI;
  kernels::Net qnet(NI, 32, taus), wnet(NI + WI + BI, 32, taus);
  // The books: one per level source and stop limit, book s G + k.
  std::vector<Adwin> record(N), limitRecord(kSources * G);
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
    std::vector<double> xs, ind, indRaw, wil, wilRaw, bol, bolRaw;  // scaled features, indicators, Wilder's and Bollinger's (scaled, raw)
    std::vector<double> e[kSources], ePlain[kSources], eTrend[kSources];  // expected pair outcomes of each source (in use, of the two fits)
    bool forecast = false;                              // the intraday forecast is fitted
    double mu = 0, adxr = 0;
  };
  auto plan = [&](std::size_t t, const std::vector<std::size_t>& cand, double medSig, double mkt) {
    const auto beta = intraday.solve();
    std::vector<std::vector<double>> ab(kSources * P), abT(kSources * P);
    for (int s0 = 0; s0 < kSources; ++s0)
      for (std::size_t q = 0; q < P; ++q) ab[s0 * P + q] = pairFit[s0][q].solve(), abT[s0 * P + q] = trendFit[s0][q].solve();
    const bool trend = useTrend();
    std::vector<TradePlan> out(N);
    for (std::size_t i : cand) {
      const auto x = features(t, i, medSig, mkt);
      if (x.empty()) continue;
      TradePlan& pl = out[i];
      pl.xs = scaler.apply(x);
      pl.indRaw = indicators(t, i);
      pl.ind = indScaler.apply(pl.indRaw);
      pl.wilRaw = wilderFeatures(t, i);
      pl.wil = wilScaler.apply(pl.wilRaw);
      pl.bolRaw = bollingerFeatures(t, i);
      pl.bol = bolScaler.apply(pl.bolRaw);
      pl.adxr = wilder(t, i, 4);
      if (beta.empty()) continue;
      pl.mu = beta[0];
      for (std::size_t j = 0; j < F; ++j) pl.mu += beta[j + 1] * pl.xs[j];
      pl.forecast = true;
      for (int s0 = 0; s0 < kSources; ++s0) {
        pl.ePlain[s0].assign(P, -kNone), pl.eTrend[s0].assign(P, -kNone);
        for (std::size_t q = 0; q < P; ++q) {
          const auto& a = ab[s0 * P + q];
          const auto& c = abT[s0 * P + q];
          if (!a.empty()) pl.ePlain[s0][q] = a[0] + a[1] * pl.mu;
          if (!c.empty()) pl.eTrend[s0][q] = c[0] + c[1] * pl.mu + (c[2] + c[3] * pl.mu) * pl.adxr;
        }
        pl.e[s0] = trend ? pl.eTrend[s0] : pl.ePlain[s0];
      }
    }
    return out;
  };
  // The network's plan: its own intraday forecast and every pair's expected outcome, from the
  // trade's features and the linear forecast.
  auto netInput = [&](const TradePlan& p) {
    std::vector<double> x = p.xs;
    x.push_back(p.mu);
    x.insert(x.end(), p.ind.begin(), p.ind.end());
    x.insert(x.end(), p.wil.begin(), p.wil.end());
    x.insert(x.end(), p.bol.begin(), p.bol.end());
    return x;
  };
  // The networks' predicted levels of every candidate, one dispatch each for the day (log
  // distances). Network 0 takes the first NI inputs, network 1 all of them (Wilder's and Bollinger's too).
  std::vector<std::vector<double>> netStop[2] = {std::vector<std::vector<double>>(N), std::vector<std::vector<double>>(N)};
  std::vector<std::vector<double>> netTake[2] = {std::vector<std::vector<double>>(N), std::vector<std::vector<double>>(N)};
  auto netLevels = [&](const std::vector<std::size_t>& cand, const std::vector<TradePlan>& pl) {
    for (int k1 = 0; k1 < 2; ++k1) {
      const kernels::Net& net = k1 == 0 ? qnet : wnet;
      std::vector<std::size_t> who;
      std::vector<double> x;
      for (std::size_t i : cand) {
        netStop[k1][i].clear(), netTake[k1][i].clear();
        if (net.steps > 0 && !pl[i].xs.empty() && !histStop[i].empty()) {
          const auto xi = netInput(pl[i]);
          x.insert(x.end(), xi.begin(), xi.begin() + std::ptrdiff_t(net.In)), who.push_back(i);
        }
      }
      const auto y = net.predict(x, who.size());
      for (std::size_t k = 0; k < who.size(); ++k) {
        const std::size_t i = who[k];
        const double sig = sigmaOf(i);
        auto& ns = netStop[k1][i];
        auto& nt = netTake[k1][i];
        ns.assign(y.begin() + std::ptrdiff_t(k * 2 * G), y.begin() + std::ptrdiff_t(k * 2 * G + G));
        nt.assign(y.begin() + std::ptrdiff_t(k * 2 * G + G), y.begin() + std::ptrdiff_t((k + 1) * 2 * G));
        for (auto* v : {&ns, &nt}) {
          for (auto& z : *v) z = std::max(0.0, z) * sig;
          std::sort(v->begin(), v->end());
        }
      }
    }
  };
  // Wilder's levels of each candidate at the close t: its moves' quantiles in true ranges, times its
  // true range at t (log distances).
  auto atrLevels = [&](std::size_t t, const std::vector<std::size_t>& cand) {
    for (std::size_t i : cand) {
      atrStop[i].clear(), atrTake[i].clear();
      const double atr = wilder(t, i, 0);
      if (!(atr > 0) || atrStopX[i].empty() || histStop[i].empty()) continue;
      for (double z : atrStopX[i]) atrStop[i].push_back(z * atr);
      for (double z : atrTakeX[i]) atrTake[i].push_back(z * atr);
    }
  };
  // Every source's levels for the day. Which source sets the levels traded is not chosen by these
  // forecasts' losses (a closer forecast of the low and high need not trade better) but by each
  // source's books' records, as the stop limit is.
  auto setLevels = [&](std::size_t t, const std::vector<std::size_t>& cand, const std::vector<TradePlan>& pl) {
    netLevels(cand, pl);
    atrLevels(t, cand);
    const std::size_t kb = bolH.empty() ? 0 : bolUnitH();
    for (std::size_t i : cand) {
      bolStop[i].clear(), bolTake[i].clear();
      const double unit = bolH.empty() ? 0.0 : bollinger(t, i, kb, 2);
      if (unit > 0 && !bolStopX[i].empty() && !histStop[i].empty()) {
        for (double z : bolStopX[i]) bolStop[i].push_back(z * unit);
        for (double z : bolTakeX[i]) bolTake[i].push_back(z * unit);
      }
    }
    for (std::size_t i : cand)
      for (int s0 = 0; s0 < kSources; ++s0) {
        const auto& st = s0 == kAtr ? atrStop[i] : s0 == kBol ? bolStop[i] : s0 >= kNet ? netStop[s0 - kNet][i] : histStop[i];
        const auto& tk = s0 == kAtr ? atrTake[i] : s0 == kBol ? bolTake[i] : s0 >= kNet ? netTake[s0 - kNet][i] : histTake[i];
        levStop[s0][i] = st.empty() ? histStop[i] : st, levTake[s0][i] = st.empty() ? histTake[i] : tk;
      }
  };
  // The opening-noise floor of stock i: no floor where its intraday path gives the exact fills.
  auto floorOf = [&](std::size_t i, bool exact) { return exact ? 0.0 : noiseN[i] >= 2 ? noise[i][vol.best()] : kNone; };
  // The risk limits: the widest stop allowed (log distance from the open).
  const double maxStop = -std::log1p(-std::clamp(risk.maxStop, 1e-9, 1.0 - 1e-9));
  // Stock i's best pair whose stop is within the limit (log distance) and the risk limits, and its
  // expected return.
  // The stop limit of stock i under candidate k at the close t (log distance): k's multiple of its
  // Bollinger band unit, or of the day's median volatility (without band limits or a band).
  auto limitOf = [&](std::size_t k, std::size_t i, std::size_t t, double medSig) {
    const double unit = risk.bandLimits && !bolH.empty() ? bollinger(t, i, bolUnitH(), 2) : 0.0;
    return mult[k] * (unit > 0 ? unit : medSig);
  };
  // The worst-day budget at the close t: the equal-weight market's typical distance from its close
  // to its lower Bollinger band (two band units at the adaptive horizon): the median of that
  // distance over every close so far, so the budget is learnt from the market but does not swing
  // with it (a budget that follows each day's bands leverages up when the market is wild). Unless
  // fixed.
  std::vector<double> mktIndex(T, 0.0);
  {
    Bar b0, b1;
    for (std::size_t u = 1; u < T; ++u) {
      double s = 0, n = 0;
      for (std::size_t i = 0; i < N; ++i)
        if (barAt(u, i, &b0) && barAt(u - 1, i, &b1)) s += std::log(b0.c / b1.c), n += 1;
      mktIndex[u] = mktIndex[u - 1] + (n > 0 ? s / n : 0.0);
    }
  }
  auto bandDistance = [&](std::size_t t) {
    const std::size_t h = bolH[bolUnitH()];
    if (t + 1 < h) return 1.0;
    double s1 = 0, s2 = 0;
    for (std::size_t u = t + 1 - h; u <= t; ++u) {
      const double c = std::exp(mktIndex[u] - mktIndex[t]);
      s1 += c, s2 += c * c;
    }
    const double mu = s1 / double(h), v = std::max(0.0, s2 / double(h) - mu * mu);
    return std::min(1.0, 2 * std::sqrt(v));  // the close is 1 on this scale
  };
  std::vector<double> pastDistances;  // the market's band distance at every close so far
  std::size_t distancesTo = 0;        // closes added
  auto budgetAt = [&](std::size_t t) {
    if (risk.maxDayLoss > 0 || bolH.empty()) return risk.maxDayLoss > 0 ? risk.maxDayLoss : 1.0;
    if (risk.dailyBudget) return bandDistance(t);
    for (; distancesTo <= t; ++distancesTo)
      if (distancesTo + 1 >= bolH[bolUnitH()]) pastDistances.push_back(bandDistance(distancesTo));
    if (pastDistances.empty()) return 1.0;
    std::vector<double> v = pastDistances;
    std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(v.size() / 2), v.end());
    return v[v.size() / 2];
  };
  double budget = 1.0;  // the day's worst-day budget (set at each close)
  auto bestPair = [&](int s0, std::size_t i, const TradePlan& pl, double limit, std::size_t* pair, bool exact) {
    double best = -kNone;
    if (!pl.forecast || levStop[s0][i].empty()) return best;
    const double floor = floorOf(i, exact);
    for (std::size_t q = 0; q < P; ++q) {
      const double dist = stopOf(s0, i, q);
      if (dist >= floor && dist <= std::min(limit, maxStop) && (!(risk.maxRiskReward > 0) || dist <= risk.maxRiskReward * takeOf(s0, i, q)) && pl.e[s0][q] > best) best = pl.e[s0][q], *pair = q;
    }
    return best;
  };
  // A book under a level source and a stop limit: down the ranking of expected net return, up to 10 trades expected to
  // gain whose stock's record beats cash; equal shares, scaled so the book's expected volatility is
  // at most the market's.
  struct Book {
    std::vector<std::size_t> pick, pair;
    std::vector<double> expect;
    double w = 0, wRaw = 0;  // the weight traded, and before the worst-day cap
    int src = kHist;
  };
  // exact(i): whether stock i's fills are exact (an intraday path) on the day traded.
  // limit(i): stock i's stop limit (log distance).
  auto ranked = [&](int s0, const std::vector<std::size_t>& cand, const std::vector<TradePlan>& pl, const auto& limit, const auto& exact) {
    std::vector<std::tuple<double, std::size_t, std::size_t, double>> v;  // (expected return, stock, pair, expect)
    for (std::size_t i : cand) {
      std::size_t q = 0;
      const double e = bestPair(s0, i, pl[i], limit(i), &q, exact(i));
      if (e > 0 && record[i].score() > 0) v.emplace_back(e * sigmaOf(i), i, q, e);
    }
    std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& c) { return std::get<0>(a) > std::get<0>(c); });
    if (v.size() > K) v.resize(K);
    return v;
  };
  // The book of the n strongest: equal shares, scaled so its expected volatility is at most the market's.
  auto makeBook = [&](int s0, const std::vector<std::tuple<double, std::size_t, std::size_t, double>>& v, std::size_t n) {
    Book bk;
    bk.src = s0;
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
    // The book's loss if every stop is hit: at most the day's budget.
    double worst = 0;
    for (std::size_t j = 0; j < bk.pick.size(); ++j) worst += bk.w * (1 - std::exp(-stopOf(s0, bk.pick[j], bk.pair[j])));
    bk.wRaw = bk.w;
    if (worst > budget) bk.w *= budget / worst;
    return bk;
  };
  // The level source and stop limit in use: the book whose record is best so far; cash while none
  // beats cash.
  auto bestLimit = [&]() {
    std::size_t k = 0;
    for (std::size_t j = 1; j < limitRecord.size(); ++j)
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
      r += bk.w * tradeReturn(b, 1.0, stopOf(bk.src, bk.pick[j], bk.pair[j]), takeOf(bk.src, bk.pick[j], bk.pair[j]), buy, sell, worstCase, &e, path, &res, halfSpread(bk.pick[j]));
      if (count) book.spreadSum += 2 * halfSpread(bk.pick[j]);
      if (count) ++book.trades, book.stops += e == 1, book.takes += e == 2, book.pathTrades += path != nullptr, book.pathResolved += res;
    }
    return r;
  };

  Bar b;
  std::vector<double> zp(kSources * P);
  for (std::size_t d = start; d < T; ++d) {
    const std::size_t t = d - 1;
    if (t > 0) addDay(t);
    // Candidates: in the day's universe, a forecast at the close t, a volatility, and a bar on day d.
    std::vector<std::size_t> cand;
    const auto inU = inUniverse(t);
    for (std::size_t i = 0; i < N; ++i)
      if (inU[i] && std::isfinite(f.E(t, i)) && sigmaOf(i) > 0 && barAt(d, i, &b)) cand.push_back(i);
    double medSig, mkt;
    dayContext(t, cand, &medSig, &mkt);
    const auto pl = plan(t, cand, medSig, mkt);
    setLevels(t, cand, pl);
    // Every source's and stop limit's book; the one in use trades.
    const std::size_t use = bestLimit(), nBooks = limitRecord.size();
    budget = budgetAt(t);
    auto limitK = [&](std::size_t k) { return [&, k](std::size_t i) { return limitOf(k % G, i, t, medSig); }; };
    const int us = int(use / G);
    const bool trade = limitRecord[use].score() > 0;
    if (trade) ++book.sourceDays[us];
    std::vector<Book> books(nBooks);
    auto exactOn = [&](std::size_t i) { return pathOf(d, i) != nullptr; };
    for (std::size_t k = 0; k < nBooks; ++k) books[k] = makeBook(int(k / G), ranked(int(k / G), cand, pl, limitK(k), exactOn), K);
    // The strongest candidates: every stock a book could trade, with and without exact fills.
    if (d >= candidatesFrom) {
      std::vector<char> seen(N, 0);
      for (int ex = 0; ex < 2; ++ex)
        for (std::size_t k = 0; k < nBooks; ++k)
          for (const auto& c : ranked(int(k / G), cand, pl, limitK(k), [&](std::size_t) { return ex == 1; }))
            if (!seen[std::get<1>(c)]) seen[std::get<1>(c)] = 1, book.candidates.emplace_back(d, std::get<1>(c));
    }
    const double r = trade ? dayReturn(books[use], d, true) : 0.0;
    // Fill test: the same trades with the opposite guess when a bar touched both levels.
    book.optimistic.push_back(trade ? dayReturn(books[use], d, false, false) : 0.0);
    book.ret.push_back(r), book.booked.push_back(trade ? int(books[use].pick.size()) : 0);
    book.exposure.push_back(trade ? books[use].w * double(books[use].pick.size()) : 0.0);
    for (std::size_t k = 0; k < nBooks; ++k) {
      // With the worst-day cap as sizing only, the books compete on their uncapped weights.
      Book bk = books[k];
      if (risk.capSizingOnly) bk.w = bk.wRaw;
      limitRecord[k].push(std::log1p(dayReturn(bk, d, false)));
    }
    // After the close d: the day's outcomes, for the stocks' records and the fits.
    for (int s0 = 0; s0 < kSources; ++s0)  // the two fits of a source see the same outcomes
      for (std::size_t q = 0; q < P; ++q) zp[s0 * P + q] = chauvenet(pairFit[s0][q].n + 1);
    const double zb = chauvenet(intraday.n + 1);
    std::vector<std::vector<double>> qx, qy;
    for (std::size_t i : cand) {
      const TradePlan& p = pl[i];
      if (p.xs.empty()) continue;
      barAt(d, i, &b);
      const double sig = sigmaOf(i);
      const auto* path = pathOf(d, i);
      book.pathDays += path != nullptr;
      int e;
      std::size_t q0 = 0;
      if (bestPair(us, i, p, kNone, &q0, path != nullptr) > 0)  // the stock's own best trade, whenever a gain is expected: its record
        record[i].push(std::log1p(tradeReturn(b, 1.0, stopOf(us, i, q0), takeOf(us, i, q0), buy, sell, true, &e, path, nullptr, halfSpread(i))));
      // The day's actual fall and rise from the open: the level predictors' targets and scores.
      if (!histStop[i].empty()) {
        std::vector<double> y(2 * G);
        const double fall = std::min(zb, std::log(b.o / b.l) / sig), rise = std::min(zb, std::log(b.h / b.o) / sig);
        for (std::size_t k = 0; k < G; ++k) y[k] = fall, y[G + k] = rise;
        auto pin = [](double u, double tq) { return u * (tq - (u < 0 ? 1.0 : 0.0)); };
        auto loss = [&](const std::vector<double>& st, const std::vector<double>& tk) {
          double l = 0;
          for (std::size_t k = 0; k < G; ++k) l += pin(fall - st[k] / sig, tauFall[k]) + pin(rise - tk[k] / sig, tauRise[k]);
          return l;
        };
        if (!atrStop[i].empty()) {
          const double lh = loss(histStop[i], histTake[i]), la = loss(atrStop[i], atrTake[i]);
          lossHA[kHist] += lh, lossHA[kAtr] += la, ++nHA;
          if (!netStop[0][i].empty() && !netStop[1][i].empty() && !bolStop[i].empty())
            lossAll[kHist] += lh, lossAll[kAtr] += la, lossAll[kBol] += loss(bolStop[i], bolTake[i]), lossAll[kNet] += loss(netStop[0][i], netTake[0][i]),
                lossAll[kWnet] += loss(netStop[1][i], netTake[1][i]), ++nAll;
        }
        qx.push_back(netInput(p)), qy.push_back(std::move(y));
      }
      std::vector<double> x1{1.0};
      x1.insert(x1.end(), p.xs.begin(), p.xs.end());
      const double raw = std::log(b.c / b.o) / sig, z = std::clamp(raw, -zb, zb);
      book.maxTarget = std::max(book.maxTarget, std::fabs(raw));
      book.bigTargets += std::fabs(raw) > zb;
      // Out-of-sample skill of the intraday forecast: its correlation with what happened.
      if (p.forecast) book.skill.add(p.mu, z);
      intraday.add(x1, z);
      if (p.forecast) {  // pair fits on genuine forecasts only
        // With intraday paths (where tight stops get traded), only honest outcomes: a stop outside
        // the opening noise, or exact fills from the path.
        const std::vector<double> mu1{1.0, p.mu}, mu2{1.0, p.mu, p.adxr, p.adxr * p.mu};
        const double floor = paths && !paths->empty() ? floorOf(i, path != nullptr) : 0.0;
        for (int s0 = 0; s0 < kSources; ++s0)
          for (std::size_t q = 0; q < P; ++q)
            if (!levStop[s0][i].empty() && stopOf(s0, i, q) >= floor) {
              const double z1 = zp[s0 * P + q];
              const double y = std::clamp(std::log1p(tradeReturn(b, 1.0, stopOf(s0, i, q), takeOf(s0, i, q), buy, sell, true, &e, path, nullptr, halfSpread(i))) / sig, -z1, z1);
              const double ep = p.ePlain[s0][q], et = p.eTrend[s0][q];
              if (std::isfinite(ep) && std::isfinite(et))  // both fits' forecasts, before they see the outcome
                errPlain += (y - ep) * (y - ep), errTrend += (y - et) * (y - et), nErr += 1;
              pairFit[s0][q].add(mu1, y), trendFit[s0][q].add(mu2, y);
            }
      }
    }
    // The network learns from the day's actual lows and highs (one pass, shuffled mini-batches).
    if (!qx.empty()) {
      std::vector<std::size_t> perm(qx.size());
      for (std::size_t k = 0; k < perm.size(); ++k) perm[k] = k;
      std::uint64_t rng = 0x2545F4914F6CDD1Dull ^ d;
      for (std::size_t k = perm.size(); k-- > 1;) {
        rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
        std::swap(perm[k], perm[rng % (k + 1)]);
      }
      for (std::size_t s0 = 0; s0 < perm.size(); s0 += 32) {
        std::vector<double> bx, by;
        const std::size_t nb = std::min(perm.size(), s0 + 32) - s0;
        for (std::size_t k = s0; k < s0 + nb; ++k) bx.insert(bx.end(), qx[perm[k]].begin(), qx[perm[k]].end()), by.insert(by.end(), qy[perm[k]].begin(), qy[perm[k]].end());
        std::vector<double> bx0;  // the first network's inputs: each sample's first NI
        for (std::size_t k = 0; k < nb; ++k) bx0.insert(bx0.end(), bx.begin() + std::ptrdiff_t(k * (NI + WI + BI)), bx.begin() + std::ptrdiff_t(k * (NI + WI + BI) + NI));
        qnet.step(bx0, by, nb), wnet.step(bx, by, nb);
      }
    }
    for (std::size_t i : cand) {
      const auto x = features(t, i, medSig, mkt);
      if (!x.empty()) scaler.add(x), indScaler.add(pl[i].indRaw), wilScaler.add(pl[i].wilRaw), bolScaler.add(pl[i].bolRaw);
    }
    for (std::size_t i = 0; i < N; ++i)
      if (barAt(d, i, &b)) logVolume[i].add(std::log(b.v));
  }

  // Orders for the next open, from the last close.
  const std::size_t t = T - 1;
  addDay(t);
  book.halfLife = halfLives[vol.best()];
  std::vector<std::size_t> cand;
  const auto inU = inUniverse(t);
  for (std::size_t i = 0; i < N; ++i)
    if (inU[i] && std::isfinite(f.E(t, i)) && sigmaOf(i) > 0 && barAt(t, i, &b)) cand.push_back(i);
  double medSig, mkt;
  dayContext(t, cand, &medSig, &mkt);
  const auto pl0 = plan(t, cand, medSig, mkt);
  setLevels(t, cand, pl0);
  const std::size_t use = bestLimit();
  const int src = int(use / G);
  book.levelsFrom = src == kWnet  ? "neural network with Wilder's and Bollinger's indicators (predicted lows and highs)"
                    : src == kNet ? "neural network (predicted lows and highs)"
                    : src == kBol ? "Bollinger bands (each stock's moves in band units, adaptive horizon)"
                    : src == kAtr ? "Wilder's average true range (each stock's moves in true ranges)"
                                  : "each stock's own daily moves";
  for (int k = 0; k < kSources; ++k) book.levelLoss[k] = nAll > 0 ? lossAll[k] / double(nAll) : k < 2 && nHA > 0 ? lossHA[k] / double(nHA) : 0.0;
  book.trendFits = useTrend();
  book.pairErr[0] = nErr > 0 ? errPlain / nErr : 0.0, book.pairErr[1] = nErr > 0 ? errTrend / nErr : 0.0;
  const auto& pl = pl0;
  auto limit = [&](std::size_t i) { return limitOf(use % G, i, t, medSig); };
  budget = budgetAt(t);
  book.dayBudget = budget;
  book.bookSize = K;
  {
    std::vector<double> lim;
    for (std::size_t i : cand) lim.push_back(1 - std::exp(-std::min(limit(i), maxStop)));
    std::sort(lim.begin(), lim.end());
    book.stopLimit = lim.empty() ? 0.0 : lim[lim.size() / 2];
  }
  book.stopLimitUnits = risk.bandLimits ? mult[use % G] : 0.0;
  book.limitScore = limitRecord[use].score();
  auto& sc = book.screen;
  sc.stocks = N, sc.candidates = cand.size();
  for (std::size_t i = 0; i < N; ++i)
    if (barAt(t, i, &b)) ++sc.bars, sc.flatBars += b.h == b.l;
  // Next open: fills are exact where intraday paths are kept (they will be fetched for these days).
  const bool exactNext = paths && !paths->empty();
  auto exactNextOf = [&](std::size_t) { return exactNext; };
  for (std::size_t i : cand) {
    std::size_t q = 0;
    const double e = bestPair(src, i, pl[i], limit(i), &q, exactNext);
    sc.forecastUp += pl[i].forecast && pl[i].mu > 0, sc.gainExpected += e > 0, sc.recordUp += record[i].score() > 0;
    sc.all += e > 0 && record[i].score() > 0;
  }
  // No trades for the next open unless the daily trades have made money over the whole backtest
  // (worst-case fills) significantly (the t-statistic of the mean daily return at least risk.minT):
  // a market where they lose, or gain no more than chance would, is not traded.
  double grown = 0;
  Moments mr;
  for (double r : book.ret) grown += std::log1p(r), mr.add(r);
  book.backtestSharpe = metrics(book.ret).sharpe;
  book.backtestT = mr.n > 1 && mr.sd() > 0 ? mr.mean() / mr.sd() * std::sqrt(mr.n - 1) : 0.0;
  book.backtestLoses = !(grown > 0) || !(book.backtestT >= risk.minT);
  if (limitRecord[use].score() > 0 && !book.backtestLoses) {
    const Book bk = makeBook(src, ranked(src, cand, pl, limit, exactNextOf), book.bookSize);
    for (std::size_t j = 0; j < bk.pick.size(); ++j) {
      const std::size_t i = bk.pick[j], q = bk.pair[j];
      DayOrder o;
      o.asset = i, o.expected = f.E(t, i), o.sigma = sigmaOf(i), o.stop = stopOf(src, i, q) / o.sigma, o.take = takeOf(src, i, q) / o.sigma;
      o.intraday = pl[i].mu * o.sigma, o.tradeReturn = std::expm1(bk.expect[j] * o.sigma), o.weight = bk.w;
      o.record = record[i].score(), o.window = record[i].window(), o.limit = 1 - std::exp(-std::min(limit(i), maxStop));
      book.orders.push_back(o);
    }
  }
  return book;
}

}  // namespace ofm
