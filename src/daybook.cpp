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

double tradeReturn(const Bar& b, double sig, double ks, double kt, double buy, double sell, bool worstCase, int* exit) {
  const double stop = std::isfinite(ks) ? b.o * std::exp(-ks * sig) : 0.0;
  const double take = std::isfinite(kt) ? b.o * std::exp(kt * sig) : kNone;
  const bool hitStop = b.l <= stop, hitTake = b.h >= take;
  double px = b.c;
  *exit = 0;
  if (hitStop && (worstCase || !hitTake)) px = stop, *exit = 1;
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

DayBook dayBook(const Market& m, const Forecast& f, std::size_t start, double buy, double sell) {
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
  Scaler scaler(F);
  LeastSquares intraday(F + 1);                            // open-to-close return / sigma on [1, features]
  std::vector<LeastSquares> pairFit(P, LeastSquares(2));    // pair outcome / sigma on [1, intraday forecast]
  std::vector<Adwin> record(N), limitRecord(G);             // each stock's record; each stop limit's book
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
    std::vector<double> xs, e;
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
      if (beta.empty()) continue;
      pl.mu = beta[0];
      for (std::size_t j = 0; j < F; ++j) pl.mu += beta[j + 1] * pl.xs[j];
      pl.e.assign(P, -kNone);
      for (std::size_t q = 0; q < P; ++q)
        if (!ab[q].empty()) pl.e[q] = ab[q][0] + ab[q][1] * pl.mu;
    }
    return out;
  };
  // Stock i's best pair whose stop is within the limit (log distance), and its expected return.
  auto bestPair = [&](std::size_t i, const TradePlan& pl, double limit, std::size_t* pair) {
    double best = -kNone;
    if (pl.e.empty()) return best;
    const double floor = noiseN[i] >= 2 ? noise[i][vol.best()] : kNone;
    for (std::size_t q = 0; q < P; ++q) {
      const double dist = mult[q / G] * sigmaOf(i);
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
  auto choose = [&](const std::vector<std::size_t>& cand, const std::vector<TradePlan>& pl, double limit) {
    std::vector<std::tuple<double, std::size_t, std::size_t, double>> v;  // (expected return, stock, pair, expect)
    for (std::size_t i : cand) {
      std::size_t q = 0;
      const double e = bestPair(i, pl[i], limit, &q);
      if (e > 0 && record[i].score() > 0) v.emplace_back(e * sigmaOf(i), i, q, e);
    }
    std::stable_sort(v.begin(), v.end(), [](const auto& a, const auto& c) { return std::get<0>(a) > std::get<0>(c); });
    if (v.size() > K) v.resize(K);
    Book bk;
    for (const auto& [r, i, q, e] : v) bk.pick.push_back(i), bk.pair.push_back(q), bk.expect.push_back(e);
    if (bk.pick.empty()) return bk;
    const double n = double(bk.pick.size());
    double var = 0;
    for (std::size_t i : bk.pick) var += sigmaOf(i) * sigmaOf(i) / (n * n);
    const double mk = vol.sigma(N);
    bk.w = (1 / n) * ((var > 0 && mk > 0) ? std::min(1.0, mk / std::sqrt(var)) : 1.0);
    return bk;
  };
  // The stop limit in use: the one whose book's record is best so far; cash while none beats cash.
  auto bestLimit = [&]() {
    std::size_t k = 0;
    for (std::size_t j = 1; j < G; ++j)
      if (limitRecord[j].score() > limitRecord[k].score()) k = j;
    return k;
  };
  auto dayReturn = [&](const Book& bk, std::size_t d, bool count) {
    double r = 0;
    Bar b;
    for (std::size_t j = 0; j < bk.pick.size(); ++j) {
      barAt(d, bk.pick[j], &b);
      int e;
      r += bk.w * tradeReturn(b, sigmaOf(bk.pick[j]), mult[bk.pair[j] / G], mult[bk.pair[j] % G], buy, sell, true, &e);
      if (count) ++book.trades, book.stops += e == 1, book.takes += e == 2;
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
    // Every stop limit's book; the one in use trades.
    const std::size_t use = bestLimit();
    const bool trade = limitRecord[use].score() > 0;
    std::vector<Book> books(G);
    for (std::size_t k = 0; k < G; ++k) books[k] = choose(cand, pl, mult[k] * medSig);
    const double r = trade ? dayReturn(books[use], d, true) : 0.0;
    book.ret.push_back(r), book.booked.push_back(trade ? int(books[use].pick.size()) : 0);
    book.exposure.push_back(trade ? books[use].w * double(books[use].pick.size()) : 0.0);
    for (std::size_t k = 0; k < G; ++k) limitRecord[k].push(std::log1p(dayReturn(books[k], d, false)));
    // After the close d: the day's outcomes, for the stocks' records and the fits.
    for (std::size_t q = 0; q < P; ++q) zp[q] = chauvenet(pairFit[q].n + 1);
    const double zb = chauvenet(intraday.n + 1);
    for (std::size_t i : cand) {
      const TradePlan& p = pl[i];
      if (p.xs.empty()) continue;
      barAt(d, i, &b);
      const double sig = sigmaOf(i);
      int e;
      std::size_t q0 = 0;
      if (bestPair(i, p, kNone, &q0) > 0)  // the stock's own best trade, whenever a gain is expected: its record
        record[i].push(std::log1p(tradeReturn(b, sig, mult[q0 / G], mult[q0 % G], buy, sell, true, &e)));
      std::vector<double> x1{1.0};
      x1.insert(x1.end(), p.xs.begin(), p.xs.end());
      const double raw = std::log(b.c / b.o) / sig, z = std::clamp(raw, -zb, zb);
      book.maxTarget = std::max(book.maxTarget, std::fabs(raw));
      book.bigTargets += std::fabs(raw) > zb;
      // Out-of-sample skill of the intraday forecast: its correlation with what happened.
      if (!p.e.empty()) book.skill.add(p.mu, z);
      intraday.add(x1, z);
      if (!p.e.empty()) {  // pair fits on genuine forecasts only
        const std::vector<double> mu1{1.0, p.mu};
        for (std::size_t q = 0; q < P; ++q)
          pairFit[q].add(mu1, std::clamp(std::log1p(tradeReturn(b, sig, mult[q / G], mult[q % G], buy, sell, true, &e)) / sig, -zp[q], zp[q]));
      }
    }
    for (std::size_t i : cand) {
      const auto x = features(t, i, medSig, mkt);
      if (!x.empty()) scaler.add(x);
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
  const auto pl = plan(t, cand, medSig, mkt);
  const std::size_t use = bestLimit();
  const double limit = mult[use] * medSig;
  book.stopLimit = 1 - std::exp(-limit), book.limitScore = limitRecord[use].score();
  auto& sc = book.screen;
  sc.stocks = N, sc.candidates = cand.size();
  for (std::size_t i = 0; i < N; ++i)
    if (barAt(t, i, &b)) ++sc.bars, sc.flatBars += b.h == b.l;
  for (std::size_t i : cand) {
    std::size_t q = 0;
    const double e = bestPair(i, pl[i], limit, &q);
    sc.forecastUp += !pl[i].e.empty() && pl[i].mu > 0, sc.gainExpected += e > 0, sc.recordUp += record[i].score() > 0;
    sc.all += e > 0 && record[i].score() > 0;
  }
  if (limitRecord[use].score() > 0) {
    const Book bk = choose(cand, pl, limit);
    for (std::size_t j = 0; j < bk.pick.size(); ++j) {
      const std::size_t i = bk.pick[j], q = bk.pair[j];
      DayOrder o;
      o.asset = i, o.expected = f.E(t, i), o.sigma = sigmaOf(i), o.stop = mult[q / G], o.take = mult[q % G];
      o.intraday = pl[i].mu * o.sigma, o.tradeReturn = std::expm1(bk.expect[j] * o.sigma), o.weight = bk.w;
      o.record = record[i].score(), o.window = record[i].window();
      book.orders.push_back(o);
    }
  }
  return book;
}

}  // namespace ofm
