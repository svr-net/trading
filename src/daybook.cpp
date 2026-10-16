#include "ofm/daybook.hpp"

#include <algorithm>
#include <cmath>

namespace ofm {
namespace day {

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
    x[i] = std::clamp((x[i] - mu) / sd, -5.0, 5.0);
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
  const std::vector<double>& G1 = grid();
  const std::size_t G = G1.size();
  if (start == 0 || start >= T || f.E.T != T) return book;
  std::vector<std::uint32_t> halfLives = f.horizons;
  if (halfLives.empty()) halfLives = {2, 4, 8, 16, 32, 64, 128, 256};
  // The pairs: every stop and take-profit both set.
  std::vector<std::size_t> pairs;
  for (std::size_t p = 0; p < G * G; ++p)
    if (std::isfinite(G1[p / G]) && std::isfinite(G1[p % G])) pairs.push_back(p);
  const std::size_t P = pairs.size();

  auto barAt = [&](std::size_t d, std::size_t i, Bar* b) {
    *b = {m.open(d, i), m.high(d, i), m.low(d, i), m.close(d, i), m.volume(d, i)};
    // A flat bar (high = low) is a stale quote, not a trading day.
    return b->o > 0 && b->h > b->l && b->l > 0 && b->c > 0 && b->l <= std::min(b->o, b->c) && b->h >= std::max(b->o, b->c) && std::isfinite(b->v) &&
           b->v > 0;
  };
  // Volatility of each stock's open-to-close return, and of the equal-weight market's (series N).
  AdaptiveVol vol(N + 1, halfLives);
  std::vector<Moments> logVolume(N);
  auto addDay = [&](std::size_t t) {
    Bar b;
    double s = 0, n = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (barAt(t, i, &b)) {
        const double r = std::log(b.c / b.o);
        vol.add(i, r), s += r, n += 1;
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
  std::vector<Adwin> record(N);
  for (auto& a : record) a.horizon = double(T - start);
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
  // Each candidate's forecast and pair at the close t: mu (intraday forecast, in its volatility),
  // the best pair and its expected net log return (in its volatility); -inf when nothing is known.
  struct TradePlan {
    std::vector<double> xs;
    double mu = 0, expect = -kNone;
    std::size_t pair = 0;
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
      for (std::size_t q = 0; q < P; ++q) {
        if (ab[q].empty()) continue;
        const double e = ab[q][0] + ab[q][1] * pl.mu;
        if (e > pl.expect) pl.expect = e, pl.pair = q;
      }
    }
    return out;
  };
  // A trade is booked only when it is expected to gain after costs and the stock's own record of
  // these trades beats cash.
  auto worthIt = [&](std::size_t i, const TradePlan& pl) { return pl.expect > 0 && record[i].score() > 0; };
  // The day's book: down the ranking of expected net return, up to 10 trades worth taking; equal
  // shares, scaled so the book's expected volatility is at most the market's.
  auto choose = [&](const std::vector<std::size_t>& cand, const std::vector<TradePlan>& pl, std::vector<std::size_t>* pick, double* w) {
    std::vector<std::size_t> v;
    for (std::size_t i : cand)
      if (worthIt(i, pl[i])) v.push_back(i);
    std::stable_sort(v.begin(), v.end(), [&](std::size_t a, std::size_t c) { return pl[a].expect * sigmaOf(a) > pl[c].expect * sigmaOf(c); });
    if (v.size() > K) v.resize(K);
    *pick = v;
    *w = 0;
    if (v.empty()) return;
    const double n = double(v.size());
    double var = 0;
    for (std::size_t i : v) var += sigmaOf(i) * sigmaOf(i) / (n * n);
    const double mk = vol.sigma(N);
    *w = (1 / n) * ((var > 0 && mk > 0) ? std::min(1.0, mk / std::sqrt(var)) : 1.0);
  };

  Bar b;
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
    std::vector<std::size_t> pick;
    double w;
    choose(cand, pl, &pick, &w);
    double r = 0;
    for (std::size_t i : pick) {
      barAt(d, i, &b);
      const std::size_t p = pairs[pl[i].pair];
      int e;
      r += w * tradeReturn(b, sigmaOf(i), G1[p / G], G1[p % G], buy, sell, true, &e);
      ++book.trades, book.stops += e == 1, book.takes += e == 2;
    }
    book.ret.push_back(r), book.booked.push_back(int(pick.size())), book.exposure.push_back(w * double(pick.size()));
    // After the close d: the day's outcomes, for the stocks' records and the fits.
    for (std::size_t i : cand) {
      const TradePlan& p = pl[i];
      if (p.xs.empty()) continue;
      barAt(d, i, &b);
      const double sig = sigmaOf(i);
      int e;
      if (p.expect > 0) {
        const std::size_t q = pairs[p.pair];
        record[i].push(std::log1p(tradeReturn(b, sig, G1[q / G], G1[q % G], buy, sell, true, &e)));
      }
      std::vector<double> x1{1.0};
      x1.insert(x1.end(), p.xs.begin(), p.xs.end());
      const double z = std::log(b.c / b.o) / sig;
      book.maxTarget = std::max(book.maxTarget, std::fabs(z));
      book.bigTargets += std::fabs(z) > 10;
      // Out-of-sample skill of the intraday forecast: its correlation with what happened.
      if (intraday.n > double(F + 1)) book.skill.add(p.mu, z);
      intraday.add(x1, z);
      if (intraday.n > double(F + 1)) {  // pair fits on genuine forecasts only
        const std::vector<double> mu1{1.0, p.mu};
        for (std::size_t q = 0; q < P; ++q) {
          const std::size_t pp = pairs[q];
          pairFit[q].add(mu1, std::log1p(tradeReturn(b, sig, G1[pp / G], G1[pp % G], buy, sell, true, &e)) / sig);
        }
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
  auto& sc = book.screen;
  sc.stocks = N, sc.candidates = cand.size();
  for (std::size_t i = 0; i < N; ++i)
    if (barAt(t, i, &b)) ++sc.bars, sc.flatBars += b.h == b.l;
  for (std::size_t i : cand) {
    sc.forecastUp += !pl[i].xs.empty() && pl[i].mu > 0, sc.gainExpected += pl[i].expect > 0, sc.recordUp += record[i].score() > 0;
    sc.all += worthIt(i, pl[i]);
  }
  std::vector<std::size_t> pick;
  double w;
  choose(cand, pl, &pick, &w);
  for (std::size_t i : pick) {
    const std::size_t p = pairs[pl[i].pair];
    DayOrder o;
    o.asset = i, o.expected = f.E(t, i), o.sigma = sigmaOf(i), o.stop = G1[p / G], o.take = G1[p % G];
    o.intraday = pl[i].mu * o.sigma, o.tradeReturn = std::expm1(pl[i].expect * o.sigma), o.weight = w;
    o.record = record[i].score(), o.window = record[i].window();
    book.orders.push_back(o);
  }
  return book;
}

}  // namespace ofm
