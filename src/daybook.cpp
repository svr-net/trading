#include "ofm/daybook.hpp"

#include <algorithm>
#include <cmath>
#include <random>

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

Mlp::Mlp(std::size_t in, std::size_t h, std::size_t out) : nIn(in), nH(h), nOut(out) {
  std::mt19937 rng(7);
  auto init = [&](std::vector<double>& w, std::size_t fanIn, std::size_t n) {
    std::normal_distribution<double> g(0.0, 1.0 / std::sqrt(double(fanIn)));
    w.resize(n);
    for (double& x : w) x = g(rng);
  };
  init(w1, in, in * h), b1.assign(h, 0.0), init(w2, h, h * h), b2.assign(h, 0.0), init(w3, h, h * out), b3.assign(out, 0.0);
  for (std::size_t q = 0; q < 6; ++q) m.emplace_back(param(q)->size(), 0.0), v.emplace_back(param(q)->size(), 0.0);
}

std::vector<double>* Mlp::param(std::size_t q) {
  std::vector<double>* p[6] = {&w1, &b1, &w2, &b2, &w3, &b3};
  return p[q];
}

void Mlp::forward(const std::vector<double>& x, std::vector<double>& h1, std::vector<double>& h2, std::vector<double>& y) const {
  h1.assign(nH, 0.0), h2.assign(nH, 0.0), y.assign(nOut, 0.0);
  for (std::size_t j = 0; j < nH; ++j) {
    double a = b1[j];
    for (std::size_t i = 0; i < nIn; ++i) a += w1[j * nIn + i] * x[i];
    h1[j] = std::tanh(a);
  }
  for (std::size_t j = 0; j < nH; ++j) {
    double a = b2[j];
    for (std::size_t i = 0; i < nH; ++i) a += w2[j * nH + i] * h1[i];
    h2[j] = std::tanh(a);
  }
  for (std::size_t k = 0; k < nOut; ++k) {
    double a = b3[k];
    for (std::size_t i = 0; i < nH; ++i) a += w3[k * nH + i] * h2[i];
    y[k] = a;
  }
}

std::vector<double> Mlp::predict(const std::vector<double>& x) const {
  std::vector<double> h1, h2, y;
  forward(x, h1, h2, y);
  return y;
}

void Mlp::train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y) {
  if (X.empty()) return;
  std::vector<std::vector<double>> g;
  for (std::size_t q = 0; q < 6; ++q) g.emplace_back(param(q)->size(), 0.0);
  std::vector<double> h1, h2, y, d3(nOut), d2(nH), d1(nH);
  for (std::size_t n = 0; n < X.size(); ++n) {
    forward(X[n], h1, h2, y);
    for (std::size_t k = 0; k < nOut; ++k) d3[k] = (y[k] - Y[n][k]) / double(nOut);
    for (std::size_t k = 0; k < nOut; ++k) {
      g[5][k] += d3[k];
      for (std::size_t i = 0; i < nH; ++i) g[4][k * nH + i] += d3[k] * h2[i];
    }
    for (std::size_t i = 0; i < nH; ++i) {
      double a = 0;
      for (std::size_t k = 0; k < nOut; ++k) a += w3[k * nH + i] * d3[k];
      d2[i] = a * (1 - h2[i] * h2[i]);
    }
    for (std::size_t j = 0; j < nH; ++j) {
      g[3][j] += d2[j];
      for (std::size_t i = 0; i < nH; ++i) g[2][j * nH + i] += d2[j] * h1[i];
    }
    for (std::size_t i = 0; i < nH; ++i) {
      double a = 0;
      for (std::size_t j = 0; j < nH; ++j) a += w2[j * nH + i] * d2[j];
      d1[i] = a * (1 - h1[i] * h1[i]);
    }
    for (std::size_t j = 0; j < nH; ++j) {
      g[1][j] += d1[j];
      for (std::size_t i = 0; i < nIn; ++i) g[0][j * nIn + i] += d1[j] * X[n][i];
    }
  }
  const double lr = 1e-3, b1m = 0.9, b2m = 0.999, eps = 1e-8;
  step += 1;
  for (std::size_t q = 0; q < 6; ++q) {
    std::vector<double>& p = *param(q);
    for (std::size_t i = 0; i < p.size(); ++i) {
      const double gi = g[q][i] / double(X.size());
      m[q][i] = b1m * m[q][i] + (1 - b1m) * gi, v[q][i] = b2m * v[q][i] + (1 - b2m) * gi * gi;
      const double mh = m[q][i] / (1 - std::pow(b1m, step)), vh = v[q][i] / (1 - std::pow(b2m, step));
      p[i] -= lr * mh / (std::sqrt(vh) + eps);
    }
  }
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

DayBook dayBook(const Market& m, const Forecast& f, std::size_t start, double buy, double sell) {
  using namespace day;
  DayBook book;
  book.buy = buy, book.sell = sell, book.start = start;
  const std::size_t T = m.T(), N = m.N(), K = 10;
  const std::vector<double>& G1 = grid();
  const std::size_t G = G1.size(), P = G * G, F = 10;
  if (start == 0 || start >= T || f.E.T != T) return book;

  auto barAt = [&](std::size_t d, std::size_t i, Bar* b) {
    *b = {m.open(d, i), m.high(d, i), m.low(d, i), m.close(d, i), m.volume(d, i)};
    return b->o > 0 && b->h > 0 && b->l > 0 && b->c > 0 && b->l <= std::min(b->o, b->c) && b->h >= std::max(b->o, b->c) && std::isfinite(b->v) &&
           b->v > 0;
  };
  // Each stock's open-to-close log-return moments, and log-volume moments, up to the current close.
  std::vector<Moments> vol(N), logVolume(N);
  auto addVol = [&](std::size_t t) {
    Bar b;
    for (std::size_t i = 0; i < N; ++i)
      if (barAt(t, i, &b)) vol[i].add(std::log(b.c / b.o));
  };
  for (std::size_t t = 0; t + 1 < start; ++t) addVol(t);
  auto sigmaOf = [&](std::size_t i) { return vol[i].sd(); };
  // Features of stock i at the close t (empty if its recent bars are missing).
  auto features = [&](std::size_t t, std::size_t i, double medSig, double mkt) {
    Bar b0, b1, b5;
    const double sig = sigmaOf(i);
    if (t < 5 || !(sig > 0) || !barAt(t, i, &b0) || !barAt(t - 1, i, &b1) || !barAt(t - 5, i, &b5) || logVolume[i].n < 2) return std::vector<double>{};
    const double lvMu = logVolume[i].mean(), lvSd = std::sqrt(std::max(1e-12, logVolume[i].s2 / logVolume[i].n - lvMu * lvMu));
    return std::vector<double>{f.E(t, i) / sig, std::log(sig / medSig), std::log(b0.c / b0.o) / sig, std::log(b0.c / b1.c) / sig,
                               std::log(b0.o / b1.c) / sig, b0.h > b0.l ? (b0.c - b0.l) / (b0.h - b0.l) : 0.5, std::log(b0.h / b0.l) / sig,
                               std::log(b0.c / b5.c) / sig, (std::log(b0.v) - lvMu) / lvSd, mkt / medSig};
  };
  Mlp net(F, 16, P);
  Scaler scaler(F);
  std::vector<Adwin> record(N);
  for (auto& a : record) a.horizon = double(T - start);
  // The pair (stop and take-profit both set) the network rates best for a stock, and its expected net log return in the stock's
  // volatility (after costs); none / none and no expectation before the network has learnt.
  auto pickOf = [&](const std::vector<double>& x, double* expect) {
    *expect = -kNone;
    if (x.empty() || net.step == 0) return P - 1;
    const auto y = net.predict(scaler.apply(x));
    // Every trade has both a stop-loss and a take-profit: the best pair with both levels set.
    std::size_t p = P - 1;
    for (std::size_t q = 0; q < P; ++q)
      if (std::isfinite(G1[q / G]) && std::isfinite(G1[q % G]) && (p == P - 1 || y[q] > y[p])) p = q;
    *expect = y[p];
    return p;
  };
  // A trade is booked only when it is expected to make money: the stock is forecast to beat the
  // market, the network expects the trade to gain after costs, and the stock's own record of these
  // trades beats cash.
  auto worthIt = [&](std::size_t t, std::size_t i, double expect) { return f.E(t, i) > 0 && expect > 0 && record[i].score() > 0; };
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
  auto ranked = [&](std::vector<std::size_t> v, std::size_t t) {
    std::stable_sort(v.begin(), v.end(), [&](std::size_t x, std::size_t y) { return f.E(t, x) > f.E(t, y); });
    return v;
  };

  Bar b;
  for (std::size_t d = start; d < T; ++d) {
    const std::size_t t = d - 1;
    if (t > 0) addVol(t);
    // Candidates: a forecast at the close t, a volatility, and a bar on day d.
    std::vector<std::size_t> cand;
    for (std::size_t i = 0; i < N; ++i)
      if (std::isfinite(f.E(t, i)) && vol[i].n >= 2 && barAt(d, i, &b)) cand.push_back(i);
    double medSig, mkt;
    dayContext(t, cand, &medSig, &mkt);
    // Each candidate's features and pair, from what was known at the close t.
    std::vector<std::vector<double>> x(N);
    std::vector<std::size_t> pick(N, P - 1);
    std::vector<double> expect(N, -kNone);
    for (std::size_t i : cand) x[i] = features(t, i, medSig, mkt), pick[i] = pickOf(x[i], &expect[i]);
    // The book: down the ranking, the trades expected to make money, until 10.
    double r = 0;
    int booked = 0;
    for (std::size_t i : ranked(cand, t)) {
      if (std::size_t(booked) == K) break;
      if (!worthIt(t, i, expect[i])) continue;
      barAt(d, i, &b);
      int e;
      r += tradeReturn(b, sigmaOf(i), G1[pick[i] / G], G1[pick[i] % G], buy, sell, true, &e);
      ++booked, ++book.trades, book.stops += e == 1, book.takes += e == 2;
    }
    book.ret.push_back(r / double(K)), book.booked.push_back(booked);
    // After the close d: each candidate's day trade goes to its record when the rule would have taken
    // it (whether booked or not), and its 64 outcomes
    // (log returns in its volatility, worst-case fills, after costs) train the network.
    std::vector<std::vector<double>> X, Y;
    for (std::size_t i : cand) {
      barAt(d, i, &b);
      const double sig = sigmaOf(i);
      int e;
      // The record holds the trades the rule would take: days its forecast was up and a gain was expected.
      if (f.E(t, i) > 0 && expect[i] > 0) record[i].push(std::log1p(tradeReturn(b, sig, G1[pick[i] / G], G1[pick[i] % G], buy, sell, true, &e)));
      if (x[i].empty()) continue;
      std::vector<double> y(P);
      for (std::size_t p = 0; p < P; ++p) y[p] = std::log1p(tradeReturn(b, sig, G1[p / G], G1[p % G], buy, sell, true, &e)) / sig;
      scaler.add(x[i]);
      X.push_back(x[i]), Y.push_back(std::move(y));
    }
    for (auto& xi : X) xi = scaler.apply(xi);
    for (std::size_t q = 0; q < X.size(); q += 32) {
      const std::size_t e = std::min(X.size(), q + 32);
      net.train({X.begin() + q, X.begin() + e}, {Y.begin() + q, Y.begin() + e});
    }
    for (std::size_t i = 0; i < N; ++i)
      if (barAt(d, i, &b)) logVolume[i].add(std::log(b.v));
  }

  // Orders for the next open, from the last close.
  const std::size_t t = T - 1;
  addVol(t);
  std::vector<std::size_t> cand;
  for (std::size_t i = 0; i < N; ++i)
    if (std::isfinite(f.E(t, i)) && vol[i].n >= 2 && barAt(t, i, &b)) cand.push_back(i);
  double medSig, mkt;
  dayContext(t, cand, &medSig, &mkt);
  auto& sc = book.screen;
  sc.stocks = N, sc.candidates = cand.size();
  for (std::size_t i = 0; i < N; ++i)
    if (barAt(t, i, &b)) ++sc.bars, sc.flatBars += b.h == b.l;
  for (std::size_t i : ranked(cand, t)) {
    double expect;
    const std::size_t p = pickOf(features(t, i, medSig, mkt), &expect);
    sc.forecastUp += f.E(t, i) > 0, sc.gainExpected += expect > 0, sc.recordUp += record[i].score() > 0;
    if (!worthIt(t, i, expect)) continue;
    ++sc.all;
    if (book.orders.size() == K) continue;
    DayOrder o;
    o.asset = i, o.expected = f.E(t, i), o.sigma = sigmaOf(i), o.stop = G1[p / G], o.take = G1[p % G];
    o.tradeReturn = std::expm1(expect * o.sigma);
    o.record = record[i].score(), o.window = record[i].window();
    book.orders.push_back(o);
  }
  return book;
}

}  // namespace ofm
