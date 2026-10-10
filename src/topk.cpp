#include "ofm/topk.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>

namespace ofm {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double rankCorrelation(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = a.size();
  if (n < 3) return kNaN;
  auto ranks = [n](const std::vector<double>& x) {
    std::vector<std::size_t> o(n);
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](std::size_t p, std::size_t q) { return x[p] < x[q]; });
    std::vector<double> r(n);
    for (std::size_t k = 0; k < n; ++k) r[o[k]] = static_cast<double>(k);
    return r;
  };
  const auto ra = ranks(a), rb = ranks(b);
  const double mu = (static_cast<double>(n) - 1) / 2;
  double sab = 0, saa = 0;
  for (std::size_t k = 0; k < n; ++k) sab += (ra[k] - mu) * (rb[k] - mu), saa += (ra[k] - mu) * (ra[k] - mu);
  return saa > 0 ? sab / saa : kNaN;
}

// What every simulation shares: the ranking, the forecast's life and the stocks' volatility.
struct Inputs {
  const Market* m = nullptr;
  Panel close;                            // carried forward
  Panel score;                            // NaN: not eligible
  std::vector<std::vector<std::size_t>> order;  // eligible stocks by score, best first
  bool returnUnits = false;               // the score is an expected daily return
  bool oneDay = false;                    // levels reset each day from the open, one day's volatility
  std::vector<double> life;               // per day (days)
  Panel width;                            // sigma sqrt(life) at each close
  std::size_t s = 0, K = 10;
  double buy = 0, sell = 0;
};

struct Sim {
  std::vector<double> daily, turnover, kS, kT;
  std::vector<TopKTrade> trades;
  std::vector<TopKPosition> positions;
  std::vector<int> orders;
};

// One run; kStop/kTake per decision day t (index t - s), or a single value for all days.
Sim simulate(const Inputs& in, const std::vector<double>& kStop, const std::vector<double>& kTake, bool record) {
  const Market& m = *in.m;
  const std::size_t T = m.T(), N = m.N(), K = in.K;
  Sim out;
  std::vector<double> h(N, 0.0), entryValue(N, 0.0), entryPx(N, 0.0), w(N, 0.0);
  std::vector<std::size_t> entryDay(N, 0);
  std::vector<int> order(N, 0);
  double cash = 1;
  auto k = [](const std::vector<double>& v, std::size_t j) { return v.size() == 1 ? v[0] : v[j]; };
  auto exitTrade = [&](std::size_t i, std::size_t d, double v, Exit why) {
    cash += v * (1 - in.sell);
    if (record) out.trades.push_back({i, entryDay[i], d, v * (1 - in.sell) / entryValue[i] - 1, why});
    h[i] = 0;
  };
  auto decide = [&](std::size_t t) {
    std::fill(order.begin(), order.end(), 0);
    const auto& ranked = in.order[t];
    std::vector<char> held(N, 0);
    std::size_t nHeld = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (h[i] > 0) {
        held[i] = 1, ++nHeld;
        if (!std::isfinite(in.score(t, i))) order[i] = -1, --nHeld;  // left the universe
      }
    if (in.returnUnits) {
      // Swap the worst held for the best outsider while the gain over the life beats a round trip.
      std::vector<std::size_t> outs, ins;
      for (std::size_t i : ranked)
        if (!held[i]) outs.push_back(i);
      for (auto it = ranked.rbegin(); it != ranked.rend(); ++it)
        if (held[*it] && order[*it] == 0) ins.push_back(*it);
      std::size_t o = 0;
      for (std::size_t q = 0; q < ins.size() && o < outs.size(); ++q, ++o) {
        if ((in.score(t, outs[o]) - in.score(t, ins[q])) * in.life[t] <= in.buy + in.sell) break;
        order[ins[q]] = -1, order[outs[o]] = 1;
      }
      std::size_t slots = K - std::min(K, nHeld);
      for (; o < outs.size() && slots > 0; ++o)
        if (order[outs[o]] == 0) order[outs[o]] = 1, --slots;
    } else {
      std::vector<char> top(N, 0);
      for (std::size_t q = 0; q < std::min(K, ranked.size()); ++q) top[ranked[q]] = 1;
      for (std::size_t i = 0; i < N; ++i)
        if (held[i] && !top[i]) order[i] = -1;
      for (std::size_t q = 0; q < std::min(K, ranked.size()); ++q)
        if (!held[ranked[q]]) order[ranked[q]] = 1;
    }
  };

  for (std::size_t t = in.s; t + 1 < T; ++t) {
    decide(t);
    const std::size_t d = t + 1, j = t - in.s;
    const double a = k(kStop, j), b = k(kTake, j);
    if (record) out.kS.push_back(a), out.kT.push_back(b);
    double eqPrev = cash;
    for (std::size_t i = 0; i < N; ++i) eqPrev += h[i];
    double traded = 0;
    // Overnight, sales and stops gapped through at the open.
    for (std::size_t i = 0; i < N; ++i) {
      if (!(h[i] > 0)) continue;
      const double o = m.open(d, i), c = in.close(t, i);
      if (!(o > 0 && c > 0)) continue;
      h[i] *= o / c;
      if (order[i] < 0) traded += h[i], exitTrade(i, d, h[i], Exit::Rotate);
      else if (!in.oneDay && o <= entryPx[i] * std::exp(-a * w[i])) traded += h[i], exitTrade(i, d, h[i], Exit::Stop);
      else if (!in.oneDay && o >= entryPx[i] * std::exp(b * w[i])) traded += h[i], exitTrade(i, d, h[i], Exit::Take);
    }
    // Purchases: the cash split equally.
    std::vector<std::size_t> buys;
    for (std::size_t i = 0; i < N; ++i)
      if (order[i] > 0 && m.open(d, i) > 0 && std::isfinite(in.width(t, i))) buys.push_back(i);
    if (!buys.empty() && cash > 0) {
      const double v = cash / static_cast<double>(buys.size()) / (1 + in.buy);
      for (std::size_t i : buys)
        h[i] = v, entryValue[i] = v * (1 + in.buy), entryPx[i] = m.open(d, i), entryDay[i] = d, w[i] = in.width(t, i), traded += v;
      cash = 0;
    }
    // One-day holding: every position's levels are set afresh from today's open.
    if (in.oneDay)
      for (std::size_t i = 0; i < N; ++i)
        if (h[i] > 0 && m.open(d, i) > 0 && std::isfinite(in.width(t, i))) entryPx[i] = m.open(d, i), w[i] = in.width(t, i) / std::sqrt(in.life[t]);
    // During the day: stop first, then take-profit; then to the close.
    for (std::size_t i = 0; i < N; ++i) {
      if (!(h[i] > 0)) continue;
      const double o = m.open(d, i), lo = m.low(d, i), hi = m.high(d, i), c = in.close(d, i);
      const double stop = entryPx[i] * std::exp(-a * w[i]), take = entryPx[i] * std::exp(b * w[i]);
      if (o > 0 && lo > 0 && lo <= stop) traded += h[i] * stop / o, exitTrade(i, d, h[i] * stop / o, Exit::Stop);
      else if (o > 0 && hi > 0 && hi >= take) traded += h[i] * take / o, exitTrade(i, d, h[i] * take / o, Exit::Take);
      else if (o > 0 && c > 0) h[i] *= c / o;
      else if (c > 0 && in.close(t, i) > 0) h[i] *= c / in.close(t, i);
    }
    double eq = cash;
    for (std::size_t i = 0; i < N; ++i) eq += h[i];
    out.daily.push_back(eq / eqPrev - 1);
    out.turnover.push_back(traded / eqPrev);
  }
  if (record) {
    const std::size_t L = T - 1;
    decide(L);
    out.orders = order;
    const double a = k(kStop, kStop.size() == 1 ? 0 : kStop.size() - 1), b = k(kTake, kTake.size() == 1 ? 0 : kTake.size() - 1);
    for (std::size_t i = 0; i < N; ++i)
      if (h[i] > 0) {
        out.trades.push_back({i, entryDay[i], L, h[i] / entryValue[i] - 1, Exit::Open});
        const double c = in.close(L, i);
        out.positions.push_back({i, entryDay[i], h[i] / entryValue[i] - 1, std::isfinite(a) ? entryPx[i] * std::exp(-a * w[i]) / c - 1 : kNaN,
                                 std::isfinite(b) ? entryPx[i] * std::exp(b * w[i]) / c - 1 : kNaN, entryPx[i] / c, w[i]});
      }
  }
  return out;
}

// Probability that a Brownian log price from 0 (drift mu, volatility sigma a day) touches lo
// before hi, and hi before lo, within `days` (Monte Carlo, fixed seed: the same inputs give the
// same answer).
std::pair<double, double> firstTouch(double mu, double sigma, double lo, double hi, double days, std::uint64_t seed) {
  if (!(sigma > 0) || !(days > 0)) return {0.0, 0.0};
  const int perDay = 48, paths = 20000;
  const int steps = std::max(1, static_cast<int>(std::lround(days * perDay)));
  const double dt = days / steps, drift = (mu - 0.5 * sigma * sigma) * dt, sd = sigma * std::sqrt(dt);
  std::uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
  auto uniform = [&x]() {
    x += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull, z = (z ^ (z >> 27)) * 0x94D049BB133111EBull, z ^= z >> 31;
    return (static_cast<double>(z >> 11) + 0.5) / 9007199254740992.0;
  };
  int nLo = 0, nHi = 0;
  for (int p = 0; p < paths; ++p) {
    double y = 0;
    for (int k = 0; k < steps; ++k) {
      const double g = std::sqrt(-2 * std::log(uniform())) * std::cos(6.283185307179586 * uniform());
      y += drift + sd * g;
      if (y <= lo) { ++nLo; break; }
      if (y >= hi) { ++nHi; break; }
    }
  }
  return {static_cast<double>(nLo) / paths, static_cast<double>(nHi) / paths};
}

// A stock's stop-loss and take-profit, learnt from its own days: how far below the open each day's
// low went (lo) and how far above the high went (hi), in the stock's volatility, and the close (log).
// The levels are those with the best return per unit of risk (the mean net log return of a day's
// trade over its standard deviation; the round trip is in every trade and a stop fills worse than
// its level by a side's cost), searched over every value the data allow, none included, by turns
// (the best stop for the current take-profit, then the best take-profit for that stop, until
// neither moves). Not trading earns nothing at no risk, so the stock is traded only while the best
// levels' ratio is above zero.
struct Learner {
  struct Obs { double lo, hi, c, sig; };
  std::vector<Obs> obs;
  std::vector<std::size_t> byLo, byHi;  // obs sorted by lo and by hi
  double kc = 0, kcs = 0;               // log of the round trip; the same for a stopped trade
  double maxDrop = 0, maxRise = 0;      // the largest fall and rise from the open ever seen (log)
  double minMove = 0;                   // the nearest a level may be to the open (log): the round trip
  double sigSum = 0;

  void add(double lo, double hi, double c, double sig) {
    obs.push_back({lo, hi, c, sig});
    maxDrop = std::max(maxDrop, lo * sig), maxRise = std::max(maxRise, hi * sig), sigSum += sig;
    const std::size_t idx = obs.size() - 1;
    auto put = [&](std::vector<std::size_t>& v, double Obs::*key) {
      const double kv = obs[idx].*key;
      v.insert(std::lower_bound(v.begin(), v.end(), kv, [&](std::size_t q, double val) { return obs[q].*key < val; }), idx);
    };
    put(byLo, &Obs::lo), put(byHi, &Obs::hi);
  }
  static double ratio(double s1, double s2, double n) {
    if (n < 2) return -kInf;
    const double mu = s1 / n, vr = std::max(0.0, s2 / n - mu * mu);
    return vr > 0 ? mu / std::sqrt(vr) : (mu > 0 ? kInf : -kInf);
  }
  double bestStop(double b, double& score) const {
    // Stopped when lo >= a: earns -a sig; otherwise the take-profit (hi >= b: b sig) or the close.
    const double n = static_cast<double>(obs.size());
    double h1 = 0, h2 = 0, t1 = 0, t2 = 0, tm = 0;  // head: sums of base, base^2; tail: sig, sig^2, count
    double a1 = 0, a2 = 0;
    for (const auto& x : obs) {
      const double v = (x.hi >= b ? b * x.sig : x.c) + kc;
      a1 += v, a2 += v * v, t1 += x.sig, t2 += x.sig * x.sig, tm += 1;
    }
    // Every trade has a stop: the best among the lows at least the round trip below the open.
    const double floorA = obs.empty() ? 0.0 : minMove / (sigSum / static_cast<double>(obs.size()));
    double bestA = byLo.empty() ? kInf : obs[byLo.back()].lo;
    score = -kInf;
    for (std::size_t q = 0; q < byLo.size(); ++q) {
      const Obs& x = obs[byLo[q]];
      const double a = x.lo;  // this low and all deeper ones stop at -a sig
      if (!(a > 0) || a < floorA) {  // nearer the open than the round trip: not a stop that can pay
        const double v = (x.hi >= b ? b * x.sig : x.c) + kc;
        h1 += v, h2 += v * v, t1 -= x.sig, t2 -= x.sig * x.sig, tm -= 1;
        continue;
      }
      const double s1 = h1 - a * t1 + tm * kcs, s2 = h2 + a * a * t2 - 2 * a * kcs * t1 + tm * kcs * kcs;
      const double rr = ratio(s1, s2, n);
      if (rr > score + 1e-12) score = rr, bestA = a;
      const double v = (x.hi >= b ? b * x.sig : x.c) + kc;
      h1 += v, h2 += v * v, t1 -= x.sig, t2 -= x.sig * x.sig, tm -= 1;
    }
    return bestA;
  }
  double bestTake(double a, double& score) const {
    // The stopped trades are fixed; among the others, hi >= b takes profit at b sig, else the close.
    const double n = static_cast<double>(obs.size());
    double c1 = 0, c2 = 0, h1 = 0, h2 = 0, t1 = 0, t2 = 0, tm = 0, all1 = 0, all2 = 0;
    for (const auto& x : obs) {
      if (x.lo >= a) {
        const double v = -a * x.sig + kcs;
        c1 += v, c2 += v * v;
      } else {
        const double v = x.c + kc;
        all1 += v, all2 += v * v, t1 += x.sig, t2 += x.sig * x.sig, tm += 1;
      }
    }
    // Every trade has a take-profit: the best among the highs at least the round trip above the open.
    const double floorB = obs.empty() ? 0.0 : minMove / (sigSum / static_cast<double>(obs.size()));
    double bestB = byHi.empty() ? kInf : obs[byHi.back()].hi;
    score = -kInf;
    for (std::size_t q = 0; q < byHi.size(); ++q) {
      const Obs& x = obs[byHi[q]];
      if (x.lo >= a) continue;
      const double b = x.hi;  // this high and all higher ones take profit at b sig
      if (!(b > 0) || b < floorB) {
        const double v = x.c + kc;
        h1 += v, h2 += v * v, t1 -= x.sig, t2 -= x.sig * x.sig, tm -= 1;
        continue;
      }
      const double s1 = c1 + h1 + b * t1 + tm * kc, s2 = c2 + h2 + b * b * t2 + 2 * b * kc * t1 + tm * kc * kc;
      const double rr = ratio(s1, s2, n);
      if (rr > score + 1e-12) score = rr, bestB = b;
      const double v = x.c + kc;
      h1 += v, h2 += v * v, t1 -= x.sig, t2 -= x.sig * x.sig, tm -= 1;
    }
    return bestB;
  }
  // Whether to trade, with the levels and their ratio.
  bool learn(double& a, double& b, double& lastScore) const {
    a = byLo.empty() ? kInf : obs[byLo.back()].lo, b = byHi.empty() ? kInf : obs[byHi.back()].hi, lastScore = kNaN;
    if (obs.size() < 2) return false;
    double score = -kInf;
    for (int it = 0; it < 50; ++it) {
      double s1, s2;
      const double nb = bestTake(a, s1), na = bestStop(nb, s2);
      score = s2;
      if (na == a && nb == b) break;
      a = na, b = nb;
    }
    lastScore = score;
    return score > 0;
  }
  std::pair<double, double> probs(double a, double b) const {
    double ns = 0, nt = 0;
    for (const auto& x : obs) ns += x.lo >= a, nt += x.lo < a && x.hi >= b;
    const double n = std::max<double>(1, static_cast<double>(obs.size()));
    return std::make_pair(obs.empty() ? 0.0 : ns / n, obs.empty() ? 0.0 : nt / n);
  }
};

// Today's 10 best by expected return, each bought at the next open and sold the same day at its
// stop-loss, its take-profit (stop first when both are touched) or the close. Levels are multiples
// of the stock's open-to-close volatility (exponentially weighted, half-life the forecast's life),
// learnt from the bars of the trades before (below). Every trade pays a round trip.
TopKResult dayTrades(const Market& m, const Forecast& f, const Inputs& in, const Costs& costs, std::size_t K, std::size_t start) {
  const std::size_t T = m.T(), N = m.N(), s = in.s;
  const double buy = costs.buyBps * 1e-4, sell = costs.sellBps * 1e-4;
  TopKResult r;
  r.name = "top 10 by expected return", r.K = K, r.start = start, r.oneDay = r.dayTrades = true;
  // Open-to-close volatility known at each close.
  Panel iv(T, N);
  std::vector<double> var(N, kNaN);
  for (std::size_t t = 0; t < T; ++t) {
    const double lam = std::pow(2.0, -1.0 / std::max(1.0, in.life[t]));
    for (std::size_t i = 0; i < N; ++i) {
      const double o = m.open(t, i), c = m.close(t, i);
      if (o > 0 && c > 0) {
        const double x = std::log(c / o);
        var[i] = std::isfinite(var[i]) ? lam * var[i] + (1 - lam) * x * x : x * x;
      }
      if (std::isfinite(var[i]) && var[i] > 0) iv(t, i) = std::sqrt(var[i]);
    }
  }
  struct Trade { std::size_t i; double o, h, l, c, sig, e, ex, v; };
  std::vector<std::vector<Trade>> days;
  std::vector<double> all;  // the day's mean open-to-close over all ranked stocks
  for (std::size_t t = s; t + 1 < T; ++t) {
    const std::size_t d = t + 1;
    double sa = 0, na = 0;
    for (std::size_t i : in.order[t]) {
      const double o = m.open(d, i), c = m.close(d, i);
      if (o > 0 && c > 0) sa += c / o - 1, na += 1;
    }
    const double mean = na > 0 ? sa / na : 0;
    std::vector<Trade> day;
    for (std::size_t i : in.order[t]) {
      if (day.size() == K) break;
      const double o = m.open(d, i), h = m.high(d, i), l = m.low(d, i), c = m.close(d, i);
      if (!(o > 0 && h > 0 && l > 0 && c > 0) || !std::isfinite(iv(t, i))) continue;
      // A bar must hold together: the low at or below the open and the close, the high at or above.
      if (l > std::min(o, c) || h < std::max(o, c)) continue;
      double vs = 0, vn = 0;  // usual volume: the mean over the forecast's life before the day
      for (std::size_t k = 1; k <= static_cast<std::size_t>(std::ceil(in.life[t])) && k <= d; ++k)
        if (std::isfinite(m.volume(d - k, i))) vs += m.volume(d - k, i), vn += 1;
      day.push_back({i, o, h, l, c, iv(t, i), f.E(t, i), c / o - 1 - mean, vn > 0 && vs > 0 ? m.volume(d, i) / (vs / vn) : kNaN});
    }
    days.push_back(std::move(day));
    all.push_back(mean);
  }
  const double kcRound = std::log((1 - sell) / (1 + buy));
  auto outcome = [&](const Trade& x, double a, double b, Exit& why) {
    // No level nearer the open than the round trip.
    const double stop = x.o * std::exp(-std::max(a * x.sig, -kcRound)), take = x.o * std::exp(std::max(b * x.sig, -kcRound));
    double px = x.c;
    why = Exit::Close;
    if (x.l <= stop) px = stop * (1 - sell), why = Exit::Stop;  // a stop fills worse than its level, by a side's cost
    else if (x.h >= take) px = take, why = Exit::Take;
    return (px / x.o) * (1 - sell) / (1 + buy) - 1;
  };
  // The state of each stock at each close, from three indicators: the parabolic SAR's trend (price
  // above or below it), the RSI (above or below 50, its neutral point) and the ADX (trend strength,
  // above or below its average so far over all stocks). Their lookback is the forecast's life,
  // rounded up to the model's next horizon (the RSI's and ADX's Wilder smoothing 1/h; the SAR's
  // acceleration rising by 1/h with each new extreme, up to 1). Eight states: the trades of each
  // state teach that state its own stop-loss and take-profit (how the levels respond to the trend,
  // its momentum and its strength).
  const std::vector<std::uint32_t>& H = f.horizons;
  auto lookback = [&](std::size_t t) {
    for (auto h : H)
      if (h >= in.life[t]) return static_cast<double>(h);
    return H.empty() ? 1.0 : static_cast<double>(H.back());
  };
  std::vector<int> state(T * N, -1);
  {
    struct Ind { double g = kNaN, l = kNaN, pdm = kNaN, ndm = kNaN, tr = kNaN, adx = kNaN, sar = kNaN, ep = kNaN, af = 0; bool up = true; };
    std::vector<Ind> st(N);
    double adxSum = 0, adxN = 0;
    for (std::size_t t = 1; t < T; ++t) {
      const double hz = lookback(t), al = 1 / hz;
      for (std::size_t i = 0; i < N; ++i) {
        const double h = m.high(t, i), l = m.low(t, i), c = m.close(t, i), ph = m.high(t - 1, i), pl = m.low(t - 1, i), pc = m.close(t - 1, i);
        if (!(h > 0 && l > 0 && c > 0 && ph > 0 && pl > 0 && pc > 0)) continue;
        Ind& x = st[i];
        auto ema = [al](double& v, double z) { v = std::isfinite(v) ? v + al * (z - v) : z; };
        // RSI
        ema(x.g, std::max(0.0, c - pc)), ema(x.l, std::max(0.0, pc - c));
        // ADX
        const double up = h - ph, dn = pl - l;
        ema(x.pdm, up > dn && up > 0 ? up : 0.0), ema(x.ndm, dn > up && dn > 0 ? dn : 0.0);
        ema(x.tr, std::max({h - l, std::fabs(h - pc), std::fabs(l - pc)}));
        if (x.tr > 0) {
          const double pdi = x.pdm / x.tr, ndi = x.ndm / x.tr;
          if (pdi + ndi > 0) ema(x.adx, 100 * std::fabs(pdi - ndi) / (pdi + ndi));
        }
        // Parabolic SAR
        if (!std::isfinite(x.sar)) {
          x.up = c >= pc, x.sar = x.up ? pl : ph, x.ep = x.up ? h : l, x.af = al;
        } else if (x.up) {
          x.sar = std::min(x.sar + x.af * (x.ep - x.sar), pl);
          if (l < x.sar) x.up = false, x.sar = x.ep, x.ep = l, x.af = al;
          else if (h > x.ep) x.ep = h, x.af = std::min(1.0, x.af + al);
        } else {
          x.sar = std::max(x.sar + x.af * (x.ep - x.sar), ph);
          if (h > x.sar) x.up = true, x.sar = x.ep, x.ep = h, x.af = al;
          else if (l < x.ep) x.ep = l, x.af = std::min(1.0, x.af + al);
        }
        if (std::isfinite(x.adx)) adxSum += x.adx, adxN += 1;
      }
      for (std::size_t i = 0; i < N; ++i) {
        const Ind& x = st[i];
        if (!std::isfinite(x.adx) || !std::isfinite(x.g) || !(x.g + x.l > 0) || adxN < 1) continue;
        const bool rsiUp = x.g / (x.g + x.l) > 0.5, strong = x.adx > adxSum / adxN;
        state[t * N + i] = (x.up ? 4 : 0) + (rsiUp ? 2 : 0) + (strong ? 1 : 0);
      }
    }
  }
  // Three ways to learn the levels, run side by side: each stock from all its own days (0), each
  // indicator state from the earlier trades in it (1), all the earlier trades together (2). Each
  // proposes its levels for every trade; the trades follow the way whose own record so far has
  // the best return per unit of risk (each way's record is what its proposals would have earned).
  const double kc = std::log((1 - sell) / (1 + buy));
  std::vector<Learner> perStock(N), perState(8), pooled(1);
  for (auto* v : {&perStock, &perState, &pooled})
    for (auto& l : *v) l.kc = kc, l.kcs = kc + std::log(1 - sell), l.minMove = -kc;
  static const Learner none;
  auto learnerOf = [&](int way, std::size_t t, std::size_t i) -> const Learner& {
    if (way == 0) return perStock[i].obs.size() < 2 ? pooled[0] : perStock[i];
    if (way == 2) return pooled[0];
    const int c = state[t * N + i];
    return c < 0 ? none : perState[static_cast<std::size_t>(c)];
  };
  auto feedStocks = [&](std::size_t d) {
    for (std::size_t i = 0; i < N; ++i) {
      const double o = m.open(d, i), h = m.high(d, i), l = m.low(d, i), c = m.close(d, i), sg = iv(d - 1, i);
      if (!(o > 0 && h > 0 && l > 0 && c > 0) || !std::isfinite(sg) || l > std::min(o, c) || h < std::max(o, c)) continue;
      perStock[i].add(-std::log(l / o) / sg, std::log(h / o) / sg, std::log(c / o), sg);
    }
  };
  for (std::size_t d = 1; d <= s; ++d) feedStocks(d);
  const char* wayName[3] = {"each stock", "each indicator state", "all trades together"};
  double wayS1[3] = {0, 0, 0}, wayS2[3] = {0, 0, 0}, wayN[3] = {0, 0, 0};
  auto bestWay = [&]() {
    int best = 2;
    double bestR = -kInf;
    for (int w = 0; w < 3; ++w) {
      const double rr = Learner::ratio(wayS1[w], wayS2[w], wayN[w]);
      if (wayN[w] >= 2 && rr > bestR) bestR = rr, best = w;
    }
    return best;
  };

  // The live portfolio, its trades and the evaluation against the bars.
  const std::size_t D = days.size();
  TopKEvaluation& ev = r.eval;
  double sxy = 0, sxx = 0, syy = 0, sx = 0, sy = 0;
  for (std::size_t j = 0; j < D; ++j) {
    const std::size_t d = s + 1 + j;
    const int way = bestWay();
    r.ways.push_back(way);
    double sum = 0, sum0 = 0, top = 0, aSum = 0, bSum = 0, nTraded = 0, waySum[3] = {0, 0, 0};
    for (const auto& x : days[j])
      for (int w = 0; w < 3; ++w) {
        double a, b, score;
        Exit why;
        if (learnerOf(w, d - 1, x.i).learn(a, b, score)) waySum[w] += outcome(x, a, b, why);
      }
    for (const auto& x : days[j]) {
      double a, b, score;
      const Learner& ln = learnerOf(way, d - 1, x.i);
      const bool trade = ln.learn(a, b, score);
      const auto [pS, pT] = ln.probs(a, b);
      Exit why, why0;
      const double ret = outcome(x, a, b, why);
      sum += trade ? ret : 0.0, sum0 += outcome(x, kInf, kInf, why0);
      if (trade) nTraded += 1, aSum += a, bSum += b;
      r.trades.push_back({x.i, d, d, ret, why});
      ++ev.trades;
      ev.predStop += pS, ev.predTake += pT, ev.realStop += why == Exit::Stop, ev.realTake += why == Exit::Take;
      ev.lowAtOpen += x.l >= x.o;
      const double e = std::isfinite(x.e) ? x.e : 0.0;
      ev.expected += e, ev.realised += x.ex, ev.signHit += (e > 0) == (x.ex > 0);
      sx += e, sy += x.ex, sxx += e * e, syy += x.ex * x.ex, sxy += e * x.ex;
      top += x.c / x.o - 1;
    }
    const double nd = static_cast<double>(days[j].size());
    r.daily.push_back(nd > 0 ? sum / nd : 0.0);  // each of the K slots: its trade, or cash
    r.traded.push_back(nTraded > 0);
    r.noStops.push_back(nd > 0 ? sum0 / nd : 0.0);
    r.kStop.push_back(nTraded > 0 ? aSum / nTraded : kNaN), r.kTake.push_back(nTraded > 0 ? bSum / nTraded : kNaN);
    r.turnover.push_back(nd > 0 ? 2.0 : 0.0);
    if (nd > 0) ++ev.days, ev.grossTop += top / nd, ev.grossAll += all[j];
    if (j + 1 == D) {
      r.lastDate = m.dates[d];
      for (const auto& x : days[j]) {
        double a, b, score;
        const Learner& ln = learnerOf(way, d - 1, x.i);
        const bool trade = ln.learn(a, b, score);
        const auto [pS, pT] = ln.probs(a, b);
        Exit why;
        const double ret = trade ? outcome(x, a, b, why) : 0.0;
        if (!trade) why = Exit::Open;  // not traded
        r.lastDay.push_back({x.i, std::isfinite(a) ? std::exp(-std::max(a * x.sig, -kcRound)) - 1 : kNaN, std::isfinite(b) ? std::exp(std::max(b * x.sig, -kcRound)) - 1 : kNaN,
                             std::isfinite(x.e) ? std::expm1(x.e) : kNaN, pS, pT, x.h / x.o - 1, x.l / x.o - 1, x.c / x.o - 1, x.v, ret, why});
      }
    }
    // Then each way's day joins its record, and the day's bars and trades join what is learnt from.
    if (!days[j].empty())
      for (int w = 0; w < 3; ++w) {
        const double v = waySum[w] / static_cast<double>(days[j].size());
        wayS1[w] += v, wayS2[w] += v * v, wayN[w] += 1;
      }
    feedStocks(d);
    for (const auto& x : days[j]) {
      const double lo = -std::log(x.l / x.o) / x.sig, hi = std::log(x.h / x.o) / x.sig, cl = std::log(x.c / x.o);
      const int c = state[(d - 1) * N + x.i];
      if (c >= 0) perState[static_cast<std::size_t>(c)].add(lo, hi, cl, x.sig);
      pooled[0].add(lo, hi, cl, x.sig);
    }
  }
  if (ev.trades) {
    const double n = static_cast<double>(ev.trades);
    ev.lowAtOpen /= n, ev.predStop /= n, ev.predTake /= n, ev.realStop /= n, ev.realTake /= n, ev.expected /= n, ev.realised /= n, ev.signHit /= n;
    const double cv = sxy - sx * sy / n, vx = sxx - sx * sx / n, vy = syy - sy * sy / n;
    ev.ic = vx > 0 && vy > 0 ? cv / std::sqrt(vx * vy) : 0.0;
  }
  if (ev.days) ev.grossTop /= static_cast<double>(ev.days), ev.grossAll /= static_cast<double>(ev.days);
  // The plan for the next day: the 10 best at the last close, each bought at the open and sold by the close.
  const std::size_t L = T - 1;
  r.tradeNext = false, r.learntRatio = kNaN;
  r.wayNow = wayName[bestWay()];
  for (int w = 0; w < 3; ++w) r.wayRatios.push_back({wayName[w], Learner::ratio(wayS1[w], wayS2[w], wayN[w]) * std::sqrt(252.0)});
  r.kStop.push_back(kNaN), r.kTake.push_back(kNaN);
  r.life = in.life[L];
  r.orders.assign(N, 0);
  for (std::size_t q = 0; q < in.order[L].size() && r.plan.size() < K; ++q) {
    const std::size_t i = in.order[L][q];
    if (!std::isfinite(iv(L, i))) continue;
    TopKOrder o;
    double a, b, score;
    const Learner& ln = learnerOf(bestWay(), L, i);
    const bool trade = ln.learn(a, b, score);
    const auto [pS, pT] = ln.probs(a, b);
    o.state = state[L * N + i];
    r.tradeNext = r.tradeNext || trade;
    o.asset = i, o.action = trade ? 1 : 0, o.rank = q + 1, o.open = 0, o.sigma = iv(L, i), o.ratio = score;
    const double e = std::isfinite(f.E(L, i)) ? f.E(L, i) : 0.0;
    // Levels between the round trip and the furthest move the trades learnt from.
    const double sd = std::isfinite(a) ? std::min(std::max(a * o.sigma, ln.minMove), std::max(ln.maxDrop, ln.minMove)) : kNaN;
    const double td = std::isfinite(b) ? std::min(std::max(b * o.sigma, ln.minMove), std::max(ln.maxRise, ln.minMove)) : kNaN;
    o.stop = std::isfinite(sd) ? std::exp(-sd) - 1 : kNaN, o.take = std::isfinite(td) ? std::exp(td) - 1 : kNaN, o.close = std::expm1(e);
    o.pStopDay = pS, o.pTakeDay = pT, o.pStopLife = o.pTakeLife = kNaN;
    r.orders[i] = trade ? 1 : 0;
    r.plan.push_back(o);
  }
  return r;
}

}  // namespace

std::vector<TopKResult> topKBacktests(const Market& m, const Forecast& f, const Costs& costs, const Backtest& bt, std::size_t K) {
  std::vector<TopKResult> results;
  const std::size_t T = m.T(), N = m.N();
  if (bt.start == 0 || bt.start + 1 >= T || N <= K) return results;
  Inputs base;
  base.m = &m, base.s = bt.start - 1, base.K = K, base.buy = costs.buyBps * 1e-4, base.sell = costs.sellBps * 1e-4;
  base.close = m.close;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t t = 1; t < T; ++t)
      if (!std::isfinite(base.close(t, i))) base.close(t, i) = base.close(t - 1, i);
  // The forecast's life: 1 / (1 - rho), rho the mean day-to-day rank correlation of E so far.
  base.life.assign(T, 1.0);
  double rhoSum = 0, rhoN = 0;
  for (std::size_t t = 1; t < T; ++t) {
    std::vector<double> a, b;
    for (std::size_t i = 0; i < N; ++i)
      if (std::isfinite(f.E(t, i)) && std::isfinite(f.E(t - 1, i))) a.push_back(f.E(t, i)), b.push_back(f.E(t - 1, i));
    const double r = rankCorrelation(a, b);
    if (std::isfinite(r)) rhoSum += r, rhoN += 1;
    if (rhoN >= 1) base.life[t] = 1.0 / (1.0 - std::max(0.0, std::min(rhoSum / rhoN, 1.0 - 1.0 / (rhoN + 1.0))));
  }
  // Volatility over the life: EWMA of squared log returns, half-life the life.
  base.width = Panel(T, N);
  std::vector<double> var(N, kNaN);
  for (std::size_t t = 1; t < T; ++t) {
    const double lam = std::pow(2.0, -1.0 / std::max(1.0, base.life[t]));
    for (std::size_t i = 0; i < N; ++i) {
      const double c0 = m.close(t - 1, i), c1 = m.close(t, i);
      if (c0 > 0 && c1 > 0) {
        const double r = std::log(c1 / c0);
        var[i] = std::isfinite(var[i]) ? lam * var[i] + (1 - lam) * r * r : r * r;
      }
      if (std::isfinite(var[i])) base.width(t, i) = std::sqrt(var[i] * base.life[t]);
    }
  }
  // Traded value over the model's longest horizon.
  const std::size_t span = f.horizons.empty() ? 1 : *std::max_element(f.horizons.begin(), f.horizons.end());
  Panel tv(T, N);
  for (std::size_t i = 0; i < N; ++i) {
    double sum = 0, n = 0;
    for (std::size_t t = 0; t < T; ++t) {
      const double x = m.close(t, i) * m.volume(t, i);
      if (std::isfinite(x)) sum += x, n += 1;
      if (t >= span) {
        const double y = m.close(t - span, i) * m.volume(t - span, i);
        if (std::isfinite(y)) sum -= y, n -= 1;
      }
      if (n > 0) tv(t, i) = sum / n;
    }
  }

  std::vector<double> gridK = {0.25, 0.5, 1, 2, 4, kInf};
  const std::size_t G = gridK.size();
  // Today's 10 best by expected return as same-day trades (dayTrades). The multi-day portfolios
  // below (1: swapping only when the gain over the forecast's life beats a round trip; 2: the 10
  // largest) choose their levels from a fixed menu of multiples and are not run.
  for (int which = 0; which < 1; ++which) {
    Inputs in = base;
    in.returnUnits = which == 1;
    in.oneDay = which == 0;
    in.score = Panel(T, N);
    in.order.assign(T, {});
    for (std::size_t t = 0; t < T; ++t) {
      for (std::size_t i = 0; i < N; ++i)
        if (std::isfinite(f.E(t, i)) && m.close(t, i) > 0) in.score(t, i) = which < 2 ? f.E(t, i) : tv(t, i);
      auto& o = in.order[t];
      for (std::size_t i = 0; i < N; ++i)
        if (std::isfinite(in.score(t, i))) o.push_back(i);
      std::stable_sort(o.begin(), o.end(), [&](std::size_t p, std::size_t q) { return in.score(t, p) > in.score(t, q); });
    }
    if (which == 0) {
      results.push_back(dayTrades(m, f, in, costs, K, bt.start));
      continue;
    }
    TopKResult r;
    r.name = which == 0 ? "top 10 by expected return" : which == 1 ? "top 10 by expected return, cost-aware swaps" : "top 10 largest (traded value)";
    r.K = K, r.start = bt.start, r.gridK = gridK;
    // Shadow portfolios for every fixed pair; the growth of each so far picks the live pair.
    std::vector<std::vector<double>> shadow(G * G);
    for (std::size_t p = 0; p < G; ++p)
      for (std::size_t q = 0; q < G; ++q) shadow[p * G + q] = simulate(in, {gridK[p]}, {gridK[q]}, false).daily;
    const std::size_t D = shadow[0].size();
    std::vector<double> kS(D + 1, kInf), kT(D + 1, kInf), growth(G * G, 0.0);
    for (std::size_t j = 0; j <= D; ++j) {
      // Decision j uses the shadows' returns before it (days 0 .. j-1).
      std::size_t best = G * G - 1;  // no stops, no take-profits
      for (std::size_t c = 0; c < G * G; ++c)
        if (growth[c] > growth[best] + 1e-12) best = c;
      kS[j] = gridK[best / G], kT[j] = gridK[best % G];
      if (j < D)
        for (std::size_t c = 0; c < G * G; ++c) growth[c] += std::log1p(shadow[c][j]);
    }
    Sim live = simulate(in, kS, kT, true);
    r.daily = std::move(live.daily), r.turnover = std::move(live.turnover), r.kStop = std::move(live.kS), r.kTake = std::move(live.kT);
    r.kStop.push_back(kS[D]), r.kTake.push_back(kT[D]);
    r.trades = std::move(live.trades), r.positions = std::move(live.positions), r.orders = std::move(live.orders);
    r.noStops = shadow[G * G - 1];
    r.oneDay = in.oneDay;
    // Before costs: each day's 10 best, bought at the next open and sold at the open after,
    // against all the stocks ranked that day (equal weights).
    for (std::size_t t = in.s; t + 2 < T; ++t) {
      double st = 0, nt = 0, sm = 0, nm = 0;
      for (std::size_t q = 0; q < in.order[t].size(); ++q) {
        const std::size_t i = in.order[t][q];
        const double o1 = m.open(t + 1, i), o2 = m.open(t + 2, i);
        if (!(o1 > 0 && o2 > 0)) continue;
        sm += o2 / o1 - 1, nm += 1;
        if (nt < static_cast<double>(K)) st += o2 / o1 - 1, nt += 1;
      }
      if (nt > 0 && nm > 0) r.grossTop.push_back(st / nt), r.grossAll.push_back(sm / nm);
    }
    // The plan for the next day.
    const std::size_t L = T - 1;
    r.life = in.life[L];
    // Where the rule has no stop (or take-profit), the best finite level so far, as advice.
    double a = kS[D], b = kT[D];
    const bool advA = !std::isfinite(a), advB = !std::isfinite(b);
    {
      std::size_t bs = G * G, bt2 = G * G;
      for (std::size_t c = 0; c < G * G; ++c) {
        if (std::isfinite(gridK[c / G]) && gridK[c % G] == b && (bs == G * G || growth[c] > growth[bs])) bs = c;
        if (std::isfinite(gridK[c % G]) && gridK[c / G] == a && (bt2 == G * G || growth[c] > growth[bt2])) bt2 = c;
      }
      if (advA && bs < G * G) a = gridK[bs / G];
      if (advB && bt2 < G * G) b = gridK[bt2 % G];
    }
    auto add = [&](std::size_t i, int action, double stop, double take) {
      TopKOrder o;
      o.asset = i, o.action = action, o.open = 0, o.advisoryStop = advA, o.advisoryTake = advB;
      const auto& ord = in.order[L];
      o.rank = static_cast<std::size_t>(std::find(ord.begin(), ord.end(), i) - ord.begin()) + 1;
      const double e = std::isfinite(f.E(L, i)) ? f.E(L, i) : 0.0;
      o.sigma = std::isfinite(in.width(L, i)) ? in.width(L, i) / std::sqrt(r.life) : kNaN;
      if (action < 0) {
        o.stop = o.take = o.close = o.pStopDay = o.pTakeDay = o.pStopLife = o.pTakeLife = kNaN;
      } else {
        o.stop = stop, o.take = take, o.close = std::expm1(e);
        const double lo = std::isfinite(stop) ? std::log1p(stop) : -kInf, hi = std::isfinite(take) ? std::log1p(take) : kInf;
        std::tie(o.pStopDay, o.pTakeDay) = firstTouch(e, o.sigma, lo, hi, 1.0, i + 1);
        std::tie(o.pStopLife, o.pTakeLife) = firstTouch(e, o.sigma, lo, hi, r.life, i + 1);
      }
      r.plan.push_back(o);
    };
    for (const auto& p : r.positions)
      if (r.orders[p.asset] >= 0) {
        // The rule's levels stay where they were set at entry; advisory ones are set from the last close.
        const double wNow = in.width(L, p.asset) / (in.oneDay ? std::sqrt(r.life) : 1.0);
        add(p.asset, 0, advA || in.oneDay ? std::exp(-a * wNow) - 1 : p.entryRel * std::exp(-a * p.width) - 1,
            advB || in.oneDay ? std::exp(b * wNow) - 1 : p.entryRel * std::exp(b * p.width) - 1);
      }
    for (std::size_t i = 0; i < N; ++i)
      if (r.orders[i] > 0) {
        const double w = in.width(L, i) / (in.oneDay ? std::sqrt(r.life) : 1.0);
        add(i, 1, std::isfinite(a) ? std::exp(-a * w) - 1 : kNaN, std::isfinite(b) ? std::exp(b * w) - 1 : kNaN);
      }
    for (const auto& p : r.positions)
      if (r.orders[p.asset] < 0) add(p.asset, -1, kNaN, kNaN);
    for (const auto& sh : shadow) r.grid.push_back(metrics(sh).sharpe);
    results.push_back(std::move(r));
  }
  return results;
}

namespace {

std::string num(double x, int digits = 6) {
  if (!std::isfinite(x)) return "null";
  char b[40];
  std::snprintf(b, sizeof b, "%.*g", digits, x);
  return b;
}

std::string str(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

std::vector<double> slice(const std::vector<double>& v, std::size_t lo, std::size_t hi) {
  return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(lo), v.begin() + static_cast<std::ptrdiff_t>(hi));
}

struct ReasonStats {
  std::size_t n = 0, wins = 0;
  double mean = 0, days = 0;
};

ReasonStats reasonStats(const TopKResult& r, Exit e) {
  ReasonStats s;
  for (const auto& t : r.trades)
    if (t.reason == e) ++s.n, s.wins += t.ret > 0, s.mean += t.ret, s.days += static_cast<double>(t.exitDay - t.entryDay);
  if (s.n) s.mean /= static_cast<double>(s.n), s.days /= static_cast<double>(s.n);
  return s;
}

const char* reasonName(Exit e) {
  switch (e) {
    case Exit::Stop: return "stop-loss";
    case Exit::Take: return "take-profit";
    case Exit::Close: return "sold at the close";
    case Exit::Rotate: return "left the top 10";
    default: return "still open";
  }
}

std::string kName(double k) { return std::isfinite(k) ? num(k, 3) + " sd" : "none"; }

}  // namespace

std::string topKJson(const Market& m, const std::vector<TopKResult>& rs, const Backtest& bt) {
  std::ostringstream o;
  o << "{\"strategies\":[";
  for (std::size_t q = 0; q < rs.size(); ++q) {
    const auto& r = rs[q];
    const std::size_t n = std::min(r.daily.size(), bt.market.size());
    const std::size_t step = std::max<std::size_t>(1, n / 600);
    const std::vector<std::pair<std::string, const std::vector<double>*>> series = {
        {r.dayTrades ? "learnt policy" : "with adaptive stops", &r.daily}, {r.dayTrades ? "every day, no stops" : "without stops", &r.noStops}, {"market (bought and held)", &bt.market}};
    o << (q ? "," : "") << "{\"name\":" << str(r.name) << ",\"K\":" << r.K << ",\"dates\":[";
    for (std::size_t k = 0; k < n; k += step) o << (k ? "," : "") << str(m.dates[r.start + k]);
    o << "],\"curves\":{";
    for (std::size_t s = 0; s < series.size(); ++s) {
      o << (s ? "," : "") << str(series[s].first) << ":[";
      double w = 1;
      for (std::size_t k = 0; k < n; ++k) {
        w *= 1 + (*series[s].second)[k];
        if (k % step == 0) o << (k ? "," : "") << num(w, 5);
      }
      o << "]";
    }
    o << "},\"parts\":[";
    const std::vector<std::pair<std::string, std::pair<std::size_t, std::size_t>>> parts = {
        {"whole period", {0, n}}, {"first half", {0, n / 2}}, {"second half", {n / 2, n}}};
    for (std::size_t p = 0; p < parts.size(); ++p) {
      const auto [lo, hi] = parts[p].second;
      o << (p ? "," : "") << "{\"name\":" << str(parts[p].first) << ",\"from\":" << str(m.dates[r.start + lo]) << ",\"to\":"
        << str(m.dates[r.start + hi - 1]) << ",\"rows\":[";
      for (std::size_t s = 0; s < series.size(); ++s) {
        const Metrics mm = metrics(slice(*series[s].second, lo, hi));
        o << (s ? "," : "") << "{\"name\":" << str(series[s].first) << ",\"annualReturn\":" << num(mm.annualReturn) << ",\"volatility\":"
          << num(mm.volatility) << ",\"sharpe\":" << num(mm.sharpe) << ",\"maxDrawdown\":" << num(mm.maxDrawdown) << "}";
      }
      o << "]}";
    }
    o << "],\"years\":[";
    std::string year;
    std::vector<double> acc(series.size(), 1.0);
    bool firstYear = true;
    for (std::size_t k = 0; k <= n; ++k) {
      const std::string y = k < n ? m.dates[r.start + k].substr(0, 4) : "";
      if (k > 0 && y != year) {
        o << (firstYear ? "" : ",") << "{\"year\":" << str(year) << ",\"returns\":[";
        for (std::size_t s = 0; s < acc.size(); ++s) o << (s ? "," : "") << num(acc[s] - 1);
        o << "]}";
        firstYear = false;
        std::fill(acc.begin(), acc.end(), 1.0);
      }
      if (k == n) break;
      year = y;
      for (std::size_t s = 0; s < acc.size(); ++s) acc[s] *= 1 + (*series[s].second)[k];
    }
    o << "],\"exits\":[";
    bool first = true;
    for (Exit e : {Exit::Stop, Exit::Take, Exit::Close, Exit::Rotate, Exit::Open}) {
      const ReasonStats s = reasonStats(r, e);
      o << (first ? "" : ",") << "{\"reason\":" << str(reasonName(e)) << ",\"count\":" << s.n << ",\"wins\":" << s.wins << ",\"meanReturn\":"
        << num(s.mean) << ",\"meanDays\":" << num(s.days) << "}";
      first = false;
    }
    const double years = static_cast<double>(n) / 252.0;
    o << "],\"turnoverPerYear\":" << num(std::accumulate(r.turnover.begin(), r.turnover.end(), 0.0) / std::max(years, 1e-9));
    o << ",\"stopNow\":" << num(r.kStop.empty() ? kNaN : r.kStop.back()) << ",\"takeNow\":" << num(r.kTake.empty() ? kNaN : r.kTake.back());
    o << ",\"gridK\":[";
    for (std::size_t k = 0; k < r.gridK.size(); ++k) o << (k ? "," : "") << num(r.gridK[k]);
    o << "],\"grid\":[";
    for (std::size_t k = 0; k < r.grid.size(); ++k) o << (k ? "," : "") << num(r.grid[k], 4);
    o << "],\"positions\":[";
    for (std::size_t k = 0; k < r.positions.size(); ++k) {
      const auto& p = r.positions[k];
      o << (k ? "," : "") << "{\"ticker\":" << str(m.tickers[p.asset]) << ",\"since\":" << str(m.dates[p.entryDay]) << ",\"ret\":" << num(p.ret)
        << ",\"stop\":" << num(p.stopDist) << ",\"take\":" << num(p.takeDist) << ",\"sell\":" << (r.orders[p.asset] < 0 ? 1 : 0) << "}";
    }
    {
      const std::size_t g = r.grossTop.size();
      double a1 = 0, a2 = 0, e = 0, e2 = 0, hit = 0;
      for (std::size_t k = 0; k < g; ++k) {
        const double x = r.grossTop[k] - r.grossAll[k];
        a1 += r.grossTop[k], a2 += r.grossAll[k], e += x, e2 += x * x, hit += x > 0;
      }
      const double gn = std::max<double>(1, static_cast<double>(g)), me = e / gn, ve = std::max(0.0, (e2 - gn * me * me) / std::max(1.0, gn - 1));
      o << "],\"oneDay\":" << (r.oneDay ? "true" : "false") << ",\"gross\":{\"days\":" << g << ",\"top\":" << num(a1 / gn) << ",\"all\":" << num(a2 / gn)
        << ",\"excess\":" << num(me) << ",\"t\":" << num(ve > 0 ? me / std::sqrt(ve / gn) : 0.0) << ",\"hitRate\":" << num(hit / gn) << "}";
    }
    if (r.dayTrades) {
      const auto& e = r.eval;
      std::size_t tradedDays = 0;
      for (bool t : r.traded) tradedDays += t;
      o << ",\"wayNow\":" << str(r.wayNow) << ",\"wayRatios\":[";
      for (std::size_t k = 0; k < r.wayRatios.size(); ++k)
        o << (k ? "," : "") << "{\"name\":" << str(r.wayRatios[k].first) << ",\"sharpe\":" << num(r.wayRatios[k].second) << "}";
      o << "]";
      o << ",\"dayTrades\":true,\"tradeNext\":" << (r.tradeNext ? "true" : "false") << ",\"tradedDays\":" << tradedDays << ",\"eval\":{\"trades\":" << e.trades << ",\"days\":" << e.days << ",\"predStop\":" << num(e.predStop)
        << ",\"realStop\":" << num(e.realStop) << ",\"predTake\":" << num(e.predTake) << ",\"realTake\":" << num(e.realTake)
        << ",\"expected\":" << num(e.expected) << ",\"realised\":" << num(e.realised) << ",\"ic\":" << num(e.ic) << ",\"signHit\":"
        << num(e.signHit) << ",\"lowAtOpen\":" << num(e.lowAtOpen) << ",\"grossTop\":" << num(e.grossTop) << ",\"grossAll\":" << num(e.grossAll) << "},\"lastDate\":" << str(r.lastDate)
        << ",\"lastDay\":[";
      for (std::size_t k = 0; k < r.lastDay.size(); ++k) {
        const auto& c = r.lastDay[k];
        o << (k ? "," : "") << "{\"ticker\":" << str(m.tickers[c.asset]) << ",\"stop\":" << num(c.stop) << ",\"take\":" << num(c.take)
          << ",\"expected\":" << num(c.expected) << ",\"pStop\":" << num(c.pStop) << ",\"pTake\":" << num(c.pTake) << ",\"high\":"
          << num(c.high) << ",\"low\":" << num(c.low) << ",\"close\":" << num(c.close) << ",\"volume\":" << num(c.volume, 3)
          << ",\"ret\":" << num(c.ret) << ",\"exit\":" << str(reasonName(c.reason)) << "}";
      }
      o << "]";
    }
    o << ",\"life\":" << num(r.life) << ",\"plan\":[";
    for (std::size_t k = 0; k < r.plan.size(); ++k) {
      const auto& p = r.plan[k];
      o << (k ? "," : "") << "{\"ticker\":" << str(m.tickers[p.asset]) << ",\"action\":" << p.action << ",\"open\":" << num(p.open)
        << ",\"stop\":" << num(p.stop) << ",\"take\":" << num(p.take) << ",\"close\":" << num(p.close) << ",\"sigma\":" << num(p.sigma)
        << ",\"pStopDay\":" << num(p.pStopDay) << ",\"pTakeDay\":" << num(p.pTakeDay) << ",\"pStopLife\":" << num(p.pStopLife)
        << ",\"pTakeLife\":" << num(p.pTakeLife) << ",\"ratio\":" << num(p.ratio) << ",\"state\":" << p.state << ",\"rank\":" << p.rank << ",\"expectedDaily\":" << num(p.close) << ",\"advisoryStop\":" << (p.advisoryStop ? "true" : "false")
        << ",\"advisoryTake\":" << (p.advisoryTake ? "true" : "false") << "}";
    }
    o << "],\"buys\":[";
    first = true;
    for (std::size_t i = 0; i < r.orders.size(); ++i)
      if (r.orders[i] > 0) o << (first ? "" : ",") << str(m.tickers[i]), first = false;
    o << "]}";
  }
  o << "]}";
  return o.str();
}

std::string topKText(const Market& m, const std::vector<TopKResult>& rs, const Backtest& bt) {
  std::ostringstream o;
  char b[640];
  for (const auto& r : rs) {
    const std::size_t n = std::min(r.daily.size(), bt.market.size());
    if (n < 4) continue;
    o << "\n== Rolling portfolio: " << r.name << " ==\n";
    const std::vector<std::pair<std::string, const std::vector<double>*>> series = {
        {r.dayTrades ? "learnt policy" : "with adaptive stops", &r.daily}, {r.dayTrades ? "every day, no stops" : "without stops", &r.noStops}, {"market (bought and held)", &bt.market}};
    for (const auto& [name, lo, hi] : std::vector<std::tuple<std::string, std::size_t, std::size_t>>{
             {"whole period", 0, n}, {"first half", 0, n / 2}, {"second half", n / 2, n}}) {
      std::snprintf(b, sizeof b, "%s: %s .. %s\n", name.c_str(), m.dates[r.start + lo].c_str(), m.dates[r.start + hi - 1].c_str());
      o << b;
      for (const auto& s : series) {
        const Metrics mm = metrics(slice(*s.second, lo, hi));
        std::snprintf(b, sizeof b, "  %-26s %7.1f%% %6.1f%% Sharpe %5.2f  max DD %5.1f%%\n", s.first.c_str(), 100 * mm.annualReturn, 100 * mm.volatility,
                      mm.sharpe, 100 * mm.maxDrawdown);
        o << b;
      }
    }
    if (!r.grossTop.empty()) {
      double a1 = 0, a2 = 0, hit = 0;
      for (std::size_t k = 0; k < r.grossTop.size(); ++k) a1 += r.grossTop[k], a2 += r.grossAll[k], hit += r.grossTop[k] > r.grossAll[k];
      const double gn = static_cast<double>(r.grossTop.size());
      std::snprintf(b, sizeof b, "before costs, open to open, %.0f days: the day's 10 best %+.2f bp a day, all ranked %+.2f bp, excess %+.2f bp, beat them on %.1f%% of days\n",
                    gn, 1e4 * a1 / gn, 1e4 * a2 / gn, 1e4 * (a1 - a2) / gn, 100 * hit / gn);
      o << b;
    }
    if (r.dayTrades && r.eval.trades) {
      const auto& e = r.eval;
      std::snprintf(b, sizeof b,
                    "daily trades vs the bars, %zu trades on %zu days: stop touched %.1f%% (expected %.1f%%), take-profit %.1f%% (expected %.1f%%)\n"
                    "  expected excess %+.2f bp, realised %+.2f bp (open to close); correlation %.3f, same sign %.1f%%; before costs the 10 %+.2f bp a day, all ranked %+.2f bp\n",
                    e.trades, e.days, 100 * e.realStop, 100 * e.predStop, 100 * e.realTake, 100 * e.predTake, 1e4 * e.expected, 1e4 * e.realised, e.ic,
                    100 * e.signHit, 1e4 * e.grossTop, 1e4 * e.grossAll);
      o << b;
      std::snprintf(b, sizeof b, "  days the low was the open (the stock never traded below its opening price): %.1f%% of trades\n", 100 * e.lowAtOpen);
      o << b;
    }
    o << "exits:\n";
    for (Exit e : {Exit::Stop, Exit::Take, Exit::Close, Exit::Rotate, Exit::Open}) {
      const ReasonStats s = reasonStats(r, e);
      std::snprintf(b, sizeof b, "  %-16s %6zu  won %5.1f%%  mean %+6.2f%%  held %5.1f days\n", reasonName(e), s.n, s.n ? 100.0 * s.wins / s.n : 0.0,
                    100 * s.mean, s.days);
      o << b;
    }
    std::snprintf(b, sizeof b, "stop now %s, take-profit now %s (of the volatility over %s%s)\n",
                  kName(r.kStop.empty() ? kInf : r.kStop.back()).c_str(), kName(r.kTake.empty() ? kInf : r.kTake.back()).c_str(),
                  r.oneDay ? "one day" : "the forecast's life", r.dayTrades ? "; learnt from the bars of the trades before" : "; the shadow that has grown most");
    if (r.dayTrades) {
      std::size_t nTrade = 0;
      for (const auto& p : r.plan) nTrade += p.action > 0;
      std::snprintf(b, sizeof b, "next open: %zu of the %zu traded (their own learnt levels beat cash after costs); the rest in cash\n", nTrade, r.plan.size());
      o << b << "levels learnt from: " << r.wayNow << " (each way's record, annualised return per unit of risk:";
      for (const auto& [name, sr] : r.wayRatios) std::snprintf(b, sizeof b, " %s %.2f;", name.c_str(), sr), o << b;
      o << ")\n";
    }
    o << b;
    if (r.gridK.empty()) continue;
    o << "hindsight Sharpe of each fixed pair (rows stop, columns take-profit):\n        ";
    const std::size_t G = r.gridK.size();
    for (double k : r.gridK) std::snprintf(b, sizeof b, "%7s", kName(k).c_str()), o << b;
    o << "\n";
    for (std::size_t p = 0; p < G; ++p) {
      std::snprintf(b, sizeof b, "  %6s", kName(r.gridK[p]).c_str());
      o << b;
      for (std::size_t q = 0; q < G; ++q) std::snprintf(b, sizeof b, "%7.2f", r.grid[p * G + q]), o << b;
      o << "\n";
    }
  }
  return o.str();
}

}  // namespace ofm
