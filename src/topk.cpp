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
      else if (o <= entryPx[i] * std::exp(-a * w[i])) traded += h[i], exitTrade(i, d, h[i], Exit::Stop);
      else if (o >= entryPx[i] * std::exp(b * w[i])) traded += h[i], exitTrade(i, d, h[i], Exit::Take);
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
  for (int which = 0; which < 2; ++which) {
    Inputs in = base;
    in.returnUnits = which == 0;
    in.score = Panel(T, N);
    in.order.assign(T, {});
    for (std::size_t t = 0; t < T; ++t) {
      for (std::size_t i = 0; i < N; ++i)
        if (std::isfinite(f.E(t, i)) && m.close(t, i) > 0) in.score(t, i) = which == 0 ? f.E(t, i) : tv(t, i);
      auto& o = in.order[t];
      for (std::size_t i = 0; i < N; ++i)
        if (std::isfinite(in.score(t, i))) o.push_back(i);
      std::stable_sort(o.begin(), o.end(), [&](std::size_t p, std::size_t q) { return in.score(t, p) > in.score(t, q); });
    }
    TopKResult r;
    r.name = which == 0 ? "top 10 by expected return" : "top 10 largest (traded value)";
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
        const double wNow = in.width(L, p.asset);
        add(p.asset, 0, advA ? std::exp(-a * wNow) - 1 : p.entryRel * std::exp(-a * p.width) - 1,
            advB ? std::exp(b * wNow) - 1 : p.entryRel * std::exp(b * p.width) - 1);
      }
    for (std::size_t i = 0; i < N; ++i)
      if (r.orders[i] > 0) {
        const double w = in.width(L, i);
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
        {"with adaptive stops", &r.daily}, {"without stops", &r.noStops}, {"market (bought and held)", &bt.market}};
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
    for (Exit e : {Exit::Stop, Exit::Take, Exit::Rotate, Exit::Open}) {
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
    o << "],\"life\":" << num(r.life) << ",\"plan\":[";
    for (std::size_t k = 0; k < r.plan.size(); ++k) {
      const auto& p = r.plan[k];
      o << (k ? "," : "") << "{\"ticker\":" << str(m.tickers[p.asset]) << ",\"action\":" << p.action << ",\"open\":" << num(p.open)
        << ",\"stop\":" << num(p.stop) << ",\"take\":" << num(p.take) << ",\"close\":" << num(p.close) << ",\"sigma\":" << num(p.sigma)
        << ",\"pStopDay\":" << num(p.pStopDay) << ",\"pTakeDay\":" << num(p.pTakeDay) << ",\"pStopLife\":" << num(p.pStopLife)
        << ",\"pTakeLife\":" << num(p.pTakeLife) << ",\"advisoryStop\":" << (p.advisoryStop ? "true" : "false")
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
  char b[256];
  for (const auto& r : rs) {
    const std::size_t n = std::min(r.daily.size(), bt.market.size());
    if (n < 4) continue;
    o << "\n== Rolling portfolio: " << r.name << " ==\n";
    const std::vector<std::pair<std::string, const std::vector<double>*>> series = {
        {"with adaptive stops", &r.daily}, {"without stops", &r.noStops}, {"market (bought and held)", &bt.market}};
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
    o << "exits:\n";
    for (Exit e : {Exit::Stop, Exit::Take, Exit::Rotate, Exit::Open}) {
      const ReasonStats s = reasonStats(r, e);
      std::snprintf(b, sizeof b, "  %-16s %6zu  won %5.1f%%  mean %+6.2f%%  held %5.1f days\n", reasonName(e), s.n, s.n ? 100.0 * s.wins / s.n : 0.0,
                    100 * s.mean, s.days);
      o << b;
    }
    std::snprintf(b, sizeof b, "stop now %s, take-profit now %s (of the volatility over the forecast's life)\n",
                  kName(r.kStop.empty() ? kInf : r.kStop.back()).c_str(), kName(r.kTake.empty() ? kInf : r.kTake.back()).c_str());
    o << b << "hindsight Sharpe of each fixed pair (rows stop, columns take-profit):\n        ";
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
