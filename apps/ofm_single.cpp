// The single-share model (research; results only): every share learns and is backtested on its own
// history alone. Nothing is pooled across shares and nothing is learnt from a portfolio's results.
//
// For each share, each close t (the past only):
//   inputs   the signals of apps/swing.hpp known at the close (losers, momentum, the close near the
//            day's low, the tug of war, ...) and the share's own volatility, each standardised by the
//            share's own past values (expanding mean and deviation)
//   trades   a limit buy for the next session at depth d of the share's own daily sigma below the
//            close (d on a grid; "open": bought in the opening auction), sold in the closing auction.
//            Filled at the open when the share opens at or below the limit, else at the limit when the
//            day's low trades half a spread through it (the share's own spread, Abdi and Ranaldo 2017),
//            else not traded (result 0). Net of the broker's costs (apps/swing.hpp).
//   model    a ridge regression per depth, of that trade's net result on the inputs, fitted on the
//            share's own settled days (refitted every --refit days). The ridge is learnt per share:
//            each of a grid of strengths trades the share on paper, and the one whose record so far is
//            best (mean shrunk by its t statistic) decides the next session.
//   decision the depth with the highest forecast, ordered when that forecast is positive.
//   gate     the share's own walk-forward record (every decision so far, on paper): it is traded only
//            while that record's shrunk mean is positive.
// Reported: every share's own result, the spread of them across shares, whether a share's first half
// tells its second (persistence), and the sum of all shares' own trades (for scale, not learnt from).
//
//   ofm_single --csv stocks.csv --costs UK|EU|US|APAC [--capital 5000] [--top 300] [--venues tools/venues.txt]
//              [--stamp-free aim.txt] [--refit 5] [--warm 250]
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "swing.hpp"

namespace {
using swing::kNaN;

constexpr std::size_t nDepth = 8;  // "open" and limits 0 .. 2 sigmas below the close
const double kDepth[nDepth] = {-1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0};  // -1: the opening auction
constexpr std::size_t nRidge = 4;
const double kRidge[nRidge] = {0.01, 0.1, 1.0, 10.0};  // times the mean diagonal of the inputs' sums

// Cholesky solve of (G + add I) X = B for several right-hand sides (the intercept, 0, not shrunk).
bool solve(std::vector<double> G, std::size_t n, double add, const std::vector<double>& B, std::size_t nb, std::vector<double>& X) {
  for (std::size_t k = 1; k < n; ++k) G[k * n + k] += add;
  for (std::size_t j = 0; j < n; ++j) {
    double s = G[j * n + j];
    for (std::size_t k = 0; k < j; ++k) s -= G[j * n + k] * G[j * n + k];
    if (!(s > 1e-12)) return false;
    G[j * n + j] = std::sqrt(s);
    for (std::size_t r = j + 1; r < n; ++r) {
      double v = G[r * n + j];
      for (std::size_t k = 0; k < j; ++k) v -= G[r * n + k] * G[j * n + k];
      G[r * n + j] = v / G[j * n + j];
    }
  }
  X.assign(n * nb, 0.0);
  std::vector<double> y(n);
  for (std::size_t c = 0; c < nb; ++c) {
    for (std::size_t r = 0; r < n; ++r) {
      double v = B[r * nb + c];
      for (std::size_t k = 0; k < r; ++k) v -= G[r * n + k] * y[k];
      y[r] = v / G[r * n + r];
    }
    for (std::size_t r = n; r-- > 0;) {
      double v = y[r];
      for (std::size_t k = r + 1; k < n; ++k) v -= G[k * n + r] * X[k * nb + c];
      X[r * nb + c] = v / G[r * n + r];
    }
  }
  return true;
}

struct Rec {  // a record of results
  double n = 0, s = 0, s2 = 0;
  void add(double x) { n += 1, s += x, s2 += x * x; }
  double mean() const { return n > 0 ? s / n : 0; }
  double sd() const { return n > 1 ? std::sqrt(std::max(0.0, s2 / n - mean() * mean())) : 0; }
  double t() const { const double d = sd(); return d > 0 ? mean() / d * std::sqrt(n) : 0; }
  double shrunk() const { const double tt = t(); return n > 1 && tt != 0 ? mean() * std::max(0.0, 1.0 - 1.0 / (tt * tt)) : 0; }
};

struct ShareResult {
  Rec all, h1, h2;        // its traded orders' results (gated), whole and by half of the days
  Rec ungated;            // every decision, ungated (the share's whole walk-forward paper record)
  Rec open;               // bought in every opening auction it could, sold at the close (no model)
  std::vector<double> day;  // its result each day it traded (gated), 0 otherwise (NaN: not tradable)
  std::size_t depthUsed[nDepth] = {};
  double fills = 0;
};
}  // namespace

int main(int argc, char** argv) {
  std::string csv, market, venues, stampFree;
  double capital = 5000;
  std::size_t top = 300, refit = 5, warm = 250;
  for (int k = 1; k + 1 < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv") csv = argv[++k];
    else if (a == "--costs") market = argv[++k];
    else if (a == "--capital") capital = std::atof(argv[++k]);
    else if (a == "--top") top = std::size_t(std::atol(argv[++k]));
    else if (a == "--venues") venues = argv[++k];
    else if (a == "--stamp-free") stampFree = argv[++k];
    else if (a == "--refit") refit = std::max<std::size_t>(1, std::size_t(std::atol(argv[++k])));
    else if (a == "--warm") warm = std::size_t(std::atol(argv[++k]));
  }
  if (csv.empty() || market.empty()) {
    std::fprintf(stderr, "usage: ofm_single --csv stocks.csv --costs UK|EU|US|APAC [--capital 5000] [--top 300] [--venues f] [--stamp-free f]\n");
    return 2;
  }
  const swing::Panel p = swing::load(csv);
  const std::size_t T = p.T, N = p.N;
  const swing::Costs cost = swing::costs(p, market, capital, venues, stampFree);
  const auto uni = swing::universe(p, top, market == "US" ? 5.0 : 0.0);
  std::vector<char> tradable(T * N, 0);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t i : uni[t]) tradable[t * N + i] = 1;
  const auto sig = swing::signals(p);
  const std::size_t S = sig.size(), F = S + 1, D = F + 1;  // the signals, the share's volatility; the intercept
  std::vector<ShareResult> res(N);
  std::atomic<std::size_t> next{0};
  const auto work = [&] {
    for (std::size_t i; (i = next++) < N;) {
      ShareResult& R = res[i];
      R.day.assign(T, kNaN);
      // The share's own inputs at each close, standardised by its own past (expanding).
      std::vector<double> x(T * D, 0.0), mu(F, 0.0), m2(F, 0.0), cnt(F, 0.0), sigma(T, kNaN), spread(T, 0.0);
      std::vector<char> has(T, 0);
      double e = 0, w = 0, lastSpread = 0;
      std::vector<double> rets;
      for (std::size_t t = 0; t < T; ++t) {
        if (t >= 1) {  // the spread (Abdi and Ranaldo 2017), from the share's own bars
          const double c0 = p.C[(t - 1) * N + i], h0 = p.H[(t - 1) * N + i], l0 = p.L[(t - 1) * N + i], h1 = p.H[t * N + i], l1 = p.L[t * N + i];
          if (c0 > 0 && h0 > 0 && l0 > 0 && h1 > 0 && l1 > 0) {
            const double c = std::log(c0), a = c - 0.5 * std::log(h0 * l0), b = c - 0.5 * std::log(h1 * l1);
            e += (4 * a * b - e) / 64, w += (1 - w) / 64;
          }
          const double s = w > 0 ? std::sqrt(std::max(0.0, e / w)) : 0.0;
          if (s > 0) lastSpread = s;
          spread[t] = lastSpread;
        }
        if (!p.ok(t, i)) continue;
        const double r = t >= 1 && p.ok(t - 1, i) ? std::log(p.C[t * N + i] / p.C[(t - 1) * N + i]) : kNaN;
        if (std::isfinite(r)) rets.push_back(r);
        if (rets.size() > 60) rets.erase(rets.begin());
        if (rets.size() < 60) continue;
        double s1 = 0, s2 = 0;
        for (double v : rets) s1 += v, s2 += v * v;
        sigma[t] = std::sqrt(std::max(0.0, s2 / 60 - (s1 / 60) * (s1 / 60)));
        if (!(sigma[t] > 0)) continue;
        has[t] = 1;
        x[t * D] = 1;
        for (std::size_t f = 0; f < F; ++f) {
          const double v = f < S ? sig[f].f(t, i) : std::log(sigma[t]);
          if (!std::isfinite(v)) continue;
          const double sd = cnt[f] > 20 ? std::sqrt(m2[f] / cnt[f]) : 0.0;
          x[t * D + 1 + f] = sd > 0 ? std::clamp((v - mu[f]) / sd, -4.0, 4.0) : 0.0;
          cnt[f] += 1;  // then the value joins the share's past (Welford)
          const double d = v - mu[f];
          mu[f] += d / cnt[f], m2[f] += d * (v - mu[f]);
        }
      }
      // The trade from the close t at each depth: its net result the next session (0: not filled).
      const auto result = [&](std::size_t t, std::size_t k, bool& filled) {
        filled = false;
        if (t + 1 >= T || !has[t] || !p.ok(t + 1, i) || !(p.O[(t + 1) * N + i] > 0)) return kNaN;
        const double c0 = p.C[t * N + i], o = p.O[(t + 1) * N + i], lo = p.L[(t + 1) * N + i], c = p.C[(t + 1) * N + i];
        double entry;
        if (kDepth[k] < 0) entry = o;
        else {
          const double lim = c0 * std::exp(-kDepth[k] * sigma[t]);
          if (o <= lim) entry = o;
          else if (lo <= lim * std::exp(-0.5 * spread[t])) entry = lim;
          else return 0.0;
        }
        filled = true;
        return std::log(c / entry) - cost.roundTrip(i, entry, c, false);
      };
      // The sums of the settled days: G (inputs) and B (inputs times each depth's result).
      std::vector<double> G(D * D, 0.0), B(D * nDepth, 0.0), beta[nRidge];
      Rec ridgeRec[nRidge], gate;
      double diag = 0;
      std::size_t nObs = 0, fitAt = 0;
      const std::size_t half = (T + std::min(T, 60 + warm)) / 2;  // the middle of the days a share can trade
      for (std::size_t t = 0; t + 1 < T; ++t) {
        // The day t - 1's trades are settled at the close t: they join the sums before deciding at t.
        if (t >= 1 && has[t - 1]) {
          double y[nDepth];
          bool ok = true, f;
          for (std::size_t k = 0; k < nDepth; ++k) ok = ok && std::isfinite(y[k] = result(t - 1, k, f));
          if (ok) {
            const double* a = &x[(t - 1) * D];
            for (std::size_t r = 0; r < D; ++r) {
              for (std::size_t c = 0; c < D; ++c) G[r * D + c] += a[r] * a[c];
              for (std::size_t k = 0; k < nDepth; ++k) B[r * nDepth + k] += a[r] * y[k];
            }
            ++nObs;
          }
        }
        if (nObs >= warm && (beta[0].empty() || nObs >= fitAt + refit)) {
          fitAt = nObs;
          diag = 0;
          for (std::size_t r = 1; r < D; ++r) diag += G[r * D + r] / double(D - 1);
          for (std::size_t q = 0; q < nRidge; ++q)
            if (!solve(G, D, kRidge[q] * diag, B, nDepth, beta[q])) beta[q].clear();
        }
        if (!has[t] || beta[0].empty()) continue;
        // Each ridge's decision at t: the depth of its highest forecast, if positive.
        const double* a = &x[t * D];
        int pick[nRidge];
        for (std::size_t q = 0; q < nRidge; ++q) {
          pick[q] = -1;
          if (beta[q].empty()) continue;
          double best = 0;
          for (std::size_t k = 0; k < nDepth; ++k) {
            double v = 0;
            for (std::size_t r = 0; r < D; ++r) v += a[r] * beta[q][r * nDepth + k];
            if (v > best) best = v, pick[q] = int(k);
          }
        }
        // The ridge the share's own record so far prefers decides.
        std::size_t use = 0;
        for (std::size_t q = 1; q < nRidge; ++q)
          if (ridgeRec[q].shrunk() > ridgeRec[use].shrunk()) use = q;
        bool f;
        double out[nRidge];
        for (std::size_t q = 0; q < nRidge; ++q) out[q] = pick[q] < 0 ? 0.0 : result(t, std::size_t(pick[q]), f);
        if (!std::isfinite(out[use])) continue;  // the next day has no bar: nothing settles
        const bool trades = tradable[t * N + i] != 0;
        const bool gateOpen = gate.shrunk() > 0;
        if (trades) {
          bool fo;
          const double ro = result(t, 0, fo);
          if (std::isfinite(ro)) R.open.add(ro);
          R.day[t + 1] = 0;
          if (pick[use] >= 0) {
            R.ungated.add(out[use]);
            if (gateOpen) {
              R.all.add(out[use]), (t < half ? R.h1 : R.h2).add(out[use]);
              R.day[t + 1] = out[use];
              R.depthUsed[pick[use]] += 1, R.fills += out[use] != 0;
            }
          }
        }
        // The records learn the outcome from the next close on (paper: tradable or not).
        for (std::size_t q = 0; q < nRidge; ++q)
          if (std::isfinite(out[q]) && pick[q] >= 0) ridgeRec[q].add(out[q]);
        if (pick[use] >= 0) gate.add(out[use]);
      }
    }
  };
  std::vector<std::thread> pool;
  const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  for (unsigned k = 0; k < nt; ++k) pool.emplace_back(work);
  for (auto& th : pool) th.join();

  // Every share's own result.
  std::size_t first = T, last = 0;
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t t = 0; t < T; ++t)
      if (std::isfinite(res[i].day[t])) first = std::min(first, t), last = std::max(last, t);
  if (first >= last) {
    std::printf("single: no share has %zu settled days of its own\n", warm);
    return 0;
  }
  Rec orders, ungated, open, h1, h2;
  double fills = 0;
  std::vector<double> means;
  std::size_t traded = 0, pos = 0, depth[nDepth] = {};
  double persistN = 0, persistSame = 0;
  Rec secondIfFirstUp, secondIfFirstDown;
  for (std::size_t i = 0; i < N; ++i) {
    const ShareResult& R = res[i];
    orders.s += R.all.s, orders.s2 += R.all.s2, orders.n += R.all.n;
    ungated.s += R.ungated.s, ungated.s2 += R.ungated.s2, ungated.n += R.ungated.n;
    open.s += R.open.s, open.s2 += R.open.s2, open.n += R.open.n;
    h1.s += R.h1.s, h1.s2 += R.h1.s2, h1.n += R.h1.n;
    h2.s += R.h2.s, h2.s2 += R.h2.s2, h2.n += R.h2.n;
    for (std::size_t k = 0; k < nDepth; ++k) depth[k] += R.depthUsed[k];
    fills += R.fills;
    if (R.all.n >= 20) means.push_back(R.all.mean()), ++traded, pos += R.all.mean() > 0;
    if (R.h1.n >= 10 && R.h2.n >= 10) {
      persistN += 1, persistSame += (R.h1.mean() > 0) == (R.h2.mean() > 0);
      (R.h1.mean() > 0 ? secondIfFirstUp : secondIfFirstDown).add(R.h2.mean());
    }
  }
  std::sort(means.begin(), means.end());
  const auto q = [&](double f) { return means.empty() ? 0.0 : means[std::min(means.size() - 1, std::size_t(f * double(means.size())))]; };
  std::printf("single: %s, %zu shares, %s to %s; each share learns and is backtested on its own bars only (top %zu traded each day, %.0f a trade)\n",
              market.c_str(), N, p.days[first].c_str(), p.days[last].c_str(), top, capital);
  const auto line = [](const char* name, const Rec& r) {
    std::printf("single: %-38s %8.0f orders, an order %+.4f%% (t %+5.1f)\n", name, r.n, 100 * r.mean(), r.t());
  };
  line("traded (own gate open)", orders);
  std::printf("single:   %.0f of them filled (%.0f%%), a fill %+.3f%%\n", fills, orders.n > 0 ? 100 * fills / orders.n : 0.0, fills > 0 ? 100 * orders.s / fills : 0.0);
  line("  first half of the days", h1);
  line("  second half of the days", h2);
  line("every decision (no gate)", ungated);
  line("opening auction every day (no model)", open);
  std::printf("single: depths used (open, 0, .25, .5, .75, 1, 1.5, 2 sigma):");
  for (std::size_t k = 0; k < nDepth; ++k) std::printf(" %zu", depth[k]);
  std::printf("\nsingle: shares with 20+ trades: %zu, positive %zu (%.0f%%); a share's mean a trade: 10%% %+.3f%%, 25%% %+.3f%%, median %+.3f%%, 75%% %+.3f%%, 90%% %+.3f%%\n",
              traded, pos, traded ? 100.0 * double(pos) / double(traded) : 0.0, 100 * q(0.1), 100 * q(0.25), 100 * q(0.5), 100 * q(0.75), 100 * q(0.9));
  std::printf("single: persistence (shares with 10+ trades in each half): %.0f shares, same sign %.0f%%; second half's mean a trade where the first was up %+.3f%% (%.0f shares), down %+.3f%% (%.0f)\n",
              persistN, persistN > 0 ? 100 * persistSame / persistN : 0.0, 100 * secondIfFirstUp.mean(), secondIfFirstUp.n, 100 * secondIfFirstDown.mean(), secondIfFirstDown.n);
  // For scale: every share's own trades together, each with the same capital (a day's mean over its trades).
  Rec daily, dh1, dh2;
  double perDay = 0, nd = 0;
  for (std::size_t t = first; t <= last; ++t) {
    double s = 0, n = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (std::isfinite(res[i].day[t]) && res[i].day[t] != 0) s += res[i].day[t], n += 1;
    const double r = n > 0 ? s / n : 0.0;
    daily.add(r), (t < (first + last) / 2 ? dh1 : dh2).add(r);
    perDay += n, nd += 1;
  }
  const auto yr = [](const Rec& r) { const double sd = r.sd(); return std::make_pair(25200 * r.mean(), sd > 0 ? r.mean() / sd * std::sqrt(252.0) : 0.0); };
  const auto [a0, s0] = yr(daily);
  const auto [a1, s1] = yr(dh1);
  const auto [a2, s2] = yr(dh2);
  std::printf("single: all shares' own trades together (%.1f a day, each the same capital): %+.1f%% a year, Sharpe %.2f | first half %+.1f%% %.2f | second half %+.1f%% %.2f\n",
              nd > 0 ? perDay / nd : 0.0, a0, s0, a1, s1, a2, s2);
  return 0;
}
