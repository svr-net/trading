// Multi-day holds on the single-share model (research; results only): every share learns, on its own
// bars alone, which limit to buy at and how long to hold. Short-term reversal is a return for supplying
// liquidity (Nagel 2012), strongest over about five days, and in Europe it pays after costs only where
// trading is cheap (large shares, few trades): holding longer spreads a round trip's costs (the UK's
// stamp duty is paid once, whatever the hold) over more days.
//
// For each share, each close t (the past only), the trades it could place for the next session:
//   entry  a limit buy for the next session at depth d of the share's own daily sigma below the close
//          (d on a grid; "open": the opening auction): filled at the open when the share opens at or
//          below the limit, else at the limit when the day's low trades --through spreads through it
//          (the share's own spread, Abdi and Ranaldo 2017), else not traded. An unfilled order is a new
//          order the next day (no fee: IBKR charges for fills only).
//   exit   sold in the closing auction after h closes (1: the same day; 2, 3, 5), or at the first close
//          above the entry price, at the latest the fifth ("up": Connors's exit on strength).
//   costs  the broker's on both sides, the buy tax once (apps/swing.hpp; a hold past the day pays the
//          overnight tax where there is one, France's financial transaction tax).
//   model  a ridge regression per (depth, exit) of the trade's net result on the share's own
//          standardised signals, fitted on its settled trades; the ridge learnt per share from its
//          own paper record; the trade with the highest forecast placed when positive; the share
//          traded only while its own record's shrunk mean is positive, and not while it holds.
// The account: --slots positions at most, the day's orders to the highest own forecasts while slots are
// free (a position keeps its slot until it is sold), marked to the close every day.
//
//   ofm_hold --csv stocks.csv --costs UK|EU|US|APAC [--capital 5000] [--top 300] [--slots 10]
//            [--venues tools/venues.txt] [--stamp-free aim.txt] [--override ibkr.csv | --ibkr-days ibkr.csv]
//            [--ibkr-only] [--holds up,1d,...] [--through 0.5] [--refit 5] [--warm 250] [--trace file]
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "ridge.hpp"
#include "swing.hpp"

namespace {
using ridge::Rec;
using ridge::solve;
using swing::kNaN;

constexpr std::size_t nDepth = 8;
const double kDepth[nDepth] = {-1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0};  // -1: the opening auction
constexpr std::size_t nHold = 5;
const int kHold[nHold] = {1, 2, 3, 5, 0};  // closes held; 0: the first close above the entry, at most 5
const char* kHoldName[nHold] = {"1d", "2d", "3d", "5d", "up"};
constexpr std::size_t nTarget = nDepth * nHold, maxHold = 5, settleLag = maxHold + 3;  // a missing bar moves an exit on, at most 3 days
constexpr std::size_t nRidge = 4;
const double kRidge[nRidge] = {0.01, 0.1, 1.0, 10.0};

struct Trade {
  std::size_t order = 0, entry = 0, exit = 0;  // the order's day, the fill's and the sale's
  int depth = -1, hold = -1;
  bool filled = false;
  double fc = 0, entryPx = 0, ret = kNaN;  // ret: net of costs (0: not filled)
};

struct ShareResult {
  std::vector<Trade> trades;  // the orders it placed (own gate open, not holding, tradable)
  std::vector<double> dec;    // each day's order forecast, settled or not (the look-ahead check)
  std::size_t depthUsed[nDepth] = {}, holdUsed[nHold] = {};
};
}  // namespace

int main(int argc, char** argv) {
  std::string csv, market, venues, stampFree, overrideCsv, ibkrDays, traceOut, holdsArg;
  double capital = 5000, through = 0.5;
  std::size_t top = 300, refit = 5, warm = 250, slots = 10;
  bool ibkrOnly = false;
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
    else if (a == "--slots") slots = std::max<std::size_t>(1, std::size_t(std::atol(argv[++k])));
    else if (a == "--override") overrideCsv = argv[++k];
    else if (a == "--ibkr-days") ibkrDays = argv[++k];
    else if (a == "--through") through = std::atof(argv[++k]);
    else if (a == "--trace") traceOut = argv[++k];
    else if (a == "--holds") holdsArg = argv[++k];  // the exits the shares may choose from (e.g. "up" or "1d,up"; default all)
  }
  for (int k = 1; k < argc; ++k)  // --ibkr-only: traded only on the share-days with IBKR's bars (learnt from every bar)
    if (std::string(argv[k]) == "--ibkr-only") ibkrOnly = true;
  if (csv.empty() || market.empty()) {
    std::fprintf(stderr, "usage: ofm_hold --csv stocks.csv --costs UK|EU|US|APAC [--capital 5000] [--top 300] [--slots 10] [--venues f] [--stamp-free f]\n");
    return 2;
  }
  swing::Panel panel = swing::load(csv);
  const std::size_t T = panel.T, N = panel.N;
  std::vector<char> ibDay(T * N, 0);
  if (!overrideCsv.empty()) std::printf("hold: IBKR's bars: %s\n", swing::override(panel, overrideCsv).c_str()), ibkrDays = overrideCsv;
  if (!ibkrDays.empty()) {
    std::map<std::string, std::size_t> di, ti;
    for (std::size_t t = 0; t < T; ++t) di[panel.days[t]] = t;
    for (std::size_t i = 0; i < N; ++i) ti[panel.tickers[i]] = i;
    std::ifstream in(ibkrDays);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto c1 = line.find(','), c2 = line.find(',', c1 + 1);
      if (c1 == std::string::npos || c2 == std::string::npos) continue;
      const auto d = di.find(line.substr(0, std::min<std::size_t>(c1, 10))), k = ti.find(line.substr(c1 + 1, c2 - c1 - 1));
      if (d != di.end() && k != ti.end() && panel.C[d->second * N + k->second] > 0) ibDay[d->second * N + k->second] = 1;
    }
  }
  bool holdOn[nHold];
  for (std::size_t h = 0; h < nHold; ++h) holdOn[h] = holdsArg.empty() || ("," + holdsArg + ",").find("," + std::string(kHoldName[h]) + ",") != std::string::npos;
  const swing::Panel& p = panel;
  const swing::Costs cost = swing::costs(p, market, capital, venues, stampFree);
  const auto uni = swing::universe(p, top, market == "US" ? 5.0 : 0.0);
  std::vector<char> tradable(T * N, 0);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t i : uni[t]) tradable[t * N + i] = !ibkrOnly || ibDay[t * N + i];
  const auto sig = swing::signals(p);
  const std::size_t S = sig.size(), F = S + 1, D = F + 1;
  std::vector<ShareResult> res(N);
  std::atomic<std::size_t> next{0};
  const auto work = [&] {
    for (std::size_t i; (i = next++) < N;) {
      ShareResult& R = res[i];
      R.dec.assign(T, kNaN);
      std::vector<double> x(T * D, 0.0), mu(F, 0.0), m2(F, 0.0), cnt(F, 0.0), sigma(T, kNaN), spread(T, 0.0);
      std::vector<char> has(T, 0);
      double e = 0, w = 0, lastSpread = 0;
      std::vector<double> rets;
      for (std::size_t t = 0; t < T; ++t) {
        if (t >= 1) {
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
          cnt[f] += 1;
          const double d = v - mu[f];
          mu[f] += d / cnt[f], m2[f] += d * (v - mu[f]);
        }
      }
      // The trade from the close t at depth k held by rule h: ret NaN while it has not settled by the
      // panel's last day (or the share has no bars to sell on).
      const auto trade = [&](std::size_t t, std::size_t k, std::size_t h) {
        Trade tr;
        tr.order = t + 1, tr.depth = int(k), tr.hold = int(h);
        if (t + 1 >= T || !has[t] || !p.ok(t + 1, i) || !(p.O[(t + 1) * N + i] > 0)) return tr;
        const double c0 = p.C[t * N + i], o = p.O[(t + 1) * N + i], lo = p.L[(t + 1) * N + i];
        double entry;
        if (kDepth[k] < 0) entry = o;
        else {
          const double lim = c0 * std::exp(-kDepth[k] * sigma[t]);
          if (o <= lim) entry = o;
          else if (lo <= lim * std::exp(-through * spread[t])) entry = lim;
          else return tr.ret = 0.0, tr.entry = tr.exit = t + 1, tr;
        }
        const std::size_t e0 = t + 1;
        tr.filled = true, tr.entry = e0, tr.entryPx = entry;  // known at the fill's close, whether or not it has been sold yet
        std::size_t xd = 0;
        if (kHold[h] > 0) {
          for (std::size_t u = t + std::size_t(kHold[h]); u < std::min(T, t + std::size_t(kHold[h]) + 4); ++u)
            if (p.ok(u, i)) { xd = u; break; }
        } else {
          std::size_t seen = 0;
          for (std::size_t u = e0; u < T && seen < maxHold; ++u) {
            if (!p.ok(u, i)) continue;
            ++seen, xd = u;
            if (p.C[u * N + i] > entry) break;
          }
          if (seen < maxHold && !(p.C[xd * N + i] > entry)) xd = 0;  // not settled yet
        }
        if (!xd) return tr;
        const double c = p.C[xd * N + i];
        tr.exit = xd;
        tr.ret = std::log(c / entry) - cost.roundTrip(i, entry, c, xd > e0);
        return tr;
      };
      std::vector<double> G(D * D, 0.0), B(D * nTarget, 0.0), beta[nRidge];
      Rec ridgeRec[nRidge], gate;
      struct Pending { std::size_t settle; int who; double v; };  // who: a ridge, or -1 for the gate
      std::vector<Pending> pending;
      std::size_t nObs = 0, fitAt = 0, nextSample = 0, holdUntil = 0;
      for (std::size_t t = 0; t < T; ++t) {
        // The samples whose every trade has settled by the close t join the sums.
        for (; nextSample + settleLag <= t; ++nextSample) {
          if (!has[nextSample]) continue;
          double y[nTarget];
          bool ok = true;
          for (std::size_t k = 0; k < nDepth && ok; ++k)
            for (std::size_t h = 0; h < nHold && ok; ++h) {
              const Trade tr = trade(nextSample, k, h);
              ok = std::isfinite(tr.ret) && tr.exit <= t, y[k * nHold + h] = tr.ret;
            }
          if (!ok) continue;
          const double* a = &x[nextSample * D];
          for (std::size_t r = 0; r < D; ++r) {
            for (std::size_t c = 0; c < D; ++c) G[r * D + c] += a[r] * a[c];
            for (std::size_t q = 0; q < nTarget; ++q) B[r * nTarget + q] += a[r] * y[q];
          }
          ++nObs;
        }
        // The records learn the outcomes settled by the close t.
        for (std::size_t q = 0; q < pending.size();)
          if (pending[q].settle <= t) {
            (pending[q].who < 0 ? gate : ridgeRec[pending[q].who]).add(pending[q].v);
            pending[q] = pending.back(), pending.pop_back();
          } else ++q;
        if (nObs >= warm && (beta[0].empty() || nObs >= fitAt + refit)) {
          fitAt = nObs;
          double diag = 0;
          for (std::size_t r = 1; r < D; ++r) diag += G[r * D + r] / double(D - 1);
          for (std::size_t q = 0; q < nRidge; ++q)
            if (!solve(G, D, kRidge[q] * diag, B, nTarget, beta[q])) beta[q].clear();
        }
        if (!has[t] || beta[0].empty() || t + 1 >= T) continue;
        const double* a = &x[t * D];
        int pick[nRidge];
        double fcast[nRidge];
        for (std::size_t q = 0; q < nRidge; ++q) {
          pick[q] = -1, fcast[q] = 0;
          if (beta[q].empty()) continue;
          for (std::size_t g = 0; g < nTarget; ++g) {
            double v = 0;
            for (std::size_t r = 0; r < D; ++r) v += a[r] * beta[q][r * nTarget + g];
            if (holdOn[g % nHold] && v > fcast[q]) fcast[q] = v, pick[q] = int(g);
          }
        }
        std::size_t use = 0;
        for (std::size_t q = 1; q < nRidge; ++q)
          if (ridgeRec[q].shrunk() > ridgeRec[use].shrunk()) use = q;
        // Paper: every ridge's decision is recorded once it settles (the records choose the ridge).
        Trade chosen;
        for (std::size_t q = 0; q < nRidge; ++q) {
          if (pick[q] < 0) continue;
          const Trade tr = trade(t, std::size_t(pick[q]) / nHold, std::size_t(pick[q]) % nHold);
          if (q == use) chosen = tr;
          if (std::isfinite(tr.ret)) pending.push_back({std::max(tr.exit, t + 1), int(q), tr.ret});
        }
        if (pick[use] < 0) continue;
        if (std::isfinite(chosen.ret)) pending.push_back({std::max(chosen.exit, t + 1), -1, chosen.ret});
        // Placed: the gate open, tradable, and not holding the share on the next session.
        if (!tradable[t * N + i] || gate.shrunk() <= 0 || holdUntil > t) continue;
        R.dec[t + 1] = fcast[use];
        if (chosen.filled) holdUntil = std::isfinite(chosen.ret) ? chosen.exit : T;  // held until sold (not yet sold at the panel's end)
        if (!std::isfinite(chosen.ret)) continue;  // not settled within the panel
        chosen.fc = fcast[use];
        R.trades.push_back(chosen);
        R.depthUsed[chosen.depth] += 1, R.holdUsed[chosen.hold] += 1;
      }
    }
  };
  std::vector<std::thread> pool;
  const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  for (unsigned k = 0; k < nt; ++k) pool.emplace_back(work);
  for (auto& th : pool) th.join();

  // Every share's own trades.
  Rec orders, fills, days, byHold[nHold], ibStore;
  std::size_t first = T, last = 0, depth[nDepth] = {}, hold[nHold] = {};
  for (std::size_t i = 0; i < N; ++i) {
    for (const Trade& tr : res[i].trades) {
      orders.add(tr.ret), first = std::min(first, tr.order), last = std::max(last, tr.exit);
      if (tr.filled) {
        fills.add(tr.ret), byHold[tr.hold].add(tr.ret);
        std::size_t held = 0;
        for (std::size_t u = tr.entry; u <= tr.exit; ++u) held += p.ok(u, i);
        days.add(double(held));
      }
      if (ibDay[tr.order * N + i]) ibStore.add(tr.ret);
    }
    for (std::size_t k = 0; k < nDepth; ++k) depth[k] += res[i].depthUsed[k];
    for (std::size_t h = 0; h < nHold; ++h) hold[h] += res[i].holdUsed[h];
  }
  if (first >= last) {
    std::printf("hold: no share has %zu settled days of its own\n", warm);
    return 0;
  }
  std::printf("hold: %s, %zu shares, %s to %s; each share learns and is backtested on its own bars only (top %zu traded each day%s, %.0f a trade; a limit fills when the low trades %.1f spreads through it)\n",
              market.c_str(), N, p.days[first].c_str(), p.days[last].c_str(), top, ibkrOnly ? ", only on IBKR's bars" : "", capital, through);
  std::printf("hold: every share's own orders: %.0f, an order %+.4f%% (t %+.1f); %.0f filled (%.0f%%), a fill %+.3f%% net, held %.1f days\n", orders.n,
              100 * orders.mean(), orders.t(), fills.n, orders.n > 0 ? 100 * fills.n / orders.n : 0.0, 100 * fills.mean(), days.mean());
  std::printf("hold: fills by exit:");
  for (std::size_t h = 0; h < nHold; ++h) std::printf(" %s %.0f (%+.3f%%)", kHoldName[h], byHold[h].n, 100 * byHold[h].mean());
  std::printf("\nhold: depths ordered (open, 0, .25, .5, .75, 1, 1.5, 2 sigma):");
  for (std::size_t k = 0; k < nDepth; ++k) std::printf(" %zu", depth[k]);
  std::printf("\n");
  // The account: slots held until sold; the day's orders to the highest forecasts while slots are free.
  std::vector<std::vector<std::pair<double, std::pair<std::size_t, std::size_t>>>> byDay(T);  // (fc, (share, trade))
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t q = 0; q < res[i].trades.size(); ++q) byDay[res[i].trades[q].order].push_back({res[i].trades[q].fc, {i, q}});
  struct Pos { std::size_t i, q; };
  std::vector<Pos> open;
  Rec acct, ah1, ah2, taken, hed, hh1, hh2;  // hed: less the market's day on the money held (equal-weighted, the day's tradable shares)
  double slotDays = 0, nd = 0;
  const std::size_t mid = (first + last) / 2;
  for (std::size_t u = first; u <= last; ++u) {
    open.erase(std::remove_if(open.begin(), open.end(), [&](const Pos& z) { return res[z.i].trades[z.q].exit < u; }), open.end());
    auto& c = byDay[u];
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (const auto& o : c) {
      if (open.size() >= slots) break;
      const Trade& tr = res[o.second.first].trades[o.second.second];
      taken.add(tr.ret);
      if (tr.filled) open.push_back({o.second.first, o.second.second});
    }
    double r = 0;
    for (const Pos& z : open) {
      const Trade& tr = res[z.i].trades[z.q];
      const std::size_t i = z.i;
      if (!p.ok(u, i)) continue;
      if (u == tr.entry) r += std::log(p.C[u * N + i] / tr.entryPx) - cost.side(i, tr.entryPx) - cost.buyTax(i, tr.exit > tr.entry);
      else {
        std::size_t pu = u - 1;
        while (pu > tr.entry && !p.ok(pu, i)) --pu;
        r += std::log(p.C[u * N + i] / p.C[pu * N + i]);
      }
      if (u == tr.exit) r -= cost.side(i, p.C[u * N + i]);
    }
    r /= double(slots);
    double ms = 0, mn = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (tradable[(u - 1) * N + i] && p.ok(u, i) && p.ok(u - 1, i)) ms += std::log(p.C[u * N + i] / p.C[(u - 1) * N + i]), mn += 1;
    const double h = r - (mn > 0 ? ms / mn : 0.0) * double(open.size()) / double(slots);
    slotDays += double(open.size()), nd += 1;
    acct.add(r), (u < mid ? ah1 : ah2).add(r);
    hed.add(h), (u < mid ? hh1 : hh2).add(h);
  }
  const auto yr = [](const Rec& r) { const double sd = r.sd(); return std::make_pair(25200 * r.mean(), sd > 0 ? r.mean() / sd * std::sqrt(252.0) : 0.0); };
  const auto [a0, s0] = yr(acct);
  const auto [a1, s1] = yr(ah1);
  const auto [a2, s2] = yr(ah2);
  std::printf("hold: the account (%zu slots, %.0f each): %.1f of them held a day on average, %.0f orders taken (an order %+.4f%%), %+.1f%% a year, Sharpe %.2f | first half %+.1f%% %.2f | second half %+.1f%% %.2f\n",
              slots, capital, nd > 0 ? slotDays / nd : 0.0, taken.n, 100 * taken.mean(), a0, s0, a1, s1, a2, s2);
  const auto [c0, d0] = yr(hed);
  const auto [c1, d1] = yr(hh1);
  const auto [c2, d2] = yr(hh2);
  std::printf("hold: the account less the market (the market's day on the money held): %+.1f%% a year, Sharpe %.2f | first half %+.1f%% %.2f | second half %+.1f%% %.2f\n", c0, d0, c1, d1, c2, d2);
  if (!ibkrDays.empty())
    std::printf("hold: orders on the share-days with IBKR's bars (%s's bars): %.0f, an order %+.4f%% (t %+.1f)\n", overrideCsv.empty() ? "the store" : "IBKR", ibStore.n,
                100 * ibStore.mean(), ibStore.t());
  if (!traceOut.empty()) {
    std::ofstream o(traceOut);
    char b[256];
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t i = 0; i < N; ++i)
        if (std::isfinite(res[i].dec[t])) std::snprintf(b, sizeof b, "%s %s %.12g\n", p.days[t].c_str(), p.tickers[i].c_str(), res[i].dec[t]), o << b;
  }
  return 0;
}
