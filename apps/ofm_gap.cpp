// Buy on gap (research; results only): Ernest Chan's buy-on-gap model (Algorithmic Trading, 2013),
// a same-day trade whose buys fill: the shares that open far below their previous day's low, bought
// just after the open and sold in the closing auction.
//
// Each day t, for every tradable share (apps/swing.hpp: the most traded):
//   gap     log(open t / low t-1), against the share's own daily sigma (its last 90 days' close-to-close
//           log returns, to the close t-1)
//   signal  gap < -k sigma, and the open above the share's 20-day mean close (to t-1): a gap in a share
//           that is not in a fall
//   trade   the N most gapped bought just after the open, paying half the share's own spread over the
//           open (Abdi and Ranaldo 2017, its own bars; a market order after the open: it always buys),
//           sold in the closing auction (no spread); net of the broker's costs (apps/swing.hpp)
//   account N slots of --budget / N; a slot without a trade stays in cash; the day's result is the mean
//           over the N slots
// k (0.5, 1, 1.5, 2) and N (3, 5, 10) learnt: every pair trades on paper each day, and the pair whose
// days over the year before (at least 60) have the best mean shrunk by its t statistic trades the
// next (before that, Chan's own: k 1, N 10). Chan's fixed pair is reported too. --trades writes each
// trade (date,ticker,ret,filled) and beside it the account's days (.days: date,ret) for apps/ofm_mc.
//
//   ofm_gap --csv stocks.csv --costs UK|EU|US|APAC [--budget 10000] [--top 300] [--venues f]
//           [--stamp-free aim.txt] [--override ibkr.csv] [--pricing tiered] [--trades log.csv]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "ridge.hpp"
#include "swing.hpp"

namespace {
using ridge::Rec;
using swing::kNaN;
constexpr std::size_t nK = 4, nN = 3;
const double kK[nK] = {0.5, 1.0, 1.5, 2.0};
const std::size_t kN[nN] = {3, 5, 10};
}  // namespace

int main(int argc, char** argv) {
  std::string csv, market, venues, stampFree, overrideCsv, tradesOut;
  double budget = 10000;
  std::size_t top = 300;
  bool tiered = false;
  for (int k = 1; k + 1 < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv") csv = argv[++k];
    else if (a == "--costs") market = argv[++k];
    else if (a == "--budget") budget = std::atof(argv[++k]);
    else if (a == "--top") top = std::size_t(std::atol(argv[++k]));
    else if (a == "--venues") venues = argv[++k];
    else if (a == "--stamp-free") stampFree = argv[++k];
    else if (a == "--override") overrideCsv = argv[++k];
    else if (a == "--pricing") tiered = std::string(argv[++k]) == "tiered";
    else if (a == "--trades") tradesOut = argv[++k];
  }
  if (csv.empty() || market.empty()) {
    std::fprintf(stderr, "usage: ofm_gap --csv stocks.csv --costs UK|EU|US|APAC [--budget 10000] [--top 300] [--venues f] [--stamp-free f] [--override f] [--pricing tiered] [--trades f]\n");
    return 2;
  }
  swing::Panel panel = swing::load(csv);
  if (!overrideCsv.empty()) std::printf("gap: IBKR's bars: %s\n", swing::override(panel, overrideCsv).c_str());
  const swing::Panel& p = panel;
  const std::size_t T = p.T, N = p.N;
  const auto uni = swing::universe(p, top, market == "US" ? 5.0 : 0.0);
  // Each slot's amount differs by N: one cost model per N (a trade's commission on budget / N).
  std::vector<swing::Costs> cost;
  for (std::size_t n = 0; n < nN; ++n) {
    cost.push_back(swing::costs(p, market, budget / double(kN[n]), venues, stampFree));
    cost.back().tiered = tiered;
  }
  // The shares' own sigma (90 days), 20-day mean close and spread, known at each close.
  std::vector<double> sig(T * N, kNaN), ma(T * N, kNaN), spr(T * N, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    std::vector<double> r, c;
    double e = 0, w = 0, last = 0;
    for (std::size_t t = 0; t < T; ++t) {
      if (t >= 1) {
        const double c0 = p.C[(t - 1) * N + i], h0 = p.H[(t - 1) * N + i], l0 = p.L[(t - 1) * N + i], h1 = p.H[t * N + i], l1 = p.L[t * N + i];
        if (c0 > 0 && h0 > 0 && l0 > 0 && h1 > 0 && l1 > 0) {
          const double cc = std::log(c0), a = cc - 0.5 * std::log(h0 * l0), b = cc - 0.5 * std::log(h1 * l1);
          e += (4 * a * b - e) / 64, w += (1 - w) / 64;
        }
        const double s = w > 0 ? std::sqrt(std::max(0.0, e / w)) : 0.0;
        if (s > 0) last = s;
        spr[t * N + i] = last;
      }
      if (!p.ok(t, i)) continue;
      if (t >= 1 && p.ok(t - 1, i)) r.push_back(std::log(p.C[t * N + i] / p.C[(t - 1) * N + i]));
      c.push_back(p.C[t * N + i]);
      if (r.size() > 90) r.erase(r.begin());
      if (c.size() > 20) c.erase(c.begin());
      if (r.size() == 90) {
        double s1 = 0, s2 = 0;
        for (double v : r) s1 += v, s2 += v * v;
        sig[t * N + i] = std::sqrt(std::max(0.0, s2 / 90 - (s1 / 90) * (s1 / 90)));
      }
      if (c.size() == 20) {
        double s = 0;
        for (double v : c) s += v;
        ma[t * N + i] = s / 20;
      }
    }
  }
  // Each day's candidates (the gap in sigmas, the share), then every pair's paper day.
  struct Trade { std::size_t i; double ret; };
  std::vector<std::vector<Trade>> pick(T * nK * nN);
  std::vector<double> day(T * nK * nN, 0.0);
  std::vector<char> live(T, 0);
  for (std::size_t t = 1; t < T; ++t) {
    std::vector<std::pair<double, std::size_t>> cand;
    for (std::size_t i : uni[t - 1]) {  // tradable at the close before
      if (!p.ok(t, i) || !p.ok(t - 1, i) || !(p.O[t * N + i] > 0)) continue;
      const double s = sig[(t - 1) * N + i], m = ma[(t - 1) * N + i];
      if (!(s > 0) || !(m > 0) || !(p.O[t * N + i] > m)) continue;
      cand.emplace_back(std::log(p.O[t * N + i] / p.L[(t - 1) * N + i]) / s, i);
    }
    std::sort(cand.begin(), cand.end());
    live[t] = 1;
    for (std::size_t a = 0; a < nK; ++a)
      for (std::size_t b = 0; b < nN; ++b) {
        const std::size_t g = (t * nK + a) * nN + b;
        double s = 0;
        for (const auto& [z, i] : cand) {
          if (z >= -kK[a] || pick[g].size() >= kN[b]) break;
          const double entry = p.O[t * N + i] * std::exp(0.5 * spr[(t - 1) * N + i]), c = p.C[t * N + i];
          const double ret = std::log(c / entry) - cost[b].roundTrip(i, entry, c, false);
          pick[g].push_back({i, ret}), s += ret;
        }
        day[g] = s / double(kN[b]);  // the slots without a trade in cash
      }
  }
  // Walk-forward: each day the pair whose year before has the best shrunk mean (Chan's before 60 days).
  const std::size_t chan = (1 * nN) + 2;  // k 1, N 10
  Rec acct, a1, a2, fixedR;
  std::vector<std::pair<std::size_t, std::size_t>> chosen;  // (day, pair)
  std::size_t first = 0, used[nK * nN] = {}, trades = 0;
  std::vector<std::size_t> liveDays;
  for (std::size_t t = 0; t < T; ++t)
    if (live[t]) liveDays.push_back(t);
  for (std::size_t q = 0; q < liveDays.size(); ++q) {
    const std::size_t t = liveDays[q];
    std::size_t best = chan;
    if (q >= 60) {
      double bs = -1e300;
      for (std::size_t g = 0; g < nK * nN; ++g) {
        Rec r;
        for (std::size_t u = q >= 250 ? q - 250 : 0; u < q; ++u) r.add(day[liveDays[u] * nK * nN + g]);
        if (r.shrunk() > bs) bs = r.shrunk(), best = g;
      }
    }
    if (q < 250) continue;  // the record begins after a year (the choice has its window)
    if (!first) first = t;
    const double r = day[t * nK * nN + best];
    acct.add(r), (q < (250 + liveDays.size()) / 2 ? a1 : a2).add(r), fixedR.add(day[t * nK * nN + chan]);
    used[best] += 1, trades += pick[t * nK * nN + best].size(), chosen.emplace_back(t, best);
  }
  if (!first) {
    std::printf("gap: fewer than 250 days\n");
    return 0;
  }
  const auto yr = [](const Rec& r) {
    const double sd = r.sd();
    return std::make_pair(std::pow(1 + r.mean(), 252.0) - 1, sd > 0 ? r.mean() / sd * std::sqrt(252.0) : 0.0);
  };
  const auto [y0, s0] = yr(acct);
  const auto [y1, s1] = yr(a1);
  const auto [y2, s2] = yr(a2);
  const auto [yf, sf] = yr(fixedR);
  std::printf("gap: %s, %zu shares, %s to %s; bought just after the open (half a spread over it), sold in the closing auction; budget %.0f in N slots\n", market.c_str(), N,
              p.days[first].c_str(), p.days[T - 1].c_str(), budget);
  std::printf("gap: learnt day by day: a day %+.4f%%, compounded %+.1f%% a year, Sharpe %.2f | first half %+.1f%% %.2f | second half %+.1f%% %.2f; %.2f trades a day\n",
              100 * acct.mean(), 100 * y0, s0, 100 * y1, s1, 100 * y2, s2, double(trades) / acct.n);
  std::printf("gap: Chan's own (k 1 sigma, 10 slots): a day %+.4f%%, compounded %+.1f%% a year, Sharpe %.2f\n", 100 * fixedR.mean(), 100 * yf, sf);
  std::printf("gap: the pairs chosen (k, N: days):");
  for (std::size_t g = 0; g < nK * nN; ++g)
    if (used[g]) std::printf(" %.1f,%zu: %zu", kK[g / nN], kN[g % nN], used[g]);
  std::printf("\n");
  if (!tradesOut.empty()) {  // for apps/ofm_mc: each trade, and the account's days
    std::ofstream o(tradesOut), d(tradesOut + ".days");
    o << "date,ticker,ret,filled\n";
    d << "date,ret\n";
    char b[256];
    for (const auto& [t, g] : chosen) {
      for (const Trade& tr : pick[t * nK * nN + g]) std::snprintf(b, sizeof b, "%s,%s,%.8g,1\n", p.days[t].c_str(), p.tickers[tr.i].c_str(), tr.ret), o << b;
      std::snprintf(b, sizeof b, "%s,%.8g\n", p.days[t].c_str(), day[t * nK * nN + g]), d << b;
    }
  }
  return 0;
}
