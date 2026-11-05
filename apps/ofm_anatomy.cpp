// Where a market's returns are, and which kinds of strategy earn them (research; results only).
//
// On daily bars (date,ticker,open,high,low,close,volume), every day's tradable shares (a price of at
// least --min-price and among the --top most traded by their median daily traded value over the last
// 20 days, the past only) are ranked by each signal known at the close, and the top tenth is held
// over each window:
//   ON    bought at the close (auction), sold at the next open (auction)
//   ID    bought at the next open, sold at its close (what the daily model trades)
//   CC    bought at the close, sold at the next close
//   H5    bought at the next open, held 5 days (a fifth of the book renewed each day)
//   H20   the same, 20 days
// The returns are net of a round trip's costs (IBKR's, apps/swing.hpp: the commission on both sides,
// the taxes). ON and CC buy in the same close's auction the signal is read at: a measure of where the
// returns are, not a tradable rule (apps/ofm_swing.cpp trades only what orders can). Every window buys and sells in the
// opening or the closing auction, which cross at one price: no spread. The book renews 1/h of itself
// a day over h days. The share list is today's (the store's): shares delisted since are missing, so
// the long side of a falling-share signal is flattered; the top less the bottom tenth is less so.
// Also: the market (every tradable share, equally weighted) over each window, and the top tenth less
// the bottom tenth (gross: a short's costs are not modelled). For each: the mean a year, the Sharpe
// ratio, the t statistic, and the first and the second half of the days apart (a pattern found in
// the first half must hold in the second to be worth anything).
//
//   ofm_anatomy --csv stocks.csv [--costs US|UK|EU|APAC] [--top 500] [--min-price 5] [--capital 5000]
//               [--venues tools/venues.txt] [--stamp-free aim.txt]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "swing.hpp"

namespace {
using swing::kNaN;

struct Series {
  double sum = 0, sum2 = 0;
  long n = 0;
  void add(double x) { sum += x, sum2 += x * x, ++n; }
  double mean() const { return n ? sum / double(n) : 0; }
  double sd() const { return n > 1 ? std::sqrt(std::max(0.0, sum2 / double(n) - mean() * mean())) : 0; }
};

struct Stat {  // a daily series: all days, first half, second half
  Series all, h1, h2;
  void add(double x, bool first) { all.add(x), (first ? h1 : h2).add(x); }
};

std::string fmt(const Series& s) {
  char b[96];
  const double sd = s.sd();
  std::snprintf(b, sizeof b, "%+7.1f%% %5.2f %+6.1f", 25200 * s.mean(), sd > 0 ? s.mean() / sd * std::sqrt(252.0) : 0.0,
                sd > 0 ? s.mean() / sd * std::sqrt(double(s.n)) : 0.0);
  return b;
}
}  // namespace

int main(int argc, char** argv) {
  std::string csv, market = "US", venues, stampFree;
  std::size_t top = 500;
  double minPrice = 0, capital = 5000;
  for (int k = 1; k + 1 < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv") csv = argv[++k];
    else if (a == "--top") top = std::size_t(std::atol(argv[++k]));
    else if (a == "--min-price") minPrice = std::atof(argv[++k]);
    else if (a == "--capital") capital = std::atof(argv[++k]);
    else if (a == "--costs") market = argv[++k];
    else if (a == "--venues") venues = argv[++k];
    else if (a == "--stamp-free") stampFree = argv[++k];
  }
  if (csv.empty()) {
    std::fprintf(stderr, "usage: ofm_anatomy --csv stocks.csv [--costs US|UK|EU|APAC] [--top 500] [--min-price 5] [--capital 5000] [--venues f] [--stamp-free f]\n");
    return 2;
  }
  const swing::Panel p = swing::load(csv);
  const std::size_t T = p.T, N = p.N;
  const auto& O = p.O;
  const auto& C = p.C;
  const auto ok = [&](std::size_t t, std::size_t i) { return p.ok(t, i); };
  const swing::Costs cost = swing::costs(p, market, capital, venues, stampFree);
  const auto commission = [&](std::size_t i, double price) { return cost.side(i, price); };
  const auto sig = swing::signals(p);
  const std::vector<std::pair<const char*, int>> windows = {{"ON", -1}, {"ID", 1}, {"CC", 0}, {"H5", 5}, {"H20", 20}};
  const std::size_t S = sig.size(), W = windows.size();
  // Each day's tradable shares and, for each signal, its top and bottom tenths (formed at the close t).
  const auto uni = swing::universe(p, top, minPrice);
  std::vector<std::vector<std::vector<std::size_t>>> topSet(S, std::vector<std::vector<std::size_t>>(T)), botSet = topSet;
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t s = 0; s < S; ++s) {
      std::vector<std::pair<double, std::size_t>> z;
      for (std::size_t i : uni[t]) {
        const double x = sig[s].f(t, i);
        if (std::isfinite(x)) z.emplace_back(x, i);
      }
      if (z.size() < 50) continue;
      std::sort(z.begin(), z.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      const std::size_t k = z.size() / 10;
      for (std::size_t r = 0; r < k; ++r) topSet[s][t].push_back(z[r].second), botSet[s][t].push_back(z[z.size() - 1 - r].second);
    }
  }
  // A share's net return on day d for a cohort formed at the close f (the window w): the return from
  // its entry (or the last close) to its exit (or this close), less its costs on its entry and exit days.
  const auto cohortDay = [&](std::size_t f, std::size_t d, std::size_t i, int h, bool net) {
    // d: the day this return ends on (close d, or open d for ON).
    const double cf = C[f * N + i];
    if (h == -1) {  // ON: close f to open f+1
      if (!(O[d * N + i] > 0)) return 0.0;
      return O[d * N + i] / cf - 1 - (net ? 2 * commission(i, cf) + cost.buyTax(i, true) : 0.0);
    }
    if (h == 0) {  // CC: close f to close f+1, both auctions
      if (!ok(d, i)) return 0.0;
      return C[d * N + i] / cf - 1 - (net ? 2 * commission(i, cf) + cost.buyTax(i, true) : 0.0);
    }
    const std::size_t age = d - f;  // 1: the entry day (open to close)
    if (!ok(d, i) || !(O[(f + 1) * N + i] > 0)) return 0.0;
    double r = age == 1 ? C[d * N + i] / O[(f + 1) * N + i] - 1 : (ok(d - 1, i) ? C[d * N + i] / C[(d - 1) * N + i] - 1 : 0.0);
    if (net) {
      if (age == 1) r -= commission(i, O[(f + 1) * N + i]) + cost.buyTax(i, h != 1);
      if (age == std::size_t(h)) r -= commission(i, C[d * N + i]);
    }
    return r;
  };
  const auto bookDay = [&](const std::vector<std::vector<std::size_t>>& sets, std::size_t d, int h, bool net) {
    // The book's return on day d: the mean of its cohorts (h of them, each 1/h of the money).
    const int n = h <= 0 ? 1 : h;
    double sum = 0;
    int have = 0;
    for (int k = 1; k <= n; ++k) {
      if (d < std::size_t(k)) break;
      const std::size_t f = d - std::size_t(k);
      if (sets[f].empty()) continue;
      double s = 0;
      for (std::size_t i : sets[f]) s += cohortDay(f, d, i, h, net);
      sum += s / double(sets[f].size()), ++have;
    }
    return have ? sum / double(n) : kNaN;
  };
  const std::size_t start = 260, mid = start + (T - start) / 2;
  std::printf("anatomy: %zu shares, %zu days (%s to %s); each day the %zu most traded at %.0f or more; costs %s on %.0f a trade, auctions only (no spread); today's share list (survivors)\n",
              N, T, p.days.front().c_str(), p.days.back().c_str(), top, minPrice, market.c_str(), capital);
  const auto first = [&](std::size_t d) { return d < mid; };
  // The market itself over each window (gross and net), every tradable share equally weighted.
  std::printf("anatomy: %-22s %-4s %24s | %24s | %24s\n", "the market (all held)", "win", "all days: a year Sh t", "first half", "second half");
  for (std::size_t w = 0; w < W; ++w)
    for (int net = 0; net < 2; ++net) {
      Stat st;
      for (std::size_t d = start; d < T; ++d) {
        const double x = bookDay(uni, d, windows[w].second, net);
        if (std::isfinite(x)) st.add(x, first(d));
      }
      std::printf("anatomy: %-22s %-4s %s | %s | %s\n", net ? "market, net" : "market, gross", windows[w].first, fmt(st.all).c_str(), fmt(st.h1).c_str(), fmt(st.h2).c_str());
    }
  std::printf("anatomy: %-22s %-4s %24s | %24s | %24s || %s\n", "signal: top tenth, net", "win", "all days: a year Sh t", "first half", "second half", "top less bottom, gross: all, 1st, 2nd");
  for (std::size_t s = 0; s < S; ++s)
    for (std::size_t w = 0; w < W; ++w) {
      Stat st, ls;
      for (std::size_t d = start; d < T; ++d) {
        const double a = bookDay(topSet[s], d, windows[w].second, true), b = bookDay(topSet[s], d, windows[w].second, false), c = bookDay(botSet[s], d, windows[w].second, false);
        if (std::isfinite(a)) st.add(a, first(d));
        if (std::isfinite(b) && std::isfinite(c)) ls.add(b - c, first(d));
      }
      std::printf("anatomy: %-22s %-4s %s | %s | %s || %s | %s | %s\n", sig[s].name, windows[w].first, fmt(st.all).c_str(), fmt(st.h1).c_str(), fmt(st.h2).c_str(),
                  fmt(ls.all).c_str(), fmt(ls.h1).c_str(), fmt(ls.h2).c_str());
    }
  return 0;
}
