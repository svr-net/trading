// Momentum book (research; results only): orders that always buy. The shares that rose most over the
// past months are bought in an opening auction (a market-on-open order: it fills, no spread) and held
// for months; a fall of the whole market empties the book.
//
// The evidence: momentum in UK shares (Liu, Strong and Xu 1999) pays after costs for holdings of six
// months or more, where the turnover is low (Agyei-Ampomah 2007), and the winners are the cheap side to
// trade (Li, Brooks and Miffre 2009); its returns come overnight (Lou, Polk and Skouras 2019), which an
// auction-to-auction holding keeps; cash while the market is under its long average (Faber 2007).
//
// Each close t, among the day's tradable shares (apps/swing.hpp: the most traded):
//   signal   the close t-21 over the close t-21-L (momentum, the last month skipped)
//   book     every R = 21 days: the N highest; a held share stays while it ranks in the top 2N (less
//            turnover); when the gate is on and the market (the tradable shares' mean close-to-close
//            return, compounded) is under its 200-day average, the book is sold and stays in cash
//   trades   in the next session's opening auction, at its open; the broker's costs on both sides and
//            the buy tax once (apps/swing.hpp); N slots of --budget / N, an empty slot in cash
// L (126, 252), N (5, 10, 20) and the gate (off, on) are learnt: every setting trades on paper each
// day, and at each rebalance the one whose days over the two years before (at least 250) have the best
// mean shrunk by its t statistic is the one whose book the account moves to. Reported: the account,
// its first and second halves, the market over the same days, and the account less the market.
// --trades writes the account's days (date,ret) for apps/ofm_mc.
//
//   ofm_mom --csv stocks.csv --costs UK|EU|US|APAC [--budget 10000] [--top 300] [--venues f]
//           [--stamp-free aim.txt] [--override ibkr.csv] [--pricing tiered] [--trades days.csv]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "swing.hpp"

namespace {
struct Cfg { std::size_t L, N; bool gate; };
struct Stat { double ann, sharpe; };
Stat stat(const std::vector<double>& r, std::size_t a, std::size_t b) {
  double lw = 0, s = 0, s2 = 0;
  const double n = double(b - a);
  for (std::size_t k = a; k < b; ++k) lw += std::log1p(r[k]), s += r[k], s2 += r[k] * r[k];
  if (n < 2) return {0, 0};
  const double m = s / n, v = s2 / n - m * m;
  return {std::expm1(lw * 252 / n), v > 0 ? m / std::sqrt(v) * std::sqrt(252.0) : 0.0};
}
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
    std::fprintf(stderr, "usage: ofm_mom --csv stocks.csv --costs UK|EU|US|APAC [--budget 10000] [--top 300] [--venues f] [--stamp-free f] [--override f] [--pricing tiered] [--trades f]\n");
    return 2;
  }
  swing::Panel panel = swing::load(csv);
  if (!overrideCsv.empty()) std::printf("mom: IBKR's bars: %s\n", swing::override(panel, overrideCsv).c_str());
  const swing::Panel& p = panel;
  const std::size_t T = p.T, N = p.N, R = 21, skip = 21, warm = 252 + skip + 1;
  const auto uni = swing::universe(p, top, 0.0);

  // The market: the tradable shares' mean close-to-close return each day, compounded; its 200-day average.
  std::vector<double> mret(T, 0.0), mlevel(T, 1.0), mavg(T, 0.0);
  for (std::size_t t = 1; t < T; ++t) {
    double s = 0, n = 0;
    for (std::size_t i : uni[t - 1])
      if (p.ok(t, i) && p.ok(t - 1, i)) s += p.C[t * N + i] / p.C[(t - 1) * N + i] - 1, n += 1;
    mret[t] = n > 0 ? s / n : 0.0, mlevel[t] = mlevel[t - 1] * (1 + mret[t]);
  }
  for (std::size_t t = 0; t < T; ++t) {
    double s = 0;
    const std::size_t a = t >= 199 ? t - 199 : 0;
    for (std::size_t u = a; u <= t; ++u) s += mlevel[u];
    mavg[t] = s / double(t - a + 1);
  }

  std::vector<Cfg> cfgs;
  for (std::size_t L : {std::size_t(126), std::size_t(252)})
    for (std::size_t n : {std::size_t(5), std::size_t(10), std::size_t(20)})
      for (bool g : {false, true}) cfgs.push_back({L, n, g});
  const std::size_t K = cfgs.size();

  // A setting's target book at the close t: its held shares kept while in the top 2N, filled to N.
  const auto target = [&](const Cfg& c, std::size_t t, const std::set<std::size_t>& held) {
    std::set<std::size_t> out;
    if (c.gate && mlevel[t] < mavg[t]) return out;
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t i : uni[t])
      if (p.ok(t, i) && p.ok(t - skip, i) && p.ok(t - skip - c.L, i)) v.emplace_back(p.C[(t - skip) * N + i] / p.C[(t - skip - c.L) * N + i], i);
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (std::size_t r = 0; r < v.size() && r < 2 * c.N; ++r)
      if (held.count(v[r].second) && out.size() < c.N) out.insert(v[r].second);
    for (std::size_t r = 0; r < v.size() && out.size() < c.N; ++r) out.insert(v[r].second);
    return out;
  };

  // One book's day t (the close t-1 to the close t), moving to `want` at the open t when given: the
  // result over its N slots, net of the trades' costs. A share without a bar keeps its last price.
  struct Book { std::set<std::size_t> held; std::size_t n = 10; double buys = 0, sells = 0; };
  const auto day = [&](Book& b, std::size_t t, const std::set<std::size_t>* want, std::size_t slots, const swing::Costs& cost) {
    double r = 0;
    const auto px = [&](std::size_t u, std::size_t i, const std::vector<double>& X) { return u < T && X[u * N + i] > 0 ? X[u * N + i] : 0.0; };
    std::set<std::size_t> next = b.held;
    if (want) {
      next.clear();
      for (std::size_t i : b.held) {
        const double o = px(t, i, p.O);
        if (want->count(i) || !(o > 0)) next.insert(i);  // kept (or no auction today: sold at a later rebalance)
      }
      for (std::size_t i : *want)
        if (!b.held.count(i) && px(t, i, p.O) > 0 && next.size() < slots) next.insert(i);
    }
    for (std::size_t i : b.held) {  // held into today: to the close, or to the open when sold there
      const double c0 = px(t - 1, i, p.C), o = px(t, i, p.O), c1 = px(t, i, p.C);
      if (!(c0 > 0)) continue;
      if (next.count(i)) r += c1 > 0 ? c1 / c0 - 1 : 0.0;
      else r += o / c0 - 1 - cost.side(i, o), b.sells += 1;
    }
    for (std::size_t i : next)
      if (!b.held.count(i)) {  // bought at the open: to the close, less the buy's costs and tax
        const double o = px(t, i, p.O), c1 = px(t, i, p.C);
        r += (c1 > 0 ? c1 / o - 1 : 0.0) - cost.side(i, o) - cost.buyTax(i, true), b.buys += 1;
      }
    b.held = next;
    return r / double(slots);
  };

  std::vector<swing::Costs> costOf;  // a slot's money differs with N
  for (std::size_t n : {std::size_t(5), std::size_t(10), std::size_t(20)}) {
    costOf.push_back(swing::costs(p, market, budget / double(n), venues, stampFree));
    costOf.back().tiered = tiered;
  }
  const auto costFor = [&](std::size_t n) -> const swing::Costs& { return costOf[n == 5 ? 0 : n == 10 ? 1 : 2]; };

  std::vector<Book> paper(K);
  for (std::size_t k = 0; k < K; ++k) paper[k].n = cfgs[k].N;
  std::vector<std::vector<double>> pr(K);  // each setting's paper days
  Book acct;
  std::size_t use = K;  // the setting the account follows (K: none yet, cash)
  std::vector<double> ar, mk;
  std::vector<std::string> ad;
  std::vector<std::size_t> usedDays(K, 0);
  double heldSum = 0;
  std::vector<std::set<std::size_t>> want(K);
  std::set<std::size_t> acctWant;
  bool acctMove = false;
  std::size_t acctSlots = 10;
  for (std::size_t t = warm; t < T; ++t) {
    const bool reb = (t - warm) % R == 0;  // the targets were set at the close t-1
    for (std::size_t k = 0; k < K; ++k) pr[k].push_back(day(paper[k], t, reb ? &want[k] : nullptr, cfgs[k].N, costFor(cfgs[k].N)));
    if (pr[0].size() > 250) {  // the account's record starts once a setting can be chosen
      ar.push_back(day(acct, t, acctMove ? &acctWant : nullptr, acctSlots, costFor(acctSlots)));
      mk.push_back(mret[t]), ad.push_back(p.days[t]), heldSum += double(acct.held.size()) / double(acctSlots);
      if (use < K) usedDays[use] += 1;
    }
    acctMove = false;
    if ((t + 1 - warm) % R == 0 && t + 1 < T) {  // the close before a rebalance: the targets, and the account's setting
      for (std::size_t k = 0; k < K; ++k) want[k] = target(cfgs[k], t, paper[k].held);
      const std::size_t n = pr[0].size();
      if (n >= 250) {
        double best = -1e300;
        for (std::size_t k = 0; k < K; ++k) {
          double s = 0, s2 = 0;
          const std::size_t a = n > 504 ? n - 504 : 0;
          for (std::size_t q = a; q < n; ++q) s += pr[k][q], s2 += pr[k][q] * pr[k][q];
          const double m = s / double(n - a), v = s2 / double(n - a) - m * m, tt = v > 0 ? m / std::sqrt(v / double(n - a)) : 0.0;
          const double sc = m * std::max(0.0, 1 - 1 / std::max(tt * tt, 1e-12));
          if (sc > best) best = sc, use = k;
        }
        if (!(best > 0)) use = K;  // no setting with a positive record: cash
        acctSlots = use < K ? cfgs[use].N : acctSlots;
        acctWant = use < K ? target(cfgs[use], t, acct.held) : std::set<std::size_t>();
        acctMove = true;
      }
    }
  }
  const std::size_t D = ar.size();
  std::printf("mom: %s, %zu shares, %s to %s; the %zu most traded a day; bought and sold in opening auctions; budget %.0f in N slots; rebalanced every %zu days\n", market.c_str(), N,
              D ? ad.front().c_str() : "-", D ? ad.back().c_str() : "-", top, budget, R);
  if (D < 20) return std::printf("mom: too few days\n"), 0;
  std::vector<double> ex(D);
  for (std::size_t q = 0; q < D; ++q) ex[q] = ar[q] - mk[q] * 1.0;
  const Stat a = stat(ar, 0, D), a1 = stat(ar, 0, D / 2), a2 = stat(ar, D / 2, D), m = stat(mk, 0, D), e = stat(ex, 0, D), e1 = stat(ex, 0, D / 2), e2 = stat(ex, D / 2, D);
  double peak = 0, w = 0, dd = 0;
  for (double r : ar) w += std::log1p(r), peak = std::max(peak, w), dd = std::max(dd, 1 - std::exp(w - peak));
  std::printf("mom: the account (learnt at each rebalance): %+.1f%% a year, Sharpe %.2f, worst drawdown %.1f%% | first half %+.1f%% %.2f | second half %+.1f%% %.2f\n", 100 * a.ann, a.sharpe,
              100 * dd, 100 * a1.ann, a1.sharpe, 100 * a2.ann, a2.sharpe);
  std::printf("mom: the market over the same days: %+.1f%% a year, Sharpe %.2f; the account less the market: %+.1f%% a year, Sharpe %.2f | first half %+.1f%% %.2f | second half %+.1f%% %.2f\n",
              100 * m.ann, m.sharpe, 100 * e.ann, e.sharpe, 100 * e1.ann, e1.sharpe, 100 * e2.ann, e2.sharpe);
  std::printf("mom: slots held %.0f%% of the time; trades: %.0f buys and %.0f sells (%.1f buys a year), every order filled (auction)\n", 100 * heldSum / double(D), acct.buys, acct.sells,
              acct.buys * 252 / double(D));
  std::printf("mom: the settings followed (L days, N, gate: days):");
  for (std::size_t k = 0; k < K; ++k)
    if (usedDays[k]) std::printf(" %zu,%zu,%s: %zu", cfgs[k].L, cfgs[k].N, cfgs[k].gate ? "on" : "off", usedDays[k]);
  std::printf("\nmom: each setting fixed in hindsight (for comparison only):");
  for (std::size_t k = 0; k < K; ++k) {
    const Stat s = stat(pr[k], pr[k].size() - D, pr[k].size());
    std::printf(" %zu,%zu,%s %+.1f%% %.2f;", cfgs[k].L, cfgs[k].N, cfgs[k].gate ? "on" : "off", 100 * s.ann, s.sharpe);
  }
  std::printf("\n");
  if (!tradesOut.empty()) {
    std::ofstream o(tradesOut);
    o << "date,ret\n";
    for (std::size_t q = 0; q < D; ++q) o << ad[q] << ',' << ar[q] << '\n';
  }
  return 0;
}
