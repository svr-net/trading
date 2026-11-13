// The UK daily model (the London Stock Exchange's live strategy): orders a person would place.
//
// Each order is a day order for one share, bought at the next open and sold in the closing auction,
// sized to what the share trades and priced so it buys where the forecast holds. An order is made only
// when the model's own out-of-sample record says it gains after every cost. Every choice below is a
// fixed, documented rule: nothing is tuned on the backtest it reports.
//
//   shares    ordinary shares in the order book's auctions (--segments: SET1, SET3, AMSM, SSMM; no
//             closed-end funds (STMM), no ETFs or ETPs, no quote-driven SETSqx shares, whose daily
//             bars show quotes more than trades). A share missing from the list (delisted since) stays
//             in the backtest, so it is not built on today's survivors only. At least 20p, 250 days of
//             history, a bar on 18 of its last 20 sessions, a median of at least GBP 250,000 traded a
//             day over its last 20 bars, no move of over 30% on the day (a bid, a placing, bad data),
//             a daily volatility (60 bars) of at most --max-vol (4%: the most volatile shares, a few
//             of whose falls took most of a small book, are left out); the --top (300) most traded of
//             those each day.
//   inputs    the signals of apps/swing.hpp known at the close (reversal over 1, 5 and 20 days,
//             momentum, the 52-week high, low volatility, the tug of war, ...), the share's volatility,
//             its liquidity and whether it pays stamp duty; each day ranked across the shares (-0.5 ..
//             0.5), so no input's scale or outliers decide.
//   target    the next session's open to close, in the share's own daily volatilities (at most 4).
//   model     one ridge regression across the shares, refitted every --refit (21) sessions on the
//             --window (750) sessions before (at least --warm, 250), fixed shrinkage.
//   expected  the forecast calibrated on the model's own out-of-sample record of the trade as placed:
//             over the last 500 sessions, among each day's highest fifth of forecasts, what buying at
//             the open and selling at the levels or the close (a triple barrier, Lopez de Prado 2018)
//             returned against what was forecast (a + b x, b shrunk by its t statistic). No record (at
//             least 120 sessions), or no skill in it, makes no orders.
//   costs     the model learns and its backtest trades before the broker's costs: nothing it learns
//             or reports is shaped by them. They come off the final orders only (the next session's):
//             IBKR's commission on the order's own amount (fixed: GBP 3 or 0.05% a side; tiered with
//             --pricing tiered) and the 0.5% stamp duty on buying (not AIM shares, --stamp-free); no
//             spread, as the orders trade in the auctions (one price for buyers and sellers); an order
//             whose costs take all it is expected to gain is not placed. (The prices the trades see stay
//             realistic: a stop sells half a spread below its level, a take-profit fills only when the
//             high trades through it.)
//   order     bought in the opening auction and only there: a limit at the last close plus --cap-sd
//             (0.5) of its daily volatility (it never chases a gap up), good only until the auction
//             is over, so it never buys a dip later in the day. The amount is the slot's capital, at
//             most --adv-share (1%) of the share's median day's value (at least --min-order, GBP 500).
//   levels    a stop-loss and a take-profit set from the price it was bought at (the bridge places
//             them once the auction has filled, so a gap down never sets the stop above the entry), one
//             cancelling the other; whatever is still held is sold in the closing auction. The stop
//             s daily volatilities below the entry (1, 1.5, 2, 3 or 4) is fixed, trailing (s below the
//             highest price since the fill: IBKR's own trailing stop) or break-even (moved to the entry
//             once the price has risen one volatility: the bridge's watch); the take-profit 1, 1.5, 2 or
//             3 above, or none. Each day the levels are chosen from these 75 by the record of the last
//             250 sessions: what each added, per session, to the day's highest fifth of forecasts
//             bought at the open, against the fixed 4-volatility stop alone, shrunk by its t statistic
//             (about two standard errors to be chosen; otherwise that stop alone).
//   paths     the levels are replayed on each share-day's 5-minute bars where --bars has them (each
//             bar's low taken before its high: the stop first); else on the daily bar both ways, the
//             low first and the high first, the worse taken (the bar cannot tell which came first).
//   choice    the orders whose expected return (before costs) is at least --hurdle (0.05%), the largest
//             expected gain in pounds first, at most --slots (X_NPICK, 5) a day; for the final orders,
//             after each one's own costs. Some days none.
// The backtest trades these orders day by day, before costs; an unfilled order and an unused slot earn 0.
//
//   ofm_uk --csv stocks.csv --costs UK [--capital 5000] [--slots 5] [--stamp-free aim.txt]
//          [--segments lse_segments.csv] [--override ibkr.csv] [--pricing fixed|tiered]
//          [--json scan.json] [--orders orders.json] [--trace orders.tr]
// Prices in pence (EODData's and IBKR's LSE quotes). X_TRADELOG=path writes every order's result
// (and path.days, the account's day by day) for tools/scan/strategies.py's budget.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ridge.hpp"
#include "swing.hpp"

namespace {
using swing::kNaN;

struct Opt {
  std::string csv, market, stampFree, segments, overrideCsv, jsonOut, ordersOut, traceOut, barsCsv, venues;
  double capital = 5000, maxVol = 0.04, minValue = 250000, advShare = 0.01, minOrder = 500, hurdle = 0.0005, capSd = 0.5, minPrice = 20, lambda = 0.5;
  std::size_t slots = 5, top = 300, refit = 21, window = 750, warm = 250, calWindow = 500, calMin = 120;
  bool tiered = false;
  std::size_t hold = 0;  // --hold H: held up to H sessions (sold at the first close above its exit level), 0: within the day
};

// Ranks of x in place (-0.5 .. 0.5, ties sharing their mean rank); NaN to 0, the middle.
void rankInPlace(std::vector<double>& x) {
  std::vector<std::pair<double, std::size_t>> v;
  for (std::size_t k = 0; k < x.size(); ++k)
    if (std::isfinite(x[k])) v.emplace_back(x[k], k);
  for (double& y : x) y = 0;
  if (v.size() < 2) return;
  std::sort(v.begin(), v.end());
  for (std::size_t a = 0; a < v.size();) {
    std::size_t b = a;
    while (b + 1 < v.size() && v[b + 1].first == v[a].first) ++b;
    const double r = 0.5 * double(a + b) / double(v.size() - 1) - 0.5;
    for (std::size_t q = a; q <= b; ++q) x[v[q].second] = r;
    a = b + 1;
  }
}

struct Order {
  std::size_t i = 0;
  double close = 0, limit = 0, stop = 0, value = 0, sigma = 0, spread = 0;
  double stopSd = 4, takeSd = kNaN;  // the levels from the entry, in daily volatilities (take NaN: none)
  double adj = 0;       // what the chosen levels added over the last 250 sessions (per trade)
  std::size_t level = 0, mode = 0;  // the grid's pair; 0 fixed, 1 trailing, 2 break-even
  double expected = 0;  // the calibrated return of the trade as placed (the open, its levels, the close), before costs
  double cost = 0;      // the round trip's costs if it buys in the auction and sells at the close
  double net = 0;       // expected - cost: the return if it buys
  double fill = kNaN;   // how often its opening auction would have bought over the share's last 250 sessions
  double exitAbove = 0, holdDays = 0;  // --hold: sold at the first close this far above the price paid (else at the last), and the days such trades were held
  bool capped = false;  // its amount cut to the share's liquidity
};

struct Fill {
  int filled = 0;  // 1 in the opening auction
  int exit = 0;    // 0 the closing auction, 1 the stop, 2 the take-profit
  double entry = 0, exitPx = 0, gross = 0, cost = 0, ret = 0;
  bool onPath = false;  // its day replayed on 5-minute bars
};
}  // namespace

int main(int argc, char** argv) {
  Opt o;
  bool minPriceSet = false;
  if (const char* e = std::getenv("X_NPICK")) o.slots = std::max<std::size_t>(1, std::size_t(std::atol(e)));
  for (int k = 1; k < argc; ++k) {  // unknown arguments (the research model's) are ignored
    const std::string a = argv[k];
    const bool v = k + 1 < argc;
    if (!v) break;
    if (a == "--csv") o.csv = argv[++k];
    else if (a == "--costs") o.market = argv[++k];
    else if (a == "--capital") o.capital = std::atof(argv[++k]);
    else if (a == "--slots") o.slots = std::max<std::size_t>(1, std::size_t(std::atol(argv[++k])));
    else if (a == "--top") o.top = std::size_t(std::atol(argv[++k]));
    else if (a == "--stamp-free") o.stampFree = argv[++k];
    else if (a == "--venues") o.venues = argv[++k];
    else if (a == "--segments") o.segments = argv[++k];
    else if (a == "--override") o.overrideCsv = argv[++k];
    else if (a == "--pricing") o.tiered = std::string(argv[++k]) == "tiered";
    else if (a == "--json") o.jsonOut = argv[++k];
    else if (a == "--orders") o.ordersOut = argv[++k];
    else if (a == "--trace") o.traceOut = argv[++k];
    else if (a == "--bars") o.barsCsv = argv[++k];
    else if (a == "--hold") o.hold = std::min<std::size_t>(5, std::size_t(std::atol(argv[++k])));
    else if (a == "--min-value") o.minValue = std::atof(argv[++k]);
    else if (a == "--max-vol") o.maxVol = std::atof(argv[++k]);
    else if (a == "--adv-share") o.advShare = std::atof(argv[++k]);
    else if (a == "--min-order") o.minOrder = std::atof(argv[++k]);
    else if (a == "--hurdle") o.hurdle = std::atof(argv[++k]);
    else if (a == "--cap-sd") o.capSd = std::atof(argv[++k]);
    else if (a == "--min-price") o.minPrice = std::atof(argv[++k]), minPriceSet = true;
    else if (a == "--lambda") o.lambda = std::atof(argv[++k]);
    else if (a == "--refit") o.refit = std::max<std::size_t>(1, std::size_t(std::atol(argv[++k])));
    else if (a == "--window") o.window = std::size_t(std::atol(argv[++k]));
    else if (a == "--warm") o.warm = std::size_t(std::atol(argv[++k]));
  }
  static const std::set<std::string> kMarkets = {"UK", "EU", "US", "APAC", "INDEX", "FOREX"};
  if (o.csv.empty() || !kMarkets.count(o.market) || ((o.market == "EU" || o.market == "APAC" || o.market == "INDEX") && o.venues.empty())) {
    std::fprintf(stderr, "usage: ofm_uk --csv stocks.csv --costs UK [--capital 5000] [--slots 5] [--stamp-free f] [--segments f] [--override f] [--json f] [--orders f]\n");
    return 2;
  }
  swing::Panel p = swing::load(o.csv);
  if (!o.overrideCsv.empty()) std::printf("uk: IBKR's bars: %s\n", swing::override(p, o.overrideCsv).c_str());
  const std::size_t T = p.T, N = p.N;
  if (T < 10 || N < 2) {
    std::fprintf(stderr, "uk: too little data (%zu days, %zu shares)\n", T, N);
    return 2;
  }
  const auto at = [N](std::size_t t, std::size_t i) { return t * N + i; };
  // The market: the UK's prices in pence (traded value in pounds: / 100), stamp duty and the LSE's segments;
  // Europe's and Asia-Pacific's (--costs EU|APAC, tickers EXCHANGE:CODE) in their exchanges' currencies, each
  // exchange's commission, fees and taxes from --venues (tools/venues.txt: the tax on buying a share held past
  // the day, as these are, for the holds); the US's in dollars (IBKR's commission a share).
  // INDEX: UCITS index ETFs on Xetra (FRA:CODE, in euros, the Frankfurt line of --venues: an EU account may
  // not buy US-listed ETFs, IBKR 201, 13 Nov); FOREX: currency pairs (research: IBKR's FX commission, no
  // volume, every pair listed, each pair's inverse beside it so the model can take either side).
  const bool fx = o.market == "FOREX", uk = o.market == "UK", us = o.market == "US";
  const char* cur = uk ? "GBP" : us || fx ? "USD" : o.market == "APAC" ? "AUD" : "EUR";  // APAC: the ASX's AUD (an SGX order's amount in SGD)
  const double priceUnit = uk ? 100.0 : 1.0;
  if (!uk && !minPriceSet) o.minPrice = fx ? 0.0 : us ? 5.0 : o.market == "APAC" ? 0.5 : 1.0;  // EUR 1; USD 5 (no penny stocks); AUD/SGD 0.50
  // FOREX: no volume. INDEX: EODData's FRA bars are the Frankfurt floor's, a sliver of the ETFs' volume on Xetra,
  // where market makers quote them (13 Nov: every ETF under the EUR 250,000 floor, no order). Neither has a
  // liquidity floor, nor an order capped by the day's volume.
  const bool noVolume = fx || o.market == "INDEX";
  if (noVolume) o.minValue = 0;
  const swing::Costs venueCosts = uk || us || fx ? swing::Costs{} : swing::costs(p, o.market, o.capital, o.venues, "");

  // Stamp duty (AIM shares are exempt) and the order book's segments.
  std::vector<char> stampFree(N, 0), segOk(N, 1);
  if (!o.stampFree.empty()) {
    std::ifstream in(o.stampFree);
    std::set<std::string> s;
    for (std::string l; std::getline(in, l);) s.insert(l.substr(0, l.find_first_of(", \r")));
    for (std::size_t i = 0; i < N; ++i) stampFree[i] = s.count(p.tickers[i]) ? 1 : 0;
  }
  std::size_t segKnown = 0, segOut = 0;
  if (!o.segments.empty()) {
    static const std::set<std::string> book = {"SET1", "SET3", "AMSM", "SSMM"};
    std::ifstream in(o.segments);
    std::map<std::string, std::string> seg;
    std::string l;
    std::getline(in, l);
    while (std::getline(in, l)) {
      const auto c1 = l.find(','), c2 = l.find(',', c1 + 1);
      if (c1 != std::string::npos) seg[l.substr(0, c1)] = l.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1);
    }
    for (std::size_t i = 0; i < N; ++i) {
      const auto it = seg.find(p.tickers[i]);
      if (it == seg.end()) continue;  // delisted since (the list is today's): kept, against survivorship
      ++segKnown;
      if (!book.count(it->second)) segOk[i] = 0, ++segOut;
    }
  }

  // Each share's volatility (60 bars of close to close), median day's value (20 bars, GBP), spread
  // (Abdi and Ranaldo 2017, smoothed over about 64 bars) and the sessions it traded, at each close.
  std::vector<double> sigma(T * N, kNaN), adv(T * N, kNaN), spread(T * N, 0.0);
  std::vector<char> eligible(T * N, 0);
  {
    std::atomic<std::size_t> next{0};
    auto work = [&] {
      for (std::size_t i; (i = next++) < N;) {
        std::vector<double> rets, vals;
        std::vector<std::size_t> barDays;
        std::size_t prev = T, hist = 0;
        double e = 0, w = 0;
        for (std::size_t t = 0; t < T; ++t) {
          if (!p.ok(t, i)) continue;
          ++hist;
          bool jump = false;
          if (prev < T) {
            const double r = std::log(p.C[at(t, i)] / p.C[at(prev, i)]);
            rets.push_back(r);
            if (rets.size() > 60) rets.erase(rets.begin());
            jump = std::fabs(std::expm1(r)) > 0.30;
            const double c = std::log(p.C[at(prev, i)]), a = c - 0.5 * std::log(p.H[at(prev, i)] * p.L[at(prev, i)]), b = c - 0.5 * std::log(p.H[at(t, i)] * p.L[at(t, i)]);
            e += (4 * a * b - e) / 64, w += (1 - w) / 64;
          }
          spread[at(t, i)] = std::min(0.2, w > 0 ? std::sqrt(std::max(0.0, e / w)) : 0.0);
          vals.push_back(p.C[at(t, i)] * p.V[at(t, i)] / priceUnit);  // the UK's pence to pounds
          if (vals.size() > 20) vals.erase(vals.begin());
          barDays.push_back(t);
          if (barDays.size() > 18) barDays.erase(barDays.begin());
          prev = t;
          if (rets.size() >= 40) {
            double s = 0, s2 = 0;
            for (double r : rets) s += r, s2 += r * r;
            const double n = double(rets.size()), m = s / n;
            sigma[at(t, i)] = std::sqrt(std::max(0.0, s2 / n - m * m));
          }
          if (vals.size() == 20) {
            std::vector<double> v = vals;
            std::nth_element(v.begin(), v.begin() + 10, v.end());
            adv[at(t, i)] = v[10];
          }
          const bool recent = barDays.size() == 18 && t - barDays.front() < 20;  // 18 bars within the last 20 sessions
          eligible[at(t, i)] = segOk[i] && recent && !jump && hist >= 250 && p.C[at(t, i)] >= o.minPrice && sigma[at(t, i)] > 0 && sigma[at(t, i)] <= o.maxVol && adv[at(t, i)] >= o.minValue &&
                               p.H[at(t, i)] > p.L[at(t, i)];
        }
      }
    };
    std::vector<std::thread> th;
    for (unsigned k = 0; k < std::max(1u, std::thread::hardware_concurrency()); ++k) th.emplace_back(work);
    for (auto& x : th) x.join();
  }
  std::vector<std::vector<std::size_t>> uni(T);
  for (std::size_t t = 0; t < T; ++t) {
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t i = 0; i < N; ++i)
      if (eligible[at(t, i)]) v.emplace_back(adv[at(t, i)], i);
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    for (std::size_t r = 0; r < std::min(o.top, v.size()); ++r) uni[t].push_back(v[r].second);
    std::sort(uni[t].begin(), uni[t].end());
  }

  // The inputs at each close, ranked across the day's shares, and the targets.
  auto sig = swing::signals(p);
  // Research switches (shadows in config/markets.json; the live strategies run without them):
  //   X_RESID=1   residual reversal (Blitz, Huij, Lansdorp and Verbeek 2013; Da, Liu and Schaumburg 2014): the
  //               1, 5 and 20-day loss less the share's market beta times the market's (beta over 60 sessions;
  //               the market: the mean return of the shares with a bar that day), so a share that fell with
  //               the market is not taken for one that fell on its own.
  //   X_VOLREGIME=1  reversal scaled by the market's volatility (Nagel 2012, "Evaporating Liquidity": reversal pays
  //               the providers of liquidity most when volatility is high): the 1 and 5-day loss ranks times
  //               the market's 20-session volatility over its median of the year before, less 1.
  //   X_TARGET=H  the fit's target the hold's own (the open after the close to the H-th close, per its sigma
  //               times the root of H) in place of the next session's open to close (Garleanu and Pedersen
  //               2013: with costs, a signal is worth what it earns over the time it is held).
  const bool xResid = std::getenv("X_RESID") && std::string(std::getenv("X_RESID")) == "1";
  const bool xRegime = std::getenv("X_VOLREGIME") && std::string(std::getenv("X_VOLREGIME")) == "1";
  const std::size_t xTarget = std::getenv("X_TARGET") ? std::min<std::size_t>(5, std::max<std::size_t>(1, std::size_t(std::atol(std::getenv("X_TARGET"))))) : 1;
  std::vector<double> mkt(T, 0.0), mbeta(T * N, kNaN), regime(T, 0.0);
  if (xResid || xRegime) {
    for (std::size_t t = 1; t < T; ++t) {  // the market's return at close t: the shares with a bar on t and t-1
      double s = 0;
      std::size_t n = 0;
      for (std::size_t i = 0; i < N; ++i) {
        const double r = p.ret(t, i, 1);
        if (std::isfinite(r) && std::fabs(r) < 0.5) s += r, ++n;
      }
      mkt[t] = n ? s / double(n) : 0.0;
    }
  }
  if (xResid) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t t = 61; t < T; ++t) {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        std::size_t n = 0;
        for (std::size_t u = t - 59; u <= t; ++u) {
          const double r = p.ret(u, i, 1);
          if (!std::isfinite(r)) continue;
          sx += mkt[u], sy += r, sxx += mkt[u] * mkt[u], sxy += mkt[u] * r, ++n;
        }
        if (n < 40) continue;
        const double vx = sxx / double(n) - (sx / double(n)) * (sx / double(n));
        if (vx > 0) mbeta[at(t, i)] = (sxy / double(n) - sx * sy / double(n) / double(n)) / vx;
      }
    const auto resid = [&p, &mkt, &mbeta, N](std::size_t t, std::size_t i, std::size_t k) {
      const double r = p.ret(t, i, k), b = mbeta[t * N + i];
      if (!std::isfinite(r) || !std::isfinite(b) || t < k) return kNaN;
      double m = 1;
      for (std::size_t u = t - k + 1; u <= t; ++u) m *= 1 + mkt[u];
      return -(r - b * (m - 1));
    };
    sig.push_back({"residual loser 1d", [resid](std::size_t t, std::size_t i) { return resid(t, i, 1); }});
    sig.push_back({"residual loser 5d", [resid](std::size_t t, std::size_t i) { return resid(t, i, 5); }});
    sig.push_back({"residual loser 20d", [resid](std::size_t t, std::size_t i) { return resid(t, i, 20); }});
  }
  std::vector<std::size_t> regimeOf;  // the signals whose rank is scaled by the market's volatility
  if (xRegime) {
    std::vector<double> mv(T, kNaN);
    for (std::size_t t = 20; t < T; ++t) {
      double s = 0, s2 = 0;
      for (std::size_t u = t - 19; u <= t; ++u) s += mkt[u], s2 += mkt[u] * mkt[u];
      mv[t] = std::sqrt(std::max(0.0, s2 / 20 - (s / 20) * (s / 20)));
    }
    for (std::size_t t = 270; t < T; ++t) {
      std::vector<double> past(mv.begin() + long(t - 250), mv.begin() + long(t) + 1);
      std::nth_element(past.begin(), past.begin() + 125, past.end());
      regime[t] = past[125] > 0 ? std::clamp(mv[t] / past[125] - 1, -1.0, 3.0) : 0.0;
    }
    const std::size_t base = sig.size();
    sig.push_back({"loser 1d x market volatility", [&p](std::size_t t, std::size_t i) { return -p.ret(t, i, 1); }});
    sig.push_back({"loser 5d x market volatility", [&p](std::size_t t, std::size_t i) { return -p.ret(t, i, 5); }});
    regimeOf = {base, base + 1};
  }
  const std::size_t S = sig.size(), F = S + 3, D = F + 1;  // signals, volatility, liquidity, stamp duty; the intercept first
  std::vector<std::vector<double>> X(T), Y(T);
  {
    std::atomic<std::size_t> next{0};
    auto work = [&] {
      for (std::size_t t; (t = next++) < T;) {
        const auto& u = uni[t];
        const std::size_t n = u.size();
        if (n < 20) continue;
        std::vector<double> col(n);
        X[t].assign(n * D, 0.0), Y[t].assign(n, kNaN);
        for (std::size_t j = 0; j < n; ++j) X[t][j * D] = 1.0;
        for (std::size_t f = 0; f < F; ++f) {
          for (std::size_t j = 0; j < n; ++j) {
            const std::size_t i = u[j];
            col[j] = f < S ? sig[f].f(t, i) : f == S ? std::log(sigma[at(t, i)]) : f == S + 1 ? std::log(adv[at(t, i)]) : (stampFree[i] ? 1.0 : 0.0);
          }
          rankInPlace(col);
          const double scale = std::find(regimeOf.begin(), regimeOf.end(), f) != regimeOf.end() ? regime[t] : 1.0;
          for (std::size_t j = 0; j < n; ++j) X[t][j * D + 1 + f] = col[j] * scale;
        }
        if (t + xTarget < T)
          for (std::size_t j = 0; j < n; ++j) {
            const std::size_t i = u[j];
            if (p.ok(t + xTarget, i) && p.hasOpen(t + 1, i))
              Y[t][j] = std::clamp((p.C[at(t + xTarget, i)] / p.O[at(t + 1, i)] - 1) / (sigma[at(t, i)] * std::sqrt(double(xTarget))), -4.0, 4.0);
          }
      }
    };
    std::vector<std::thread> th;
    for (unsigned k = 0; k < std::max(1u, std::thread::hardware_concurrency()); ++k) th.emplace_back(work);
    for (auto& x : th) x.join();
  }
  // Each day's sums for the ridge (prefix sums: a window's in one subtraction).
  const std::size_t G = D * D + D + 1;
  std::vector<double> cum((T + 1) * G, 0.0);
  for (std::size_t t = 0; t < T; ++t) {
    double* c = &cum[(t + 1) * G];
    const double* c0 = &cum[t * G];
    for (std::size_t g = 0; g < G; ++g) c[g] = c0[g];
    for (std::size_t j = 0; j < Y[t].size(); ++j) {
      if (!std::isfinite(Y[t][j])) continue;
      const double* x = &X[t][j * D];
      for (std::size_t a = 0; a < D; ++a) {
        for (std::size_t b = 0; b < D; ++b) c[a * D + b] += x[a] * x[b];
        c[D * D + a] += x[a] * Y[t][j];
      }
      c[G - 1] += 1;
    }
  }

  // Walk-forward: the coefficients refitted every o.refit sessions from the sessions settled by then.
  std::vector<std::vector<double>> pred(T);
  std::vector<double> beta;
  double fitted = 0;
  std::size_t fits = 0;
  for (std::size_t t = 0; t < T; ++t) {
    if (t % o.refit == 0 && t >= o.warm) {
      // samples u with their target known at t: u + xTarget <= t
      const std::size_t hi = t + 1 >= xTarget ? t + 1 - xTarget : 0, lo = hi > o.window ? hi - o.window : 0;
      const double n = cum[hi * G + G - 1] - cum[lo * G + G - 1];
      std::size_t daysWith = 0;
      for (std::size_t u = lo; u < hi; ++u) daysWith += !Y[u].empty();
      if (daysWith >= o.warm && n > 50.0 * double(D)) {
        std::vector<double> gram(D * D), rhs(D), b;
        for (std::size_t g = 0; g < D * D; ++g) gram[g] = cum[hi * G + g] - cum[lo * G + g];
        for (std::size_t a = 0; a < D; ++a) rhs[a] = cum[hi * G + D * D + a] - cum[lo * G + D * D + a];
        if (ridge::solve(gram, D, o.lambda * n / 12.0, rhs, 1, b)) beta = b, fitted = n, ++fits;
      }
    }
    if (beta.empty() || Y[t].empty()) continue;
    pred[t].assign(Y[t].size(), 0.0);
    for (std::size_t j = 0; j < Y[t].size(); ++j) {
      double s = 0;
      for (std::size_t a = 0; a < D; ++a) s += beta[a] * X[t][j * D + a];
      pred[t][j] = s;
    }
  }

  // The forecast's rank correlation with the outcome, each day (out of sample).
  std::vector<double> ic(T, kNaN);
  for (std::size_t u = 0; u < T; ++u) {
    if (pred[u].empty()) continue;
    std::vector<double> a, b;
    for (std::size_t j = 0; j < pred[u].size(); ++j)
      if (std::isfinite(Y[u][j])) a.push_back(pred[u][j]), b.push_back(Y[u][j]);
    if (a.size() < 20) continue;
    rankInPlace(a), rankInPlace(b);
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t q = 0; q < a.size(); ++q) sab += a[q] * b[q], saa += a[q] * a[q], sbb += b[q] * b[q];
    ic[u] = saa > 0 && sbb > 0 ? sab / std::sqrt(saa * sbb) : kNaN;
  }

  // The levels' grid: a stop s daily volatilities below the entry, fixed, trailing (s below the highest
  // price since the fill) or moved to the entry once the price has risen one volatility (break-even);
  // a take-profit k above the entry, or none. The base: the fixed 4-volatility stop alone.
  static const double kStop[] = {1, 1.5, 2, 3, 4}, kTake[] = {1, 1.5, 2, 3, kNaN};
  static const char* const kMode[] = {"fixed", "trailing", "break-even"};
  constexpr std::size_t nS = 5, nK = 5, nM = 3, nL = nM * nS * nK, kBase = 4 * nK + 4;
  const auto modeOf = [](std::size_t g) { return g / (nS * nK); };
  const auto stopOf = [](std::size_t g) { return kStop[(g / nK) % nS]; };
  const auto takeOf = [](std::size_t g) { return kTake[g % nK]; };
  const auto levelName = [&](std::size_t g) {
    char b[64];
    if (std::isfinite(takeOf(g))) std::snprintf(b, sizeof b, "%s %g/%g", kMode[modeOf(g)], stopOf(g), takeOf(g));
    else std::snprintf(b, sizeof b, "%s %g/none", kMode[modeOf(g)], stopOf(g));
    return std::string(b);
  };

  // The 5-minute bars (--bars date,ticker,time,high,low,close[,volume]) of the share-days the levels are
  // tested on (the day after each close, for the day's highest 2/5 of forecasts): each a path of prices
  // in time order, every bar's low before its high (the order within a bar is not known: the stop first).
  std::vector<float> pathPts;
  std::map<std::size_t, std::pair<std::size_t, std::size_t>> pathOf;  // (day, share) -> (offset, points)
  std::size_t pathDays = 0, pathBad = 0;
  if (!o.barsCsv.empty()) {
    std::vector<char> want(T * N, 0);
    for (std::size_t u = 0; u + 1 < T; ++u) {
      if (pred[u].empty()) continue;
      std::vector<std::pair<double, std::size_t>> v;
      for (std::size_t j = 0; j < pred[u].size(); ++j) v.emplace_back(pred[u][j], uni[u][j]);
      std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
      for (std::size_t q = 0; q < v.size() * 2 / 5; ++q) want[at(u + 1, v[q].second)] = 1;
    }
    std::map<std::string, std::size_t> di, ti;
    for (std::size_t t = 0; t < T; ++t) di[p.days[t]] = t;
    for (std::size_t i = 0; i < N; ++i) ti[p.tickers[i]] = i;
    std::map<std::size_t, std::vector<std::pair<std::string, std::pair<float, float>>>> raw;
    std::ifstream in(o.barsCsv);
    std::string l;
    std::getline(in, l);
    while (std::getline(in, l)) {
      std::size_t c[6], k = 0;
      for (std::size_t q = 0; q < l.size() && k < 6; ++q)
        if (l[q] == ',') c[k++] = q;
      if (k < 5) continue;
      const auto d = di.find(l.substr(0, std::min<std::size_t>(c[0], 10))), tk = ti.find(l.substr(c[0] + 1, c[1] - c[0] - 1));
      if (d == di.end() || tk == ti.end() || !want[at(d->second, tk->second)]) continue;
      const float h = float(std::atof(l.c_str() + c[2] + 1)), lo = float(std::atof(l.c_str() + c[3] + 1));
      if (!(h > 0) || !(lo > 0) || h < lo) continue;
      raw[at(d->second, tk->second)].push_back({l.substr(c[1] + 1, c[2] - c[1] - 1), {lo, h}});
    }
    for (auto& [key, bars] : raw) {
      const std::size_t t = key / N, i = key % N;
      std::sort(bars.begin(), bars.end());
      // in the daily bar's scale (the bridge's recordings are IBKR's prices: an LSE share in pence or
      // pounds), and the same prices as the daily bar: not a stale or mis-scaled day
      double scale = 1;
      if (!bars.empty() && p.O[key] > 0)
        for (double sc : {1.0, 100.0, 0.01})
          if (std::fabs(std::log(p.O[key] / (sc * double(bars.front().second.first)))) < 0.1) scale = sc;
      if (scale != 1)
        for (auto& b : bars) b.second.first = float(b.second.first * scale), b.second.second = float(b.second.second * scale);
      if (bars.size() < 10 || !(p.O[key] > 0) || std::fabs(std::log(double(bars.front().second.first) / p.O[key])) > 0.1 ||
          std::fabs(std::log(double(bars.back().second.second) / p.C[at(t, i)])) > 0.1) {
        ++pathBad;
        continue;
      }
      pathOf[key] = {pathPts.size(), 2 * bars.size()};
      for (const auto& b : bars) pathPts.push_back(b.second.first), pathPts.push_back(b.second.second);
      ++pathDays;
    }
  }
  struct Exit { int code; double px, slip; };
  // A trade on a path of prices: stopped when a price reaches the stop (a stop sells into the bid), out at
  // the take-profit when a price trades through it, else at the close; the stop moves up after each price.
  const auto simulate = [&](const float* pts, std::size_t m, double C, double entry, double sg, double hs, std::size_t g) {
    const double sSd = stopOf(g), kSd = takeOf(g);
    const std::size_t mode = modeOf(g);
    double stop = entry * (1 - sSd * sg), hw = entry;
    const double take = std::isfinite(kSd) ? entry * (1 + kSd * sg) : kNaN;
    for (std::size_t q = 0; q < m; ++q) {
      const double x = pts[q];
      if (x <= stop) return Exit{1, stop, hs};
      if (std::isfinite(take) && x >= take * (1 + hs)) return Exit{2, take, 0.0};
      if (x > hw) {
        hw = x;
        if (mode == 1) stop = std::max(stop, hw * (1 - sSd * sg));
        if (mode == 2 && hw >= entry * (1 + sg)) stop = std::max(stop, entry);
      }
    }
    return Exit{0, C, 0.0};
  };
  // The exit of a share bought at `entry` in the opening auction of session d with the levels g: on its
  // 5-minute path where there is one; else on the daily bar both ways (the low first, the high first),
  // the worse taken, as the bar cannot tell.
  const auto exitOf = [&](std::size_t d, std::size_t i, double entry, double sg, double spr, std::size_t g) {
    const double L = p.L[at(d, i)], H = p.H[at(d, i)], C = p.C[at(d, i)], hs = 0.5 * spr;
    const auto it = pathOf.find(at(d, i));
    if (it != pathOf.end()) return simulate(&pathPts[it->second.first], it->second.second, C, entry, sg, hs, g);
    const float lowFirst[2] = {float(L), float(H)}, highFirst[2] = {float(H), float(L)};
    const Exit a = simulate(lowFirst, 2, C, entry, sg, hs, g), b = simulate(highFirst, 2, C, entry, sg, hs, g);
    return a.px * (1 - a.slip) <= b.px * (1 - b.slip) ? a : b;
  };
  // Each session, the day's highest fifth of forecasts bought at the open (when the auction was under
  // the limit), as a triple barrier (Lopez de Prado 2018: the stop, the take-profit, the close): every
  // pair's mean result that day and its difference from the base, and for the calibration each trade's
  // forecast against its result under each pair, in the share's daily volatilities.
  std::vector<double> lvD(T + 1, 0.0);
  std::vector<double> lvS((T + 1) * nL, 0.0), lvS2((T + 1) * nL, 0.0), lvR((T + 1) * nL, 0.0);
  std::vector<std::array<double, 6>> calG((T + 1) * nL, {0, 0, 0, 0, 0, 0});
  for (std::size_t u = 0; u < T; ++u) {
    lvD[u + 1] = lvD[u];
    for (std::size_t g = 0; g < nL; ++g)
      lvS[(u + 1) * nL + g] = lvS[u * nL + g], lvS2[(u + 1) * nL + g] = lvS2[u * nL + g], lvR[(u + 1) * nL + g] = lvR[u * nL + g], calG[(u + 1) * nL + g] = calG[u * nL + g];
    if (pred[u].empty() || u + 1 >= T) continue;
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t j = 0; j < pred[u].size(); ++j) v.emplace_back(pred[u][j], j);
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    double sum[nL] = {}, n = 0;
    std::array<double, 6> day[nL] = {};
    for (std::size_t q = 0; q < v.size() / 5; ++q) {
      const std::size_t i = uni[u][v[q].second];
      if (!p.ok(u + 1, i) || !p.hasOpen(u + 1, i)) continue;
      const double O = p.O[at(u + 1, i)], sg = sigma[at(u, i)], spr = spread[at(u, i)], x = v[q].first;
      if (!(O <= p.C[at(u, i)] * (1 + o.capSd * sg))) continue;
      for (std::size_t g = 0; g < nL; ++g) {
        const Exit e = exitOf(u + 1, i, O, sg, spr, g);
        const double r = e.px / O - 1 - e.slip, y = std::clamp(r / sg, -4.0, 4.0);
        sum[g] += r;
        auto& c = day[g];
        c[0] += 1, c[1] += x, c[2] += y, c[3] += x * x, c[4] += x * y, c[5] += y * y;
      }
      n += 1;
    }
    if (n < 3) continue;
    lvD[u + 1] += 1;
    for (std::size_t g = 0; g < nL; ++g) {
      const double r = sum[g] / n, dlt = r - sum[kBase] / n;
      lvS[(u + 1) * nL + g] += dlt, lvS2[(u + 1) * nL + g] += dlt * dlt, lvR[(u + 1) * nL + g] += r;
      for (int k = 0; k < 6; ++k) calG[(u + 1) * nL + g][k] += day[g][k];
    }
  }
  // The pair for the session after the close t (the sessions settled by then) and what it added a trade.
  const auto levelsAt = [&](std::size_t t, double& adj) {
    adj = 0;
    const std::size_t hi = t, lo = t > 250 ? t - 250 : 0;
    const double n = lvD[hi] - lvD[lo];
    if (n < 60) return kBase;
    std::size_t best = kBase;
    for (std::size_t g = 0; g < nL; ++g) {
      const double m = (lvS[hi * nL + g] - lvS[lo * nL + g]) / n, v = (lvS2[hi * nL + g] - lvS2[lo * nL + g]) / n - m * m;
      if (!(m > 0) || !(v > 0)) continue;
      const double ts = m / std::sqrt(v / n), sc = m * std::max(0.0, 1.0 - 4.0 / (ts * ts));
      if (sc > adj) adj = sc, best = g;
    }
    return best;
  };
  // The calibration known at the close t for the pair g (outcomes of the sessions up to t): the result
  // of the trade as placed (its triple barrier), a + b x in daily volatilities, b shrunk by its t statistic.
  const auto calibration = [&](std::size_t t, std::size_t g, double& a, double& b) {
    const std::size_t hi = t, lo = t > o.calWindow ? t - o.calWindow : 0;  // days u in [lo, hi): outcome at u + 1 <= t
    if (std::size_t(lvD[hi] - lvD[lo]) < o.calMin) return false;
    double c[6];
    for (int k = 0; k < 6; ++k) c[k] = calG[hi * nL + g][k] - calG[lo * nL + g][k];
    const double n = c[0], mx = c[1] / n, my = c[2] / n, sxx = c[3] - n * mx * mx, sxy = c[4] - n * mx * my, syy = c[5] - n * my * my;
    if (!(n > 100) || !(sxx > 0)) return false;
    double slope = sxy / sxx;
    const double s2 = std::max(0.0, syy - slope * sxy) / (n - 2), ts = s2 > 0 ? slope / std::sqrt(s2 / sxx) : 0.0;  // its t statistic
    slope = ts > 0 ? std::min(1.5, slope) * std::max(0.0, 1.0 - 1.0 / (ts * ts)) : 0.0;
    b = slope, a = my - slope * mx;
    return true;
  };

  // --hold H: bought in the opening auction and sold at the first close at least theta daily volatilities
  // above the price paid, else at the H-th close (as few days as the trade needs, never more than H). theta
  // is chosen each day from a grid by the record of the day's highest fifth of forecasts: their return
  // per day held against theta 0 (the first close above the entry), shrunk by its t statistic; the
  // expected return is calibrated on the whole trade, a + b x in daily volatilities. A forecast day's
  // trades are all known H sessions later, so the record at the close t holds the days up to t - H.
  static const double kTh[] = {0, 0.25, 0.5, 1, 1.5};
  constexpr std::size_t nTh = 5;
  const std::size_t H = std::max<std::size_t>(1, o.hold);
  struct HoldTrade { bool ok = false; double r = 0; std::size_t days = 0, exitDay = 0; bool atTarget = false; };
  const auto holdTrade = [&](std::size_t u, std::size_t i, double entry, double sg, double th) {
    HoldTrade h;
    double last = entry;
    for (std::size_t k = 0; k < H; ++k) {
      const std::size_t d = u + 1 + k;
      if (d >= T) return h;  // not settled yet
      if (p.ok(d, i)) last = p.C[at(d, i)];
      const bool target = p.ok(d, i) && last >= entry * (1 + th * sg);
      if (target || k + 1 == H) {
        h.ok = true, h.r = last / entry - 1, h.days = k + 1, h.exitDay = d, h.atTarget = target;
        return h;
      }
    }
    return h;
  };
  std::vector<double> hD(T + 1, 0.0), hS((T + 1) * nTh, 0.0), hS2((T + 1) * nTh, 0.0), hR((T + 1) * nTh, 0.0), hDays((T + 1) * nTh, 0.0);
  std::vector<std::array<double, 6>> calH((T + 1) * nTh, {0, 0, 0, 0, 0, 0});
  if (o.hold)
    for (std::size_t u = 0; u < T; ++u) {
      hD[u + 1] = hD[u];
      for (std::size_t g = 0; g < nTh; ++g) {
        const std::size_t a = (u + 1) * nTh + g, b = u * nTh + g;
        hS[a] = hS[b], hS2[a] = hS2[b], hR[a] = hR[b], hDays[a] = hDays[b], calH[a] = calH[b];
      }
      if (pred[u].empty()) continue;
      std::vector<std::pair<double, std::size_t>> v;
      for (std::size_t j = 0; j < pred[u].size(); ++j) v.emplace_back(pred[u][j], j);
      std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
      double perDay[nTh] = {}, sumR[nTh] = {}, sumD[nTh] = {}, n = 0;
      std::array<double, 6> day[nTh] = {};
      bool settled = true;
      for (std::size_t q = 0; q < v.size() / 5 && settled; ++q) {
        const std::size_t i = uni[u][v[q].second];
        if (u + 1 >= T || !p.ok(u + 1, i) || !p.hasOpen(u + 1, i)) continue;
        const double O = p.O[at(u + 1, i)], sg = sigma[at(u, i)], x = v[q].first;
        if (!(O <= p.C[at(u, i)] * (1 + o.capSd * sg))) continue;
        for (std::size_t g = 0; g < nTh && settled; ++g) {
          const HoldTrade h = holdTrade(u, i, O, sg, kTh[g]);
          if (!h.ok) { settled = false; break; }
          perDay[g] += h.r / double(h.days), sumR[g] += h.r, sumD[g] += double(h.days);
          const double y = std::clamp(h.r / sg, -6.0, 6.0);
          auto& c = day[g];
          c[0] += 1, c[1] += x, c[2] += y, c[3] += x * x, c[4] += x * y, c[5] += y * y;
        }
        n += 1;
      }
      if (!settled || n < 3) continue;
      hD[u + 1] += 1;
      for (std::size_t g = 0; g < nTh; ++g) {
        const std::size_t a = (u + 1) * nTh + g;
        const double dlt = perDay[g] / n - perDay[0] / n;
        hS[a] += dlt, hS2[a] += dlt * dlt, hR[a] += sumR[g] / n, hDays[a] += sumD[g] / n;
        for (int k = 0; k < 6; ++k) calH[a][k] += day[g][k];
      }
    }
  // The days whose trades are all known at the close t: forecast days u < t + 1 - H.
  const auto known = [&](std::size_t t) { return t + 1 >= H ? t + 1 - H : 0; };
  const auto holdAt = [&](std::size_t t, double& adj) {
    adj = 0;
    const std::size_t hi = known(t), lo = hi > 250 ? hi - 250 : 0;
    const double n = hD[hi] - hD[lo];
    std::size_t best = 0;
    if (n < 60) return best;
    for (std::size_t g = 1; g < nTh; ++g) {
      const double m = (hS[hi * nTh + g] - hS[lo * nTh + g]) / n, v = (hS2[hi * nTh + g] - hS2[lo * nTh + g]) / n - m * m;
      if (!(m > 0) || !(v > 0)) continue;
      const double ts = m / std::sqrt(v / n), sc = m * std::max(0.0, 1.0 - 4.0 / (ts * ts));
      if (sc > adj) adj = sc, best = g;
    }
    return best;
  };
  const auto calibrationH = [&](std::size_t t, std::size_t g, double& a, double& b, double& daysHeld) {
    const std::size_t hi = known(t), lo = hi > o.calWindow ? hi - o.calWindow : 0;
    const double nd = hD[hi] - hD[lo];
    if (std::size_t(nd) < o.calMin) return false;
    double c[6];
    for (int k = 0; k < 6; ++k) c[k] = calH[hi * nTh + g][k] - calH[lo * nTh + g][k];
    const double n = c[0], mx = c[1] / n, my = c[2] / n, sxx = c[3] - n * mx * mx, sxy = c[4] - n * mx * my, syy = c[5] - n * my * my;
    if (!(n > 100) || !(sxx > 0)) return false;
    double slope = sxy / sxx;
    const double s2 = std::max(0.0, syy - slope * sxy) / (n - 2), ts = s2 > 0 ? slope / std::sqrt(s2 / sxx) : 0.0;
    slope = ts > 0 ? std::min(1.5, slope) * std::max(0.0, 1.0 - 1.0 / (ts * ts)) : 0.0;
    b = slope, a = my - slope * mx;
    daysHeld = (hDays[hi * nTh + g] - hDays[lo * nTh + g]) / nd;
    return true;
  };

  // The broker's costs of one side of an order of `value` (pounds; euros in Europe; dollars in the US) at
  // `price` a share: on the final orders only. Europe: the exchange's line (a rate, at least a minimum, a
  // further fee's rate, a tax on both sides). The US: IBKR's fixed USD 0.005 a share, at least USD 1, at
  // most 1% (tiered: USD 0.0035, at least 0.35, and about 0.0005 a share of exchange and clearing fees).
  const auto commission = [&](double value, std::size_t i, double price) {
    if (!(value > 0)) return 0.0;
    if (fx) return std::max(2.0, 0.00002 * value);  // IBKR's IDEALPRO: 0.2 basis points, at least USD 2
    if (us) {
      const double shares = price > 0 ? std::max(1.0, std::floor(value / price)) : 1.0;
      return o.tiered ? std::max(0.35, 0.0035 * shares) + 0.0005 * shares : std::min(0.01 * value, std::max(1.0, 0.005 * shares));
    }
    if (!uk) {
      const auto& v = venueCosts.venue[i];
      return std::max(v[2], v[1] * value) + (v[3] + v[5]) * value;
    }
    return o.tiered ? std::max(1.0, 0.0005 * value) + std::max(0.10, 0.000045 * value) + 0.06 : std::max(3.0, 0.0005 * value);
  };
  // The tax on buying: the UK's stamp duty (none on AIM shares); Europe's on a share held past the day
  // (the French and Spanish FTT, the Portuguese) or, within the day, its same-day one.
  const auto stamp = [&](std::size_t i) { return uk ? (stampFree[i] ? 0.0 : 0.005) : us || fx ? 0.0 : venueCosts.venue[i][o.hold ? 6 : 4]; };
  // The US regulators' fees on a sale (the SEC's USD 27.80 a million, FINRA's USD 0.000166 a share up to USD 8.30).
  const auto saleFees = [&](double value, double price) {
    if (!us || !(value > 0)) return 0.0;
    const double shares = price > 0 ? std::floor(value / price) : 0.0;
    return 0.0000278 * value + std::min(8.30, 0.000166 * shares);
  };
  const auto exchangeOf = [&](std::size_t i) {
    if (uk) return std::string(stampFree[i] ? "AIM" : "main");
    if (us || fx) {
      const auto k = p.tickers[i].find(':');
      return k == std::string::npos ? std::string(o.market) : p.tickers[i].substr(0, k);
    }
    const auto k = p.tickers[i].find(':');
    return k == std::string::npos ? std::string("?") : p.tickers[i].substr(0, k);
  };
  // Both sides' commission and the taxes. No spread: the orders buy in the opening auction and sell in
  // the closing auction (or the opening one), where buyers and sellers trade at one price, so none is crossed.
  const auto costOf = [&](const Order& r) { return (2 * commission(r.value, r.i, r.close) + saleFees(r.value, r.close)) / r.value + stamp(r.i); };

  // The orders for the session after the close t, before costs (the model and its backtest), or with
  // the broker's costs applied (withCosts: the final orders): each order's expected return less its own
  // costs, those that still gain kept, the highest expected gain in pounds first.
  const auto ordersAt = [&](std::size_t t, bool withCosts) {
    std::vector<Order> out;
    double a = 0, b = 0, lvAdj = 0, daysHeld = 0;
    const std::size_t lv = o.hold ? holdAt(t, lvAdj) : levelsAt(t, lvAdj);
    if (pred[t].empty() || !(o.hold ? calibrationH(t, lv, a, b, daysHeld) : calibration(t, lv, a, b))) return out;
    const auto& u = uni[t];
    for (std::size_t j = 0; j < u.size(); ++j) {
      const std::size_t i = u[j];
      Order r;
      r.i = i, r.close = p.C[at(t, i)], r.sigma = sigma[at(t, i)], r.spread = spread[at(t, i)];
      r.expected = (a + b * pred[t][j]) * r.sigma;  // the trade as placed: bought at the open, out at its stop, its take-profit or the close
      r.value = noVolume ? o.capital : std::min(o.capital, o.advShare * adv[at(t, i)]);  // FOREX, INDEX: no volume to cap it by
      r.capped = r.value < o.capital;
      if (r.value < o.minOrder || r.expected < o.hurdle) continue;
      r.limit = r.close * (1 + o.capSd * r.sigma);
      r.level = lv, r.adj = lvAdj, r.stopSd = stopOf(lv), r.takeSd = takeOf(lv), r.mode = modeOf(lv);
      if (o.hold) r.stopSd = kNaN, r.takeSd = kNaN, r.mode = 0, r.exitAbove = kTh[lv] * r.sigma, r.holdDays = daysHeld;
      r.stop = o.hold ? kNaN : r.close * (1 - r.stopSd * r.sigma);  // shown from the close; placed from the price bought at
      r.cost = withCosts ? costOf(r) : 0.0;
      r.net = r.expected - r.cost;
      if (withCosts && !(r.net > 0)) continue;  // its costs take all it is expected to gain: no order
      out.push_back(r);
    }
    std::sort(out.begin(), out.end(), [](const Order& x, const Order& y) { return x.net * x.value > y.net * y.value || (x.net * x.value == y.net * y.value && x.i < y.i); });
    if (out.size() > o.slots) out.resize(o.slots);
    for (auto& r : out) {  // how often its opening auction would have bought over the share's last 250 sessions
      double n = 0, f = 0;
      for (std::size_t q = t > 250 ? t - 250 : 0; q < t; ++q) {
        if (!p.ok(q, r.i) || !p.ok(q + 1, r.i) || !(sigma[at(q, r.i)] > 0)) continue;
        const double lim = p.C[at(q, r.i)] * (1 + o.capSd * sigma[at(q, r.i)]);
        n += 1, f += p.hasOpen(q + 1, r.i) && p.O[at(q + 1, r.i)] <= lim;
      }
      r.fill = n > 0 ? f / n : kNaN;
    }
    return out;
  };
  // An order's session (the bars of t + 1), before costs; f.cost: what the broker would have charged
  // (reported, never taken off).
  const auto play = [&](std::size_t t, const Order& r) {
    Fill f;
    const std::size_t i = r.i, d = t + 1;
    if (d >= T || !p.ok(d, i) || !p.hasOpen(d, i)) return f;
    const double O = p.O[at(d, i)];
    if (!(O <= r.limit)) return f;  // the auction above the limit: not bought (the order expires with the auction)
    f.filled = 1, f.entry = O;
    const Exit e = exitOf(d, i, O, r.sigma, r.spread, r.level);
    f.onPath = pathOf.count(at(d, i)) > 0;
    f.exit = e.code, f.exitPx = e.px;
    f.gross = f.exitPx / f.entry - 1 - e.slip;
    f.cost = costOf(r);
    f.ret = f.gross;
    return f;
  };

  // The backtest: every session's orders from the close before, traded on the session's bars.
  std::FILE* trace = o.traceOut.empty() ? nullptr : std::fopen(o.traceOut.c_str(), "w");
  std::FILE* tlog = std::getenv("X_TRADELOG") ? std::fopen(std::getenv("X_TRADELOG"), "w") : nullptr;
  if (tlog) std::fprintf(tlog, "date,ticker,close,limit,stopRel,takeRel,value,expected,cost,filled,entry,exit,exitPrice,gross,ret\n");
  std::vector<double> ret;
  std::vector<std::string> days;
  double nOrders = 0, nFilled = 0, nOnPath = 0, nAuction = 0, nStopped = 0, nTaken = 0, nCapped = 0, sumGross = 0, sumCost = 0, sumNet = 0, nWin = 0, sumValue = 0, sumExpected = 0;
  std::size_t daysWithOrders = 0, firstDay = T, levelDays[nL] = {};
  std::map<std::string, std::pair<std::vector<double>, double>> byYear;  // each year's daily returns, IC sum
  std::vector<double> baseline;  // every share of the day's list, bought at the open and sold at the close (gross): the market the orders come from
  const auto traceOrder = [&](std::size_t t, const Order& r) {
    if (trace) std::fprintf(trace, "%s,%s,%.6g,%zu,%.6g,%.6g\n", p.days[t].c_str(), p.tickers[r.i].c_str(), r.limit, r.level, r.value, r.net);
  };
  struct Check { std::size_t i; Order r; Fill f; };
  std::vector<Check> lastCheck;
  double sumHeld = 0, nDropped = 0;  // --hold: the days the trades were held; the trades the daily update sold at the open
  if (o.hold) {
    // The book: at most --slots positions; each morning the day's orders take the free slots (shares not
    // held already), bought in the opening auction; every position marked to each close and sold at its
    // exit level's first close, else at its H-th; an unused slot earns 0.
    struct Pos { Order r; double entry, last; std::size_t k, from; };
    std::vector<Pos> book;
    for (std::size_t t = 0; t + 1 < T; ++t) {
      double a = 0, b = 0, adj = 0, dh = 0;
      const std::size_t g = holdAt(t, adj);
      const bool live = !pred[t].empty() && calibrationH(t, g, a, b, dh);
      if (!live && book.empty()) continue;
      if (live && firstDay == T) firstDay = t + 1;
      if (live) ++levelDays[g];
      const std::size_t d = t + 1;
      std::vector<Order> ords = live ? ordersAt(t, false) : std::vector<Order>{};
      for (const auto& r : ords) traceOrder(t, r);
      std::size_t free = o.slots > book.size() ? o.slots - book.size() : 0;
      double day = 0;
      const double unit = double(o.slots) * o.capital;
      // how it ended: 0 its last day's close, 2 its exit level's close, 3 sold at the open (the model no longer wanted it)
      const auto settle = [&](const Pos& q, int how) {
        const bool target = how == 2;
        const double gr = q.last / q.entry - 1;
        nFilled += 1, nTaken += target, nDropped += how == 3, sumNet += gr, sumGross += gr, sumCost += costOf(q.r), nWin += gr > 0, sumValue += q.r.value, sumHeld += double(q.k);
        if (tlog)
          std::fprintf(tlog, "%s,%s,%.6g,%.6g,0,%.6g,%.6g,%.6g,%.6g,1,%.6g,%d,%.6g,%.6g,%.6g\n", p.days[q.from].c_str(), p.tickers[q.r.i].c_str(), q.r.close, q.r.limit, q.r.exitAbove,
                       q.r.value, q.r.expected, costOf(q.r), q.entry, how, q.last, gr, gr);
      };
      // Each morning every position held is looked at again with the last close's forecast (the active
      // update): one the model would no longer buy (its expected return as a new trade, before costs: the
      // costs paid are spent) is sold in the opening auction; the others keep, their exit level set again
      // from the day's (theta times the share's volatility now, above the price paid).
      if (live)
        for (std::size_t q = 0; q < book.size();) {
          Pos& ps = book[q];
          const auto it = std::lower_bound(uni[t].begin(), uni[t].end(), ps.r.i);
          if (it == uni[t].end() || *it != ps.r.i || !(sigma[at(t, ps.r.i)] > 0)) { ++q; continue; }  // no forecast for it: kept as planned
          const double sg = sigma[at(t, ps.r.i)], e = (a + b * pred[t][std::size_t(it - uni[t].begin())]) * sg;
          if (e > 0) {
            ps.r.exitAbove = kTh[g] * sg;
            ++q;
            continue;
          }
          ps.k += 1;
          if (p.hasOpen(d, ps.r.i)) day += ps.r.value * (p.O[at(d, ps.r.i)] / ps.last - 1) / unit, ps.last = p.O[at(d, ps.r.i)];
          settle(ps, 3), book.erase(book.begin() + std::ptrdiff_t(q));
        }
      // the positions held from earlier days: marked to this close, sold at their exit level or their last day
      for (std::size_t q = 0; q < book.size();) {
        Pos& ps = book[q];
        ps.k += 1;
        if (p.ok(d, ps.r.i)) day += ps.r.value * (p.C[at(d, ps.r.i)] / ps.last - 1) / unit, ps.last = p.C[at(d, ps.r.i)];
        const bool target = p.ok(d, ps.r.i) && ps.last >= ps.entry * (1 + ps.r.exitAbove);
        if (target || ps.k >= H) settle(ps, target ? 2 : 0), book.erase(book.begin() + std::ptrdiff_t(q));
        else ++q;
      }
      // the day's orders into the free slots (bought in the opening auction, or not at all)
      for (const auto& r : ords) {
        if (!free) break;
        bool held = false;
        for (const auto& ps : book) held |= ps.r.i == r.i;
        if (held) continue;
        --free;
        nOrders += 1, nCapped += r.capped, sumExpected += (std::isfinite(r.fill) ? r.fill : 1.0) * r.net;
        if (!p.ok(d, r.i) || !p.hasOpen(d, r.i) || !(p.O[at(d, r.i)] <= r.limit)) {
          if (tlog)
            std::fprintf(tlog, "%s,%s,%.6g,%.6g,0,%.6g,%.6g,%.6g,%.6g,0,0,0,0,0,0\n", p.days[d].c_str(), p.tickers[r.i].c_str(), r.close, r.limit, r.exitAbove, r.value, r.expected, costOf(r));
          continue;
        }
        Pos ps{r, p.O[at(d, r.i)], p.C[at(d, r.i)], 1, d};
        nAuction += 1;
        day += r.value * (ps.last / ps.entry - 1) / unit;
        const bool target = ps.last >= ps.entry * (1 + r.exitAbove);
        if (target || H == 1) settle(ps, target ? 2 : 0);
        else book.push_back(ps);
      }
      daysWithOrders += !ords.empty();
      ret.push_back(day), days.push_back(p.days[d]);
      auto& y = byYear[p.days[d].substr(0, 4)];
      y.first.push_back(day);
      if (std::isfinite(ic[t])) y.second += ic[t];
      double bs = 0, bn = 0;
      for (std::size_t i : uni[t])
        if (p.ok(d, i) && p.hasOpen(d, i)) bs += p.C[at(d, i)] / p.O[at(d, i)] - 1, bn += 1;
      baseline.push_back(bn > 0 ? bs / bn : 0.0);
    }
  }
  for (std::size_t t = 0; t + 1 < T && !o.hold; ++t) {
    const auto ords = ordersAt(t, false);
    double a = 0, b = 0, adj = 0;
    const std::size_t lvNow = levelsAt(t, adj);
    const bool live = !pred[t].empty() && calibration(t, lvNow, a, b);
    if (!live) continue;
    if (firstDay == T) firstDay = t + 1;
    ++levelDays[lvNow];
    double day = 0;
    lastCheck.clear();
    for (const auto& r : ords) {
      traceOrder(t, r);
      const Fill f = play(t, r);
      nOrders += 1, nCapped += r.capped, sumExpected += (std::isfinite(r.fill) ? r.fill : 1.0) * r.net;
      if (f.filled) {
        nFilled += 1, nOnPath += f.onPath, nAuction += f.filled == 1, nStopped += f.exit == 1, nTaken += f.exit == 2, sumGross += f.gross, sumCost += f.cost, sumNet += f.ret, nWin += f.ret > 0, sumValue += r.value;
        day += r.value * f.ret / (double(o.slots) * o.capital);
      }
      if (tlog)
        std::fprintf(tlog, "%s,%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,%d,%.6g,%.6g,%.6g\n", p.days[t + 1].c_str(), p.tickers[r.i].c_str(), r.close, r.limit, -r.stopSd * r.sigma,
                     std::isfinite(r.takeSd) ? r.takeSd * r.sigma : 0.0, r.value, r.expected,
                     r.cost, f.filled, f.entry, f.exit, f.exitPx, f.gross, f.ret);
      lastCheck.push_back({r.i, r, f});
    }
    daysWithOrders += !ords.empty();
    ret.push_back(day), days.push_back(p.days[t + 1]);
    auto& y = byYear[p.days[t + 1].substr(0, 4)];
    y.first.push_back(day);
    if (std::isfinite(ic[t])) y.second += ic[t];
    double bs = 0, bn = 0;
    for (std::size_t i : uni[t])
      if (p.ok(t + 1, i) && p.hasOpen(t + 1, i)) bs += p.C[at(t + 1, i)] / p.O[at(t + 1, i)] - 1, bn += 1;
    baseline.push_back(bn > 0 ? bs / bn : 0.0);
  }
  if (tlog) {
    std::fclose(tlog);
    if (std::FILE* f = std::fopen((std::string(std::getenv("X_TRADELOG")) + ".days").c_str(), "w")) {
      std::fprintf(f, "date,ret\n");
      for (std::size_t q = 0; q < ret.size(); ++q) std::fprintf(f, "%s,%.8g\n", days[q].c_str(), ret[q]);
      std::fclose(f);
    }
  }
  // The next open's orders, from the last close: the model's (before costs, as every day of the backtest;
  // traced, for the look-ahead check), then the final orders with the broker's costs applied.
  for (const auto& r : ordersAt(T - 1, false)) traceOrder(T - 1, r);
  const auto next = ordersAt(T - 1, true);
  if (trace) std::fclose(trace);

  struct Stats { double annual = 0, sharpe = 0, dd = 0, mean = 0; };
  const auto stats = [](const std::vector<double>& x) {
    Stats s;
    if (x.empty()) return s;
    double w = 0, s1 = 0, s2 = 0, peak = 0;
    for (double y : x) {
      w += std::log1p(y), s1 += y, s2 += y * y;
      peak = std::max(peak, w), s.dd = std::max(s.dd, 1 - std::exp(w - peak));
    }
    const double n = double(x.size()), m = s1 / n, sd = std::sqrt(std::max(0.0, s2 / n - m * m));
    s.annual = std::expm1(w * 252.0 / n), s.sharpe = sd > 0 ? m / sd * std::sqrt(252.0) : 0.0, s.mean = m;
    return s;
  };
  // The probabilistic Sharpe ratio (Bailey and Lopez de Prado 2012): the chance the true daily Sharpe
  // is above 0, given the record's length, skew and fat tails.
  const auto psr = [](const std::vector<double>& x) {
    const double n = double(x.size());
    if (n < 30) return 0.0;
    double m = 0, m2 = 0, m3 = 0, m4 = 0;
    for (double y : x) m += y / n;
    for (double y : x) { const double d = y - m; m2 += d * d / n, m3 += d * d * d / n, m4 += d * d * d * d / n; }
    if (!(m2 > 0)) return 0.0;
    const double sr = m / std::sqrt(m2), sk = m3 / std::pow(m2, 1.5), ku = m4 / (m2 * m2);
    const double den = std::sqrt(std::max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr * sr));
    return 0.5 * std::erfc(-sr * std::sqrt(n - 1) / den / std::sqrt(2.0));
  };
  const Stats all = stats(ret), base = stats(baseline);
  std::size_t uniMean = 0;
  for (const auto& u : uni) uniMean += u.size();
  std::printf("uk: %zu shares, %zu sessions (%s .. %s); %zu in the exchange's list, %zu of them outside the order book's auctions (funds, ETFs, SETSqx): left out\n", N, T,
              p.days.front().c_str(), p.days.back().c_str(), segKnown, segOut);
  std::printf("uk: the day's list: %.0f shares on average (at most %zu; at least %s %.0f traded a day); %zu fits of %zu inputs (the last on %.0f share-days)\n",
              double(uniMean) / double(T), o.top, cur, o.minValue, fits, F, fitted);
  if (ret.empty()) {
    std::printf("uk: no out-of-sample record yet (at least %zu sessions to fit and %zu to calibrate): no orders\n", o.warm, o.calMin);
  } else {
    std::printf("uk: the account (%s .. %s, %zu slots of %s %.0f): %+.1f%% a year, Sharpe %.2f, max drawdown %.1f%%, before the broker's costs; "
                "the chance its true Sharpe is above 0: %.1f%% (probabilistic Sharpe ratio, Bailey and Lopez de Prado 2012; %zu %s tried)\n",
                days.front().c_str(), days.back().c_str(), o.slots, cur, o.capital, 100 * all.annual, all.sharpe, 100 * all.dd, 100 * psr(ret), o.hold ? nTh : nL,
                o.hold ? "exit levels" : "stop / take-profit pairs");
    std::printf("uk:   for scale: every share of the day's list bought at the open and sold at the close %+.1f%% a year (Sharpe %.2f), before costs\n", 100 * base.annual, base.sharpe);
    if (o.hold) {
      std::printf("uk:   held up to %zu sessions: orders %.0f on %zu of %zu sessions (%.2f a day, into the book's free slots); bought in the opening auction %.1f%%; "
                  "trades %.0f, held %.2f sessions on average, %.1f%% sold at their exit level's first close, %.1f%% at the open by the daily update, the rest at their last close; "
                  "%.0f%% cut to the share's liquidity\n",
                  H, nOrders, daysWithOrders, ret.size(), nOrders / double(ret.size()), nOrders > 0 ? 100 * nAuction / nOrders : 0.0, nFilled, nFilled > 0 ? sumHeld / nFilled : 0.0,
                  nFilled > 0 ? 100 * nTaken / nFilled : 0.0, nFilled > 0 ? 100 * nDropped / nFilled : 0.0, nOrders > 0 ? 100 * nCapped / nOrders : 0.0);
      std::printf("uk:   exit levels used (the first close this many daily volatilities above the price paid):");
      for (std::size_t g = 0; g < nTh; ++g)
        if (levelDays[g]) std::printf(" %g on %zu sessions;", kTh[g], levelDays[g]);
      std::printf("\n");
      const std::size_t lo = firstDay > 0 ? firstDay - 1 : 0, hi = known(T - 1);
      const double n = hi > lo ? hD[hi] - hD[lo] : 0;
      if (n > 1) {
        std::printf("uk:   each exit level over the backtest (the day's top fifth): a trade, its days held, per day:");
        for (std::size_t g = 0; g < nTh; ++g)
          std::printf("%s %g: %+.3f%%, %.2f days, %+.3f%% a day", g ? ";" : "", kTh[g], 100 * (hR[hi * nTh + g] - hR[lo * nTh + g]) / n, (hDays[hi * nTh + g] - hDays[lo * nTh + g]) / n,
                      100 * (hR[hi * nTh + g] - hR[lo * nTh + g]) / std::max(1e-9, hDays[hi * nTh + g] - hDays[lo * nTh + g]));
        std::printf("\n");
      }
    } else {
    std::printf("uk:   orders: %.0f on %zu of %zu sessions (%.2f a day); bought in the opening auction %.1f%%; of the trades %.1f%% stopped, %.1f%% took profit, the rest sold at the close; %.0f%% cut to the share's liquidity\n",
                nOrders, daysWithOrders, ret.size(), nOrders / double(ret.size()), nOrders > 0 ? 100 * nFilled / nOrders : 0.0, nFilled > 0 ? 100 * nStopped / nFilled : 0.0,
                nFilled > 0 ? 100 * nTaken / nFilled : 0.0, nOrders > 0 ? 100 * nCapped / nOrders : 0.0);
    // The levels: which pair each session used (chosen walk-forward), and every pair's record over the
    // whole period (for reading only: the choice each day saw only the 250 sessions before it).
    std::printf("uk:   levels used (stop / take-profit, daily volatilities from the entry):");
    for (std::size_t g = 0; g < nL; ++g)
      if (levelDays[g]) std::printf(" %s on %zu sessions;", levelName(g).c_str(), levelDays[g]);
    std::printf("\n");
    std::printf("uk:   5-minute bars: %zu share-days (%zu left out: too few bars or not the daily bar's prices); %.0f%% of the filled trades replayed on them, "
                "the rest on the daily bar with the worse of its two orders\n", pathDays, pathBad, nFilled > 0 ? 100 * nOnPath / nFilled : 0.0);
    {  // every pair's record over the whole period (for reading only: each day's choice saw only the 250 sessions before it), the best ten
      const std::size_t lo = firstDay > 0 ? firstDay - 1 : 0;
      const double n = lvD[T] - lvD[lo];
      if (n > 1) {
        std::vector<std::pair<double, std::size_t>> best;
        std::vector<double> mean(nL), tst(nL);
        for (std::size_t g = 0; g < nL; ++g) {
          const double m = (lvS[T * nL + g] - lvS[lo * nL + g]) / n, v = (lvS2[T * nL + g] - lvS2[lo * nL + g]) / n - m * m;
          mean[g] = m, tst[g] = v > 0 ? m / std::sqrt(v / n) : 0.0;
          if (g != kBase) best.emplace_back(m, g);
        }
        std::sort(best.rbegin(), best.rend());
        std::printf("uk:   the best levels over the backtest, added a trade against the fixed 4-volatility stop alone (t):");
        for (std::size_t q = 0; q < std::min<std::size_t>(10, best.size()); ++q)
          std::printf("%s %s %+.3f%% (%+.1f)", q ? "," : "", levelName(best[q].second).c_str(), 100 * best[q].first, tst[best[q].second]);
        for (std::size_t md = 0; md < nM; ++md) {  // each kind's best
          std::size_t bg = kBase;
          double bm = -1e9;
          for (std::size_t g = md * nS * nK; g < (md + 1) * nS * nK; ++g)
            if (mean[g] > bm) bm = mean[g], bg = g;
          std::printf("%s best %s: %s %+.3f%%", md ? ";" : "\nuk:   by kind:", kMode[md], levelName(bg).c_str(), 100 * bm);
        }
        std::printf("\n");
      }
    }
    }  // not --hold
    if (nFilled > 0)
      std::printf("uk:   a %s trade on average: %s %.0f, %+.3f%%, %.1f%% gained; expected per order %+.3f%%, realised per order %+.3f%% (before costs); "
                  "the broker's costs would have been %.3f%% a trade (not charged here: they come off the final orders)\n",
                  o.hold ? "completed" : "filled", cur, sumValue / nFilled, 100 * sumNet / nFilled, 100 * nWin / nFilled, 100 * sumExpected / std::max(1.0, nOrders), 100 * sumNet / std::max(1.0, nOrders), 100 * sumCost / nFilled);
    for (const auto& [yr, v] : byYear) {
      const Stats s = stats(v.first);
      std::printf("uk:   %s: %+6.1f%% (Sharpe %5.2f, %3zu sessions), the forecast's mean daily rank correlation %+.3f\n", yr.c_str(), 100 * s.annual, s.sharpe, v.first.size(),
                  v.second / double(v.first.size()));
    }
  }
  std::printf("uk: next session (after the close of %s): %zu orders\n", p.days.back().c_str(), next.size());
  {  // the best before costs and what each would cost: why an order was or was not placed
    const std::size_t keep = o.slots;
    o.slots = 10;
    const auto best = ordersAt(T - 1, false);
    o.slots = keep;
    double ab = 0, bb = 0, dh = 0;
    const bool cal = !pred[T - 1].empty() && (o.hold ? calibrationH(T - 1, holdAt(T - 1, ab), ab, bb, dh) : calibration(T - 1, levelsAt(T - 1, ab), ab, bb));
    if (!cal) std::printf("uk:   no calibrated record at the last close: no candidates\n");
    for (const auto& r : best) {
      const double cm = (2 * commission(r.value, r.i, r.close) + saleFees(r.value, r.close)) / r.value, st = stamp(r.i);
      std::printf("uk:   candidate %-6s %s: expected %+.2f%% before costs; commission%s %.2f%%, %s %.2f%%: %+.2f%% after (its spread %.2f%%, not paid in the auctions); %s %.0f%s\n",
                  p.tickers[r.i].c_str(), exchangeOf(r.i).c_str(), 100 * r.expected, us ? " and fees" : "", 100 * cm, uk ? "stamp duty" : "tax", 100 * st, 100 * (r.expected - cm - st), 100 * r.spread, cur, r.value,
                  r.expected - cm - st > 0 ? " (gains after costs)" : "");
    }
  }
  for (const auto& r : next)
    if (o.hold)
      std::printf("uk:   %-6s buy %s %5.0f%s at the open, at most %.4g (close %.4g %+.2f%%); sold at the first close %+.2f%% above the price paid, else at the close of day %zu "
                  "(such trades held %.1f days); expected %+.2f%% before costs, its costs %.2f%%, %+.2f%% after%s\n",
                  p.tickers[r.i].c_str(), cur, r.value, r.capped ? " (its liquidity)" : "", r.limit, r.close, 100 * (r.limit / r.close - 1), 100 * r.exitAbove, H, r.holdDays, 100 * r.expected,
                  100 * r.cost, 100 * r.net, uk && stampFree[r.i] ? " (AIM: no stamp duty)" : "");
    else
    std::printf("uk:   %-6s buy %s %5.0f%s at the open, at most %.4g (close %.4g %+.2f%%); a %s stop %.1f%% and %s from the price bought at; expected %+.2f%% before costs (levels %+.2f%%), its costs %.2f%%, %+.2f%% after; "
                "the auction under the limit on %.0f%% of its last sessions%s\n",
                p.tickers[r.i].c_str(), cur, r.value, r.capped ? " (its liquidity)" : "", r.limit, r.close, 100 * (r.limit / r.close - 1),
                r.mode == 1 ? "trailing" : r.mode == 2 ? "break-even (to the entry once up 1 volatility)" : "fixed", -100 * r.stopSd * r.sigma, std::isfinite(r.takeSd) ? ("take-profit +" + std::to_string(100 * r.takeSd * r.sigma).substr(0, 4) + "%").c_str() : "no take-profit", 100 * r.expected, 100 * r.adj, 100 * r.cost,
                100 * r.net, 100 * (std::isfinite(r.fill) ? r.fill : 0.0), uk && stampFree[r.i] ? " (AIM: no stamp duty)" : "");

  // The orders for the bridge (prices: to the private store, never the page): each carries its amount
  // (capital, at most maxCapital, the share's liquidity), its auction-only limit ("entry": "open"), and
  // its levels as fractions of the price it is bought at (stopRel, takeRel: the bridge sets them from the
  // fill), how its stop moves (stopMode: fixed; trail, trailRel below the highest price since the fill;
  // breakeven, to the entry once the price is breakevenRel above it); stop and take are the same levels
  // from the last close, for reading.
  if (!o.ordersOut.empty()) {
    std::ofstream f(o.ordersOut);
    char b[1024];
    f << "{\"model\":\"uk\",\"asOf\":\"" << p.days.back() << "\",\"slots\":" << o.slots << (o.hold ? ",\"book\":\"hold\",\"days\":" + std::to_string(H) : std::string()) << ",\"orders\":[";
    for (std::size_t q = 0; q < next.size(); ++q) {
      const auto& r = next[q];
      if (o.hold) {  // a multi-day book's order (the bridge's --holds): bought in the opening auction, sold at the first close exitAbove above the price paid, else at the days-th close
        std::snprintf(b, sizeof b,
                      "%s{\"ticker\":\"%s\",\"close\":%.6g,\"entry\":\"open\",\"open\":true,\"limit\":%.6g,\"stop\":null,\"take\":null,\"exitAbove\":%.6g,\"days\":%zu,\"holdDays\":%.3g,"
                      "\"expected\":%.6g,\"ifFilled\":%.6g,\"fill\":%.4g,\"capital\":%.2f,\"maxCapital\":%.2f,\"rebuy\":0,\"spread\":%.6g,\"cost\":%.6g,\"costAmount\":%.4g,\"stampFree\":%s}",
                      q ? "," : "", p.tickers[r.i].c_str(), r.close, r.limit, r.exitAbove, H, r.holdDays, (std::isfinite(r.fill) ? r.fill : 1.0) * r.net, r.net,
                      std::isfinite(r.fill) ? r.fill : 0.0, r.value, (noVolume ? o.capital : o.advShare * adv[at(T - 1, r.i)]), r.spread, r.cost, r.cost * r.value, stampFree[r.i] ? "true" : "false");
        f << b;
        continue;
      }
      std::snprintf(b, sizeof b,
                    "%s{\"ticker\":\"%s\",\"close\":%.6g,\"entry\":\"open\",\"limit\":%.6g,\"stop\":%.6g,\"take\":%s,\"stopRel\":%.6g,\"takeRel\":%s,\"stopMode\":\"%s\",\"trailRel\":%s,\"breakevenRel\":%s,\"exit\":\"close\",\"expected\":%.6g,\"ifFilled\":%.6g,\"fill\":%.4g,\"capital\":%.2f,"
                    "\"maxCapital\":%.2f,\"rebuy\":0,\"rebuyCost\":0,\"spread\":%.6g,\"cost\":%.6g,\"costAmount\":%.4g,\"stampFree\":%s}",
                    q ? "," : "", p.tickers[r.i].c_str(), r.close, r.limit, r.stop,
                    std::isfinite(r.takeSd) ? std::to_string(r.close * (1 + r.takeSd * r.sigma)).c_str() : "null", -r.stopSd * r.sigma,
                    std::isfinite(r.takeSd) ? std::to_string(r.takeSd * r.sigma).c_str() : "null", r.mode == 1 ? "trail" : r.mode == 2 ? "breakeven" : "fixed",
                    r.mode == 1 ? std::to_string(r.stopSd * r.sigma).c_str() : "null", r.mode == 2 ? std::to_string(r.sigma).c_str() : "null", (std::isfinite(r.fill) ? r.fill : 1.0) * r.net, r.net, std::isfinite(r.fill) ? r.fill : 0.0, r.value,
                    (noVolume ? o.capital : o.advShare * adv[at(T - 1, r.i)]), r.spread, r.cost, r.cost * r.value, stampFree[r.i] ? "true" : "false");
      f << b;
    }
    if (o.hold) {  // the daily update of the positions held (the bridge's hold book): for every share on the day's list, kept or sold at the open, and its exit level now
      f << "],\"updates\":[";
      double ua = 0, ub = 0, uadj = 0, udh = 0;
      const std::size_t ug = holdAt(T - 1, uadj);
      if (!pred[T - 1].empty() && calibrationH(T - 1, ug, ua, ub, udh))
        for (std::size_t j = 0, n = 0; j < uni[T - 1].size(); ++j) {
          const std::size_t i = uni[T - 1][j];
          const double sg = sigma[at(T - 1, i)], e = (ua + ub * pred[T - 1][j]) * sg;
          if (!(sg > 0)) continue;
          std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"keep\":%s,\"exitAbove\":%.6g,\"expected\":%.6g}", n++ ? "," : "", p.tickers[i].c_str(), e > 0 ? "true" : "false", kTh[ug] * sg, e);
          f << b;
        }
    }
    f << "],\"candidates\":[";
    {  // the day's list by forecast, for the bridge's recording of the opening auction
      std::vector<std::pair<double, std::size_t>> v;
      if (!pred[T - 1].empty())
        for (std::size_t j = 0; j < uni[T - 1].size(); ++j) v.emplace_back(pred[T - 1][j], uni[T - 1][j]);
      std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
      for (std::size_t q = 0; q < std::min<std::size_t>(60, v.size()); ++q) {
        std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"close\":%.6g,\"score\":%.6g}", q ? "," : "", p.tickers[v[q].second].c_str(), p.C[at(T - 1, v[q].second)], v[q].first);
        f << b;
      }
    }
    f << "],\"check\":{\"date\":\"" << (days.empty() ? std::string() : days.back()) << "\",\"trades\":[";
    for (std::size_t q = 0; q < lastCheck.size(); ++q) {
      const auto& c = lastCheck[q];
      std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"limit\":%.6g,\"filled\":%d,\"entry\":%.6g,\"exit\":%d,\"exitPrice\":%.6g,\"ret\":%.6g}", q ? "," : "", p.tickers[c.i].c_str(), c.r.limit,
                    c.f.filled, c.f.entry, c.f.exit, c.f.exitPx, c.f.ret);
      f << b;
    }
    f << "]},\"pool\":[";
    for (std::size_t q = 0; q < uni[T - 1].size(); ++q) f << (q ? ",\"" : "\"") << p.tickers[uni[T - 1][q]] << '"';
    f << "]}\n";
  }

  // The page's scan (results only: returns and levels relative to the last close, no prices).
  if (!o.jsonOut.empty()) {
    std::ofstream f(o.jsonOut);
    char b[512];
    f << "{\"model\":\"uk\",\"asOf\":\"" << p.days.back() << "\",\"stocks\":" << N << ",\"lag\":0,\"top\":" << o.top << (o.hold ? ",\"book\":\"hold\",\"days\":" + std::to_string(H) : std::string());
    std::snprintf(b, sizeof b, ",\"costs\":{\"broker\":\"IBKR\",\"market\":\"%s\",\"net\":false,\"capital\":%.6g,\"stamp\":%g,\"stampFree\":%ld,\"venues\":%s}", o.market.c_str(), o.capital,
                  uk ? 0.005 : 0.0, long(std::count(stampFree.begin(), stampFree.end(), 1)), uk || us ? "false" : "true");
    f << b;
    std::snprintf(b, sizeof b, ",\"backtest\":{\"from\":\"%s\",\"to\":\"%s\",\"annualReturn\":%.6g,\"sharpe\":%.6g,\"maxDrawdown\":%.6g,\"orders\":%.0f,\"trades\":%.0f,\"avgTrade\":%.6g,\"winRate\":%.6g}",
                  days.empty() ? "" : days.front().c_str(), days.empty() ? "" : days.back().c_str(), all.annual, all.sharpe, all.dd, nOrders, nFilled, nFilled > 0 ? sumNet / nFilled : 0.0,
                  nFilled > 0 ? nWin / nFilled : 0.0);
    f << b << ",\"orders\":[";
    for (std::size_t q = 0; q < next.size(); ++q) {
      const auto& r = next[q];
      if (o.hold) {
        std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"expected\":%.6g,\"ifFilled\":%.6g,\"gross\":%.6g,\"cost\":%.6g,\"entry\":\"open\",\"limit\":%.6g,\"stop\":null,\"take\":null,"
                      "\"exitAbove\":%.6g,\"days\":%zu,\"holdDays\":%.3g,\"rebuy\":0,\"fill\":%.4g,\"amount\":%.0f}", q ? "," : "", p.tickers[r.i].c_str(),
                      (std::isfinite(r.fill) ? r.fill : 1.0) * r.net, r.net, r.expected, r.cost, r.limit / r.close - 1, r.exitAbove, H, r.holdDays, std::isfinite(r.fill) ? r.fill : 0.0, r.value);
        f << b;
        continue;
      }
      // the levels from the entry (the open), the limit from the close
      std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"expected\":%.6g,\"ifFilled\":%.6g,\"gross\":%.6g,\"cost\":%.6g,\"entry\":\"open\",\"limit\":%.6g,\"stop\":%.6g,\"stopMode\":\"%s\",\"breakeven\":%s,"
                    "\"take\":%s,\"rebuy\":0,\"fill\":%.4g,\"amount\":%.0f}", q ? "," : "",
                    p.tickers[r.i].c_str(), (std::isfinite(r.fill) ? r.fill : 1.0) * r.net, r.net, r.expected, r.cost, r.limit / r.close - 1, -r.stopSd * r.sigma,
                    r.mode == 1 ? "trail" : r.mode == 2 ? "breakeven" : "fixed", r.mode == 2 ? std::to_string(r.sigma).c_str() : "null",
                    std::isfinite(r.takeSd) ? std::to_string(r.takeSd * r.sigma).c_str() : "null", std::isfinite(r.fill) ? r.fill : 0.0, r.value);
      f << b;
    }
    f << "],\"recent\":[";
    for (std::size_t q = ret.size() > 20 ? ret.size() - 20 : 0, s = q; q < ret.size(); ++q) {
      std::snprintf(b, sizeof b, "%s{\"date\":\"%s\",\"day\":%.6g}", q == s ? "" : ",", days[q].c_str(), ret[q]);
      f << b;
    }
    // The chart: the account's growth on a log scale, heights and positions as shares of the chart.
    const std::size_t n = ret.size(), step = std::max<std::size_t>(1, n / 600);
    std::vector<double> lv;
    double lo = 0, hi = 0, w = 0;
    for (std::size_t q = 0; q < n; ++q) {
      w += std::log1p(ret[q]);
      if (q % step == 0) lv.push_back(w), lo = std::min(lo, w), hi = std::max(hi, w);
    }
    const double span = std::max(hi - lo, 1e-9);
    std::snprintf(b, sizeof b, "],\"chart\":{\"series\":[{\"name\":\"daily trades\",\"final\":\"%.2f×\",\"y\":[", std::exp(w));
    f << b;
    for (std::size_t q = 0; q < lv.size(); ++q) std::snprintf(b, sizeof b, "%s%.4f", q ? "," : "", (lv[q] - lo) / span), f << b;
    f << "]}],\"ticks\":[";
    bool lead = true;
    for (double v : {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0, 64.0}) {
      const double y = (std::log(v) - lo) / span;
      if (y < -1e-9 || y > 1 + 1e-9) continue;
      std::snprintf(b, sizeof b, "%s{\"label\":\"%g×\",\"y\":%.4f}", lead ? "" : ",", v, y), f << b, lead = false;
    }
    f << "],\"years\":[";
    lead = true;
    for (std::size_t q = 0, j = 0; q < n; q += step, ++j)
      if (j == 0 || days[q].substr(0, 4) != days[q - step].substr(0, 4))
        std::snprintf(b, sizeof b, "%s{\"label\":\"%s\",\"x\":%.4f}", lead ? "" : ",", days[q].substr(0, 4).c_str(), lv.size() > 1 ? double(j) / double(lv.size() - 1) : 0.0), f << b,
            lead = false;
    std::set<std::string> exs;
    if (uk) exs.insert("LSE");
    else if (o.market == "US") exs.insert("NYSE"), exs.insert("NASDAQ");
    else
      for (const auto& tk : p.tickers)
        if (tk.find(':') != std::string::npos) exs.insert(tk.substr(0, tk.find(':')));
    f << "]},\"exchanges\":[";
    for (auto it = exs.begin(); it != exs.end(); ++it) f << (it == exs.begin() ? "\"" : ",\"") << *it << '"';
    f << "]}\n";
  }
  (void)firstDay;
  return 0;
}
