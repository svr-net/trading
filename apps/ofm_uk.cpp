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
//             day over its last 20 bars, no move of over 30% on the day (a bid, a placing, bad data);
//             the --top (300) most traded of those each day.
//   inputs    the signals of apps/swing.hpp known at the close (reversal over 1, 5 and 20 days,
//             momentum, the 52-week high, low volatility, the tug of war, ...), the share's volatility,
//             its liquidity and whether it pays stamp duty; each day ranked across the shares (-0.5 ..
//             0.5), so no input's scale or outliers decide.
//   target    the next session's open to close, in the share's own daily volatilities (at most 4).
//   model     one ridge regression across the shares, refitted every --refit (21) sessions on the
//             --window (750) sessions before (at least --warm, 250), fixed shrinkage.
//   expected  the forecast calibrated on the model's own out-of-sample record: over the last 500
//             sessions, among each day's highest fifth of forecasts, what the share then did against
//             what was forecast (a + b x, b shrunk by its t statistic). No record (at least 120
//             sessions), or no skill in it, makes no orders.
//   costs     IBKR's commission on the order's own amount (fixed: GBP 3 or 0.05% a side; tiered with
//             --pricing tiered), the 0.5% stamp duty on buying (not AIM shares, --stamp-free), a
//             quarter of the share's spread (Abdi and Ranaldo 2017) at each auction and half of it on a
//             stop's sale in the day.
//   order     bought in the opening auction and only there: a limit at the last close plus --cap-sd
//             (0.5) of its daily volatility (it never chases a gap up), good only until the auction
//             is over, so it never buys a dip later in the day. The amount is the slot's capital, at
//             most --adv-share (1%) of the share's median day's value (at least --min-order, GBP 500).
//   levels    a stop-loss and a take-profit set from the price it was bought at (the bridge places
//             them once the auction has filled, so a gap down never sets the stop above the entry), one
//             cancelling the other; whatever is still held is sold in the closing auction. Each day
//             the pair is chosen from a grid (stop 1, 1.5, 2, 3 or 4 daily volatilities below the
//             entry; take-profit 1, 1.5, 2 or 3 above, or none) by the record of the last 250 sessions:
//             what each pair added, per session, to the day's highest fifth of forecasts bought at the
//             open, against the 4-volatility stop alone, shrunk by its t statistic (a pair must have
//             added about two standard errors to be chosen; otherwise the 4-volatility stop alone).
//             A day's bar cannot tell which came first: a share that touched both is taken as stopped.
//             A stop sells half a spread below its level; a take-profit fills only when the high
//             clears it by half a spread.
//   choice    the orders whose expected return after costs (the forecast, plus what the chosen levels
//             added) is at least --hurdle (0.05%), the largest expected gain in pounds first, at most
//             --slots (X_NPICK, 5) a day. Some days none.
// The backtest trades exactly these orders day by day; an unfilled order and an unused slot earn 0.
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
  std::string csv, market, stampFree, segments, overrideCsv, jsonOut, ordersOut, traceOut;
  double capital = 5000, minValue = 250000, advShare = 0.01, minOrder = 500, hurdle = 0.0005, capSd = 0.5, minPrice = 20, lambda = 0.5;
  std::size_t slots = 5, top = 300, refit = 21, window = 750, warm = 250, calWindow = 500, calMin = 120;
  bool tiered = false;
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
  std::size_t level = 0;
  double expected = 0;  // the calibrated return from the entry to the close (with the levels), before costs
  double cost = 0;      // the round trip's costs if it buys in the auction and sells at the close
  double net = 0;       // expected - cost: the return if it buys
  double fill = kNaN;   // how often its opening auction would have bought over the share's last 250 sessions
  bool capped = false;  // its amount cut to the share's liquidity
};

struct Fill {
  int filled = 0;  // 1 in the opening auction
  int exit = 0;    // 0 the closing auction, 1 the stop, 2 the take-profit
  double entry = 0, exitPx = 0, gross = 0, cost = 0, ret = 0;
};
}  // namespace

int main(int argc, char** argv) {
  Opt o;
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
    else if (a == "--segments") o.segments = argv[++k];
    else if (a == "--override") o.overrideCsv = argv[++k];
    else if (a == "--pricing") o.tiered = std::string(argv[++k]) == "tiered";
    else if (a == "--json") o.jsonOut = argv[++k];
    else if (a == "--orders") o.ordersOut = argv[++k];
    else if (a == "--trace") o.traceOut = argv[++k];
    else if (a == "--min-value") o.minValue = std::atof(argv[++k]);
    else if (a == "--adv-share") o.advShare = std::atof(argv[++k]);
    else if (a == "--min-order") o.minOrder = std::atof(argv[++k]);
    else if (a == "--hurdle") o.hurdle = std::atof(argv[++k]);
    else if (a == "--cap-sd") o.capSd = std::atof(argv[++k]);
    else if (a == "--min-price") o.minPrice = std::atof(argv[++k]);
    else if (a == "--lambda") o.lambda = std::atof(argv[++k]);
    else if (a == "--refit") o.refit = std::max<std::size_t>(1, std::size_t(std::atol(argv[++k])));
    else if (a == "--window") o.window = std::size_t(std::atol(argv[++k]));
    else if (a == "--warm") o.warm = std::size_t(std::atol(argv[++k]));
  }
  if (o.csv.empty() || o.market != "UK") {
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
          vals.push_back(p.C[at(t, i)] * p.V[at(t, i)] / 100.0);  // pence to pounds
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
          eligible[at(t, i)] = segOk[i] && recent && !jump && hist >= 250 && p.C[at(t, i)] >= o.minPrice && sigma[at(t, i)] > 0 && adv[at(t, i)] >= o.minValue &&
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
  const auto sig = swing::signals(p);
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
          for (std::size_t j = 0; j < n; ++j) X[t][j * D + 1 + f] = col[j];
        }
        if (t + 1 < T)
          for (std::size_t j = 0; j < n; ++j) {
            const std::size_t i = u[j];
            if (p.ok(t + 1, i) && p.hasOpen(t + 1, i)) Y[t][j] = std::clamp((p.C[at(t + 1, i)] / p.O[at(t + 1, i)] - 1) / sigma[at(t, i)], -4.0, 4.0);
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
      // samples u with their target known at t: u + 1 <= t
      const std::size_t hi = t, lo = t > o.window ? t - o.window : 0;
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

  // The calibration's sums per day: among the day's highest fifth of forecasts, forecast against outcome.
  std::vector<std::array<double, 6>> calCum(T + 1, {0, 0, 0, 0, 0, 0});
  std::vector<char> calDay(T + 1, 0);
  std::vector<double> ic(T, kNaN);  // the forecast's rank correlation with the outcome, each day (out of sample)
  for (std::size_t u = 0; u < T; ++u) {
    calCum[u + 1] = calCum[u], calDay[u + 1] = 0;
    if (pred[u].empty()) continue;
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t j = 0; j < pred[u].size(); ++j)
      if (std::isfinite(Y[u][j])) v.emplace_back(pred[u][j], j);
    if (v.size() < 20) continue;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    for (std::size_t q = 0; q < v.size() / 5; ++q) {
      const double x = v[q].first, y = Y[u][v[q].second];
      auto& c = calCum[u + 1];
      c[0] += 1, c[1] += x, c[2] += y, c[3] += x * x, c[4] += x * y, c[5] += y * y;
    }
    calDay[u + 1] = 1;
    std::vector<double> a(v.size()), b(v.size());
    for (std::size_t q = 0; q < v.size(); ++q) a[q] = v[q].first, b[q] = Y[u][v[q].second];
    rankInPlace(a), rankInPlace(b);
    double sab = 0, saa = 0, sbb = 0;
    for (std::size_t q = 0; q < v.size(); ++q) sab += a[q] * b[q], saa += a[q] * a[q], sbb += b[q] * b[q];
    ic[u] = saa > 0 && sbb > 0 ? sab / std::sqrt(saa * sbb) : kNaN;
  }
  std::vector<int> calDays(T + 1, 0);
  for (std::size_t u = 0; u < T; ++u) calDays[u + 1] = calDays[u] + calDay[u + 1];
  // The calibration known at the close t (outcomes of the sessions up to t): {a, b}, or none.
  const auto calibration = [&](std::size_t t, double& a, double& b) {
    const std::size_t hi = t, lo = t > o.calWindow ? t - o.calWindow : 0;  // days u in [lo, hi): outcome at u + 1 <= t
    if (std::size_t(calDays[hi] - calDays[lo]) < o.calMin) return false;
    double c[6];
    for (int k = 0; k < 6; ++k) c[k] = calCum[hi][k] - calCum[lo][k];
    const double n = c[0], mx = c[1] / n, my = c[2] / n, sxx = c[3] - n * mx * mx, sxy = c[4] - n * mx * my, syy = c[5] - n * my * my;
    if (!(n > 100) || !(sxx > 0)) return false;
    double slope = sxy / sxx;
    const double s2 = std::max(0.0, syy - slope * sxy) / (n - 2), ts = s2 > 0 ? slope / std::sqrt(s2 / sxx) : 0.0;  // its t statistic
    slope = ts > 0 ? std::min(1.5, slope) * std::max(0.0, 1.0 - 1.0 / (ts * ts)) : 0.0;
    b = slope, a = my - slope * mx;
    return true;
  };

  // The levels' grid and the exit of a share bought at `entry` in the opening auction of session d.
  static const double kStop[] = {1, 1.5, 2, 3, 4}, kTake[] = {1, 1.5, 2, 3, kNaN};
  constexpr std::size_t nS = 5, nK = 5, nL = nS * nK, kBase = 4 * nK + 4;  // the base: the 4-volatility stop alone
  struct Exit { int code; double px, slip; };
  const auto exitOf = [&](std::size_t d, std::size_t i, double entry, double sg, double spr, double sSd, double kSd) {
    const double L = p.L[at(d, i)], H = p.H[at(d, i)], C = p.C[at(d, i)], hs = 0.5 * spr;
    const double stop = entry * (1 - sSd * sg), take = std::isfinite(kSd) ? entry * (1 + kSd * sg) : kNaN;
    if (L <= stop) return Exit{1, stop, hs};                               // touched: stopped (first, if the take-profit was touched too)
    if (std::isfinite(take) && H >= take * (1 + hs)) return Exit{2, take, 0.0};  // a resting limit sell: no spread
    return Exit{0, C, 0.25 * spr};                                          // the closing auction
  };
  // Each session's record of every pair: the day's highest fifth of forecasts bought at the open (when
  // the auction was under the limit), each pair's mean result that day and its difference from the base.
  std::vector<double> lvD(T + 1, 0.0);
  std::vector<double> lvS((T + 1) * nL, 0.0), lvS2((T + 1) * nL, 0.0), lvR((T + 1) * nL, 0.0);
  for (std::size_t u = 0; u < T; ++u) {
    lvD[u + 1] = lvD[u];
    for (std::size_t g = 0; g < nL; ++g) lvS[(u + 1) * nL + g] = lvS[u * nL + g], lvS2[(u + 1) * nL + g] = lvS2[u * nL + g], lvR[(u + 1) * nL + g] = lvR[u * nL + g];
    if (pred[u].empty() || u + 1 >= T) continue;
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t j = 0; j < pred[u].size(); ++j) v.emplace_back(pred[u][j], j);
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    double sum[nL] = {}, n = 0;
    for (std::size_t q = 0; q < v.size() / 5; ++q) {
      const std::size_t i = uni[u][v[q].second];
      if (!p.ok(u + 1, i) || !p.hasOpen(u + 1, i)) continue;
      const double O = p.O[at(u + 1, i)], sg = sigma[at(u, i)], spr = spread[at(u, i)];
      if (!(O <= p.C[at(u, i)] * (1 + o.capSd * sg))) continue;
      for (std::size_t g = 0; g < nL; ++g) {
        const Exit e = exitOf(u + 1, i, O, sg, spr, kStop[g / nK], kTake[g % nK]);
        sum[g] += e.px / O - 1 - e.slip;
      }
      n += 1;
    }
    if (n < 3) continue;
    lvD[u + 1] += 1;
    for (std::size_t g = 0; g < nL; ++g) {
      const double r = sum[g] / n, dlt = r - sum[kBase] / n;
      lvS[(u + 1) * nL + g] += dlt, lvS2[(u + 1) * nL + g] += dlt * dlt, lvR[(u + 1) * nL + g] += r;
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

  // The costs of an order of `value` pounds.
  const auto commission = [&](double value) {
    if (!(value > 0)) return 0.0;
    return o.tiered ? std::max(1.0, 0.0005 * value) + std::max(0.10, 0.000045 * value) + 0.06 : std::max(3.0, 0.0005 * value);
  };
  const auto stamp = [&](std::size_t i) { return stampFree[i] ? 0.0 : 0.005; };

  // The orders for the session after the close t.
  const auto ordersAt = [&](std::size_t t) {
    std::vector<Order> out;
    double a = 0, b = 0;
    if (pred[t].empty() || !calibration(t, a, b)) return out;
    double lvAdj = 0;
    const std::size_t lv = levelsAt(t, lvAdj);
    const auto& u = uni[t];
    for (std::size_t j = 0; j < u.size(); ++j) {
      const std::size_t i = u[j];
      Order r;
      r.i = i, r.close = p.C[at(t, i)], r.sigma = sigma[at(t, i)], r.spread = spread[at(t, i)];
      r.expected = (a + b * pred[t][j]) * r.sigma;
      r.value = std::min(o.capital, o.advShare * adv[at(t, i)]);
      r.capped = r.value < o.capital;
      if (r.value < o.minOrder) continue;
      r.limit = r.close * (1 + o.capSd * r.sigma);
      r.level = lv, r.adj = lvAdj, r.stopSd = kStop[lv / nK], r.takeSd = kTake[lv % nK], r.expected += lvAdj;
      r.stop = r.close * (1 - r.stopSd * r.sigma);  // shown from the close; placed from the price bought at
      r.cost = 2 * commission(r.value) / r.value + stamp(i) + 0.5 * r.spread;  // a quarter-spread at each auction (the levels' own slippage is in what they added)
      r.net = r.expected - r.cost;
      if (r.net >= o.hurdle) out.push_back(r);
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
  // An order's session (the bars of t + 1).
  const auto play = [&](std::size_t t, const Order& r) {
    Fill f;
    const std::size_t i = r.i, d = t + 1;
    if (d >= T || !p.ok(d, i) || !p.hasOpen(d, i)) return f;
    const double O = p.O[at(d, i)];
    if (!(O <= r.limit)) return f;  // the auction above the limit: not bought (the order expires with the auction)
    f.filled = 1, f.entry = O;
    const Exit e = exitOf(d, i, O, r.sigma, r.spread, r.stopSd, r.takeSd);
    f.exit = e.code, f.exitPx = e.px;
    const double slip = 0.25 * r.spread + e.slip;  // the auction's price against the book's, and the exit's
    f.gross = f.exitPx / f.entry - 1;
    f.cost = 2 * commission(r.value) / r.value + stamp(i) + slip;
    f.ret = f.gross - f.cost;
    return f;
  };

  // The backtest: every session's orders from the close before, traded on the session's bars.
  std::FILE* trace = o.traceOut.empty() ? nullptr : std::fopen(o.traceOut.c_str(), "w");
  std::FILE* tlog = std::getenv("X_TRADELOG") ? std::fopen(std::getenv("X_TRADELOG"), "w") : nullptr;
  if (tlog) std::fprintf(tlog, "date,ticker,close,limit,stopRel,takeRel,value,expected,cost,filled,entry,exit,exitPrice,gross,ret\n");
  std::vector<double> ret;
  std::vector<std::string> days;
  double nOrders = 0, nFilled = 0, nAuction = 0, nStopped = 0, nTaken = 0, nCapped = 0, sumGross = 0, sumCost = 0, sumNet = 0, nWin = 0, sumValue = 0, sumExpected = 0;
  std::size_t daysWithOrders = 0, firstDay = T, levelDays[nL] = {};
  std::map<std::string, std::pair<std::vector<double>, double>> byYear;  // each year's daily returns, IC sum
  std::vector<double> baseline;  // every share of the day's list, bought at the open and sold at the close (gross): the market the orders come from
  const auto traceOrder = [&](std::size_t t, const Order& r) {
    if (trace) std::fprintf(trace, "%s,%s,%.6g,%.6g,%.6g,%.6g,%.6g\n", p.days[t].c_str(), p.tickers[r.i].c_str(), r.limit, r.stopSd, r.takeSd, r.value, r.net);
  };
  struct Check { std::size_t i; Order r; Fill f; };
  std::vector<Check> lastCheck;
  for (std::size_t t = 0; t + 1 < T; ++t) {
    const auto ords = ordersAt(t);
    double a = 0, b = 0;
    const bool live = !pred[t].empty() && calibration(t, a, b);
    if (!live) continue;
    if (firstDay == T) firstDay = t + 1;
    {
      double adj = 0;
      ++levelDays[levelsAt(t, adj)];
    }
    double day = 0;
    lastCheck.clear();
    for (const auto& r : ords) {
      traceOrder(t, r);
      const Fill f = play(t, r);
      nOrders += 1, nCapped += r.capped, sumExpected += (std::isfinite(r.fill) ? r.fill : 1.0) * r.net;
      if (f.filled) {
        nFilled += 1, nAuction += f.filled == 1, nStopped += f.exit == 1, nTaken += f.exit == 2, sumGross += f.gross, sumCost += f.cost, sumNet += f.ret, nWin += f.ret > 0, sumValue += r.value;
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
  // The next open's orders, from the last close.
  const auto next = ordersAt(T - 1);
  for (const auto& r : next) traceOrder(T - 1, r);
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
  const Stats all = stats(ret), base = stats(baseline);
  std::size_t uniMean = 0;
  for (const auto& u : uni) uniMean += u.size();
  std::printf("uk: %zu shares, %zu sessions (%s .. %s); %zu in the exchange's list, %zu of them outside the order book's auctions (funds, ETFs, SETSqx): left out\n", N, T,
              p.days.front().c_str(), p.days.back().c_str(), segKnown, segOut);
  std::printf("uk: the day's list: %.0f shares on average (at most %zu; at least GBP %.0f traded a day); %zu fits of %zu inputs (the last on %.0f share-days)\n",
              double(uniMean) / double(T), o.top, o.minValue, fits, F, fitted);
  if (ret.empty()) {
    std::printf("uk: no out-of-sample record yet (at least %zu sessions to fit and %zu to calibrate): no orders\n", o.warm, o.calMin);
  } else {
    std::printf("uk: the account (%s .. %s, %zu slots of GBP %.0f): %+.1f%% a year, Sharpe %.2f, max drawdown %.1f%%, net of costs\n", days.front().c_str(), days.back().c_str(),
                o.slots, o.capital, 100 * all.annual, all.sharpe, 100 * all.dd);
    std::printf("uk:   for scale: every share of the day's list bought at the open and sold at the close %+.1f%% a year (Sharpe %.2f), before costs\n", 100 * base.annual, base.sharpe);
    std::printf("uk:   orders: %.0f on %zu of %zu sessions (%.2f a day); bought in the opening auction %.1f%%; of the trades %.1f%% stopped, %.1f%% took profit, the rest sold at the close; %.0f%% cut to the share's liquidity\n",
                nOrders, daysWithOrders, ret.size(), nOrders / double(ret.size()), nOrders > 0 ? 100 * nFilled / nOrders : 0.0, nFilled > 0 ? 100 * nStopped / nFilled : 0.0,
                nFilled > 0 ? 100 * nTaken / nFilled : 0.0, nOrders > 0 ? 100 * nCapped / nOrders : 0.0);
    // The levels: which pair each session used (chosen walk-forward), and every pair's record over the
    // whole period (for reading only: the choice each day saw only the 250 sessions before it).
    std::printf("uk:   levels used (stop / take-profit, daily volatilities from the entry):");
    for (std::size_t g = 0; g < nL; ++g)
      if (levelDays[g]) {
        char k[16];
        std::snprintf(k, sizeof k, std::isfinite(kTake[g % nK]) ? "%g" : "none", kTake[g % nK]);
        std::printf(" %g/%s on %zu sessions;", kStop[g / nK], k, levelDays[g]);
      }
    std::printf("\n");
    {
      const std::size_t lo = firstDay > 0 ? firstDay - 1 : 0;
      const double n = lvD[T] - lvD[lo];
      if (n > 1) {
        std::printf("uk:   each pair over the backtest, added a trade against the %g-volatility stop alone (t):", kStop[kBase / nK]);
        for (std::size_t g = 0; g < nL; ++g) {
          if (g == kBase) continue;
          const double m = (lvS[T * nL + g] - lvS[lo * nL + g]) / n, v = (lvS2[T * nL + g] - lvS2[lo * nL + g]) / n - m * m;
          char k[16];
          std::snprintf(k, sizeof k, std::isfinite(kTake[g % nK]) ? "%g" : "none", kTake[g % nK]);
          std::printf("%s %g/%s %+.3f%% (%+.1f)", g % nK ? "," : "\nuk:    ", kStop[g / nK], k, 100 * m, v > 0 ? m / std::sqrt(v / n) : 0.0);
        }
        std::printf("\n");
      }
    }
    if (nFilled > 0)
      std::printf("uk:   a filled trade on average: GBP %.0f, %+.3f%% before costs, costs %.3f%%, %+.3f%% after; %.1f%% gained; expected per order %+.3f%%, realised per order %+.3f%%\n",
                  sumValue / nFilled, 100 * sumGross / nFilled, 100 * sumCost / nFilled, 100 * sumNet / nFilled, 100 * nWin / nFilled, 100 * sumExpected / std::max(1.0, nOrders),
                  100 * sumNet / std::max(1.0, nOrders));
    for (const auto& [yr, v] : byYear) {
      const Stats s = stats(v.first);
      std::printf("uk:   %s: %+6.1f%% (Sharpe %5.2f, %3zu sessions), the forecast's mean daily rank correlation %+.3f\n", yr.c_str(), 100 * s.annual, s.sharpe, v.first.size(),
                  v.second / double(v.first.size()));
    }
  }
  std::printf("uk: next session (after the close of %s): %zu orders\n", p.days.back().c_str(), next.size());
  for (const auto& r : next)
    std::printf("uk:   %-6s buy GBP %5.0f%s at the open, at most %.4g (close %.4g %+.2f%%); stop %.1f%% and %s from the price bought at; expected %+.2f%% (levels %+.2f%%), costs %.2f%%, net %+.2f%%; "
                "the auction under the limit on %.0f%% of its last sessions%s\n",
                p.tickers[r.i].c_str(), r.value, r.capped ? " (its liquidity)" : "", r.limit, r.close, 100 * (r.limit / r.close - 1), -100 * r.stopSd * r.sigma,
                std::isfinite(r.takeSd) ? ("take-profit +" + std::to_string(100 * r.takeSd * r.sigma).substr(0, 4) + "%").c_str() : "no take-profit", 100 * r.expected, 100 * r.adj, 100 * r.cost,
                100 * r.net, 100 * (std::isfinite(r.fill) ? r.fill : 0.0), stampFree[r.i] ? " (AIM: no stamp duty)" : "");

  // The orders for the bridge (prices: to the private store, never the page): each carries its amount
  // (capital, at most maxCapital, the share's liquidity), its auction-only limit ("entry": "open"), and
  // its levels as fractions of the price it is bought at (stopRel, takeRel: the bridge sets them from the
  // fill); stop and take are the same levels from the last close, for reading.
  if (!o.ordersOut.empty()) {
    std::ofstream f(o.ordersOut);
    char b[1024];
    f << "{\"model\":\"uk\",\"asOf\":\"" << p.days.back() << "\",\"slots\":" << o.slots << ",\"orders\":[";
    for (std::size_t q = 0; q < next.size(); ++q) {
      const auto& r = next[q];
      std::snprintf(b, sizeof b,
                    "%s{\"ticker\":\"%s\",\"close\":%.6g,\"entry\":\"open\",\"limit\":%.6g,\"stop\":%.6g,\"take\":%s,\"stopRel\":%.6g,\"takeRel\":%s,\"exit\":\"close\",\"expected\":%.6g,\"ifFilled\":%.6g,\"fill\":%.4g,\"capital\":%.2f,"
                    "\"maxCapital\":%.2f,\"rebuy\":0,\"rebuyCost\":0,\"spread\":%.6g,\"cost\":%.6g,\"costAmount\":%.4g,\"stampFree\":%s}",
                    q ? "," : "", p.tickers[r.i].c_str(), r.close, r.limit, r.stop,
                    std::isfinite(r.takeSd) ? std::to_string(r.close * (1 + r.takeSd * r.sigma)).c_str() : "null", -r.stopSd * r.sigma,
                    std::isfinite(r.takeSd) ? std::to_string(r.takeSd * r.sigma).c_str() : "null", (std::isfinite(r.fill) ? r.fill : 1.0) * r.net, r.net, std::isfinite(r.fill) ? r.fill : 0.0, r.value,
                    o.advShare * adv[at(T - 1, r.i)], r.spread, r.cost, r.cost * r.value, stampFree[r.i] ? "true" : "false");
      f << b;
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
    f << "{\"model\":\"uk\",\"asOf\":\"" << p.days.back() << "\",\"stocks\":" << N << ",\"lag\":0,\"top\":" << o.top;
    std::snprintf(b, sizeof b, ",\"costs\":{\"broker\":\"IBKR\",\"market\":\"UK\",\"net\":true,\"capital\":%.6g,\"stamp\":0.005,\"stampFree\":%ld,\"venues\":false}", o.capital,
                  long(std::count(stampFree.begin(), stampFree.end(), 1)));
    f << b;
    std::snprintf(b, sizeof b, ",\"backtest\":{\"from\":\"%s\",\"to\":\"%s\",\"annualReturn\":%.6g,\"sharpe\":%.6g,\"maxDrawdown\":%.6g,\"orders\":%.0f,\"trades\":%.0f,\"avgTrade\":%.6g,\"winRate\":%.6g}",
                  days.empty() ? "" : days.front().c_str(), days.empty() ? "" : days.back().c_str(), all.annual, all.sharpe, all.dd, nOrders, nFilled, nFilled > 0 ? sumNet / nFilled : 0.0,
                  nFilled > 0 ? nWin / nFilled : 0.0);
    f << b << ",\"orders\":[";
    for (std::size_t q = 0; q < next.size(); ++q) {
      const auto& r = next[q];
      // the levels from the entry (the open), the limit from the close
      std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"expected\":%.6g,\"ifFilled\":%.6g,\"limit\":%.6g,\"stop\":%.6g,\"take\":%s,\"rebuy\":0,\"fill\":%.4g,\"amount\":%.0f}", q ? "," : "",
                    p.tickers[r.i].c_str(), (std::isfinite(r.fill) ? r.fill : 1.0) * r.net, r.net, r.limit / r.close - 1, -r.stopSd * r.sigma,
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
    f << "]},\"exchanges\":[\"LSE\"]}\n";
  }
  (void)firstDay;
  return 0;
}
