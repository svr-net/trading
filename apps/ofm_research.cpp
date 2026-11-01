// The predictive model's research: each OHLCV feature's power to predict the next day's moves.
//   ofm_research --csv stocks.csv [--series futures.csv]
// Prints, for every feature and target, the mean of its day-by-day cross-sectional correlation
// with the target and its t-statistic (the features and the correlations run in fused kernels,
// kernels/v2).
#include <algorithm>
#include <array>
#include <map>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "net.hpp"
#include "invoke.hpp"
#include "ofm/data.hpp"
#include "v2_kernels.hpp"

namespace {
using ofm::kNaN;
std::string readFile(const std::string& p) {
  std::ifstream in(p);
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}
}  // namespace

int main(int argc, char** argv) {
  std::string csv, seriesCsv, jsonOut, ordersOut;
  std::size_t topN = 0;  // trade only each day's topN most traded (0: all)
  std::size_t lag = 0;   // a check on the data: trade on the predictions made lag closes earlier
  bool closeOnly = false, safetyOnly = false;  // --exit close: sold at the close; --exit safety: a safety stop only (see levelsAt)
  bool scanOnly = false;  // --scan: only the page's model (no comparison models), for the daily job
  // --costs UK|US: the broker's costs on every filled trade (kernels/v2/levels.wgsl), --capital the
  // amount a trade, --stamp the UK stamp duty on buying, --stamp-free a file of the tickers exempt
  // from it (AIM shares, one ticker a line).
  std::string stampFree, venuesFile, costMarket;
  bool fixedPricing = true;  // --pricing fixed (the default: IBKR's fixed commissions) or tiered  // --venues: each exchange's costs (see fee below)
  double costKind = 0, capital = 5000, stamp = 0.005;  // --exit close: no stop-loss or take-profit, every trade sold at the close
  for (int k = 1; k + 1 < argc; ++k) {
    if (std::string(argv[k]) == "--csv") csv = argv[k + 1];
    if (std::string(argv[k]) == "--series") seriesCsv = argv[k + 1];
    if (std::string(argv[k]) == "--top") topN = std::size_t(std::atol(argv[k + 1]));
    if (std::string(argv[k]) == "--lag") lag = std::size_t(std::atol(argv[k + 1]));
    if (std::string(argv[k]) == "--json") jsonOut = argv[k + 1];
    if (std::string(argv[k]) == "--orders") ordersOut = argv[k + 1];
    if (std::string(argv[k]) == "--scan") scanOnly = true;
    if (std::string(argv[k]) == "--exit") closeOnly = std::string(argv[k + 1]) == "close", safetyOnly = std::string(argv[k + 1]) == "safety";
    if (std::string(argv[k]) == "--costs") {
      costMarket = argv[k + 1];
      costKind = costMarket == "UK" ? 1 : costMarket == "US" ? 2 : costMarket == "APAC" ? 3 : 4;  // 4: by --venues only
    }
    if (std::string(argv[k]) == "--capital") capital = std::atof(argv[k + 1]);
    if (std::string(argv[k]) == "--stamp") stamp = std::atof(argv[k + 1]);
    if (std::string(argv[k]) == "--stamp-free") stampFree = argv[k + 1];
    if (std::string(argv[k]) == "--pricing") fixedPricing = std::string(argv[k + 1]) != "tiered";
    if (std::string(argv[k]) == "--venues") venuesFile = argv[k + 1];
  }
  for (int k = 1; k < argc; ++k)
    if (std::string(argv[k]) == "--scan") scanOnly = true;
  ofm::Market mk = ofm::parseMarketCsv(readFile(csv));
  // Impossible bars (a low above the open or the close, a high below them, a price not above zero)
  // are dropped as missing.
  std::size_t dropped = 0;
  for (std::size_t t = 0; t < mk.T(); ++t)
    for (std::size_t i = 0; i < mk.N(); ++i) {
      const double o = mk.open(t, i), h = mk.high(t, i), l = mk.low(t, i), c = mk.close(t, i);
      if (!std::isfinite(o) || !std::isfinite(c)) continue;
      if (o > 0 && h > 0 && l > 0 && c > 0 && l <= std::min(o, c) * (1 + 1e-9) && h >= std::max(o, c) * (1 - 1e-9)) continue;
      mk.open(t, i) = mk.high(t, i) = mk.low(t, i) = mk.close(t, i) = mk.volume(t, i) = kNaN, ++dropped;
    }
  const ofm::Market& m = mk;
  const std::size_t T = m.T(), N = m.N(), F = 27, K = F + 5;
  // Each stock's stamp duty on buying (none for the tickers in --stamp-free).
  std::vector<double> stampOf(N, costKind == 1 ? stamp : 0.0);
  if (!stampFree.empty()) {
    std::istringstream in(readFile(stampFree));
    std::set<std::string> free;
    for (std::string w; in >> w;) free.insert(w);
    std::size_t nf = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (free.count(m.tickers[i])) stampOf[i] = 0, ++nf;
    std::printf("stamp duty: %zu of %zu stocks exempt (%s)\n", nf, N, stampFree.c_str());
  }
  // Each stock's costs (7 fields, as kernels/v2/levels.wgsl's costOf): kind (1 a rate with a minimum
  // an order, 2 the US's commission a share), the amount a trade, the commission's rate and minimum,
  // a fee's rate, the tax on buying, the tax on both sides. By --costs; with --venues, a stock whose
  // ticker is EXCHANGE:CODE takes its exchange's line (EXCHANGE amount rate minimum fee buy-tax
  // both-tax, the amount in the exchange's currency).
  std::vector<double> fee(N * 7, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    double* f = &fee[i * 7];
    // IBKR's pricing (--pricing fixed, the default, or tiered). UK fixed: GBP 3 an order up to GBP
    // 6,000, 0.05% above, the exchange's and clearing fees included; tiered: 0.05%, at least GBP 1,
    // the LSE's fee 0.0045% (at least GBP 0.10) and clearing GBP 0.06 an order. Stamp duty either way.
    if (costKind == 1 && fixedPricing) f[0] = 1, f[1] = capital, f[2] = 0.0005, f[3] = 3, f[4] = 0, f[5] = stampOf[i];
    if (costKind == 1 && !fixedPricing) f[0] = 1, f[1] = capital, f[2] = 0.0005, f[3] = 1, f[4] = std::max(0.000045, 0.10 / capital) + 0.06 / capital, f[5] = stampOf[i];
    // US, a share: fixed USD 0.005, at least USD 1 an order (fees included); tiered USD 0.0035, at
    // least USD 0.35, + USD 0.0005 a share of fees; at most 1% of the value; the SEC's and FINRA's
    // fees on selling either way.
    if (costKind == 2) f[0] = 2, f[1] = capital, f[2] = fixedPricing ? 0.005 : 0.0035, f[3] = fixedPricing ? 1.0 : 0.35, f[4] = fixedPricing ? 0.0 : 0.0005;
    if (costKind == 3) f[0] = 1, f[1] = capital, f[2] = 0.0008, f[3] = 6;                                    // ASX: 0.08%, at least AUD 6
  }
  if (!venuesFile.empty()) {
    std::istringstream in(readFile(venuesFile));
    std::map<std::string, std::array<double, 6>> venue;
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ls(line);
      std::string ex;
      std::array<double, 6> v{};
      if (ls >> ex >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5]) venue[ex] = v;
    }
    std::size_t hit = 0;
    for (std::size_t i = 0; i < N; ++i) {
      const auto c = m.tickers[i].find(':');
      const auto it = c == std::string::npos ? venue.end() : venue.find(m.tickers[i].substr(0, c));
      if (it == venue.end()) continue;
      double* f = &fee[i * 7];
      f[0] = 1;
      for (std::size_t k = 0; k < 6; ++k) f[1 + k] = it->second[k];
      ++hit;
    }
    std::printf("venues: %zu of %zu stocks priced by their exchange's line (%s)\n", hit, N, venuesFile.c_str());
  }
  std::vector<double> bars(5 * T * N, 0.0), feat(T * N * K, 0.0), ic(T * F * 4, 0.0);
  const ofm::Panel* pn[5] = {&m.open, &m.high, &m.low, &m.close, &m.volume};
  for (std::size_t f = 0; f < 5; ++f)
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t i = 0; i < N; ++i) {
        const double x = (*pn[f])(t, i);
        bars[(f * T + t) * N + i] = std::isfinite(x) ? x : 0.0;
      }
  ofm::v2::gen::V2Header<double> P;
  P.T = std::uint32_t(T), P.N = std::uint32_t(N), P.K = std::uint32_t(K), P.NT = std::uint32_t(F);
  ofm::v2::gen::Features<double> kf;
  kf.P = P, kf.bars = bars.data(), kf.out = feat.data();
  ofm::detail::dispatch(kf, ofm::detail::groups(std::uint32_t(N), 64));
  ofm::v2::gen::Ic<double> ki;
  ki.P = P, ki.feat = feat.data(), ki.ic = ic.data();
  ofm::detail::dispatch(ki, ofm::detail::groups(std::uint32_t(T * F * 4), 64));
  const char* fn[27] = {"momentum 2d", "momentum 4d", "momentum 8d", "momentum 16d", "momentum 32d", "momentum 64d", "volume shock 4d", "volume shock 16d",
                        "volume shock 64d", "range shock 4d", "range shock 16d", "range shock 64d", "open-to-close", "close-to-close", "gap",
                        "close in range", "upper wick", "lower wick", "signed volume shock", "liquidity (traded value)", "volatility level", "intraday momentum 4d", "intraday momentum 16d", "intraday momentum 64d", "overnight momentum 4d", "overnight momentum 16d", "overnight momentum 64d"};
  const char* tn[4] = {"next open-to-close", "next open-to-high", "next open-to-low (fall)", "next close-to-close"};
  // Each day's tradable stocks: with --top, the topN most traded by their own traded value.
  std::vector<char> tradable(T * N, 1);
  if (topN > 0)
    for (std::size_t t = 0; t < T; ++t) {
      std::vector<std::pair<double, std::size_t>> v;
      for (std::size_t i = 0; i < N; ++i)
        if (feat[(t * N + i) * K + K - 1] > 0) v.emplace_back(feat[(t * N + i) * K + 19], i);
      std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      for (std::size_t i = 0; i < N; ++i) tradable[t * N + i] = 0;
      for (std::size_t r = 0; r < std::min(topN, v.size()); ++r) tradable[t * N + v[r].second] = 1;
    }
  // Each stock's bid-ask spread as known at each close (Abdi and Ranaldo 2017, from the closes and
  // the mid-ranges: s^2 = 4 E[(c_t - eta_t)(c_t - eta_t+1)] in logs, averaged over about 64 days).
  // A stop-loss sells half a spread below its level; a take-profit sells only when the high clears
  // it by half a spread.
  std::vector<double> spread(T * N, 0.0);
  for (std::size_t i = 0; i < N; ++i) {
    double e = 0, w = 0;
    for (std::size_t t = 1; t < T; ++t) {
      const double c0 = m.close(t - 1, i), h0 = m.high(t - 1, i), l0 = m.low(t - 1, i), h1 = m.high(t, i), l1 = m.low(t, i);
      if (c0 > 0 && h0 > 0 && l0 > 0 && h1 > 0 && l1 > 0) {
        const double c = std::log(c0), a = c - 0.5 * std::log(h0 * l0), b = c - 0.5 * std::log(h1 * l1);
        e += (4 * a * b - e) / 64, w += (1 - w) / 64;
      }
      spread[t * N + i] = w > 0 ? std::sqrt(std::max(0.0, e / w)) : 0.0;
    }
  }
  // Each stock's parabolic SAR for the next day as known at each close (Wilder: acceleration 0.02,
  // rising by 0.02 to 0.2 with each new extreme), while it trails below the price; NaN in a
  // downtrend.
  std::vector<double> sar(T * N, kNaN);
  for (std::size_t i = 0; i < N; ++i) {
    bool up = true, init = false;
    double ps = 0, ep = 0, af = 0.02, l1 = 0, l2 = 0, h1 = 0, h2 = 0;
    for (std::size_t t = 0; t < T; ++t) {
      const double h = m.high(t, i), l = m.low(t, i);
      if (!(h > 0 && l > 0)) continue;
      if (!init) {
        init = true, ps = l, ep = h, l1 = l2 = l, h1 = h2 = h;
      } else {
        ps += af * (ep - ps);
        if (up) ps = std::min({ps, l1, l2});
        else ps = std::max({ps, h1, h2});
        if (up && l < ps) up = false, ps = ep, ep = l, af = 0.02;
        else if (!up && h > ps) up = true, ps = ep, ep = h, af = 0.02;
        else if (up && h > ep) ep = h, af = std::min(0.2, af + 0.02);
        else if (!up && l < ep) ep = l, af = std::min(0.2, af + 0.02);
        l2 = l1, l1 = l, h2 = h1, h1 = h;
      }
      double nx = ps + af * (ep - ps);
      nx = up ? std::min({nx, l1, l2}) : nx;
      if (up) sar[t * N + i] = nx;
    }
  }
  // Outlier shares, left out of the trades each day: from each share's last 60 days up to the close
  // (at least 40 with a bar), its mean log traded value, its share of days the close did not move
  // and its mean log range (high / low); a share more than 3 robust standard deviations (1.4826
  // times the median absolute deviation; the mean absolute deviation times 1.2533 where that is
  // zero) from the day's median, on the side of illiquid (low value), stale (unmoved closes) or
  // erratic (wide range), is an outlier.
  {
    std::vector<double> stat(3 * N), ok(N);
    std::size_t out = 0, seen = 0;
    for (std::size_t t = 0; t < T; ++t) {
      for (std::size_t i = 0; i < N; ++i) {
        double n = 0, lv = 0, st = 0, sn = 0, rg = 0;
        for (std::size_t u = t >= 59 ? t - 59 : 0; u <= t; ++u) {
          const double c = m.close(u, i), v = m.volume(u, i), h = m.high(u, i), l = m.low(u, i);
          if (!(c > 0) || !(h > 0) || !(l > 0)) continue;
          n += 1, lv += std::log(c * std::max(v, 1.0)), rg += std::log(h / l);
          if (u > 0 && m.close(u - 1, i) > 0) st += c == m.close(u - 1, i), sn += 1;
        }
        ok[i] = n >= 40 && sn > 0;
        if (ok[i]) stat[i] = lv / n, stat[N + i] = st / sn, stat[2 * N + i] = rg / n;
      }
      double med[3], sc[3];
      for (std::size_t k = 0; k < 3; ++k) {
        std::vector<double> z;
        for (std::size_t i = 0; i < N; ++i)
          if (ok[i]) z.push_back(stat[k * N + i]);
        if (z.size() < 30) { med[k] = 0, sc[k] = 0; continue; }
        auto mid = [](std::vector<double> y) { std::nth_element(y.begin(), y.begin() + std::ptrdiff_t(y.size() / 2), y.end()); return y[y.size() / 2]; };
        med[k] = mid(z);
        std::vector<double> d;
        double mean = 0;
        for (double x : z) d.push_back(std::abs(x - med[k])), mean += std::abs(x - med[k]) / double(z.size());
        sc[k] = 1.4826 * mid(d);
        if (!(sc[k] > 0)) sc[k] = 1.2533 * mean;
      }
      for (std::size_t i = 0; i < N; ++i) {
        if (!ok[i]) { tradable[t * N + i] = 0; continue; }
        const bool o = (sc[0] > 0 && stat[i] < med[0] - 3 * sc[0]) || (sc[1] > 0 && stat[N + i] > med[1] + 3 * sc[1]) || (sc[2] > 0 && stat[2 * N + i] > med[2] + 3 * sc[2]);
        if (o) tradable[t * N + i] = 0;
        if (t + 1 == T) out += o, ++seen;
      }
    }
    std::printf("impossible bars dropped: %zu; outlier shares at the last close: %zu of %zu\n", dropped, out, seen);
  }
  // Days with correlations defined.
  std::vector<std::size_t> days;
  for (std::size_t t = 0; t < T; ++t)
    if (ic[t * F * 4] != 0.0) days.push_back(t);
  std::printf("%zu stocks, %zu days with features (%s .. %s)\n", N, days.size(), days.empty() ? "" : m.dates[days.front()].c_str(),
              days.empty() ? "" : m.dates[days.back()].c_str());
  for (std::size_t k = 0; k < 4; ++k) {
    std::printf("\n%s: mean daily correlation (t)\n", tn[k]);
    for (std::size_t j = 0; j < F; ++j) {
      double s1 = 0, s2 = 0, n = 0;
      for (std::size_t t : days) {
        const double x = ic[(t * F + j) * 4 + k];
        s1 += x, s2 += x * x, n += 1;
      }
      const double mu = n > 1 ? s1 / n : 0.0, sd = n > 1 ? std::sqrt(std::max(0.0, s2 / n - mu * mu)) : 0.0;
      std::printf("  %-22s %+.4f (%+6.1f)\n", fn[j], mu, sd > 0 ? mu / sd * std::sqrt(n) : 0.0);
    }
  }
  // The walk-forward model: each day's sums (fused kernels), accumulated over the days before
  // each close, solved for each target; predictions; then the plainest test of the direction: each
  // day buy the 10 stocks predicted highest at the open, sell at the close (always, and only on
  // days the 10 are predicted to gain on average). ft: [T][N][Km] with Fm inputs (the first nOwn
  // standardised within each day, the rest market-wide), 4 targets and the defined flag.
  struct Run {
    std::vector<double> ret, mkt;
    std::vector<std::string> dates;
    // The backtest's trades in the page's terms: the orders, those that bought, their net returns' sum
    // (from the buy price, after costs) and how many gained.
    double orders = 0, filled = 0, filledSum = 0, gained = 0;
    struct Pick {
      std::size_t stock;
      double expected, stop, take;  // the calibrated expected gain from the buy price if the order buys, net of its costs; the stop and take-profit from the entry (NaN: none)
      double limit;                 // the buy limit from the last close
    };
    std::vector<Pick> picks;
    std::vector<std::pair<std::size_t, double>> candidates;  // the pool's highest ranked for the next open (stock, score), for recording their opening auctions
    std::vector<std::size_t> pool;                           // the learnt pool for the next open (IBKR's daily bars are fetched for it)
    // The model's own result for the latest day's trades (the orders placed that morning, as the
    // backtest trades them), to compare with the broker's fills: stock, limit, filled (0 no, 1 at the
    // open, 2 during the day), entry, exit (0 close, 1 stop, 2 take-profit), exit price, net return.
    struct Check {
      std::size_t stock;
      double limit;
      int filled;
      double entry;
      int exit;
      double exitPx, ret;
    };
    std::string checkDate;
    std::vector<Check> check;
  };
  Run lastRun;
  const std::size_t nPick = std::getenv("X_NPICK") ? std::size_t(std::atol(std::getenv("X_NPICK"))) : 10;
  const double ridgeK = std::getenv("X_RIDGE") ? std::atof(std::getenv("X_RIDGE")) : 1e-6;
  const double forget = std::getenv("X_HALF") ? std::exp2(-1.0 / std::atof(std::getenv("X_HALF"))) : 1.0;
  auto model = [&](const std::vector<double>& ft, std::size_t Fm, std::size_t Km, std::size_t nOwn, const char* label) {
    ofm::v2::gen::V2Header<double> Q = P;
    Q.K = std::uint32_t(Km), Q.NT = std::uint32_t(Fm);
    const std::size_t D = Fm + 1, E = D * (D + 4);
    std::vector<double> zs(T * Fm * 2, 0.0), gram(T * E, 0.0), beta(T * 4 * D, 0.0), pred(T * N * 4, 0.0);
    ofm::v2::gen::Zscore<double> kz;
    kz.P = Q, kz.feat = const_cast<double*>(ft.data()), kz.zs = zs.data();
    ofm::detail::dispatch(kz, ofm::detail::groups(std::uint32_t(T * Fm), 64));
    for (std::size_t t = 0; t < T; ++t)  // market-wide inputs enter as they are
      for (std::size_t f = nOwn; f < Fm; ++f) zs[(t * Fm + f) * 2] = 0, zs[(t * Fm + f) * 2 + 1] = 1;
    ofm::v2::gen::Gram<double> kg;
    kg.P = Q, kg.feat = const_cast<double*>(ft.data()), kg.zs = zs.data(), kg.out = gram.data();
    if (D > 64) { std::fprintf(stderr, "too many inputs for the kernels (%zu > 63)\n", D - 1); std::exit(1); }
    ofm::detail::dispatch(kg, std::uint32_t(T));  // one workgroup per day (kernels/v2/gram.wgsl)
    // The sums of the days before each day (exponentially forgotten with --half), then every day's and
    // target's coefficients at once (kernels/v2/solve.wgsl).
    std::vector<double> acc(T * E, 0.0), rk = {ridgeK};
    for (std::size_t t = 2; t < T; ++t)
      for (std::size_t e = 0; e < E; ++e) acc[t * E + e] = forget * acc[(t - 1) * E + e] + gram[(t - 2) * E + e];
    {
      ofm::v2::gen::V2Header<double> S = Q;
      S.T = std::uint32_t(T), S.u0 = std::uint32_t(D), S.u1 = std::uint32_t(E);
      ofm::v2::gen::Solve<double> ks;
      ks.P = S, ks.acc = acc.data(), ks.rk = rk.data(), ks.beta = beta.data();
      ofm::detail::dispatch(ks, ofm::detail::groups(std::uint32_t(T * 4), 64));
    }
    ofm::v2::gen::Predict<double> kp;
    kp.P = Q, kp.feat = const_cast<double*>(ft.data()), kp.zs = zs.data(), kp.beta = beta.data(), kp.out = pred.data();
    ofm::detail::dispatch(kp, ofm::detail::groups(std::uint32_t(T * N), 64));
    const std::size_t warm = 250;
    std::vector<double> always, gated, mkt;
    // The trades' settings, learnt continuously: the pool (the 250 * 2^c most traded by traded value,
    // c 0 to 3), the ranking (the expected return net of the share's own costs, over its volatility to
    // the power 1 - beta: 0 ranks in volatilities, 1 by the expected net return), the entry (a limit buy d standard
    // deviations below the previous close, d -2 to 2: filled at the open when the stock opens at or
    // below it, else during the day when its low trades half a spread through it, else not traded;
    // after a fill during the day the take-profit is not credited, its high may have come first), and
    // the levels, from the entry, relative to each stock's
    // own moves from the open over its last 100 days (log low/open and log high/open): the stop-loss
    // at the fall it reached on a fraction p_s of those days, blended geometrically with its parabolic
    // SAR by alpha (when the SAR trails below the open), and the take-profit at the rise it reached on
    // a fraction p_t of them (p 1/2 to 1/10: levels the stock reaches on its typical days, so they are
    // hit). The stop first when the day's bar touched both; a stop
    // sells half a spread below its level, a take-profit fills only when the high clears it by half a
    // spread; otherwise sold at the close. Every setting on a lattice keeps a record; each day the pool,
    // ranking and entry with the best record (mean shrunk by its t-statistic, averaged over the level
    // settings) and, at them, the levels with
    // the best mean over downside deviation (Sortino) so far are refined to a continuous optimum by a
    // parabola through their neighbours along each axis, and today's trades use them.
    // The UK's daily cap on AIM shares (learnt: 0, 2, .., 10 of the day's trades; with --stamp-free
    // only: AIM shares pay no stamp duty, so the net ranking favours them, and the cap weighs that
    // against concentrating in small companies).
    const bool aimCap = costKind == 1 && !stampFree.empty();
    constexpr std::size_t nC = 4, nB = 5, nE = 9, nP = 7, nA = 5, nLv = nP * nP * nA, nD = 7;
    // The selection's code k (one discrete axis): the AIM cap 2 * (k / 2) (UK; else none) and the gate
    // k % 2 (1: only trades expected to gain after their costs; 0: the top 10 whatever their
    // expectation, as the forecasts understate the gains of some), learnt like the rest.
    const std::size_t nK = (aimCap ? 6 : 1) * 2, nR = nC * nB * nK, nV = nR * nE * nLv;
    const auto capOf = [&](double k) { return aimCap ? 2.0 * std::floor(std::floor(k + 0.5) / 2) : double(nPick); };
    const auto gateOf = [&](double k) { return std::size_t(std::floor(k + 0.5)) % 2 == 1; };
    const auto axis = [&](std::size_t d, double x) {  // lattice coordinate -> setting
      if (d == 0) return 250.0 * std::exp2(x);
      if (d == 1) return x / double(nB - 1);
      if (d == 2) return x;  // the selection's code (capOf, gateOf)
      if (d == 3) return -2.0 + 0.5 * x;  // the limit's depth d (standard deviations below the previous close)
      if (d == 6) return safetyOnly ? 1.5 * std::pow(std::sqrt(2.0), x) : x / double(nA - 1);  // the SAR's weight; for a safety stop (no SAR) its learnt limit: at most this many of the share's own daily sigmas below the anchor
      if (d == 4 && safetyOnly) return 0.05 * std::pow(0.2, x / double(nP - 1));  // a safety stop's tail, 1/20 .. 1/100
      return 0.5 * std::pow(0.2, x / double(nP - 1));  // the tail fraction p, 1/2 .. 1/10
    };
    const std::size_t dims[nD] = {nC, nB, nK, nE, nP, nP, nA};
    std::vector<double> s1v(nV, 0.0), s2v(nV, 0.0), dnv(nV, 0.0);
    std::vector<double> scv(nV, 0.0), sov(nV, 0.0);  // each setting's scores for the next day (kernels/v2/records.wgsl)
    std::vector<double> dvals(nE), avals(nA);  // the lattice's limit depths and SAR weights, for the kernel
    for (std::size_t q = 0; q < nE; ++q) dvals[q] = axis(3, double(q));
    for (std::size_t q = 0; q < nA; ++q) avals[q] = axis(6, double(q));  // sums of the results, their squares, the losses' squares
    double nDays = 0;
    std::vector<double> adaptiveLevels, byRank[2];
    std::vector<std::array<double, nD>> used;
    std::vector<std::array<double, 5>> paths;  // per candidate trade: filled, the stop and the take-profit from the entry, stopped, took profit
    std::vector<std::string> sd;
    std::vector<Run::Check> check;  // the latest day's trades (Run::check)
    double tradeOrders = 0, tradeFilled = 0, tradeSum = 0, tradeGained = 0;  // Run's trade counts
    std::string checkDate;
    std::FILE* tradeLog = nullptr;
    if (std::getenv("X_TRADELOG") && std::string(label).find("interactions") != std::string::npos) {
      tradeLog = std::fopen(std::getenv("X_TRADELOG"), "w");
      std::fprintf(tradeLog, "date,ticker,rank,pred,sigma,liq_rank,spread,low,high,close,stop,take,market,exit,ret,gap,limit,filled\n");
    }
    // The record's scores, written by records.wgsl with each day: its mean shrunk by its t-statistic,
    // and its mean over its downside deviation (Sortino); 0 before 2 days.
    const auto score = [&](std::size_t g) { return scv[g]; };
    const auto ddScore = [&](std::size_t g) { return sov[g]; };
    const auto index = [&](const std::size_t* c) { return (((((c[0] * nB + c[1]) * nK + c[2]) * nE + c[3]) * nP + c[4]) * nP + c[5]) * nA + c[6]; };
    const auto coords = [&](std::size_t g, std::size_t* c) {
      c[6] = g % nA, g /= nA, c[5] = g % nP, g /= nP, c[4] = g % nP, g /= nP, c[3] = g % nE, g /= nE, c[2] = g % nK, g /= nK, c[1] = g % nB, c[0] = g / nB;
    };
    // The continuous settings {pool, beta, AIM cap, d, p_s, p_t, alpha} from the records so far.
    // The pool, ranking and entry by their records' mean over every level setting, as the levels are
    // chosen after them (the single best of 44,100 settings is mostly luck); X_SINGLEBEST=1
    // (research): by the single best setting's record, as before.
    const bool robust = std::getenv("X_SINGLEBEST") == nullptr;
    std::vector<double> entryScore(nR * nE, 0.0), entrySortino(nR * nE, 0.0);
    const bool capBySortino = std::getenv("X_CAPRET") == nullptr;
    const auto learnt = [&] {
      std::size_t best = 0;
      if (robust) {
        for (std::size_t r = 0; r < nR * nE; ++r) {
          double sm = 0;
          for (std::size_t g = r * nLv; g < (r + 1) * nLv; ++g) sm += score(g);
          entryScore[r] = sm / double(nLv);
        }
        for (std::size_t r = 1; r < nR * nE; ++r)
          if (entryScore[r] > entryScore[best / nLv]) best = r * nLv;
      } else {
        for (std::size_t g = 1; g < nV; ++g)
          if (score(g) > score(best)) best = g;
      }
      // The AIM cap, a limit on risk, by Sortino (averaged over the level settings), at the pool,
      // ranking and entry chosen; X_CAPRET=1 (research): by return with them.
      if (nK > 1 && capBySortino) {
        for (std::size_t r = 0; r < nR * nE; ++r) {
          double sm = 0;
          for (std::size_t g = r * nLv; g < (r + 1) * nLv; ++g) sm += ddScore(g);
          entrySortino[r] = sm / double(nLv);
        }
        std::size_t c[nD];
        coords(best, c);
        const auto at = [&](std::size_t k) { c[2] = k; return entrySortino[index(c) / nLv]; };
        std::size_t bk = 0;
        for (std::size_t k = 1; k < nK; ++k)
          if (at(k) > at(bk)) bk = k;
        c[2] = bk;
        best = index(c);
      }
      const std::size_t r0 = best / nLv;
      std::size_t lv = r0 * nLv;
      for (std::size_t g = lv + 1; g < (r0 + 1) * nLv; ++g)
        if (ddScore(g) > ddScore(lv)) lv = g;
      std::size_t bc[nD], lc[nD];
      coords(best, bc), coords(lv, lc);
      std::array<double, nD> x{};
      for (std::size_t d = 0; d < nD; ++d) {
        const bool pick = d < 4;  // the pool, ranking, AIM cap and entry by return; the levels by Sortino
        const auto f = [&](std::size_t g) {
          if (d == 2 && nK > 1 && capBySortino) return entrySortino[g / nLv];
          return pick ? (robust ? entryScore[g / nLv] : score(g)) : ddScore(g);
        };
        std::size_t at[nD];
        for (std::size_t e = 0; e < nD; ++e) at[e] = pick ? bc[e] : lc[e];
        double off = 0;
        if (d != 2 && at[d] > 0 && at[d] + 1 < dims[d]) {  // the selection's code is discrete
          std::size_t lo[nD], hi[nD];
          for (std::size_t e = 0; e < nD; ++e) lo[e] = hi[e] = at[e];
          --lo[d], ++hi[d];
          const double fm = f(index(lo)), f0 = f(index(at)), fp = f(index(hi)), den = fm - 2 * f0 + fp;
          if (den < 0) off = std::min(0.5, std::max(-0.5, 0.5 * (fm - fp) / den));
        }
        x[d] = axis(d, double(at[d]) + off);
      }
      return x;
    };
    // Each stock's place by traded value (feature 19) among the day's stocks with a prediction (and,
    // in the backtest, a known outcome).
    std::vector<double> liqRank(N);
    const auto rankLiquidity = [&](std::size_t t, bool outcome) {
      std::vector<std::pair<double, std::size_t>> l;
      for (std::size_t i = 0; i < N; ++i) {
        liqRank[i] = 1e300;
        const double liq = ft[(t * N + i) * Km + 19];
        if ((!outcome || ft[(t * N + i) * Km + Km - 1] > 0) && std::isfinite(liq) && t >= lag && ft[((t - lag) * N + i) * Km + Km - 1] > 0) l.emplace_back(liq, i);
      }
      std::sort(l.begin(), l.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      for (std::size_t r = 0; r < l.size(); ++r) liqRank[l[r].second] = double(r);
    };
    // A filled trade's costs as a share of its value (as kernels/v2/levels.wgsl's costOf), stock i's.
    const auto costOf = [&](double entry, double r, std::size_t i) {
      const double* f = &fee[i * 7];
      const double cap = f[1];
      if (f[0] == 1) {
        const double sell = cap * (1 + r);
        return (std::max(f[3], f[2] * cap) + f[4] * cap + std::max(f[3], f[2] * sell) + f[4] * sell) / cap + f[5] + f[6] * (2 + r);
      }
      if (f[0] == 2) {
        const double n = std::max(1.0, std::floor(cap / entry)), v = n * entry, sell = v * (1 + r);
        const double cb = std::min(std::max(f[3], f[2] * n), 0.01 * v) + f[4] * n, cs = std::min(std::max(f[3], f[2] * n), 0.01 * sell) + f[4] * n;
        return (cb + cs + 0.0000278 * sell + std::min(0.000166 * n, 8.3)) / v;
      }
      return 0.0;
    };
    // The ranking's score: the expected open-to-close net of the share's own costs (at its last close;
    // the commissions per share make cheap shares dearer in the US, the stamp duty non-AIM shares in
    // the UK), in units of its volatility to the power beta - 1 (beta 1: the expected net return,
    // 0: in volatilities). X_GROSSRANK=1 (research): ranked before costs, as before.
    const bool netRank = !std::getenv("X_GROSSRANK");
    const bool fillAtLimit = std::getenv("X_FILLATLIMIT") != nullptr;  // research: an open fill pays the limit, not the open
    const bool verify = std::getenv("X_VERIFY") != nullptr;
    std::size_t verifyN = 0, verifyBad = 0;
    const auto rankScore = [&](std::size_t t, std::size_t i, double b) {
      const std::size_t src = (t - lag) * N + i;
      const double ls = ft[src * Km + 20], c = netRank && fee[i * 7] > 0 ? costOf(m.close(t, i), 0.0, i) : 0.0;
      return (pred[src * 4] * std::exp(ls) - c) * std::exp((b - 1) * ls);
    };
    // The expected return, calibrated walk-forward: the model's raw expectation (its open-to-close
    // forecast less the share's costs) ranks well but is shrunk towards zero, so a pick's calibrated
    // expectation is what the backtest's earlier picks with a similar raw expectation (bands of 0.2%,
    // widened to at least 100 picks) actually returned net of costs, an unfilled one as 0. Only picks
    // whose day is over enter (calAdd after the day's decisions); before any, the raw expectation.
    constexpr int calBins = 41;
    constexpr double calW = 0.002;
    std::vector<double> calN(calBins, 0.0), calS(calBins, 0.0), calF(calBins, 0.0);  // picks, their net returns' sum, those filled
    const auto rawExpected = [&](std::size_t t, std::size_t i) {
      const std::size_t src = (t - lag) * N + i;
      const double c = fee[i * 7] > 0 ? costOf(m.close(t, i), 0.0, i) : 0.0;
      return std::expm1(pred[src * 4] * std::exp(ft[src * Km + 20])) - c;
    };
    const auto calBin = [&](double e) { return int(std::min<double>(calBins - 1, std::max(0.0, std::floor(e / calW + 0.5) + calBins / 2))); };
    // filled: the mean over the picks that bought (the gain from the buy price, what the page shows);
    // else over every pick (an unfilled one 0; the gate's). The same bands, so the same sign.
    const auto calibrated = [&](double e, bool filled = false) {
      const int b = calBin(e);
      double n = calN[b], sm = calS[b], nf = calF[b];
      for (int w = 1; n < 100 && w < calBins; ++w) {
        if (b - w >= 0) n += calN[b - w], sm += calS[b - w], nf += calF[b - w];
        if (b + w < calBins) n += calN[b + w], sm += calS[b + w], nf += calF[b + w];
      }
      if (filled) return nf > 0 ? sm / nf : e;
      return n > 0 ? sm / n : e;
    };
    const auto calAdd = [&](double e, double ret, bool filled) { calN[calBin(e)] += 1, calS[calBin(e)] += ret, calF[calBin(e)] += filled; };
    // The day's trades: up to 10 among the pool most traded, ranked with beta, at most cap of them AIM
    // shares; gated: only those whose calibrated expectation is a gain.
    const auto capTo = [&](std::size_t t, std::vector<std::pair<double, std::size_t>>& v, double cap, bool gate) {
      std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      const std::size_t most = std::size_t(std::floor(cap + 0.5));
      std::vector<std::pair<double, std::size_t>> w;
      std::size_t aim = 0;
      for (const auto& e : v) {
        if (w.size() >= nPick) break;
        if (gate && !(calibrated(rawExpected(t, e.second)) > 0)) continue;  // gated: only trades expected to gain after their costs
        const bool isAim = aimCap && stampOf[e.second] == 0;
        if (isAim && aim >= most) continue;
        aim += isAim, w.push_back(e);
      }
      return w;
    };
    // A share the trade's amount buys at least one of (at its last close; LSE prices in pence): a
    // dearer one (BRK.A at USD 1,000) cannot be ordered, so it is not picked, live or in the backtest.
    const auto affordable = [&](std::size_t t, std::size_t i) {
      const double px = m.close(t, i) * (costKind == 1 ? 0.01 : 1.0), cap = fee[i * 7] > 0 ? fee[i * 7 + 1] : capital;
      return px > 0 && std::floor(cap / px) >= 1;
    };
    const auto ranked = [&](std::size_t t, double pool, double b, double code) {
      std::vector<std::pair<double, std::size_t>> v;
      for (std::size_t i = 0; i < N; ++i) {
        if (!(liqRank[i] < pool) || !tradable[t * N + i] || !affordable(t, i)) continue;
        v.emplace_back(rankScore(t, i, b), i);
      }
      return capTo(t, v, capOf(code), gateOf(code));
    };
    // A stock's moves from the open over its last 100 days to the close t, sorted: the falls to the
    // low and the rises to the high (logs), kept for the day.
    std::vector<std::size_t> tailDay(N, ~std::size_t(0));
    std::vector<std::vector<double>> falls(N), rises(N);
    const auto tails = [&](std::size_t t, std::size_t i) {
      if (tailDay[i] != t) {
        tailDay[i] = t, falls[i].clear(), rises[i].clear();
        for (std::size_t u = t >= 99 ? t - 99 : 0; u <= t; ++u) {
          const double o = m.open(u, i), h = m.high(u, i), l = m.low(u, i);
          if (o > 0 && h >= o && l > 0 && l <= o) falls[i].push_back(std::log(l / o)), rises[i].push_back(std::log(h / o));
        }
        std::sort(falls[i].begin(), falls[i].end()), std::sort(rises[i].begin(), rises[i].end());
      }
      return falls[i].size() >= 20;
    };
    const auto quantile = [](const std::vector<double>& z, double q) {
      const double x = q * double(z.size() - 1);
      const std::size_t k = std::min(z.size() - 2, std::size_t(x));
      return z[k] + (x - double(k)) * (z[k + 1] - z[k]);
    };
    // The stop and the take-profit as fractions of the reference price (the open; the last close for
    // the next open's orders); none (0 and infinity) without 20 days of history.
    const auto levelsAt = [&](std::size_t t, std::size_t i, double ref, double ps, double pt, double al) {
      if (closeOnly || !tails(t, i)) return std::make_pair(0.0, 1e300);
      if (safetyOnly) return std::make_pair(std::exp(std::max(quantile(falls[i], ps), -al * std::exp(ft[(t * N + i) * Km + 20]))), 1e300);  // a rare day's fall; no take-profit, else sold at the close
      double st = quantile(falls[i], ps);
      if (std::isfinite(sar[t * N + i])) st = (1 - al) * st + al * std::log(std::min(1.0, sar[t * N + i] / ref));  // a SAR above the open: stopped at the open
      return std::make_pair(std::exp(st), std::exp(quantile(rises[i], 1 - pt)));
    };
    // One trade: the limit (a fraction of the previous close), whether it filled and where, its exit
    // (0 close, 1 stop, 2 take-profit) and its return from the entry.
    struct Trade {
      double limit, entry;
      int filled, exit;
      double ret, stop, take, exitPx;
    };
    const auto trade = [&](std::size_t t, std::size_t i, double d, double ps, double pt, double al) {
      Trade r{};
      const double pc = m.close(t, i), sig = std::exp(ft[(t * N + i) * Km + 20]);
      const double o = m.open(t + 1, i), hi = m.high(t + 1, i), lo = m.low(t + 1, i), c = m.close(t + 1, i), half = std::exp(0.5 * spread[t * N + i]);
      const double L = pc * std::exp(-d * sig);
      r.limit = L / pc - 1;
      bool intraday = false;
      if (o <= L) r.entry = fillAtLimit ? L : o, r.filled = 1;
      else if (lo <= L / half) r.entry = L, r.filled = 2, intraday = true;
      else return r;
      // The levels as the orders set them before the open (as kernels/v2/levels.wgsl): from the anchor,
      // the limit or the previous close when the limit is above it.
      const double anchor = std::min(L, pc);
      const auto [fs, fk] = levelsAt(t, i, anchor, ps, pt, al);
      if (fs * half * half > 1) return Trade{r.limit, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0};  // a stop within a spread of the anchor: no order (cash)
      const double stop = anchor * fs, take = anchor * fk;
      r.stop = stop / r.entry - 1, r.take = fk < 1e300 ? take / r.entry - 1 : 0.0;
      double px = c;
      if (!intraday && o <= stop) px = o / half, r.exit = 1;  // opened through the stop: sold at the open
      else if (lo <= stop) px = stop / half, r.exit = 1;
      else if (!intraday && hi >= take * half) px = std::max(take, o), r.exit = 2;  // a take-profit below the open fills at the open
      r.ret = px / r.entry - 1, r.exitPx = px;
      r.ret -= costOf(r.entry, r.ret, i);
      return r;
    };
    // The day's result: every trade one unit, a trade not filled stays in cash.
    const auto dayResult = [&](std::size_t t, const std::vector<std::pair<double, std::size_t>>& w, double d, double ps, double pt, double al) {
      double rg = 0;
      for (const auto& e : w) rg += trade(t, e.second, d, ps, pt, al).ret / double(nPick);  // a tenth each: a missing trade stays in cash
      return rg;
    };
    for (std::size_t q = warm; q < days.size(); ++q) {
      const std::size_t t = days[q];
      if (t + 1 >= T || t < lag) continue;
      // Only what is known at the close t decides: the stocks with a close and a prediction (whether a
      // stock trades tomorrow is not known; one that does not stays unfilled, in cash).
      double ms = 0, mn = 0, known = 0;
      for (std::size_t i = 0; i < N; ++i) {
        if (!(tradable[t * N + i] && std::isfinite(m.close(t, i)) && m.close(t, i) > 0 && ft[((t - lag) * N + i) * Km + Km - 1] > 0)) continue;
        known += 1;
        if (std::isfinite(m.close(t + 1, i)) && m.close(t + 1, i) > 0) ms += m.close(t + 1, i) / m.close(t, i) - 1, mn += 1;  // the market's day (a diagnostic)
      }
      if (known < 30) continue;
      rankLiquidity(t, false);
      std::vector<std::vector<std::pair<double, std::size_t>>> v(nR);
      for (std::size_t k = 0; k < nR; ++k) v[k] = ranked(t, axis(0, double(k / (nB * nK))), axis(1, double(k / nK % nB)), axis(2, double(k % nK)));
      const auto& all0 = v[((nC - 1) * nB) * nK + nK - 2], &all1 = v[((nC - 1) * nB + nB - 1) * nK + nK - 2];  // no AIM cap, no gate
      double r = 0, pm = 0;
      const auto oc = [&](std::size_t i) { const double x = m.close(t + 1, i) / m.open(t + 1, i) - 1; return std::isfinite(x) ? x : 0.0; };  // no bar: cash
      for (const auto& e : all0) r += oc(e.second) / double(all0.size()), pm += e.first / double(all0.size());
      always.push_back(r), gated.push_back(pm > 0 ? r : 0.0), mkt.push_back(mn > 0 ? ms / mn : 0.0), sd.push_back(m.dates[t + 1]);
      double rk = 0;
      for (const auto& e : all1) rk += oc(e.second) / double(all1.size());
      byRank[1].push_back(rk);
      // Today's trades with the settings learnt so far; then every lattice point's record takes today.
      const auto x = learnt();
      const auto today = ranked(t, x[0], x[1], x[2]);
      adaptiveLevels.push_back(dayResult(t, today, x[3], x[4], x[5], x[6])), used.push_back(x);
      for (std::size_t k = 0; k < today.size(); ++k) {  // the trades, for the diagnostics and the log
        const std::size_t i = today[k].second, src = (t - lag) * N + i;
        const double o = m.open(t + 1, i);
        const Trade tr = trade(t, i, x[3], x[4], x[5], x[6]);
        calAdd(rawExpected(t, i), tr.ret, tr.filled > 0);
        tradeOrders += 1;
        if (tr.filled > 0) tradeFilled += 1, tradeSum += tr.ret, tradeGained += tr.ret > 0;  // the day is over: this pick's result joins the calibration (used from the next day's decisions)
        if (t + 2 == T) {  // the latest day: the morning's orders, for the comparison with the broker's fills
          if (k == 0) check.clear(), checkDate = m.dates[t + 1];
          check.push_back({i, m.close(t, i) * (1 + tr.limit), tr.filled, tr.entry, tr.exit, tr.exitPx, tr.ret});
        }
        paths.push_back({double(tr.filled > 0), tr.stop, tr.take, double(tr.exit == 1), double(tr.exit == 2)});
        if (tradeLog)  // one line per trade, for the single-trade analysis (X_TRADELOG; local, not published)
          std::fprintf(tradeLog, "%s,%s,%zu,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,%.6g,%.6g,%d\n", m.dates[t + 1].c_str(), m.tickers[i].c_str(), k,
                       pred[src * 4], std::exp(ft[src * Km + 20]), liqRank[i], spread[t * N + i], m.low(t + 1, i) / o - 1, m.high(t + 1, i) / o - 1,
                       m.close(t + 1, i) / o - 1, tr.stop, tr.take, ms / mn, tr.exit, tr.ret, m.close(t, i) > 0 ? o / m.close(t, i) - 1 : 0.0, tr.limit, tr.filled);
      }
      // Every lattice point's record takes today, in two fused kernels: levels.wgsl trades each of the
      // day's candidates (a slot: its prices, sigma, half-spread, SAR and level quantiles at the
      // lattice's tails) once under every entry and exit setting; records.wgsl adds each setting's
      // day (its ranking's candidates) to its record and scores it for the next day.
      {
        std::vector<double> cand(nR * nPick, -1.0), st;
        std::vector<long> slotOf(N, -1);
        std::size_t ns = 0;
        for (std::size_t k = 0; k < nR; ++k)
          for (std::size_t j = 0; j < v[k].size(); ++j) {
            const std::size_t i = v[k][j].second;
            if (slotOf[i] < 0) {
              slotOf[i] = long(ns++);
              const bool ok = tails(t, i);
              st.insert(st.end(), {m.open(t + 1, i), m.high(t + 1, i), m.low(t + 1, i), m.close(t + 1, i), m.close(t, i), std::exp(ft[(t * N + i) * Km + 20]),
                                   std::exp(0.5 * spread[t * N + i]), std::isfinite(sar[t * N + i]) ? sar[t * N + i] : 0.0, ok ? (fillAtLimit ? 2.0 : 1.0) : 0.0});
              st.insert(st.end(), &fee[i * 7], &fee[i * 7] + 7);
              for (std::size_t q = 0; q < nP; ++q) st.push_back(ok ? quantile(falls[i], axis(4, double(q))) : 0.0);
              for (std::size_t q = 0; q < nP; ++q) st.push_back(ok ? (safetyOnly ? 50.0 : quantile(rises[i], 1 - axis(5, double(q)))) : 0.0);  // 50: no take-profit
            }
            cand[k * nPick + j] = double(slotOf[i]);
          }
        const std::size_t nLe = nE * nLv;  // entry and exit settings per share
        std::vector<double> ret(ns * nLe, 0.0);
        ofm::v2::gen::V2Header<double> H{};
        H.T = std::uint32_t(nLe), H.N = std::uint32_t(ns), H.K = std::uint32_t(16 + 2 * nP);
        H.t0 = std::uint32_t(nE), H.u0 = std::uint32_t(nP), H.u1 = std::uint32_t(nA), H.u2 = closeOnly ? 1u : safetyOnly ? 2u : 0u;
        ofm::v2::gen::Levels<double> kl;
        kl.P = H, kl.st = st.data(), kl.dv = dvals.data(), kl.av = avals.data(), kl.ret = ret.data();
        ofm::detail::dispatch(kl, ofm::detail::groups(std::uint32_t(ns * nLe), 64));
        // X_VERIFY (a check): every slot under a spread of settings, the kernel's return against the
        // trade path's (trade(), the one the published trades and the check use); they must agree.
        if (verify) {
          std::vector<std::size_t> slotStock(ns);
          for (std::size_t i = 0; i < N; ++i)
            if (slotOf[i] >= 0) slotStock[std::size_t(slotOf[i])] = i;
          for (std::size_t sl = 0; sl < ns; ++sl)
            for (std::size_t le = sl % 97; le < nLe; le += 97) {
              const std::size_t ca = le % nA, ct = le / nA % nP, cs = le / (nA * nP) % nP, ce = le / (nA * nP * nP);
              const double want = trade(t, slotStock[sl], dvals[ce], axis(4, double(cs)), axis(5, double(ct)), avals[ca]).ret, got = ret[sl * nLe + le];
              ++verifyN;
              if (std::fabs(want - got) > 1e-9) {
                if (++verifyBad <= 5) std::printf("  verify: %s %s setting %zu: kernel %+.6f, trade %+.6f\n", m.dates[t + 1].c_str(), m.tickers[slotStock[sl]].c_str(), le, got, want);
              }
            }
        }
        ofm::v2::gen::V2Header<double> R{};
        R.T = std::uint32_t(nV), R.u0 = std::uint32_t(nLe), R.NT = std::uint32_t(nPick), R.t0 = std::uint32_t(nDays + 1);
        ofm::v2::gen::Records<double> kr;
        kr.P = R, kr.cand = cand.data(), kr.ret = ret.data(), kr.s1 = s1v.data(), kr.s2 = s2v.data(), kr.dn = dnv.data(), kr.sc = scv.data(), kr.so = sov.data();
        ofm::detail::dispatch(kr, ofm::detail::groups(std::uint32_t(nV), 64));
      }
      nDays += 1;
    }
    if (tradeLog) std::fclose(tradeLog);
    std::printf("calibration (raw expectation band: picks, mean net return a pick (unfilled as 0), filled, mean net return a filled trade):");
    for (int b = 0; b < calBins; ++b)
      if (calN[b] >= 100) std::printf(" %+.1f%%: %.0f, %+.3f%%, %.0f, %+.3f%%;", 100 * (b - calBins / 2) * calW, calN[b], 100 * calS[b] / calN[b], calF[b], calF[b] > 0 ? 100 * calS[b] / calF[b] : 0.0);
    std::printf("\n");
    if (verify) std::printf("verify: %zu kernel trades against the trade path, %zu differ\n", verifyN, verifyBad);
    if (verifyBad > 0) {  // nothing is published from a backtest whose two implementations disagree
      std::fprintf(stderr, "verify failed: the fused kernel and the trade path disagree on %zu trades\n", verifyBad);
      std::exit(3);
    }
    auto stats = [](const std::vector<double>& x) {
      double w = 0, s1 = 0, s2 = 0;
      for (double y : x) w += std::log1p(y), s1 += y, s2 += y * y;
      const double n = double(x.size()), mu = s1 / n, sdv = std::sqrt(std::max(0.0, s2 / n - mu * mu));
      return std::make_pair(std::expm1(w * 252.0 / n), sdv > 0 ? mu / sdv * std::sqrt(252.0) : 0.0);
    };
    const auto a1 = stats(always), a2 = stats(gated);
    std::size_t traded = 0;
    for (double y : gated) traded += y != 0.0;
    std::printf("\n%s (%s .. %s), the 10 highest predicted, open to close:\n", label, sd.front().c_str(), sd.back().c_str());
    std::printf("  every day                        %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a1.first, a1.second);
    std::printf("  ranked by expected return        %+7.1f%% a year (Sharpe %5.2f)\n", 100 * stats(byRank[1]).first, stats(byRank[1]).second);
    const auto fx = learnt();
    {
      const auto a = stats(adaptiveLevels);
      double mx[nD] = {};
      for (const auto& x : used)
        for (std::size_t d = 0; d < nD; ++d) mx[d] += x[d] / double(used.size());
      double gated = 0;
      for (const auto& x : used) gated += gateOf(x[2]) / double(used.size());
      std::printf("  self-adaptive pool, ranking, levels %+6.1f%% a year (Sharpe %5.2f); mean pool %.0f, beta %.2f, gated on %.0f%% of days, limit %.2f sd, stop 1 day in %.1f, take 1 day in %.1f, %s %.2f\n",
                  100 * a.first, a.second, mx[0], mx[1], 100 * gated, mx[3], 1 / mx[4], 1 / mx[5], safetyOnly ? "stop at most (sd)" : "SAR weight", mx[6]);
      std::printf("  learnt for the next open: pool %.0f, beta %.2f, AIM cap %.0f, gate %s, limit %.2f sd, stop 1 day in %.1f, take 1 day in %.1f, %s %.2f\n", fx[0], fx[1],
                  capOf(fx[2]), gateOf(fx[2]) ? "on" : "off", fx[3], 1 / fx[4], 1 / fx[5], safetyOnly ? "stop at most (sd)" : "SAR weight", fx[6]);
      // The trades' levels from the open and how often they were touched.
      auto qt = [&](std::size_t d, double p) {
        std::vector<double> z;
        for (const auto& x : paths) z.push_back(x[d]);
        std::nth_element(z.begin(), z.begin() + std::ptrdiff_t(p * double(z.size() - 1)), z.end());
        return z[std::size_t(p * double(z.size() - 1))];
      };
      double fl = 0, st = 0, tk = 0;
      for (const auto& x : paths) fl += x[0], st += x[3], tk += x[4];
      if (closeOnly)
        std::printf("  candidate trades: %zu, filled %.1f%%; sold at the close\n", paths.size(), 100 * fl / double(paths.size()));
      else
        std::printf("  candidate trades: %zu, filled %.1f%%; stop from the entry: median %+.2f%% (10%% %+.2f%%, 90%% %+.2f%%), touched %.1f%% of filled; take-profit: median "
                  "%+.2f%% (10%% %+.2f%%, 90%% %+.2f%%), taken %.1f%% of filled\n",
                  paths.size(), 100 * fl / double(paths.size()), 100 * qt(1, 0.5), 100 * qt(1, 0.1), 100 * qt(1, 0.9), 100 * st / std::max(fl, 1.0), 100 * qt(2, 0.5),
                  100 * qt(2, 0.1), 100 * qt(2, 0.9), 100 * tk / std::max(fl, 1.0));
    }
    std::printf("  only days predicted to gain      %+7.1f%% a year (Sharpe %5.2f), %zu of %zu days\n", 100 * a2.first, a2.second, traded, gated.size());
    // The next open's trades: the 10 highest predictions lag closes before the last close, among
    // the stocks with a bar at the last close (their targets are not known yet), from the pool and
    // ranked and with the levels by the settings learnt over the whole backtest. Expected open-to-close: the prediction
    // times the stock's volatility (feature 20, its log). The SAR stop is from the last close (the
    // open is not known yet).
    lastRun = Run{adaptiveLevels, mkt, sd, {}};
    lastRun.check = check, lastRun.checkDate = checkDate;
    lastRun.orders = tradeOrders, lastRun.filled = tradeFilled, lastRun.filledSum = tradeSum, lastRun.gained = tradeGained;
    {
      const std::size_t t = T - 1;
      std::vector<std::pair<double, std::size_t>> v;
      if (t >= lag) {
        rankLiquidity(t, false);
        for (std::size_t i = 0; i < N; ++i) {
          if (!(liqRank[i] < fx[0]) || !std::isfinite(m.close(t, i)) || !(m.close(t, i) > 0) || !affordable(t, i)) continue;
          if (!tradable[t * N + i]) continue;
          v.emplace_back(rankScore(t, i, fx[1]), i);
        }
      }
      constexpr std::size_t nCand = 60;
      std::partial_sort(v.begin(), v.begin() + std::ptrdiff_t(std::min<std::size_t>(std::max(nPick, nCand), v.size())), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      for (std::size_t q = 0; q < std::min<std::size_t>(nCand, v.size()); ++q) lastRun.candidates.emplace_back(v[q].second, v[q].first);
      for (const auto& e : v) lastRun.pool.push_back(e.second);
      const auto picks = capTo(t, v, capOf(fx[2]), gateOf(fx[2]));  // as learnt: the AIM cap and the gate, the backtest's selection
      for (std::size_t q = 0; q < picks.size(); ++q) {
        const std::size_t i = picks[q].second;
        // The limit from the last close.
        const double limit = std::exp(-fx[3] * std::exp(ft[(t * N + i) * Km + 20]));
        // The levels from the expected entry: the limit, or the last close when the limit is above it
        // (a limit above the close buys at about the open, not at the limit).
        const auto [fs, fk] = levelsAt(t, i, m.close(t, i) * std::min(limit, 1.0), fx[4], fx[5], fx[6]);
        if (fs * std::exp(spread[t * N + i]) > 1) continue;  // a stop within a spread of the entry: skipped, as in the backtest
        lastRun.picks.push_back({i, calibrated(rawExpected(t, i), true), fs > 0 ? fs - 1 : kNaN, fk < 1e300 ? fk - 1 : kNaN, limit - 1});  // calibrated: the gain from the buy price if it buys, net of costs
      }
    }
    return pred;
  };
  // The features with their interactions with liquidity and volatility (kernels/v2/interact.wgsl):
  // momentum 2d, volume shock 4d, open-to-close, gap, close in range, intraday and overnight
  // momentum 64d.
  const std::vector<double> sigs = {0, 6, 12, 14, 15, 23, 26};
  // The residual reversal (5 and 21 days), the tug of war, the largest return, the distance from the
  // 52-week high and the idiosyncratic volatility (kernels/v2/resid.wgsl); X_RES picks them (a mask, research).
  const std::string resMask = std::getenv("X_RES") ? std::getenv("X_RES") : "111101";  // the 52-week high did not help (UK and US)
  std::vector<std::size_t> resUsed;
  for (std::size_t f = 0; f < 6 && f < resMask.size(); ++f)
    if (resMask[f] == '1') resUsed.push_back(f);
  const std::size_t NR = std::getenv("X_NORES") ? 0 : resUsed.size();
  const std::size_t MI = sigs.size(), FI = F + 2 * MI + NR, KI = FI + 5;
  std::vector<double> ftI(T * N * KI, 0.0);
  {
    std::vector<double> mk(T, 0.0), rs(T * N * 6, 0.0);
    ofm::v2::gen::Mret<double> km;
    km.P = P, km.bars = bars.data(), km.out = mk.data();
    ofm::detail::dispatch(km, ofm::detail::groups(std::uint32_t(T), 64));
    ofm::v2::gen::Resid<double> kr;
    kr.P = P, kr.bars = bars.data(), kr.mk = mk.data(), kr.out = rs.data();
    ofm::detail::dispatch(kr, ofm::detail::groups(std::uint32_t(N), 64));
    std::vector<double> zs(T * F * 2, 0.0), ia(T * N * 2 * MI, 0.0);
    ofm::v2::gen::Zscore<double> kz;
    kz.P = P, kz.feat = feat.data(), kz.zs = zs.data();
    ofm::detail::dispatch(kz, ofm::detail::groups(std::uint32_t(T * F), 64));
    ofm::v2::gen::V2Header<double> Q = P;
    Q.u0 = std::uint32_t(MI);
    ofm::v2::gen::Interact<double> kx;
    kx.P = Q, kx.feat = feat.data(), kx.zs = zs.data(), kx.sig = const_cast<double*>(sigs.data()), kx.out = ia.data();
    ofm::detail::dispatch(kx, ofm::detail::groups(std::uint32_t(T * N), 64));
    for (std::size_t r = 0; r < T * N; ++r) {
      const double* a = &feat[r * K];
      double* o = &ftI[r * KI];
      for (std::size_t f = 0; f < F; ++f) o[f] = a[f];
      for (std::size_t f = 0; f < 2 * MI; ++f) o[F + f] = ia[r * 2 * MI + f];
      for (std::size_t f = 0; f < NR; ++f) o[F + 2 * MI + f] = rs[r * 6 + resUsed[f]];
      for (std::size_t f = 0; f < 5; ++f) o[FI + f] = a[F + f];
    }
  }
  if (!scanOnly) model(feat, F, K, F, "Walk-forward model on the stocks' own OHLCV");
  const auto linPred = model(ftI, FI, KI, FI, "Walk-forward model with liquidity and volatility interactions");
  // The page's scan (--json): the model above (lagged by --lag), its trades for the next open, its
  // backtest against the market, the chart laid out here and the last 20 days. Results only.
  // The next open's orders in prices (--orders; prices come from EODData's data, so the file goes to
  // the private store, never the page): each trade's last close, its buy limit and its stop-loss and
  // take-profit from the lower of the limit and the last close, for tools/ibkr/submit.py; and the 60 highest
  // ranked of the pool with their last closes, whose opening auctions the bridge records (--auction),
  // and the pool's tickers, whose daily bars the bridge fetches from IBKR (--bars); and the model's own
  // result for the latest day's orders (check), which the bridge sets beside the broker's fills.
  if (!ordersOut.empty()) {
    std::ofstream o(ordersOut);
    o << "{\"asOf\":\"" << m.dates[T - 1] << "\",\"orders\":[";
    char b[512];
    for (std::size_t q = 0; q < lastRun.picks.size(); ++q) {
      const auto& pk = lastRun.picks[q];
      const double c = m.close(T - 1, pk.stock), lim = c * (1 + pk.limit), ref = std::min(lim, c);  // the levels from the expected entry
      auto px = [&](double x) { return std::isfinite(x) ? std::to_string(ref * (1 + x)) : std::string("null"); };
      std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"close\":%.6g,\"limit\":%.6g,\"stop\":%s,\"take\":%s,\"expected\":%.6g,\"capital\":%.6g}", q ? "," : "",
                    m.tickers[pk.stock].c_str(), c, lim, px(pk.stop).c_str(), px(pk.take).c_str(), pk.expected, fee[pk.stock * 7 + 1] > 0 ? fee[pk.stock * 7 + 1] : capital);
      o << b;
    }
    o << "],\"candidates\":[";
    for (std::size_t q = 0; q < lastRun.candidates.size(); ++q) {
      const auto [i, sc] = lastRun.candidates[q];
      std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"close\":%.6g,\"score\":%.6g}", q ? "," : "", m.tickers[i].c_str(), m.close(T - 1, i), sc);
      o << b;
    }
    o << "],\"check\":{\"date\":\"" << lastRun.checkDate << "\",\"trades\":[";
    for (std::size_t q = 0; q < lastRun.check.size(); ++q) {
      const auto& c = lastRun.check[q];
      std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"limit\":%.6g,\"filled\":%d,\"entry\":%.6g,\"exit\":%d,\"exitPrice\":%.6g,\"ret\":%.6g}", q ? "," : "",
                    m.tickers[c.stock].c_str(), c.limit, c.filled, c.entry, c.exit, c.exitPx, c.ret);
      o << b;
    }
    o << "]},\"pool\":[";
    for (std::size_t q = 0; q < lastRun.pool.size(); ++q) o << (q ? ",\"" : "\"") << m.tickers[lastRun.pool[q]] << '"';
    o << "]}\n";
  }
  if (!jsonOut.empty()) {
    const Run& R = lastRun;
    auto metr = [](const std::vector<double>& x) {
      double w = 0, s1 = 0, s2 = 0, peak = 0, dd = 0;
      for (double y : x) {
        w += std::log1p(y), s1 += y, s2 += y * y;
        peak = std::max(peak, w), dd = std::max(dd, 1 - std::exp(w - peak));
      }
      const double n = double(x.size()), mu = s1 / n, sdv = std::sqrt(std::max(0.0, s2 / n - mu * mu));
      return std::array<double, 3>{std::expm1(w * 252.0 / n), sdv > 0 ? mu / sdv * std::sqrt(252.0) : 0.0, dd};
    };
    const auto ms = metr(R.ret);
    std::ofstream o(jsonOut);
    char b[256];
    o << "{\"model\":\"v2\",\"asOf\":\"" << m.dates[T - 1] << "\",\"stocks\":" << N << ",\"lag\":" << lag << ",\"top\":" << topN;
    if (costKind > 0) o << ",\"costs\":{\"broker\":\"IBKR\",\"market\":\"" << costMarket << "\",\"capital\":" << capital << ",\"stamp\":" << (costKind == 1 ? stamp : 0.0)
                      << ",\"stampFree\":" << std::count(stampOf.begin(), stampOf.end(), 0.0) * (costKind == 1) << ",\"venues\":" << (venuesFile.empty() ? "false" : "true") << "}";
    std::snprintf(b, sizeof b, ",\"backtest\":{\"from\":\"%s\",\"to\":\"%s\",\"annualReturn\":%.6g,\"sharpe\":%.6g,\"maxDrawdown\":%.6g,"
                  "\"orders\":%.0f,\"trades\":%.0f,\"avgTrade\":%.6g,\"winRate\":%.6g}", R.dates.front().c_str(), R.dates.back().c_str(), ms[0], ms[1], ms[2],
                  R.orders, R.filled, R.filled > 0 ? R.filledSum / R.filled : 0.0, R.filled > 0 ? R.gained / R.filled : 0.0);
    o << b;
    o << b << ",\"orders\":[";
    for (std::size_t q = 0; q < R.picks.size(); ++q) {
      const auto& pk = R.picks[q];
      auto lvl = [](double x) { return std::isfinite(x) ? std::to_string(x) : std::string("null"); };
      std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"expected\":%.6g,\"limit\":%.6g,\"stop\":%s,\"take\":%s}", q ? "," : "", m.tickers[pk.stock].c_str(),
                    pk.expected, pk.limit, lvl(pk.stop).c_str(), lvl(pk.take).c_str());
      o << b;
    }
    o << "],\"recent\":[";
    for (std::size_t q = R.ret.size() > 20 ? R.ret.size() - 20 : 0, f = q; q < R.ret.size(); ++q) {
      std::snprintf(b, sizeof b, "%s{\"date\":\"%s\",\"day\":%.6g}", q == f ? "" : ",", R.dates[q].c_str(), R.ret[q]);
      o << b;
    }
    // The chart: the daily trades' growth on a log scale, heights and positions as shares of the chart.
    const std::size_t n = R.ret.size(), step = std::max<std::size_t>(1, n / 600);
    std::vector<double> lv[1];
    double lo = 0, hi = 0;
    for (int k = 0; k < 1; ++k) {
      double w = 0;
      const auto& x = R.ret;
      for (std::size_t q = 0; q < n; ++q) {
        w += std::log1p(x[q]);
        if (q % step == 0) lv[k].push_back(w), lo = std::min(lo, w), hi = std::max(hi, w);
      }
    }
    const double span = std::max(hi - lo, 1e-9);
    o << "],\"chart\":{\"series\":[";
    const char* nm[1] = {"daily trades"};
    for (int k = 0; k < 1; ++k) {
      double w = 0;
      for (double y : R.ret) w += std::log1p(y);
      std::snprintf(b, sizeof b, "%s{\"name\":\"%s\",\"final\":\"%.2f×\",\"y\":[", k ? "," : "", nm[k], std::exp(w));
      o << b;
      for (std::size_t q = 0; q < lv[k].size(); ++q) std::snprintf(b, sizeof b, "%s%.4f", q ? "," : "", (lv[k][q] - lo) / span), o << b;
      o << "]}";
    }
    o << "],\"ticks\":[";
    bool lead = true;
    for (double v : {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0}) {
      const double y = (std::log(v) - lo) / span;
      if (y < -1e-9 || y > 1 + 1e-9) continue;
      std::snprintf(b, sizeof b, "%s{\"label\":\"%g×\",\"y\":%.4f}", lead ? "" : ",", v, y), o << b, lead = false;
    }
    o << "],\"years\":[";
    lead = true;
    for (std::size_t q = 0, j = 0; q < n; q += step, ++j)
      if (j == 0 || R.dates[q].substr(0, 4) != R.dates[q - step].substr(0, 4))
        std::snprintf(b, sizeof b, "%s{\"label\":\"%s\",\"x\":%.4f}", lead ? "" : ",", R.dates[q].substr(0, 4).c_str(),
                      lv[0].size() > 1 ? double(j) / double(lv[0].size() - 1) : 0.0),
            o << b, lead = false;
    o << "]}}";
  }
  // The same inputs through a neural network (two hidden layers of rectified linear units, the
  // fused kernels of src/net): each day it predicts every stock's next open-to-close from
  // the day's standardised features, then learns from that day's outcomes once they are known
  // (shuffled mini-batches of 32, one pass), so every prediction is out of sample.
  if (!scanOnly) {
    std::vector<double> zs(T * F * 2, 0.0);
    ofm::v2::gen::Zscore<double> kz;
    kz.P = P, kz.feat = feat.data(), kz.zs = zs.data();
    ofm::detail::dispatch(kz, ofm::detail::groups(std::uint32_t(T * F), 64));
    // Settings under test (NN_HIDDEN, NN_LR, NN_RANK: the target as the day's rank).
    const auto env = [](const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; };
    ofm::net::Net net(F, std::size_t(env("NN_HIDDEN", 16)), {0.5});
    net.lr = env("NN_LR", 3e-4);
    const bool rankTarget = env("NN_RANK", 0) > 0;
    std::vector<double> memX, memY;  // every earlier day's samples (replay)
    std::vector<double> always, ens, mkt, adaptive, linOnly;
    // The self-adaptive choice: each day the variant (linear, network, ensemble) whose record so far
    // is best, its mean shrunk by its t-statistic, m max(0, 1 - 1/t^2).
    double vs1[3] = {0, 0, 0}, vs2[3] = {0, 0, 0}, vn = 0;
    std::size_t used[3] = {0, 0, 0};
    std::vector<std::string> sd;
    std::uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (std::size_t q = 0; q < days.size(); ++q) {
      const std::size_t t = days[q];
      std::vector<double> x, y;
      std::vector<std::size_t> who;
      for (std::size_t i = 0; i < N; ++i) {
        const double* a = &feat[(t * N + i) * K];
        if (!(a[K - 1] > 0)) continue;
        for (std::size_t f = 0; f < F; ++f) x.push_back(std::clamp((a[f] - zs[(t * F + f) * 2]) / zs[(t * F + f) * 2 + 1], -4.0, 4.0));
        y.push_back(std::clamp(a[F], -6.0, 6.0)), who.push_back(i);
      }
      if (who.size() < 30) continue;
      // The target relative to the day's average (the ranking is what is traded), or the day's rank.
      if (rankTarget) {
        std::vector<std::pair<double, std::size_t>> o(y.size());
        for (std::size_t r = 0; r < y.size(); ++r) o[r] = {y[r], r};
        std::sort(o.begin(), o.end());
        for (std::size_t r = 0; r < o.size(); ++r) y[o[r].second] = (double(r) + 0.5) / double(o.size()) - 0.5;
      } else {
        double my = 0;
        for (double v : y) my += v / double(y.size());
        for (double& v : y) v -= my;
      }
      if (q >= 250 && t + 1 < T) {
        const auto pr = net.predict(x, who.size());
        std::vector<std::pair<double, std::size_t>> v;
        for (std::size_t r = 0; r < who.size(); ++r)
          if (tradable[t * N + who[r]]) v.emplace_back(pr[r], who[r]);
        std::partial_sort(v.begin(), v.begin() + 10, v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        double r1 = 0, ms = 0;
        for (std::size_t r = 0; r < 10; ++r) r1 += (m.close(t + 1, v[r].second) / m.open(t + 1, v[r].second) - 1) / 10;
        // The ensemble: each stock's rank under the network plus its rank under the linear model.
        std::vector<std::pair<double, std::size_t>> rn(who.size()), rl(who.size());
        for (std::size_t r = 0; r < who.size(); ++r) rn[r] = {pr[r], r}, rl[r] = {linPred[(t * N + who[r]) * 4], r};
        std::sort(rn.begin(), rn.end()), std::sort(rl.begin(), rl.end());
        std::vector<std::pair<double, std::size_t>> sc(who.size());
        for (std::size_t r = 0; r < who.size(); ++r) sc[r].second = who[r];
        for (std::size_t r = 0; r < who.size(); ++r) sc[rn[r].second].first += double(r), sc[rl[r].second].first += double(r);
        sc.erase(std::remove_if(sc.begin(), sc.end(), [&](const auto& e) { return !tradable[t * N + e.second]; }), sc.end());
        std::partial_sort(sc.begin(), sc.begin() + 10, sc.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        double r2 = 0;
        for (std::size_t r = 0; r < 10; ++r) r2 += (m.close(t + 1, sc[r].second) / m.open(t + 1, sc[r].second) - 1) / 10;
        ens.push_back(r2);
        // The linear model's own top 10.
        std::vector<std::pair<double, std::size_t>> lv(who.size());
        for (std::size_t r = 0; r < who.size(); ++r) lv[r] = {linPred[(t * N + who[r]) * 4], who[r]};
        lv.erase(std::remove_if(lv.begin(), lv.end(), [&](const auto& e) { return !tradable[t * N + e.second]; }), lv.end());
        std::partial_sort(lv.begin(), lv.begin() + 10, lv.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        double r0 = 0;
        for (std::size_t r = 0; r < 10; ++r) r0 += (m.close(t + 1, lv[r].second) / m.open(t + 1, lv[r].second) - 1) / 10;
        linOnly.push_back(r0);
        const double rv[3] = {r0, r1, r2};
        std::size_t best = 0;
        double bs = -1e300;
        for (std::size_t k = 0; k < 3; ++k) {
          const double mu = vn > 1 ? vs1[k] / vn : 0.0, var = vn > 1 ? std::max(0.0, vs2[k] / vn - mu * mu) : 0.0;
          const double sc = vn > 1 && mu != 0 ? mu * std::max(0.0, 1.0 - var / vn / (mu * mu)) : 0.0;
          if (sc > bs) bs = sc, best = k;
        }
        adaptive.push_back(rv[best]), ++used[best];
        for (std::size_t k = 0; k < 3; ++k) vs1[k] += rv[k], vs2[k] += rv[k] * rv[k];
        vn += 1;
        for (std::size_t i : who) ms += (m.close(t + 1, i) / m.close(t, i) - 1) / double(who.size());
        always.push_back(r1), mkt.push_back(ms), sd.push_back(m.dates[t + 1]);
      }
      // Learn from the day's outcomes (known at the next close, before the next prediction).
      std::vector<std::size_t> perm(who.size());
      for (std::size_t r = 0; r < perm.size(); ++r) perm[r] = r;
      for (std::size_t r = perm.size(); r-- > 1;) {
        rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
        std::swap(perm[r], perm[rng % (r + 1)]);
      }
      // Replay: as many samples again drawn from every earlier day, so the network keeps learning
      // from the whole history rather than drifting with the last days.
      const std::size_t nNew = who.size(), nOld = std::min(memY.size(), nNew);
      std::vector<double> tx(x), ty(y);
      for (std::size_t r = 0; r < nOld; ++r) {
        rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
        const std::size_t k = rng % memY.size();
        tx.insert(tx.end(), memX.begin() + std::ptrdiff_t(k * F), memX.begin() + std::ptrdiff_t((k + 1) * F)), ty.push_back(memY[k]);
      }
      memX.insert(memX.end(), x.begin(), x.end()), memY.insert(memY.end(), y.begin(), y.end());
      perm.resize(ty.size());
      for (std::size_t r = 0; r < perm.size(); ++r) perm[r] = r;
      for (std::size_t r = perm.size(); r-- > 1;) {
        rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
        std::swap(perm[r], perm[rng % (r + 1)]);
      }
      for (std::size_t b0 = 0; b0 < perm.size(); b0 += 32) {
        const std::size_t nb = std::min<std::size_t>(32, perm.size() - b0);
        std::vector<double> bx, by;
        for (std::size_t r = b0; r < b0 + nb; ++r) bx.insert(bx.end(), tx.begin() + std::ptrdiff_t(perm[r] * F), tx.begin() + std::ptrdiff_t((perm[r] + 1) * F)), by.push_back(ty[perm[r]]);
        net.step(bx, by, nb);
      }
    }
    auto stats = [](const std::vector<double>& v) {
      double w = 0, s1 = 0, s2 = 0;
      for (double r : v) w += std::log1p(r), s1 += r, s2 += r * r;
      const double n = double(v.size()), mu = s1 / n, sdv = std::sqrt(std::max(0.0, s2 / n - mu * mu));
      return std::make_pair(std::expm1(w * 252.0 / n), sdv > 0 ? mu / sdv * std::sqrt(252.0) : 0.0);
    };
    const auto a1 = stats(always), a2 = stats(ens), a4 = stats(adaptive), a0 = stats(linOnly);
    std::printf("\nNeural network on the stocks' own OHLCV (%s .. %s), the 10 highest predicted, open to close:\n", sd.front().c_str(), sd.back().c_str());
    std::printf("  every day                        %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a1.first, a1.second);
    std::printf("  ensemble with the linear model   %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a2.first, a2.second);
    std::printf("  linear model alone               %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a0.first, a0.second);
    std::printf("  self-adaptive choice             %+7.1f%% a year (Sharpe %5.2f); days linear %zu, network %zu, ensemble %zu\n", 100 * a4.first, a4.second, used[0], used[1], used[2]);
  }
  // With the market-wide series (futures and indices) as further inputs.
  if (!seriesCsv.empty()) {
    const auto ser = ofm::parseSeriesCsv(readFile(seriesCsv), m.dates);
    std::vector<std::string> names;
    std::vector<double> closes;
    for (const auto& [nm, c] : ser) {
      names.push_back(nm);
      for (std::size_t t = 0; t < T; ++t) closes.push_back(t < c.size() && std::isfinite(c[t]) ? c[t] : 0.0);
    }
    const std::size_t M = names.size(), F2 = F + 2 * M, K2 = F2 + 5;
    std::vector<double> mk(T * 2 * M, 0.0), ft2(T * N * K2, 0.0);
    ofm::v2::gen::V2Header<double> Q = P;
    Q.u0 = std::uint32_t(M);
    ofm::v2::gen::Market<double> km;
    km.P = Q, km.closes = closes.data(), km.out = mk.data();
    ofm::detail::dispatch(km, ofm::detail::groups(std::uint32_t(M), 64));
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t i = 0; i < N; ++i) {
        const double* a = &feat[(t * N + i) * K];
        double* o = &ft2[(t * N + i) * K2];
        for (std::size_t f = 0; f < F; ++f) o[f] = a[f];
        for (std::size_t f = 0; f < 2 * M; ++f) o[F + f] = mk[t * 2 * M + f];
        for (std::size_t f = 0; f < 5; ++f) o[F2 + f] = a[F + f];
      }
    std::printf("\nmarket series (%zu):", M);
    for (const auto& nm : names) std::printf(" %s", nm.c_str());
    std::printf("\n");
    model(ft2, F2, K2, F, "Walk-forward model with the market series");
  }
  return 0;
}
