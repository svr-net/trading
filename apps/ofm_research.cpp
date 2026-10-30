// The predictive model's research: each OHLCV feature's power to predict the next day's moves.
//   ofm_research --csv stocks.csv [--series futures.csv]
// Prints, for every feature and target, the mean of its day-by-day cross-sectional correlation
// with the target and its t-statistic (the features and the correlations run in fused kernels,
// kernels/v2).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
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
  std::string csv, seriesCsv, jsonOut;
  std::size_t topN = 0;  // trade only each day's topN most traded (0: all)
  std::size_t lag = 0;   // a check on the data: trade on the predictions made lag closes earlier
  for (int k = 1; k + 1 < argc; ++k) {
    if (std::string(argv[k]) == "--csv") csv = argv[k + 1];
    if (std::string(argv[k]) == "--series") seriesCsv = argv[k + 1];
    if (std::string(argv[k]) == "--top") topN = std::size_t(std::atol(argv[k + 1]));
    if (std::string(argv[k]) == "--lag") lag = std::size_t(std::atol(argv[k + 1]));
    if (std::string(argv[k]) == "--json") jsonOut = argv[k + 1];
  }
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
    struct Pick {
      std::size_t stock;
      double expected, stop, take;  // expected open-to-close; the stop and take-profit from the entry (NaN: none)
      double limit;                 // the buy limit from the last close
    };
    std::vector<Pick> picks;
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
    ofm::detail::dispatch(kg, ofm::detail::groups(std::uint32_t(T * E), 64));
    std::vector<double> acc(E, 0.0);
    for (std::size_t t = 0; t < T; ++t) {
      for (std::size_t k = 0; k < 4; ++k) {
        std::vector<double> M(acc.begin(), acc.begin() + std::ptrdiff_t(D * D)), v(D);
        for (std::size_t q = 0; q < D; ++q) v[q] = acc[D * D + k * D + q];
        double tr = 0;
        for (std::size_t q = 0; q < D; ++q) tr += M[q * D + q];
        if (!(tr > 0)) continue;
        for (std::size_t q = 0; q < D; ++q) M[q * D + q] += ridgeK * tr;
        for (std::size_t c = 0; c < D; ++c)
          for (std::size_t r = 0; r < D; ++r)
            if (r != c) {
              const double q2 = M[r * D + c] / M[c * D + c];
              for (std::size_t q = 0; q < D; ++q) M[r * D + q] -= q2 * M[c * D + q];
              v[r] -= q2 * v[c];
            }
        for (std::size_t q = 0; q < D; ++q) beta[(t * 4 + k) * D + q] = v[q] / M[q * D + q];
      }
      if (t > 0)
        for (std::size_t e = 0; e < E; ++e) acc[e] = forget * acc[e] + gram[(t - 1) * E + e];
    }
    ofm::v2::gen::Predict<double> kp;
    kp.P = Q, kp.feat = const_cast<double*>(ft.data()), kp.zs = zs.data(), kp.beta = beta.data(), kp.out = pred.data();
    ofm::detail::dispatch(kp, ofm::detail::groups(std::uint32_t(T * N), 64));
    const std::size_t warm = 250;
    std::vector<double> always, gated, mkt;
    // The trades' settings, learnt continuously: the pool (the 250 * 2^c most traded by traded value,
    // c 0 to 3), the ranking (the prediction times the volatility to the power beta: 0 ranks by the
    // prediction in volatilities, 1 by the expected return), the entry (a limit buy d standard
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
    // spread; otherwise sold at the close. Every setting on a lattice keeps a record; each day the pool
    // and ranking with the best record (mean shrunk by its t-statistic) and, at them, the levels with
    // the best mean over downside deviation (Sortino) so far are refined to a continuous optimum by a
    // parabola through their neighbours along each axis, and today's trades use them.
    constexpr std::size_t nC = 4, nB = 5, nE = 9, nP = 7, nA = 5, nR = nC * nB, nLv = nP * nP * nA, nV = nR * nE * nLv, nD = 6;
    const auto axis = [](std::size_t d, double x) {  // lattice coordinate -> setting
      if (d == 0) return 250.0 * std::exp2(x);
      if (d == 1) return x / double(nB - 1);
      if (d == 2) return -2.0 + 0.5 * x;  // the limit's depth d (standard deviations below the previous close)
      if (d == 5) return x / double(nA - 1);
      return 0.5 * std::pow(0.2, x / double(nP - 1));  // the tail fraction p, 1/2 .. 1/10
    };
    const std::size_t dims[nD] = {nC, nB, nE, nP, nP, nA};
    std::vector<double> s1v(nV, 0.0), s2v(nV, 0.0), dnv(nV, 0.0);  // sums of the results, their squares, the losses' squares
    double nDays = 0;
    std::vector<double> adaptiveLevels, byRank[2];
    std::vector<std::array<double, nD>> used;
    std::vector<std::array<double, 5>> paths;  // per candidate trade: filled, the stop and the take-profit from the entry, stopped, took profit
    std::vector<std::string> sd;
    std::FILE* tradeLog = nullptr;
    if (std::getenv("X_TRADELOG") && std::string(label).find("interactions") != std::string::npos) {
      tradeLog = std::fopen(std::getenv("X_TRADELOG"), "w");
      std::fprintf(tradeLog, "date,ticker,rank,pred,sigma,liq_rank,spread,low,high,close,stop,take,market,exit,ret,gap,limit,filled\n");
    }
    const auto score = [&](std::size_t g) {
      if (nDays < 2) return 0.0;
      const double mu = s1v[g] / nDays, var = std::max(0.0, s2v[g] / nDays - mu * mu);
      return mu != 0 ? mu * std::max(0.0, 1.0 - var / nDays / (mu * mu)) : 0.0;
    };
    const auto ddScore = [&](std::size_t g) { return nDays < 2 || !(dnv[g] > 0) ? 0.0 : s1v[g] / nDays / std::sqrt(dnv[g] / nDays); };
    const auto index = [&](const std::size_t* c) { return ((((c[0] * nB + c[1]) * nE + c[2]) * nP + c[3]) * nP + c[4]) * nA + c[5]; };
    const auto coords = [&](std::size_t g, std::size_t* c) {
      c[5] = g % nA, g /= nA, c[4] = g % nP, g /= nP, c[3] = g % nP, g /= nP, c[2] = g % nE, g /= nE, c[1] = g % nB, c[0] = g / nB;
    };
    // The continuous settings {pool, beta, d, p_s, p_t, alpha} from the records so far.
    const auto learnt = [&] {
      std::size_t best = 0;
      for (std::size_t g = 1; g < nV; ++g)
        if (score(g) > score(best)) best = g;
      const std::size_t r0 = best / nLv;
      std::size_t lv = r0 * nLv;
      for (std::size_t g = lv + 1; g < (r0 + 1) * nLv; ++g)
        if (ddScore(g) > ddScore(lv)) lv = g;
      std::size_t bc[nD], lc[nD];
      coords(best, bc), coords(lv, lc);
      std::array<double, nD> x{};
      for (std::size_t d = 0; d < nD; ++d) {
        const bool pick = d < 3;  // the pool, ranking and entry by return; the levels by Sortino
        const auto f = [&](std::size_t g) { return pick ? score(g) : ddScore(g); };
        std::size_t at[nD];
        for (std::size_t e = 0; e < nD; ++e) at[e] = pick ? bc[e] : lc[e];
        double off = 0;
        if (at[d] > 0 && at[d] + 1 < dims[d]) {
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
    // The day's 10 trades: among the pool most traded, ranked with beta.
    const auto ranked = [&](std::size_t t, double pool, double b) {
      std::vector<std::pair<double, std::size_t>> v;
      for (std::size_t i = 0; i < N; ++i) {
        if (!(liqRank[i] < pool) || !tradable[t * N + i]) continue;
        const std::size_t src = (t - lag) * N + i;
        v.emplace_back(pred[src * 4] * std::exp(b * ft[src * Km + 20]), i);
      }
      std::partial_sort(v.begin(), v.begin() + std::ptrdiff_t(std::min<std::size_t>(nPick, v.size())), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      v.resize(std::min<std::size_t>(nPick, v.size()));
      return v;
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
      if (!tails(t, i)) return std::make_pair(0.0, 1e300);
      double st = quantile(falls[i], ps);
      if (std::isfinite(sar[t * N + i])) st = (1 - al) * st + al * std::log(std::min(1.0, sar[t * N + i] / ref));  // a SAR above the open: stopped at the open
      return std::make_pair(std::exp(st), std::exp(quantile(rises[i], 1 - pt)));
    };
    // One trade: the limit (a fraction of the previous close), whether it filled and where, its exit
    // (0 close, 1 stop, 2 take-profit) and its return from the entry.
    struct Trade {
      double limit, entry;
      int filled, exit;
      double ret, stop, take;
    };
    const auto trade = [&](std::size_t t, std::size_t i, double d, double ps, double pt, double al) {
      Trade r{};
      const double pc = m.close(t, i), sig = std::exp(ft[(t * N + i) * Km + 20]);
      const double o = m.open(t + 1, i), hi = m.high(t + 1, i), lo = m.low(t + 1, i), c = m.close(t + 1, i), half = std::exp(0.5 * spread[t * N + i]);
      const double L = pc * std::exp(-d * sig);
      r.limit = L / pc - 1;
      bool intraday = false;
      if (o <= L) r.entry = o, r.filled = 1;
      else if (lo <= L / half) r.entry = L, r.filled = 2, intraday = true;
      else return r;
      const auto [fs, fk] = levelsAt(t, i, r.entry, ps, pt, al);
      const double stop = r.entry * fs, take = r.entry * fk;
      r.stop = fs - 1, r.take = fk < 1e300 ? fk - 1 : 0.0;
      double px = c;
      if (lo <= stop) px = stop / half, r.exit = 1;
      else if (!intraday && hi >= take * half) px = take, r.exit = 2;
      r.ret = px / r.entry - 1;
      return r;
    };
    // The day's result: every trade one unit, a trade not filled stays in cash.
    const auto dayResult = [&](std::size_t t, const std::vector<std::pair<double, std::size_t>>& w, double d, double ps, double pt, double al) {
      double rg = 0;
      for (const auto& e : w) rg += trade(t, e.second, d, ps, pt, al).ret / double(w.size());
      return w.empty() ? 0.0 : rg;
    };
    for (std::size_t q = warm; q < days.size(); ++q) {
      const std::size_t t = days[q];
      if (t + 1 >= T || t < lag) continue;
      double ms = 0, mn = 0;
      for (std::size_t i = 0; i < N; ++i)
        if (ft[(t * N + i) * Km + Km - 1] > 0 && tradable[t * N + i] && ft[((t - lag) * N + i) * Km + Km - 1] > 0) ms += m.close(t + 1, i) / m.close(t, i) - 1, mn += 1;
      if (mn < 30) continue;
      rankLiquidity(t, true);
      std::vector<std::pair<double, std::size_t>> v[nR];
      for (std::size_t k = 0; k < nR; ++k) v[k] = ranked(t, axis(0, double(k / nB)), axis(1, double(k % nB)));
      const auto& all0 = v[(nC - 1) * nB], &all1 = v[(nC - 1) * nB + nB - 1];
      double r = 0, pm = 0;
      for (const auto& e : all0) r += (m.close(t + 1, e.second) / m.open(t + 1, e.second) - 1) / double(all0.size()), pm += e.first / double(all0.size());
      always.push_back(r), gated.push_back(pm > 0 ? r : 0.0), mkt.push_back(ms / mn), sd.push_back(m.dates[t + 1]);
      double rk = 0;
      for (const auto& e : all1) rk += (m.close(t + 1, e.second) / m.open(t + 1, e.second) - 1) / double(all1.size());
      byRank[1].push_back(rk);
      // Today's trades with the settings learnt so far; then every lattice point's record takes today.
      const auto x = learnt();
      const auto today = ranked(t, x[0], x[1]);
      adaptiveLevels.push_back(dayResult(t, today, x[2], x[3], x[4], x[5])), used.push_back(x);
      for (std::size_t k = 0; k < today.size(); ++k) {  // the trades, for the diagnostics and the log
        const std::size_t i = today[k].second, src = (t - lag) * N + i;
        const double o = m.open(t + 1, i);
        const Trade tr = trade(t, i, x[2], x[3], x[4], x[5]);
        paths.push_back({double(tr.filled > 0), tr.stop, tr.take, double(tr.exit == 1), double(tr.exit == 2)});
        if (tradeLog)  // one line per trade, for the single-trade analysis (X_TRADELOG; local, not published)
          std::fprintf(tradeLog, "%s,%s,%zu,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,%.6g,%.6g,%d\n", m.dates[t + 1].c_str(), m.tickers[i].c_str(), k,
                       pred[src * 4], std::exp(ft[src * Km + 20]), liqRank[i], spread[t * N + i], m.low(t + 1, i) / o - 1, m.high(t + 1, i) / o - 1,
                       m.close(t + 1, i) / o - 1, tr.stop, tr.take, ms / mn, tr.exit, tr.ret, m.close(t, i) > 0 ? o / m.close(t, i) - 1 : 0.0, tr.limit, tr.filled);
      }
      for (std::size_t g = 0; g < nV; ++g) {
        std::size_t c[nD];
        coords(g, c);
        const double y = dayResult(t, v[g / (nE * nLv)], axis(2, double(c[2])), axis(3, double(c[3])), axis(4, double(c[4])), axis(5, double(c[5])));
        s1v[g] += y, s2v[g] += y * y, dnv[g] += std::min(y, 0.0) * std::min(y, 0.0);
      }
      nDays += 1;
    }
    if (tradeLog) std::fclose(tradeLog);
    auto stats = [](const std::vector<double>& x) {
      double w = 0, s1 = 0, s2 = 0;
      for (double y : x) w += std::log1p(y), s1 += y, s2 += y * y;
      const double n = double(x.size()), mu = s1 / n, sdv = std::sqrt(std::max(0.0, s2 / n - mu * mu));
      return std::make_pair(std::expm1(w * 252.0 / n), sdv > 0 ? mu / sdv * std::sqrt(252.0) : 0.0);
    };
    const auto a1 = stats(always), a2 = stats(gated), a3 = stats(mkt);
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
      std::printf("  self-adaptive pool, ranking, levels %+6.1f%% a year (Sharpe %5.2f); mean pool %.0f, beta %.2f, limit %.2f sd, stop 1 day in %.1f, take 1 day in %.1f, SAR weight %.2f\n",
                  100 * a.first, a.second, mx[0], mx[1], mx[2], 1 / mx[3], 1 / mx[4], mx[5]);
      std::printf("  learnt for the next open: pool %.0f, beta %.2f, limit %.2f sd, stop 1 day in %.1f, take 1 day in %.1f, SAR weight %.2f\n", fx[0], fx[1], fx[2], 1 / fx[3],
                  1 / fx[4], fx[5]);
      // The trades' levels from the open and how often they were touched.
      auto qt = [&](std::size_t d, double p) {
        std::vector<double> z;
        for (const auto& x : paths) z.push_back(x[d]);
        std::nth_element(z.begin(), z.begin() + std::ptrdiff_t(p * double(z.size() - 1)), z.end());
        return z[std::size_t(p * double(z.size() - 1))];
      };
      double fl = 0, st = 0, tk = 0;
      for (const auto& x : paths) fl += x[0], st += x[3], tk += x[4];
      std::printf("  candidate trades: %zu, filled %.1f%%; stop from the entry: median %+.2f%% (10%% %+.2f%%, 90%% %+.2f%%), touched %.1f%% of filled; take-profit: median "
                  "%+.2f%% (10%% %+.2f%%, 90%% %+.2f%%), taken %.1f%% of filled\n",
                  paths.size(), 100 * fl / double(paths.size()), 100 * qt(1, 0.5), 100 * qt(1, 0.1), 100 * qt(1, 0.9), 100 * st / std::max(fl, 1.0), 100 * qt(2, 0.5),
                  100 * qt(2, 0.1), 100 * qt(2, 0.9), 100 * tk / std::max(fl, 1.0));
    }
    std::printf("  only days predicted to gain      %+7.1f%% a year (Sharpe %5.2f), %zu of %zu days\n", 100 * a2.first, a2.second, traded, gated.size());
    std::printf("  market (all stocks, held)        %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a3.first, a3.second);
    // The next open's trades: the 10 highest predictions lag closes before the last close, among
    // the stocks with a bar at the last close (their targets are not known yet), from the pool and
    // ranked and with the levels by the settings learnt over the whole backtest. Expected open-to-close: the prediction
    // times the stock's volatility (feature 20, its log). The SAR stop is from the last close (the
    // open is not known yet).
    lastRun = Run{adaptiveLevels, mkt, sd, {}};
    {
      const std::size_t t = T - 1;
      std::vector<std::pair<double, std::size_t>> v;
      if (t >= lag) {
        rankLiquidity(t, false);
        for (std::size_t i = 0; i < N; ++i) {
          const double* a = &ft[((t - lag) * N + i) * Km];
          if (!(liqRank[i] < fx[0]) || !std::isfinite(m.close(t, i)) || !(m.close(t, i) > 0)) continue;
          if (!tradable[t * N + i]) continue;
          v.emplace_back(pred[((t - lag) * N + i) * 4] * std::exp(fx[1] * a[20]), i);
        }
      }
      std::partial_sort(v.begin(), v.begin() + std::ptrdiff_t(std::min<std::size_t>(nPick, v.size())), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      for (std::size_t q = 0; q < std::min<std::size_t>(nPick, v.size()); ++q) {
        const std::size_t i = v[q].second, src = (t - lag) * N + i;
        const double sig = std::exp(ft[src * Km + 20]);
        // The limit from the last close; the levels from the limit (the entry at worst).
        const double limit = std::exp(-fx[2] * std::exp(ft[(t * N + i) * Km + 20]));
        const auto [fs, fk] = levelsAt(t, i, m.close(t, i) * limit, fx[3], fx[4], fx[5]);
        if (v[q].first > 0) lastRun.picks.push_back({i, std::expm1(pred[src * 4] * sig), fs > 0 ? fs - 1 : kNaN, fk < 1e300 ? fk - 1 : kNaN, limit - 1});
      }
    }
    return pred;
  };
  // The features with their interactions with liquidity and volatility (kernels/v2/interact.wgsl):
  // momentum 2d, volume shock 4d, open-to-close, gap, close in range, intraday and overnight
  // momentum 64d.
  const std::vector<double> sigs = {0, 6, 12, 14, 15, 23, 26};
  const std::size_t NR = std::getenv("X_NORES") ? 0 : 3;  // the residual reversal (5 and 21 days) and the tug of war (kernels/v2/resid.wgsl)
  const std::size_t MI = sigs.size(), FI = F + 2 * MI + NR, KI = FI + 5;
  std::vector<double> ftI(T * N * KI, 0.0);
  {
    std::vector<double> mk(T, 0.0), rs(T * N * 3, 0.0);
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
      for (std::size_t f = 0; f < NR; ++f) o[F + 2 * MI + f] = rs[r * 3 + f];
      for (std::size_t f = 0; f < 5; ++f) o[FI + f] = a[F + f];
    }
  }
  model(feat, F, K, F, "Walk-forward model on the stocks' own OHLCV");
  const auto linPred = model(ftI, FI, KI, FI, "Walk-forward model with liquidity and volatility interactions");
  // The page's scan (--json): the model above (lagged by --lag), its trades for the next open, its
  // backtest against the market, the chart laid out here and the last 20 days. Results only.
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
    const auto ms = metr(R.ret), mm = metr(R.mkt);
    std::ofstream o(jsonOut);
    char b[256];
    o << "{\"model\":\"v2\",\"asOf\":\"" << m.dates[T - 1] << "\",\"stocks\":" << N << ",\"lag\":" << lag << ",\"top\":" << topN;
    std::snprintf(b, sizeof b, ",\"backtest\":{\"from\":\"%s\",\"to\":\"%s\",\"annualReturn\":%.6g,\"sharpe\":%.6g,\"maxDrawdown\":%.6g}", R.dates.front().c_str(),
                  R.dates.back().c_str(), ms[0], ms[1], ms[2]);
    o << b;
    std::snprintf(b, sizeof b, ",\"market\":{\"annualReturn\":%.6g,\"sharpe\":%.6g,\"maxDrawdown\":%.6g}", mm[0], mm[1], mm[2]);
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
      std::snprintf(b, sizeof b, "%s{\"date\":\"%s\",\"day\":%.6g,\"market\":%.6g}", q == f ? "" : ",", R.dates[q].c_str(), R.ret[q], R.mkt[q]);
      o << b;
    }
    // The chart: both growth curves on a log scale, heights and positions as shares of the chart.
    const std::size_t n = R.ret.size(), step = std::max<std::size_t>(1, n / 600);
    std::vector<double> lv[2];
    double lo = 0, hi = 0;
    for (int k = 0; k < 2; ++k) {
      double w = 0;
      const auto& x = k == 0 ? R.ret : R.mkt;
      for (std::size_t q = 0; q < n; ++q) {
        w += std::log1p(x[q]);
        if (q % step == 0) lv[k].push_back(w), lo = std::min(lo, w), hi = std::max(hi, w);
      }
    }
    const double span = std::max(hi - lo, 1e-9);
    o << "],\"chart\":{\"series\":[";
    const char* nm[2] = {"daily trades", "market (bought and held)"};
    for (int k = 0; k < 2; ++k) {
      double w = 0;
      for (double y : k == 0 ? R.ret : R.mkt) w += std::log1p(y);
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
  {
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
    const auto a1 = stats(always), a2 = stats(ens), a3 = stats(mkt), a4 = stats(adaptive), a0 = stats(linOnly);
    std::printf("\nNeural network on the stocks' own OHLCV (%s .. %s), the 10 highest predicted, open to close:\n", sd.front().c_str(), sd.back().c_str());
    std::printf("  every day                        %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a1.first, a1.second);
    std::printf("  ensemble with the linear model   %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a2.first, a2.second);
    std::printf("  linear model alone               %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a0.first, a0.second);
    std::printf("  self-adaptive choice             %+7.1f%% a year (Sharpe %5.2f); days linear %zu, network %zu, ensemble %zu\n", 100 * a4.first, a4.second, used[0], used[1], used[2]);
    std::printf("  market (all stocks, held)        %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a3.first, a3.second);
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
