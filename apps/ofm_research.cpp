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
  const ofm::Market m = ofm::parseMarketCsv(readFile(csv));
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
  // Each stock's Bollinger width as known at each close: the standard deviation of its last 20
  // daily log returns.
  std::vector<double> boll(T * N, 0.0);
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t t = 1; t < T; ++t) {
      double s1 = 0, s2 = 0, n = 0;
      for (std::size_t u = t >= 20 ? t - 19 : 1; u <= t; ++u)
        if (m.close(u, i) > 0 && m.close(u - 1, i) > 0) {
          const double r = std::log(m.close(u, i) / m.close(u - 1, i));
          s1 += r, s2 += r * r, n += 1;
        }
      boll[t * N + i] = n > 1 ? std::sqrt(std::max(0.0, (s2 - s1 * s1 / n) / (n - 1))) : 0.0;
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
      double expected, stop, take;  // expected open-to-close; the stop and take-profit from the open (NaN: none)
    };
    std::vector<Pick> picks;
    double size = 1;  // units in each of the next open's trades
  };
  Run lastRun;
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
        for (std::size_t q = 0; q < D; ++q) M[q * D + q] += 1e-6 * tr;
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
        for (std::size_t e = 0; e < E; ++e) acc[e] += gram[(t - 1) * E + e];
    }
    ofm::v2::gen::Predict<double> kp;
    kp.P = Q, kp.feat = const_cast<double*>(ft.data()), kp.zs = zs.data(), kp.beta = beta.data(), kp.out = pred.data();
    ofm::detail::dispatch(kp, ofm::detail::groups(std::uint32_t(T * N), 64));
    const std::size_t warm = 250;
    std::vector<double> always, gated, mkt;
    // The trades' settings, learnt continuously: the ranking (the prediction times the volatility to
    // the power beta: 0 ranks by the prediction in volatilities, 1 by the expected return), the
    // take-profit k_t and the stop-loss k_s standard deviations of the stock's last 20 daily returns
    // from the open (Bollinger), the stop blended geometrically with the stock's parabolic SAR by
    // alpha (0: the Bollinger stop, 1: the SAR, when the SAR trails below the open). The stop first
    // when the day's bar touched both; a stop sells half a spread below its level, a take-profit only
    // when the high clears it by half a spread. Every setting on a lattice keeps a record; each day the
    // ranking with the best record (mean shrunk by its t-statistic) and, at it, the levels with the
    // best mean over downside deviation (Sortino) so far are refined to a continuous optimum by a parabola through their
    // neighbours along each axis, and today's trades use them.
    constexpr std::size_t nB = 5, nK = 13, nA = 5, nV = nB * nK * nK * nA;
    const auto axis = [](std::size_t d, double x) {  // lattice coordinate -> setting
      return d == 0 ? x / double(nB - 1) : d == 3 ? x / double(nA - 1) : std::exp2(-2.0 + 0.5 * x);
    };
    const std::size_t dims[4] = {nB, nK, nK, nA};
    std::vector<double> s1v(nV, 0.0), s2v(nV, 0.0), dnv(nV, 0.0);  // sums of the results, their squares, the losses' squares
    double nDays = 0;
    std::vector<double> adaptiveLevels, byRank[2];
    std::vector<std::array<double, 4>> used;
    std::vector<std::array<double, 7>> paths;  // per trade: low and high from the open (sd), open-to-close, sd, spread, stopped, took profit
    std::vector<std::string> sd;
    const auto score = [&](std::size_t g) {
      if (nDays < 2) return 0.0;
      const double mu = s1v[g] / nDays, var = std::max(0.0, s2v[g] / nDays - mu * mu);
      return mu != 0 ? mu * std::max(0.0, 1.0 - var / nDays / (mu * mu)) : 0.0;
    };
    const auto index = [&](const std::size_t* c) { return ((c[0] * nK + c[1]) * nK + c[2]) * nA + c[3]; };
    // The continuous settings {beta, k_s, k_t, alpha} from the records so far.
    // The levels' score: the record's mean over its downside deviation (Sortino).
    const auto ddScore = [&](std::size_t g) { return nDays < 2 || !(dnv[g] > 0) ? 0.0 : s1v[g] / nDays / std::sqrt(dnv[g] / nDays); };
    const auto learnt = [&] {
      // The ranking from the best record (mean); the levels, at that ranking, with the best return
      // for their downside.
      std::size_t best = 0;
      for (std::size_t g = 1; g < nV; ++g)
        if (score(g) > score(best)) best = g;
      const std::size_t b0 = best / (nK * nK * nA);
      std::size_t lv = b0 * nK * nK * nA;
      for (std::size_t g = lv + 1; g < (b0 + 1) * nK * nK * nA; ++g)
        if (ddScore(g) > ddScore(lv)) lv = g;
      std::size_t c[4] = {b0, lv / (nK * nA) % nK, lv / nA % nK, lv % nA};
      std::array<double, 4> x{};
      for (std::size_t d = 0; d < 4; ++d) {
        const auto f = [&](std::size_t g) { return d == 0 ? score(g) : ddScore(g); };
        std::size_t at[4] = {c[0], c[1], c[2], c[3]};
        if (d == 0) at[1] = best / (nK * nA) % nK, at[2] = best / nA % nK, at[3] = best % nA;
        double off = 0;
        if (at[d] > 0 && at[d] + 1 < dims[d]) {
          std::size_t lo[4] = {at[0], at[1], at[2], at[3]}, hi[4] = {at[0], at[1], at[2], at[3]};
          --lo[d], ++hi[d];
          const double fm = f(index(lo)), f0 = f(index(at)), fp = f(index(hi)), den = fm - 2 * f0 + fp;
          if (den < 0) off = std::min(0.5, std::max(-0.5, 0.5 * (fm - fp) / den));
        }
        x[d] = axis(d, double(at[d]) + off);
      }
      return x;
    };

    // The day's 10 trades ranked with beta, and their average result at the levels.
    const auto ranked = [&](std::size_t t, double b) {
      std::vector<std::pair<double, std::size_t>> v;
      for (std::size_t i = 0; i < N; ++i) {
        if (!(ft[(t * N + i) * Km + Km - 1] > 0) || !tradable[t * N + i] || !(ft[((t - lag) * N + i) * Km + Km - 1] > 0)) continue;
        const std::size_t src = (t - lag) * N + i;
        v.emplace_back(pred[src * 4] * std::exp(b * ft[src * Km + 20]), i);
      }
      std::partial_sort(v.begin(), v.begin() + std::ptrdiff_t(std::min<std::size_t>(10, v.size())), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      v.resize(std::min<std::size_t>(10, v.size()));
      return v;
    };
    const bool noSpread = std::getenv("NO_SPREAD") != nullptr;
    const auto levels = [&](std::size_t t, std::size_t i, double ks, double kt, double al) {
      double st = -ks * boll[t * N + i];
      if (std::isfinite(sar[t * N + i])) st = (1 - al) * st + al * std::log(std::min(1.0, sar[t * N + i] / m.open(t + 1, i)));  // a SAR above the open: stopped at the open
      return std::make_pair(std::exp(st), std::exp(kt * boll[t * N + i]));
    };
    const auto dayResult = [&](std::size_t t, const std::vector<std::pair<double, std::size_t>>& w, double ks, double kt, double al) {
      double rg = 0;
      for (const auto& e : w) {
        const std::size_t i = e.second;
        const double o = m.open(t + 1, i), hi = m.high(t + 1, i), lo = m.low(t + 1, i), c = m.close(t + 1, i);
        const auto [fs, fk] = levels(t, i, ks, kt, al);
        const double stop = o * fs, take = o * fk, half = noSpread ? 1.0 : std::exp(0.5 * spread[t * N + i]);
        rg += ((lo <= stop ? stop / half : hi >= take * half ? take : c) / o - 1) / double(w.size());
      }
      return rg;
    };
    for (std::size_t q = warm; q < days.size(); ++q) {
      const std::size_t t = days[q];
      if (t + 1 >= T || t < lag) continue;
      double ms = 0, mn = 0;
      for (std::size_t i = 0; i < N; ++i)
        if (ft[(t * N + i) * Km + Km - 1] > 0 && tradable[t * N + i] && ft[((t - lag) * N + i) * Km + Km - 1] > 0) ms += m.close(t + 1, i) / m.close(t, i) - 1, mn += 1;
      if (mn < 30) continue;
      std::vector<std::pair<double, std::size_t>> v[nB];
      for (std::size_t b = 0; b < nB; ++b) v[b] = ranked(t, axis(0, double(b)));
      double r = 0, pm = 0;
      for (const auto& e : v[0]) r += (m.close(t + 1, e.second) / m.open(t + 1, e.second) - 1) / 10, pm += e.first / 10;
      always.push_back(r), gated.push_back(pm > 0 ? r : 0.0), mkt.push_back(ms / mn), sd.push_back(m.dates[t + 1]);
      for (std::size_t k = 0; k < 2; ++k) {
        double rk = 0;
        for (const auto& e : v[k ? nB - 1 : 0]) rk += (m.close(t + 1, e.second) / m.open(t + 1, e.second) - 1) / 10;
        byRank[k].push_back(rk);
      }
      // Today's trades with the settings learnt so far; then every lattice point's record takes today.
      const auto x = learnt();
      const auto today = ranked(t, x[0]);
      adaptiveLevels.push_back(dayResult(t, today, x[1], x[2], x[3])), used.push_back(x);
      for (const auto& e : today) {  // the trades' paths, for the diagnostics
        const std::size_t i = e.second;
        const double o = m.open(t + 1, i), sg = boll[t * N + i];
        if (!(sg > 0)) continue;
        const auto [fs, fk] = levels(t, i, x[1], x[2], x[3]);
        paths.push_back({std::log(m.low(t + 1, i) / o) / sg, std::log(m.high(t + 1, i) / o) / sg, m.close(t + 1, i) / o - 1, sg, spread[t * N + i],
                         double(m.low(t + 1, i) <= o * fs), double(m.high(t + 1, i) >= o * fk)});
      }
      for (std::size_t g = 0; g < nV; ++g) {
        const double y = dayResult(t, v[g / (nK * nK * nA)], axis(1, double(g / (nK * nA) % nK)), axis(2, double(g / nA % nK)), axis(3, double(g % nA)));
        s1v[g] += y, s2v[g] += y * y, dnv[g] += std::min(y, 0.0) * std::min(y, 0.0);
      }
      nDays += 1;
    }
    // Position sizing, learnt: each day's units in every trade, f = min(1, (vbar / v)^gamma) *
    // exp(-lambda * dd), from the trades' record up to the last close: v its recent volatility (about
    // 20 days), vbar that volatility's average so far, dd its drawdown from its peak. gamma (0 to 2)
    // and lambda (0 to 64) on a lattice, each point's sized record kept; each day the point with the
    // best Sortino so far (scaling every day alike leaves it unchanged, so it learns when to trade
    // less, not to trade less) is refined by parabolas, as the levels are.
    constexpr std::size_t nG = 9, nL = 11, nZ = nG * nL;
    const auto gAxis = [](double x) { return 0.25 * x; };
    const auto lAxis = [](double x) { return x <= 0 ? 0.0 : std::exp2(x - 4.0); };  // 0, 1/8 .. 64
    std::vector<double> z1(nZ, 0.0), zd(nZ, 0.0), sized;
    std::vector<std::array<double, 2>> sizeUsed;
    double ev = 0, ew = 0, vsum = 0, vn = 0, eq = 0, peak = 0;
    const auto sizeScore = [&](std::size_t g) { return zd[g] > 0 ? z1[g] / std::sqrt(zd[g]) : 0.0; };
    const auto sizeOf = [&](double gm, double lm) {
      if (ew <= 0 || vn < 1) return 1.0;
      const double vol = std::sqrt(ev / ew), vbar = vsum / vn, dd = 1 - std::exp(eq - peak);
      return std::min(1.0, vol > 0 ? std::pow(vbar / vol, gm) : 1.0) * std::exp(-lm * dd);
    };
    const auto sizeLearnt = [&] {
      std::size_t best = 0;
      for (std::size_t g = 1; g < nZ; ++g)
        if (sizeScore(g) > sizeScore(best)) best = g;
      const std::size_t c[2] = {best / nL, best % nL}, dims2[2] = {nG, nL};
      std::array<double, 2> x{};
      for (std::size_t d = 0; d < 2; ++d) {
        double off = 0;
        if (c[d] > 0 && c[d] + 1 < dims2[d]) {
          const std::size_t lo = d == 0 ? best - nL : best - 1, hi = d == 0 ? best + nL : best + 1;
          const double fm = sizeScore(lo), f0 = sizeScore(best), fp = sizeScore(hi), den = fm - 2 * f0 + fp;
          if (den < 0) off = std::min(0.5, std::max(-0.5, 0.5 * (fm - fp) / den));
        }
        x[d] = d == 0 ? gAxis(double(c[d]) + off) : lAxis(double(c[d]) + off);
      }
      return x;
    };
    for (double y : adaptiveLevels) {
      const auto x = sizeLearnt();
      const double f = sizeOf(x[0], x[1]);
      sized.push_back(f * y), sizeUsed.push_back({x[0], f});
      for (std::size_t g = 0; g < nZ; ++g) {
        const double yg = sizeOf(gAxis(double(g / nL)), lAxis(double(g % nL))) * y;
        z1[g] += yg, zd[g] += std::min(yg, 0.0) * std::min(yg, 0.0);
      }
      // The record takes today: its volatility (about 20 days), that volatility's average, its drawdown.
      ev += (y * y - ev) / 20, ew += (1 - ew) / 20;
      if (ew > 0.6) vsum += std::sqrt(ev / ew), vn += 1;
      eq += std::log1p(y), peak = std::max(peak, eq);
    }
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
      double mx[4] = {0, 0, 0, 0};
      for (const auto& x : used)
        for (std::size_t d = 0; d < 4; ++d) mx[d] += x[d] / double(used.size());
      std::printf("  self-adaptive ranking and levels %+7.1f%% a year (Sharpe %5.2f); mean beta %.2f, stop %.2f sd, take %.2f sd, SAR weight %.2f\n", 100 * a.first,
                  a.second, mx[0], mx[1], mx[2], mx[3]);
      std::printf("  learnt for the next open: beta %.2f, stop %.2f sd, take %.2f sd, SAR weight %.2f\n", fx[0], fx[1], fx[2], fx[3]);
      // Where the trades went within the day: the low and the high from the open in standard
      // deviations, how often the learnt levels were touched, and what holding to the close gave
      // after a dip of k sd against selling at the dip (less half the spread).
      auto qt = [&](std::size_t d, double p) {
        std::vector<double> z;
        for (const auto& x : paths) z.push_back(x[d]);
        std::nth_element(z.begin(), z.begin() + std::ptrdiff_t(p * double(z.size() - 1)), z.end());
        return z[std::size_t(p * double(z.size() - 1))];
      };
      double st = 0, tk = 0;
      for (const auto& x : paths) st += x[5], tk += x[6];
      std::printf("  trades: %zu; sd median %.2f%%, spread median %.2f%%; learnt stop touched %.1f%%, take-profit %.1f%%\n", paths.size(), 100 * qt(3, 0.5),
                  100 * qt(4, 0.5), 100 * st / double(paths.size()), 100 * tk / double(paths.size()));
      std::printf("  low from the open (sd): median %.2f, 10%% %.2f, 1%% %.2f; high: median %.2f, 90%% %.2f, 99%% %.2f\n", qt(0, 0.5), qt(0, 0.1), qt(0, 0.01), qt(1, 0.5),
                  qt(1, 0.9), qt(1, 0.99));
      for (double k : {0.5, 1.0, 2.0, 3.0}) {
        double n = 0, hold = 0, sell = 0, nu = 0, hu = 0, tu = 0;
        for (const auto& x : paths) {
          if (x[0] <= -k) n += 1, hold += x[2], sell += std::exp(-k * x[3] - 0.5 * x[4]) - 1;
          if (x[1] >= k) nu += 1, hu += x[2], tu += std::exp(k * x[3]) - 1;
        }
        std::printf("  dipped %.1f sd: %4.1f%% of trades, held to the close %+.2f%%, stopped there %+.2f%% | rose %.1f sd: %4.1f%%, held %+.2f%%, taken there %+.2f%%\n", k,
                    100 * n / double(paths.size()), n ? 100 * hold / n : 0.0, n ? 100 * sell / n : 0.0, k, 100 * nu / double(paths.size()), nu ? 100 * hu / nu : 0.0,
                    nu ? 100 * tu / nu : 0.0);
      }
    }
    {
      const auto a = stats(sized);
      double mf = 0, wd = 0, pk = 0, dmax = 0, w0 = 0, pk0 = 0, d0 = 0;
      for (std::size_t q = 0; q < sized.size(); ++q) {
        mf += sizeUsed[q][1] / double(sized.size());
        wd += std::log1p(sized[q]), pk = std::max(pk, wd), dmax = std::max(dmax, 1 - std::exp(wd - pk));
        w0 += std::log1p(adaptiveLevels[q]), pk0 = std::max(pk0, w0), d0 = std::max(d0, 1 - std::exp(w0 - pk0));
      }
      const auto fz = sizeLearnt();
      std::printf("  with learnt position sizes       %+7.1f%% a year (Sharpe %5.2f), max drawdown %.0f%% (unsized %.0f%%); mean size %.2f; next open: gamma %.2f, lambda %.2f, size %.2f\n",
                  100 * a.first, a.second, 100 * dmax, 100 * d0, mf, fz[0], fz[1], sizeOf(fz[0], fz[1]));
    }
    std::printf("  only days predicted to gain      %+7.1f%% a year (Sharpe %5.2f), %zu of %zu days\n", 100 * a2.first, a2.second, traded, gated.size());
    std::printf("  market (all stocks, held)        %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a3.first, a3.second);
    // The next open's trades: the 10 highest predictions lag closes before the last close, among
    // the stocks with a bar at the last close (their targets are not known yet), ranked and with
    // levels by the settings learnt over the whole backtest. Expected open-to-close: the prediction
    // times the stock's volatility (feature 20, its log). The SAR stop is from the last close (the
    // open is not known yet).
    lastRun = Run{sized, mkt, sd, {}};
    {
      const auto fz = sizeLearnt();
      lastRun.size = sizeOf(fz[0], fz[1]);
    }
    {
      const std::size_t t = T - 1;
      std::vector<std::pair<double, std::size_t>> v;
      if (t >= lag)
        for (std::size_t i = 0; i < N; ++i) {
          const double* a = &ft[((t - lag) * N + i) * Km];
          if (!(a[Km - 1] > 0) || !std::isfinite(m.close(t, i)) || !(m.close(t, i) > 0)) continue;
          if (topN > 0 && !tradable[(t - lag) * N + i]) continue;
          v.emplace_back(pred[((t - lag) * N + i) * 4] * std::exp(fx[0] * a[20]), i);
        }
      std::partial_sort(v.begin(), v.begin() + std::ptrdiff_t(std::min<std::size_t>(10, v.size())), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      for (std::size_t q = 0; q < std::min<std::size_t>(10, v.size()); ++q) {
        const std::size_t i = v[q].second, src = (t - lag) * N + i;
        const double sig = std::exp(ft[src * Km + 20]);
        double st = -fx[1] * boll[t * N + i];
        if (std::isfinite(sar[t * N + i])) st = (1 - fx[3]) * st + fx[3] * std::log(std::min(1.0, sar[t * N + i] / m.close(t, i)));
        if (v[q].first > 0) lastRun.picks.push_back({i, std::expm1(pred[src * 4] * sig), std::expm1(st), std::expm1(fx[2] * boll[t * N + i])});
      }
    }
    return pred;
  };
  // The features with their interactions with liquidity and volatility (kernels/v2/interact.wgsl):
  // momentum 2d, volume shock 4d, open-to-close, gap, close in range, intraday and overnight
  // momentum 64d.
  const std::vector<double> sigs = {0, 6, 12, 14, 15, 23, 26};
  const std::size_t MI = sigs.size(), FI = F + 2 * MI, KI = FI + 5;
  std::vector<double> ftI(T * N * KI, 0.0);
  {
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
    std::snprintf(b, sizeof b, ",\"size\":%.4f", R.size);
    o << b << ",\"orders\":[";
    for (std::size_t q = 0; q < R.picks.size(); ++q) {
      const auto& pk = R.picks[q];
      auto lvl = [](double x) { return std::isfinite(x) ? std::to_string(x) : std::string("null"); };
      std::snprintf(b, sizeof b, "%s{\"ticker\":\"%s\",\"expected\":%.6g,\"stop\":%s,\"take\":%s}", q ? "," : "", m.tickers[pk.stock].c_str(), pk.expected,
                    lvl(pk.stop).c_str(), lvl(pk.take).c_str());
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
