// The predictive model's research: each OHLCV feature's power to predict the next day's moves.
//   ofm_research --csv stocks.csv [--series futures.csv]
// Prints, for every feature and target, the mean of its day-by-day cross-sectional correlation
// with the target and its t-statistic (the features and the correlations run in fused kernels,
// kernels/v2).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "daykernels.hpp"
#include "invoke.hpp"
#include "ofm/data.hpp"
#include "v2_kernels.hpp"

namespace {
std::string readFile(const std::string& p) {
  std::ifstream in(p);
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}
}  // namespace

int main(int argc, char** argv) {
  std::string csv, seriesCsv;
  for (int k = 1; k + 1 < argc; ++k) {
    if (std::string(argv[k]) == "--csv") csv = argv[k + 1];
    if (std::string(argv[k]) == "--series") seriesCsv = argv[k + 1];
  }
  const ofm::Market m = ofm::parseMarketCsv(readFile(csv));
  const std::size_t T = m.T(), N = m.N(), F = 21, K = F + 5;
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
  const char* fn[21] = {"momentum 2d", "momentum 4d", "momentum 8d", "momentum 16d", "momentum 32d", "momentum 64d", "volume shock 4d", "volume shock 16d",
                        "volume shock 64d", "range shock 4d", "range shock 16d", "range shock 64d", "open-to-close", "close-to-close", "gap",
                        "close in range", "upper wick", "lower wick", "signed volume shock", "liquidity (traded value)", "volatility level"};
  const char* tn[4] = {"next open-to-close", "next open-to-high", "next open-to-low (fall)", "next close-to-close"};
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
    std::vector<std::string> sd;
    for (std::size_t q = warm; q < days.size(); ++q) {
      const std::size_t t = days[q];
      if (t + 1 >= T) continue;
      std::vector<std::pair<double, std::size_t>> v;
      double ms = 0, mn = 0;
      for (std::size_t i = 0; i < N; ++i) {
        if (!(ft[(t * N + i) * Km + Km - 1] > 0)) continue;
        v.emplace_back(pred[(t * N + i) * 4 + 0], i);
        ms += m.close(t + 1, i) / m.close(t, i) - 1, mn += 1;
      }
      if (v.size() < 30) continue;
      std::partial_sort(v.begin(), v.begin() + 10, v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      double r = 0, pm = 0;
      for (std::size_t q2 = 0; q2 < 10; ++q2) r += (m.close(t + 1, v[q2].second) / m.open(t + 1, v[q2].second) - 1) / 10, pm += v[q2].first / 10;
      always.push_back(r), gated.push_back(pm > 0 ? r : 0.0), mkt.push_back(ms / mn), sd.push_back(m.dates[t + 1]);
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
    std::printf("  only days predicted to gain      %+7.1f%% a year (Sharpe %5.2f), %zu of %zu days\n", 100 * a2.first, a2.second, traded, gated.size());
    std::printf("  market (all stocks, held)        %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a3.first, a3.second);
    return pred;
  };
  const auto linPred = model(feat, F, K, F, "Walk-forward model on the stocks' own OHLCV");
  // The same inputs through a neural network (two hidden layers of rectified linear units, the
  // fused kernels of src/daykernels): each day it predicts every stock's next open-to-close from
  // the day's standardised features, then learns from that day's outcomes once they are known
  // (shuffled mini-batches of 32, one pass), so every prediction is out of sample.
  {
    std::vector<double> zs(T * F * 2, 0.0);
    ofm::v2::gen::Zscore<double> kz;
    kz.P = P, kz.feat = feat.data(), kz.zs = zs.data();
    ofm::detail::dispatch(kz, ofm::detail::groups(std::uint32_t(T * F), 64));
    ofm::day::kernels::Net net(F, 32, {0.5});
    std::vector<double> memX, memY;  // every earlier day's samples (replay)
    std::vector<double> always, ens, mkt;
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
      // The target relative to the day's average (the ranking is what is traded).
      {
        double my = 0;
        for (double v : y) my += v / double(y.size());
        for (double& v : y) v -= my;
      }
      if (q >= 250 && t + 1 < T) {
        const auto pr = net.predict(x, who.size());
        std::vector<std::pair<double, std::size_t>> v;
        for (std::size_t r = 0; r < who.size(); ++r) v.emplace_back(pr[r], who[r]);
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
        std::partial_sort(sc.begin(), sc.begin() + 10, sc.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        double r2 = 0;
        for (std::size_t r = 0; r < 10; ++r) r2 += (m.close(t + 1, sc[r].second) / m.open(t + 1, sc[r].second) - 1) / 10;
        ens.push_back(r2);
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
    const auto a1 = stats(always), a2 = stats(ens), a3 = stats(mkt);
    std::printf("\nNeural network on the stocks' own OHLCV (%s .. %s), the 10 highest predicted, open to close:\n", sd.front().c_str(), sd.back().c_str());
    std::printf("  every day                        %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a1.first, a1.second);
    std::printf("  ensemble with the linear model   %+7.1f%% a year (Sharpe %5.2f)\n", 100 * a2.first, a2.second);
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
