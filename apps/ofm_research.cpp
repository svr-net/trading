// The predictive model's research: each OHLCV feature's power to predict the next day's moves.
//   ofm_research --csv stocks.csv
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
  std::string csv;
  for (int k = 1; k + 1 < argc; ++k)
    if (std::string(argv[k]) == "--csv") csv = argv[k + 1];
  const ofm::Market m = ofm::parseMarketCsv(readFile(csv));
  const std::size_t T = m.T(), N = m.N(), F = 19, K = F + 5;
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
  const char* fn[19] = {"momentum 2d", "momentum 4d", "momentum 8d", "momentum 16d", "momentum 32d", "momentum 64d", "volume shock 4d", "volume shock 16d",
                        "volume shock 64d", "range shock 4d", "range shock 16d", "range shock 64d", "open-to-close", "close-to-close", "gap",
                        "close in range", "upper wick", "lower wick", "signed volume shock"};
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
  // day buy the 10 stocks predicted highest at the open, sell at the close.
  const std::size_t D = F + 1, E = D * (D + 4);
  std::vector<double> zs(T * F * 2, 0.0), gram(T * E, 0.0), beta(T * 4 * D, 0.0), pred(T * N * 4, 0.0);
  ofm::v2::gen::Zscore<double> kz;
  kz.P = P, kz.feat = feat.data(), kz.zs = zs.data();
  ofm::detail::dispatch(kz, ofm::detail::groups(std::uint32_t(T * F), 64));
  ofm::v2::gen::Gram<double> kg;
  kg.P = P, kg.feat = feat.data(), kg.zs = zs.data(), kg.out = gram.data();
  ofm::detail::dispatch(kg, ofm::detail::groups(std::uint32_t(T * E), 64));
  std::vector<double> acc(E, 0.0);
  for (std::size_t t = 0; t < T; ++t) {
    // Coefficients for close t from the days before (their targets are known by then).
    for (std::size_t k = 0; k < 4; ++k) {
      std::vector<double> M(acc.begin(), acc.begin() + std::ptrdiff_t(D * D)), v(D);
      for (std::size_t j = 0; j < D; ++j) v[j] = acc[D * D + k * D + j];
      double tr = 0;
      for (std::size_t j = 0; j < D; ++j) tr += M[j * D + j];
      if (!(tr > 0)) continue;
      for (std::size_t j = 0; j < D; ++j) M[j * D + j] += 1e-6 * tr;
      for (std::size_t c = 0; c < D; ++c)  // Gauss-Jordan (the matrix is positive definite)
        for (std::size_t r = 0; r < D; ++r)
          if (r != c) {
            const double q = M[r * D + c] / M[c * D + c];
            for (std::size_t j = 0; j < D; ++j) M[r * D + j] -= q * M[c * D + j];
            v[r] -= q * v[c];
          }
      for (std::size_t j = 0; j < D; ++j) beta[(t * 4 + k) * D + j] = v[j] / M[j * D + j];
    }
    if (t > 0)
      for (std::size_t e = 0; e < E; ++e) acc[e] += gram[(t - 1) * E + e];
  }
  ofm::v2::gen::Predict<double> kp;
  kp.P = P, kp.feat = feat.data(), kp.zs = zs.data(), kp.beta = beta.data(), kp.out = pred.data();
  ofm::detail::dispatch(kp, ofm::detail::groups(std::uint32_t(T * N), 64));
  const std::size_t warm = 250;
  std::vector<double> strat, mkt;
  std::vector<std::string> sd;
  for (std::size_t q = warm; q < days.size(); ++q) {
    const std::size_t t = days[q];
    if (t + 1 >= T) continue;
    std::vector<std::pair<double, std::size_t>> v;
    double ms = 0, mn = 0;
    for (std::size_t i = 0; i < N; ++i) {
      if (!(feat[(t * N + i) * K + K - 1] > 0)) continue;
      v.emplace_back(pred[(t * N + i) * 4 + 0], i);
      ms += m.close(t + 1, i) / m.close(t, i) - 1, mn += 1;
    }
    if (v.size() < 30) continue;
    std::partial_sort(v.begin(), v.begin() + 10, v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    double r = 0;
    for (std::size_t j = 0; j < 10; ++j) r += (m.close(t + 1, v[j].second) / m.open(t + 1, v[j].second) - 1) / 10;
    strat.push_back(r), mkt.push_back(ms / mn), sd.push_back(m.dates[t + 1]);
  }
  auto stats = [](const std::vector<double>& x, std::size_t lo, std::size_t hi) {
    double w = 0, s1 = 0, s2 = 0;
    for (std::size_t k = lo; k < hi; ++k) w += std::log1p(x[k]), s1 += x[k], s2 += x[k] * x[k];
    const double n = double(hi - lo), mu = s1 / n, sdv = std::sqrt(std::max(0.0, s2 / n - mu * mu));
    return std::make_pair(std::expm1(w * 252.0 / n), sdv > 0 ? mu / sdv * std::sqrt(252.0) : 0.0);
  };
  const std::size_t n = strat.size();
  std::printf("\nWalk-forward model, top 10 by predicted open-to-close, bought at the open and sold at the close (%s .. %s):\n", sd.front().c_str(), sd.back().c_str());
  {
    const auto a1 = stats(strat, 0, n), a2 = stats(mkt, 0, n);
    std::printf("  strategy %+6.1f%% a year (Sharpe %5.2f) | market %+6.1f%% (Sharpe %5.2f)\n", 100 * a1.first, a1.second, 100 * a2.first, a2.second);
  }
  return 0;
}
