// Tests of the predictive model's kernels, the neural network and the market data.
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "invoke.hpp"
#include "net.hpp"
#include "ofm/data.hpp"
#include "v2_kernels.hpp"

namespace {

std::vector<std::pair<std::string, std::function<void()>>>& tests() {
  static std::vector<std::pair<std::string, std::function<void()>>> t;
  return t;
}
struct Reg {
  Reg(const char* n, std::function<void()> f) { tests().push_back({n, std::move(f)}); }
};
#define TEST(name)                     \
  static void name();                  \
  static Reg reg_##name(#name, &name); \
  static void name()
#define CHECK(c) \
  if (!(c)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": CHECK(" #c ")")

// The features kernel's table (kernels/v2/features.wgsl) of a market.
std::vector<double> features(const ofm::Market& m, std::size_t K) {
  const std::size_t T = m.T(), N = m.N();
  std::vector<double> bars(5 * T * N, 0.0), out(T * N * K, 0.0);
  const ofm::Panel* pn[5] = {&m.open, &m.high, &m.low, &m.close, &m.volume};
  for (std::size_t f = 0; f < 5; ++f)
    for (std::size_t t = 0; t < T; ++t)
      for (std::size_t i = 0; i < N; ++i) bars[(f * T + t) * N + i] = std::isfinite((*pn[f])(t, i)) ? (*pn[f])(t, i) : 0.0;
  ofm::v2::gen::V2Header<double> P;
  P.T = std::uint32_t(T), P.N = std::uint32_t(N), P.K = std::uint32_t(K), P.NT = std::uint32_t(K - 5);
  ofm::v2::gen::Features<double> k;
  k.P = P, k.bars = bars.data(), k.out = out.data();
  ofm::detail::dispatch(k, ofm::detail::groups(std::uint32_t(N), 64));
  return out;
}

}  // namespace

TEST(features_use_only_the_past) {
  const auto syn = ofm::syntheticMarket(7, 40, 400);
  const std::size_t K = 32, F = K - 5, T = syn.market.T(), N = syn.market.N();
  const auto a = features(syn.market, K);
  // Changing the last 50 days leaves every earlier day's features the same.
  auto m2 = syn.market;
  for (std::size_t t = T - 50; t < T; ++t)
    for (std::size_t i = 0; i < N; ++i) m2.close(t, i) *= 1.03, m2.high(t, i) *= 1.03, m2.volume(t, i) *= 2;
  const auto b = features(m2, K);
  std::size_t defined = 0;
  for (std::size_t t = 0; t + 50 < T; ++t)
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t f = 0; f < F; ++f) {
        CHECK(a[(t * N + i) * K + f] == b[(t * N + i) * K + f]);
        defined += a[(t * N + i) * K + K - 1] > 0;
      }
  CHECK(defined > 0);
}

// The walk-forward model's kernels (zscore, gram, solve, predict, as apps/ofm_research.cpp chains
// them) on inputs with a known relation: the coefficients recover it, equal a plain ridge solve of the
// days before (z-scored within each day), and never see a later day's target.
namespace {
struct Fit {
  std::vector<double> beta, pred;
};
Fit walkForward(const std::vector<double>& ft, std::size_t T, std::size_t N, std::size_t F, double ridge) {
  const std::size_t K = F + 5, D = F + 1, E = D * (D + 4);
  ofm::v2::gen::V2Header<double> Q;
  Q.T = std::uint32_t(T), Q.N = std::uint32_t(N), Q.K = std::uint32_t(K), Q.NT = std::uint32_t(F);
  std::vector<double> zs(T * F * 2), gram(T * E), acc(T * E, 0.0), rk = {ridge};
  Fit r{std::vector<double>(T * 4 * D, 0.0), std::vector<double>(T * N * 4, 0.0)};
  ofm::v2::gen::Zscore<double> kz;
  kz.P = Q, kz.feat = const_cast<double*>(ft.data()), kz.zs = zs.data();
  ofm::detail::dispatch(kz, ofm::detail::groups(std::uint32_t(T * F), 64));
  ofm::v2::gen::Gram<double> kg;
  kg.P = Q, kg.feat = const_cast<double*>(ft.data()), kg.zs = zs.data(), kg.out = gram.data();
  ofm::detail::dispatch(kg, std::uint32_t(T));
  for (std::size_t t = 2; t < T; ++t)  // as the app: day t's sums are the days up to t - 2 (their targets known at t)
    for (std::size_t e = 0; e < E; ++e) acc[t * E + e] = acc[(t - 1) * E + e] + gram[(t - 2) * E + e];
  ofm::v2::gen::V2Header<double> S = Q;
  S.T = std::uint32_t(T), S.u0 = std::uint32_t(D), S.u1 = std::uint32_t(E);
  ofm::v2::gen::Solve<double> ks;
  ks.P = S, ks.acc = acc.data(), ks.rk = rk.data(), ks.beta = r.beta.data();
  ofm::detail::dispatch(ks, ofm::detail::groups(std::uint32_t(T * 4), 64));
  ofm::v2::gen::Predict<double> kp;
  kp.P = Q, kp.feat = const_cast<double*>(ft.data()), kp.zs = zs.data(), kp.beta = r.beta.data(), kp.out = r.pred.data();
  ofm::detail::dispatch(kp, ofm::detail::groups(std::uint32_t(T * N), 64));
  return r;
}
}  // namespace

TEST(walk_forward_fit_is_a_ridge_of_the_past) {
  const std::size_t T = 160, N = 120, F = 3, K = F + 5, D = F + 1;
  const double w[F] = {0.5, -0.3, 0.0}, ridge = 1e-6;
  std::vector<double> ft(T * N * K, 0.0);
  std::uint64_t z = 12345;
  const auto u = [&] { z = z * 6364136223846793005ull + 1442695040888963407ull; return double(z >> 11) / double(1ull << 53); };
  const auto g = [&] { return std::sqrt(-2 * std::log(u() + 1e-300)) * std::cos(6.283185307179586 * u()); };
  for (std::size_t r = 0; r < T * N; ++r) {
    double* a = &ft[r * K];
    double y = 0;
    for (std::size_t f = 0; f < F; ++f) a[f] = 2 * g() + 0.5, y += w[f] * a[f] / 2;  // the inputs' sd is 2: z = (x - 0.5) / 2
    for (std::size_t k = 0; k < 4; ++k) a[F + k] = y + 0.5 * g();
    a[K - 1] = 1;
  }
  const Fit fit = walkForward(ft, T, N, F, ridge);
  // 1. The latest coefficients recover the relation (on z-scored inputs).
  for (std::size_t f = 0; f < F; ++f) CHECK(std::fabs(fit.beta[((T - 1) * 4) * D + 1 + f] - w[f]) < 0.03);
  // 2. Day t's coefficients equal a plain ridge solve of the days before t - 1, z-scored within each day.
  for (std::size_t t : {std::size_t(10), std::size_t(80), T - 1}) {
    std::vector<double> A(D * D, 0.0), b(D, 0.0);
    for (std::size_t d = 0; d + 2 <= t; ++d) {
      double mu[F], sd[F];
      for (std::size_t f = 0; f < F; ++f) {
        double s1 = 0, s2 = 0;
        for (std::size_t i = 0; i < N; ++i) s1 += ft[(d * N + i) * K + f], s2 += ft[(d * N + i) * K + f] * ft[(d * N + i) * K + f];
        mu[f] = s1 / double(N), sd[f] = std::sqrt(std::max(s2 / double(N) - mu[f] * mu[f], 1e-12));
      }
      for (std::size_t i = 0; i < N; ++i) {
        double x[D];
        x[0] = 1;
        for (std::size_t f = 0; f < F; ++f) x[1 + f] = std::min(std::max((ft[(d * N + i) * K + f] - mu[f]) / sd[f], -4.0), 4.0);
        const double y = std::min(std::max(ft[(d * N + i) * K + F], -6.0), 6.0);
        for (std::size_t p = 0; p < D; ++p) {
          b[p] += x[p] * y;
          for (std::size_t q = 0; q < D; ++q) A[p * D + q] += x[p] * x[q];
        }
      }
    }
    double tr = 0;
    for (std::size_t p = 0; p < D; ++p) tr += A[p * D + p];
    for (std::size_t p = 0; p < D; ++p) A[p * D + p] += ridge * tr;
    for (std::size_t c = 0; c < D; ++c)  // Gaussian elimination with back substitution
      for (std::size_t r = c + 1; r < D; ++r) {
        const double q = A[r * D + c] / A[c * D + c];
        for (std::size_t k = c; k < D; ++k) A[r * D + k] -= q * A[c * D + k];
        b[r] -= q * b[c];
      }
    double x[D];
    for (std::size_t p = D; p-- > 0;) {
      double s = b[p];
      for (std::size_t k = p + 1; k < D; ++k) s -= A[p * D + k] * x[k];
      x[p] = s / A[p * D + p];
    }
    for (std::size_t p = 0; p < D; ++p) CHECK(std::fabs(fit.beta[(t * 4) * D + p] - x[p]) < 1e-9);
  }
  // 3. Changing the targets from day t - 1 on leaves day t's coefficients and predictions as they were.
  const std::size_t t0 = 100;
  auto ft2 = ft;
  for (std::size_t r = (t0 - 1) * N; r < T * N; ++r)
    for (std::size_t k = 0; k < 4; ++k) ft2[r * K + F + k] = -ft2[r * K + F + k] + 3;
  const Fit fit2 = walkForward(ft2, T, N, F, ridge);
  for (std::size_t t = 0; t <= t0; ++t)
    for (std::size_t p = 0; p < 4 * D; ++p) CHECK(fit.beta[t * 4 * D + p] == fit2.beta[t * 4 * D + p]);
  for (std::size_t r = 0; r < (t0 + 1) * N * 4; ++r) CHECK(fit.pred[r] == fit2.pred[r]);
}

TEST(network_learns_a_simple_relation) {
  // y = 0.5 x0 - 0.3 x1 on random inputs: the median network gets close after some steps.
  ofm::net::Net net(2, 16, {0.5});
  net.lr = 3e-3;
  std::uint64_t r = 1;
  auto u = [&]() { r ^= r << 13, r ^= r >> 7, r ^= r << 17; return double(r >> 11) / double(1ull << 53) * 2 - 1; };
  for (int s = 0; s < 3000; ++s) {
    std::vector<double> x, y;
    for (int k = 0; k < 32; ++k) {
      const double a = u(), b = u();
      x.push_back(a), x.push_back(b), y.push_back(0.5 * a - 0.3 * b);
    }
    net.step(x, y, 32);
  }
  const auto p = net.predict({0.8, -0.5}, 1);
  CHECK(std::fabs(p[0] - (0.5 * 0.8 + 0.3 * 0.5)) < 0.08);
}

TEST(csv_round_trip) {
  const auto m = ofm::parseMarketCsv(
      "date,ticker,open,high,low,close,volume\n2024-01-02,AAA,10,11,9,10.5,100\n2024-01-03,AAA,10.5,12,10,11,120\n2024-01-04,AAA,11,11.5,10.5,11.2,90\n"
      "2024-01-02,BBB,5,5.5,4.5,5,50\n2024-01-04,BBB,5,5.2,4.9,5.1,60\n");
  CHECK(m.T() == 3 && m.N() == 2);
  CHECK(m.close(1, 0) == 11 && std::isnan(m.close(1, 1)) && m.close(2, 1) == 5.1);
}

int main() {
  int failed = 0;
  for (const auto& [name, f] : tests()) {
    try {
      f();
      std::printf("[  OK  ] %s\n", name.c_str());
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[ FAIL ] %s: %s\n", name.c_str(), e.what());
    }
  }
  std::printf("%zu passed, %d failed\n", tests().size() - std::size_t(failed), failed);
  return failed ? 1 : 0;
}
