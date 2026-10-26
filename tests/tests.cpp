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
