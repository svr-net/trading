// Tests of the model, the kernels' CPU emulation and the trading.
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "ofm/kernels.hpp"
#include "ofm/report.hpp"
#include "ofm/structure.hpp"
#include "ofm/topk.hpp"

namespace {

std::vector<std::pair<std::string, std::function<void()>>>& tests() {
  static std::vector<std::pair<std::string, std::function<void()>>> t;
  return t;
}
struct Reg {
  Reg(const char* n, std::function<void()> f) { tests().push_back({n, std::move(f)}); }
};
#define TEST(name)                         \
  static void name();                      \
  static Reg reg_##name(#name, &name);     \
  static void name()
#define CHECK(c) \
  if (!(c)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": CHECK(" #c ")")

const ofm::Synthetic& sample() {
  static const ofm::Synthetic s = ofm::syntheticMarket(7, 60, 1200);
  return s;
}

}  // namespace

TEST(plan_uses_dyadic_horizons_up_to_an_eighth_of_the_history) {
  const ofm::Plan p = ofm::compilePlan(sample().market);
  CHECK(p.horizons.front() == 2);
  CHECK(p.horizons.back() == 32);   // 60 stocks support 5 horizons (50 signals < 60 stocks)
  CHECK(p.K == ofm::kFamilies * p.horizons.size());
  CHECK(p.first == 32);
  CHECK(ofm::compilePlan(ofm::syntheticMarket(1, 200, 1200).market).horizons.back() == 128);  // 1200 / 8 = 150 -> 128
  CHECK(p.stride == p.K * (p.K + 1) / 2 + p.K + 2);
  for (std::size_t i = 0; i < p.N; ++i) CHECK(p.tables[4 * p.T * p.N + (p.first - 1) * p.N + i] == 0.0f);
}

TEST(emulated_kernels_match_the_reference) {
  const ofm::Plan p = ofm::compilePlan(sample().market);
  std::vector<double> zr, gr;
  std::vector<float> ze, ge;
  ofm::referenceChunk(p, 0, zr, gr);
  ofm::kernels::emulateZscore(p, 0, ze);
  ofm::kernels::emulateGram(p, 0, ze, ge);
  CHECK(zr.size() == ze.size() && gr.size() == ge.size());
  std::size_t differ = 0;
  for (std::size_t k = 0; k < zr.size(); ++k) differ += std::fabs(zr[k] - ze[k]) > 1e-3 ? 1 : 0;
  // Single precision reorders near ties (frequent in short-horizon indicators such as a 2-day RSI
  // or Bollinger z, which take few distinct values): a few percent of z-scores move.
  std::printf("  z-scores differing: %.2f%%\n", 100.0 * static_cast<double>(differ) / static_cast<double>(zr.size()));
  CHECK(differ < zr.size() / 20);
  double worst = 0;
  std::size_t at = 0;
  for (std::size_t k = 0; k < gr.size(); ++k) {
    // Relative to the scale of the day's Gram matrix: its diagonal is the number of stocks.
    const double scale = gr[(k / p.stride) * p.stride + p.pairs() + p.K];
    const double d = std::fabs(gr[k] - ge[k]) / std::max(scale, 1.0);
    if (d > worst) worst = d, at = k;
  }
  if (!(worst < 0.05)) std::printf("  gram worst %.4g at %zu (slot %zu of %zu): %.6g vs %.6g\n", worst, at, at % p.stride, p.stride, gr[at], ge[at]);
  CHECK(worst < 0.01);
  const auto a = ofm::runModel(sample().market, "reference"), b = ofm::runModel(sample().market, "emulated");
  double sab = 0, saa = 0, sbb = 0;
  for (std::size_t k = 0; k < a.E.v.size(); ++k)
    if (std::isfinite(a.E.v[k]) && std::isfinite(b.E.v[k])) sab += a.E.v[k] * b.E.v[k], saa += a.E.v[k] * a.E.v[k], sbb += b.E.v[k] * b.E.v[k];
  CHECK(saa > 0 && sab / std::sqrt(saa * sbb) > 0.999);
}

TEST(expected_returns_are_relative_to_the_market) {
  const auto f = ofm::runModel(sample().market);
  const std::size_t t = f.E.T - 1;
  double s = 0, n = 0, a = 0;
  for (std::size_t i = 0; i < f.E.N; ++i)
    if (std::isfinite(f.E(t, i))) s += f.E(t, i), a += std::fabs(f.E(t, i)), n += 1;
  CHECK(n > 0 && std::fabs(s) < 1e-9 + 1e-6 * a);
}

TEST(nothing_uses_the_future) {
  ofm::Market m = sample().market;
  const auto f1 = ofm::runModel(m);
  const ofm::Costs c;
  const auto b1 = ofm::backtest(m, f1, c, sample().series);
  const std::size_t X = m.T() - 100;
  for (std::size_t t = X; t < m.T(); ++t)
    for (std::size_t i = 0; i < m.N(); ++i) {
      const double k = 1.0 + 0.5 * std::sin(static_cast<double>(t * 7 + i));
      m.open(t, i) *= k, m.high(t, i) *= k * 1.1, m.low(t, i) *= k, m.close(t, i) *= k, m.volume(t, i) *= 3;
    }
  auto series = sample().series;
  for (auto& [n, v] : series)
    for (std::size_t t = X; t < v.size(); ++t) v[t] *= 1.3;
  const auto f2 = ofm::runModel(m);
  for (std::size_t t = 0; t < X; ++t)
    for (std::size_t i = 0; i < m.N(); ++i) {
      const double x = f1.E(t, i), y = f2.E(t, i);
      CHECK((std::isnan(x) && std::isnan(y)) || x == y);
    }
  const auto b2 = ofm::backtest(m, f2, c, series);
  // Day X's return is the first one that may see the change (positions held into day X).
  for (std::size_t k = 0; k + b1.start + 1 < X; ++k) CHECK(b1.model[k] == b2.model[k] && b1.hedged[k] == b2.hedged[k]);
}

TEST(finds_the_built_in_effects_and_beats_the_market_on_the_sample) {
  const auto s = ofm::syntheticMarket(11, 80, 2000);
  const auto f = ofm::runModel(s.market);
  std::size_t nonzero = 0;
  for (double p : f.premium) nonzero += p != 0 ? 1 : 0;
  CHECK(nonzero > 0);
  const auto bt = ofm::backtest(s.market, f, ofm::Costs::fromPercent(0.10, 0.50), s.series);
  CHECK(ofm::metrics(bt.model).sharpe > ofm::metrics(bt.market).sharpe);
}

TEST(pure_noise_gives_little_trading) {
  // Shuffle the sample's returns across days per stock: no effect survives.
  auto s = ofm::syntheticMarket(3, 50, 1000);
  ofm::Market& m = s.market;
  std::uint64_t x = 12345;
  for (std::size_t i = 0; i < m.N(); ++i) {
    std::vector<double> r;
    for (std::size_t t = 1; t < m.T(); ++t) r.push_back(m.close(t, i) / m.close(t - 1, i));
    for (std::size_t k = r.size(); k > 1; --k) {
      x = x * 6364136223846793005ull + 1442695040888963407ull;
      std::swap(r[k - 1], r[(x >> 33) % k]);
    }
    for (std::size_t t = 1; t < m.T(); ++t) {
      const double c = m.close(t - 1, i) * r[t - 1];
      m.close(t, i) = c, m.open(t, i) = c, m.high(t, i) = c * 1.002, m.low(t, i) = c / 1.002;
    }
  }
  const auto f = ofm::runModel(m);
  const auto bt = ofm::backtest(m, f, ofm::Costs::fromPercent(0.10, 0.50), {});
  double turn = 0;
  for (double v : bt.turnover) turn += v;
  CHECK(turn / (static_cast<double>(bt.turnover.size()) / 252.0) < 3.0);
}

TEST(csv_round_trip_and_report) {
  const std::string csv =
      "date,ticker,open,high,low,close,volume\n2020-01-01,A,1,1,1,1,10\n2020-01-02,A,1,1.2,0.9,1.1,10\n2020-01-03,A,1.1,1.2,1,1.05,10\n"
      "2020-01-01,B,2,2,2,2,5\n2020-01-03,B,2,2.1,1.9,2.05,5\n";
  const auto m = ofm::parseMarketCsv(csv);
  CHECK(m.N() == 2 && m.T() == 3 && std::isnan(m.close(1, 1)) && m.close(2, 0) == 1.05);
  const auto ew = ofm::equalWeightIndex(m);
  CHECK(ew[0] == 1.0 && std::fabs(ew[1] - 1.1) < 1e-12 && std::fabs(ew[2] - 1.1 * (1.05 / 1.1)) < 1e-12);  // B has no bar on day 1
  const auto s = ofm::parseSeriesCsv("date,ticker,close\n2020-01-02,F,100\n", m.dates);
  CHECK(std::isnan(s.at("F")[0]) && s.at("F")[1] == 100 && std::isnan(s.at("F")[2]));
  const auto syn = ofm::syntheticMarket(5, 40, 800);
  const auto f = ofm::runModel(syn.market);
  const auto bt = ofm::backtest(syn.market, f, ofm::Costs{}, syn.series);
  const std::string j = ofm::reportJson(syn.market, f, bt, ofm::Costs{});
  CHECK(j.front() == '{' && j.back() == '}' && j.find("\"expected\":[") != std::string::npos);
}

TEST(market_structure_sees_one_factor_and_independent_stocks) {
  // Independent stocks: effective dimension near 1, long tree; one common factor: dimension near
  // 1/n, short tree.
  auto make = [](double common) {
    ofm::Market m;
    const std::size_t T = 400, N = 30;
    m.close = m.open = m.high = m.low = m.volume = ofm::Panel(T, N, 1.0);
    std::uint64_t x = 99;
    auto nrm = [&]() {
      x = x * 6364136223846793005ull + 1442695040888963407ull;
      const double u1 = (static_cast<double>(x >> 11) + 0.5) * 0x1.0p-53;
      x = x * 6364136223846793005ull + 1442695040888963407ull;
      const double u2 = (static_cast<double>(x >> 11) + 0.5) * 0x1.0p-53;
      return std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    };
    for (std::size_t t = 1; t < T; ++t) {
      const double f = nrm();
      for (std::size_t i = 0; i < N; ++i) m.close(t, i) = m.close(t - 1, i) * (1 + 0.01 * (common * f + (1 - common) * nrm()));
    }
    return ofm::marketStructure(m, {64}, {});
  };
  const auto ind = make(0.0), one = make(0.98);
  const std::size_t t = 399;
  CHECK(ind.dimension(t, 0) > 0.4 && one.dimension(t, 0) < 0.1);
  CHECK(ind.treeLength(t, 0) > 1.2 && one.treeLength(t, 0) < 0.5);
}

TEST(kernel_sources_declare_their_entry_points) {
  for (const std::string* s : {&ofm::kernels::zscoreSource(), &ofm::kernels::gramSource(), &ofm::kernels::expectSource()})
    CHECK(s->find("fn main") != std::string::npos && s->find("struct Header") != std::string::npos);
}

int main() {
  int failed = 0;
  for (auto& [name, fn] : tests()) {
    try {
      fn();
      std::printf("[  OK  ] %s\n", name.c_str());
    } catch (const std::exception& e) {
      ++failed;
      std::printf("[ FAIL ] %s\n         %s\n", name.c_str(), e.what());
    }
  }
  std::printf("%zu passed, %d failed\n", tests().size() - static_cast<std::size_t>(failed), failed);
  return failed ? 1 : 0;
}

TEST(rolling_top_ten_holds_ten_and_uses_no_future) {
  const auto& s = sample();
  const auto f = ofm::runModel(s.market);
  const auto bt = ofm::backtest(s.market, f, ofm::Costs{}, s.series);
  const auto r = ofm::topKBacktests(s.market, f, ofm::Costs{}, bt, 10);
  CHECK(r.size() == 1 && r[0].dayTrades);
  // The daily ranking holds exactly the 10 best at the last close after the next open's orders.
  std::size_t kept = 0;
  for (const auto& o : r[0].plan)
    if (o.action >= 0) {
      CHECK(o.rank >= 1 && o.rank <= 10);
      ++kept;
    }
  CHECK(kept == 10);
  for (const auto& x : r) {
    CHECK(x.daily.size() == bt.model.size() && x.noStops.size() == bt.model.size());
    CHECK(x.positions.size() <= 10 && (x.dayTrades || !x.positions.empty()));
    CHECK(x.plan.size() >= 10 || !x.dayTrades);
    CHECK(x.grid.size() == x.gridK.size() * x.gridK.size());
    for (double v : x.daily) CHECK(std::isfinite(v) && v > -1);
  }
  // Changing the last day's bars leaves every earlier day's return unchanged.
  ofm::Market m2 = s.market;
  const std::size_t L = m2.T() - 1;
  for (std::size_t i = 0; i < m2.N(); ++i) m2.close(L, i) *= 1.5, m2.high(L, i) *= 1.5;
  const auto f2 = ofm::runModel(m2);
  const auto bt2 = ofm::backtest(m2, f2, ofm::Costs{}, s.series);
  const auto r2 = ofm::topKBacktests(m2, f2, ofm::Costs{}, bt2, 10);
  for (std::size_t q = 0; q < r.size(); ++q)
    for (std::size_t k = 0; k + 1 < r[q].daily.size(); ++k) CHECK(r[q].daily[k] == r2[q].daily[k]);
}
