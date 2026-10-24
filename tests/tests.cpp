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
#include "../src/daykernels.hpp"

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
  const std::vector<double> td(p.tables.begin(), p.tables.end());
  ofm::kernels::zscore<double>(p, 0, td.data(), zr);
  ofm::kernels::gram<double>(p, 0, td.data(), zr, gr);
  ofm::kernels::zscore<float>(p, 0, p.tables.data(), ze);
  ofm::kernels::gram<float>(p, 0, p.tables.data(), ze, ge);
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

TEST(intraday_path_decides_which_level_came_first) {
  using namespace ofm::day;
  const Bar b{100, 110, 90, 100, 1};
  const double sig = 0.05;  // stop at 100 e^-0.1 = 90.5, take-profit at 100 e^0.05 = 105.1
  int e;
  bool res;
  tradeReturn(b, sig, 2, 1, 0, 0, true, &e);
  CHECK(e == 1);  // daily bar only: the stop first
  const std::vector<Tick> up{{101, 99}, {106, 100}, {104, 90}};
  tradeReturn(b, sig, 2, 1, 0, 0, true, &e, &up, &res);
  CHECK(e == 2 && res);  // the take-profit was reached two ticks before the stop
  const std::vector<Tick> down{{101, 90}, {110, 100}};
  tradeReturn(b, sig, 2, 1, 0, 0, false, &e, &down, &res);
  CHECK(e == 1 && res);
  const std::vector<Tick> same{{110, 90}};
  tradeReturn(b, sig, 2, 1, 0, 0, true, &e, &same, &res);
  CHECK(e == 1 && !res);  // the same tick: worst case
}

TEST(wilder_indicators_match_a_plain_reference) {
  // One stock, a random walk with a trend that turns: ATR and RSI against Wilder's recursions
  // written out plainly, and the Parabolic's rules (the next day's stop never inside this day's or
  // the previous day's range; it flips when the price crosses it).
  const std::size_t T = 300, N = 1;
  std::vector<double> bars(5 * T * N, 0.0), o(T), h(T), l(T), c(T);
  std::uint64_t x = 99;
  auto unif = [&]() { x = x * 6364136223846793005ull + 1442695040888963407ull; return double(x >> 11) / double(1ull << 53); };
  double p = 100;
  for (std::size_t t = 0; t < T; ++t) {
    o[t] = p * std::exp(0.004 * (unif() - 0.5));
    p = o[t] * std::exp((t < 150 ? 0.003 : -0.003) + 0.02 * (unif() - 0.5));
    c[t] = p, h[t] = std::max(o[t], c[t]) * (1 + 0.01 * unif()), l[t] = std::min(o[t], c[t]) * (1 - 0.01 * unif());
    const double v[5] = {o[t], h[t], l[t], c[t], 1};
    for (std::size_t f = 0; f < 5; ++f) bars[f * T + t] = v[f];
  }
  bars[2 * T + 40] = 0;  // a missing bar is skipped
  std::vector<double> w;
  ofm::day::kernels::wilder(bars, T, N, w);
  const std::size_t WI = ofm::day::kernels::kWilder;
  CHECK(w[40 * WI] == 0 && w[13 * WI] == 0 && w[14 * WI] > 0);
  double atr = 0, au = 0, ad = 0;
  std::size_t k = 0, prev = 0, flips = 0;
  for (std::size_t t = 0; t < T; ++t) {
    if (t == 40) continue;
    if (k > 0) {
      const double tr = std::max(h[t], c[prev]) - std::min(l[t], c[prev]), r = c[t] - c[prev];
      const double u = std::max(r, 0.0), d = std::max(-r, 0.0);
      if (k <= 14) atr += tr / 14, au += u / 14, ad += d / 14;
      else atr = (13 * atr + tr) / 14, au = (13 * au + u) / 14, ad = (13 * ad + d) / 14;
      if (k >= 14) {
        CHECK(std::fabs(w[t * WI] - atr / c[t]) < 1e-12 && std::fabs(w[t * WI + 1] - (au - ad) / (au + ad)) < 1e-12);
        const double sar = c[t] - w[t * WI + 6] * atr, side = w[t * WI + 5];
        CHECK(side == 1 || side == -1);
        if (side > 0) CHECK(sar <= std::min(l[t], l[prev]) + 1e-9);
        if (side < 0) CHECK(sar >= std::max(h[t], h[prev]) - 1e-9);
        const std::size_t t1 = t + 1 == 40 ? 41 : t + 1;
        if (t1 < T && w[t1 * WI + 5] != side) {
          ++flips;
          CHECK(side > 0 ? l[t1] <= sar : h[t1] >= sar);  // crossed
        }
      }
    }
    prev = t, ++k;
  }
  CHECK(flips > 0);
  // ADX in [0, 1], ADXR the mean of ADX now and 13 days before.
  for (std::size_t t = 60; t < T; ++t) {
    CHECK(w[t * WI + 3] >= 0 && w[t * WI + 3] <= 1);
    if (w[(t - 13) * WI + 3] > 0) CHECK(std::fabs(w[t * WI + 4] - 0.5 * (w[t * WI + 3] + w[(t - 13) * WI + 3])) < 1e-12);
  }
}

TEST(wilder_levels_win_when_volatility_changes) {
  // The sample's intraday moves stretched and shrunk by turns (x2 and x0.5, every 60 days): the
  // stops and take-profits measured in true ranges follow; the fixed quantiles of the last moves lag.
  ofm::Market m = sample().market;
  for (std::size_t t = 0; t < m.T(); ++t)
    for (std::size_t i = 0; i < m.N(); ++i) {
      const double k = (t / 60) % 2 ? 2.0 : 0.5, o = m.open(t, i);
      if (!(o > 0)) continue;
      m.high(t, i) = o * std::pow(m.high(t, i) / o, k), m.low(t, i) = o * std::pow(m.low(t, i) / o, k), m.close(t, i) = o * std::pow(m.close(t, i) / o, k);
    }
  const auto f = ofm::runModel(m);
  const auto bt = ofm::backtest(m, f, ofm::Costs{}, {});
  const auto d = ofm::dayBook(m, f, bt.start, 5e-4, 5e-4);
  std::printf("  level losses: history %.4f, Wilder's true range %.4f, Bollinger %.4f, network %.4f, with Wilder's and Bollinger's %.4f; pair fits %s\n", d.levelLoss[0],
              d.levelLoss[1], d.levelLoss[2], d.levelLoss[3], d.levelLoss[4], d.trendFits ? "with ADXR" : "plain");
  CHECK(d.levelLoss[1] > 0 && d.levelLoss[1] < d.levelLoss[0]);
}

TEST(bollinger_bands_match_a_plain_reference) {
  // One stock with a missing bar: each horizon's mean and standard deviation of the last h closes
  // (bars, not days), the close's place in the bands, their width and the band unit.
  const std::size_t T = 120, N = 1;
  std::vector<double> bars(5 * T * N, 0.0), c(T);
  std::uint64_t x = 5;
  double p = 50;
  for (std::size_t t = 0; t < T; ++t) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    p *= std::exp(0.02 * (double(x >> 11) / double(1ull << 53) - 0.5));
    c[t] = p;
    const double v[5] = {p, p * 1.01, p * 0.99, p, 1};
    for (std::size_t f = 0; f < 5; ++f) bars[f * T + t] = v[f];
  }
  bars[3 * T + 30] = 0;  // no close on day 30
  const std::vector<std::uint32_t> hz{8, 16};
  std::vector<double> b;
  ofm::day::kernels::bollinger(bars, T, N, hz, b);
  for (std::size_t k = 0; k < hz.size(); ++k)
    for (std::size_t t = 0; t < T; ++t) {
      const double* r = &b[t * 6 + 3 * k];
      std::vector<double> w;
      for (std::size_t u = 0; u <= t; ++u)
        if (u != 30) w.push_back(c[u]);
      if (t == 30 || w.size() <= hz[k]) {
        CHECK(r[0] == 0 && r[1] == 0 && r[2] == 0);
        continue;
      }
      double m = 0, v = 0;
      for (std::size_t j = w.size() - hz[k]; j < w.size(); ++j) m += w[j] / hz[k];
      for (std::size_t j = w.size() - hz[k]; j < w.size(); ++j) v += (w[j] - m) * (w[j] - m) / hz[k];
      const double sd = std::sqrt(v);
      CHECK(std::fabs(r[0] - (c[t] - m) / sd) < 1e-6 && std::fabs(r[1] - std::log(sd / c[t])) < 1e-6 && std::fabs(r[2] - sd / c[t]) < 1e-9);
    }
}

TEST(daily_book_uses_only_the_past) {
  ofm::Market m = sample().market;
  const auto f1 = ofm::runModel(m);
  const auto b1 = ofm::backtest(m, f1, ofm::Costs{}, sample().series);
  const auto d1 = ofm::dayBook(m, f1, b1.start, 5e-4, 5e-4);
  CHECK(d1.ret.size() == b1.model.size() && d1.trades > 0 && d1.orders.size() <= 10);
  for (const auto& o : d1.orders) CHECK(o.tradeReturn > 0 && o.weight > 0 && o.weight <= 1 && o.sigma > 0 && o.record > 0 && std::isfinite(o.stop) && std::isfinite(o.take));
  // The risk limits: each stop within its stock's learnt limit, and no more than the day's budget
  // lost if every stop is hit.
  double worst = 0;
  for (const auto& o : d1.orders) {
    const double stop = 1 - std::exp(-o.stop * o.sigma);
    CHECK(stop <= o.limit + 1e-12);
    worst += o.weight * stop;
  }
  CHECK(d1.dayBudget > 0 && worst <= d1.dayBudget + 1e-12);
  // Change the bars from day X on: the book's returns before X are the same.
  const std::size_t X = m.T() - 100;
  for (std::size_t t = X; t < m.T(); ++t)
    for (std::size_t i = 0; i < m.N(); ++i) {
      const double k = 1.0 + 0.5 * std::sin(static_cast<double>(t * 7 + i));
      m.open(t, i) *= k, m.high(t, i) *= k * 1.1, m.low(t, i) *= k, m.close(t, i) *= k, m.volume(t, i) *= 3;
    }
  const auto f2 = ofm::runModel(m);
  const auto d2 = ofm::dayBook(m, f2, b1.start, 5e-4, 5e-4);
  for (std::size_t k = 0; b1.start + k < X; ++k) CHECK(d1.ret[k] == d2.ret[k]);
  // Dearer dealing: the learners adapt (other trades), so only the rules are checked: the same
  // days, and every order still expected to gain after the costs.
  const auto d3 = ofm::dayBook(sample().market, f1, b1.start, 2e-3, 2e-3);
  CHECK(d3.ret.size() == d1.ret.size());
  for (const auto& o : d3.orders) CHECK(o.tradeReturn > 0);
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
