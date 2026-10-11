// Daily top-10 test against the real bars, compared with the model portfolio (research only).
//
//   ofm_daytest --csv stocks.csv [--series futures.csv] [--cost-pct 0.10] [--stamp-pct 0]
//
// Each close t, the 10 stocks with the best expected return (the model's forecast, known at the
// close) that have a valid bar on day t+1 are bought at its open, one tenth of the capital each,
// and sold the same day: at the stop-loss, the take-profit or the close, read from that day's
// high, low and close. Every trade pays the purchase and the sale cost; the book compounds daily.
//
// Stops and take-profits are multiples of the stock's own open-to-close volatility (all its days
// before t). The multiples are learnt walk-forward: each day the pair whose record (Sharpe ratio
// of the 10-trade book over all earlier days, worst-case fills) is best is used; "none" is one of
// the choices. Daily bars cannot tell whether the stop or the take-profit was touched first when
// both were; the main result assumes the stop (worst case), the optimistic one the take-profit.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ofm/data.hpp"
#include "ofm/model.hpp"
#include "ofm/report.hpp"
#include "ofm/structure.hpp"
#include "ofm/trade.hpp"

namespace {

std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

constexpr double kInf = std::numeric_limits<double>::infinity();
const std::vector<double> kGrid = {0.5, 1.0, 1.5, 2.0, 3.0, kInf};  // multiples of the volatility

struct Bar { double o, h, l, c, v; };

// One trade's net return: bought at the open, sold at the stop, the take-profit or the close.
// worstCase: when both levels were touched, the stop is taken to come first.
double tradeReturn(const Bar& b, double sig, double ks, double kt, double buy, double sell, bool worstCase, int* exit) {
  const double stop = std::isfinite(ks) ? b.o * std::exp(-ks * sig) : 0.0;
  const double take = std::isfinite(kt) ? b.o * std::exp(kt * sig) : kInf;
  const bool hitStop = b.l <= stop, hitTake = b.h >= take;
  double px = b.c;
  *exit = 0;
  if (hitStop && (worstCase || !hitTake)) px = stop, *exit = 1;
  else if (hitTake) px = take, *exit = 2;
  return (px / b.o) * (1 - sell) / (1 + buy) - 1;
}

struct Stats {
  double s = 0, s2 = 0, n = 0;
  void add(double x) { s += x, s2 += x * x, n += 1; }
  double sharpe() const {
    if (n < 2) return -kInf;
    const double m = s / n, v = (s2 - n * m * m) / (n - 1);
    return v > 0 ? m / std::sqrt(v) : (m > 0 ? kInf : -kInf);
  }
};

void row(const char* name, const std::vector<double>& r, std::size_t lo, std::size_t hi) {
  const ofm::Metrics m = ofm::metrics(std::vector<double>(r.begin() + lo, r.begin() + hi));
  double w = 1;
  for (std::size_t k = lo; k < hi; ++k) w *= 1 + r[k];
  std::printf("  %-44s %7.1f%% %7.1f%% %7.2f %7.1f%% %9.2fx\n", name, 100 * m.annualReturn, 100 * m.volatility, m.sharpe, 100 * m.maxDrawdown, w);
}

// The mean daily difference a - b, its t-statistic, and the share of days a was ahead.
void compare(const char* name, const std::vector<double>& a, const std::vector<double>& b) {
  Stats d;
  double ahead = 0;
  for (std::size_t k = 0; k < a.size(); ++k) d.add(a[k] - b[k]), ahead += a[k] > b[k];
  const double m = d.s / d.n, v = (d.s2 - d.n * m * m) / (d.n - 1);
  std::printf("  %-44s %+8.3f%% a day, t %+5.2f, ahead on %4.1f%% of days\n", name, 100 * m, m / std::sqrt(v / d.n), 100 * ahead / d.n);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string csv, series;
    double costPct = ofm::kDealingPct, stampPct = ofm::kStampPct;
    for (int k = 1; k < argc; ++k) {
      const std::string a = argv[k];
      auto next = [&]() -> std::string {
        if (k + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++k];
      };
      if (a == "--csv") csv = next();
      else if (a == "--series") series = next();
      else if (a == "--cost-pct") costPct = std::stod(next());
      else if (a == "--stamp-pct") stampPct = std::stod(next());
      else throw std::invalid_argument("unknown option " + a);
    }
    const ofm::Costs costs = ofm::Costs::fromPercent(costPct, stampPct, 1.0);
    const double buy = costs.buyBps * 1e-4, sell = costs.sellBps * 1e-4;
    ofm::Market m;
    std::map<std::string, std::vector<double>> s;
    if (csv.empty()) {
      auto syn = ofm::syntheticMarket();
      m = std::move(syn.market), s = std::move(syn.series);
    } else {
      m = ofm::parseMarketCsv(readFile(csv));
      if (!series.empty()) s = ofm::parseSeriesCsv(readFile(series), m.dates);
    }
    const ofm::Forecast f = ofm::runModel(m, "reference");
    const ofm::MarketStructure ms = ofm::marketStructure(m, f.horizons, f.familyReturns);
    const ofm::Backtest bt = ofm::backtest(m, f, costs, s, &ms);
    const std::size_t T = m.T(), N = m.N(), K = 10, G = kGrid.size();
    if (bt.start == 0 || bt.model.empty()) throw std::runtime_error("no backtest");

    auto barAt = [&](std::size_t d, std::size_t i, Bar* b) {
      *b = {m.open(d, i), m.high(d, i), m.low(d, i), m.close(d, i), m.volume(d, i)};
      return b->o > 0 && b->h > 0 && b->l > 0 && b->c > 0 && b->l <= std::min(b->o, b->c) && b->h >= std::max(b->o, b->c) &&
             std::isfinite(b->v) && b->v > 0;
    };
    // Each stock's open-to-close log-return moments up to the current day.
    std::vector<Stats> vol(N);
    auto addVol = [&](std::size_t t) {
      Bar b;
      for (std::size_t i = 0; i < N; ++i)
        if (barAt(t, i, &b)) vol[i].add(std::log(b.c / b.o));
    };
    for (std::size_t t = 0; t + 1 < bt.start; ++t) addVol(t);

    // Strategies: no stops; learnt pair (worst-case fills); the same pair with optimistic fills.
    std::vector<double> plain, learnt, optimistic, gross, model, market;
    std::vector<Stats> pairRecord(G * G);
    std::size_t days = 0, trades = 0, both = 0, lowIsOpen = 0, stops = 0, takes = 0, full = 0;
    std::map<std::string, int> pairUse;
    for (std::size_t k = 0; k < bt.model.size(); ++k) {
      const std::size_t d = bt.start + k, t = d - 1;
      if (t > 0) addVol(t);  // everything known at the close t
      // The day's 10: best expected return with a valid bar on day d.
      std::vector<std::size_t> order;
      Bar b;
      for (std::size_t i = 0; i < N; ++i)
        if (std::isfinite(f.E(t, i)) && vol[i].n >= 2 && barAt(d, i, &b)) order.push_back(i);
      std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return f.E(t, x) > f.E(t, y); });
      if (order.size() > K) order.resize(K);
      full += order.size() == K;
      // Today's pair: the best record so far (none/none before any record).
      std::size_t best = G * G - 1;
      double bestSharpe = -kInf;
      for (std::size_t p = 0; p < G * G; ++p)
        if (pairRecord[p].sharpe() > bestSharpe + 1e-12) bestSharpe = pairRecord[p].sharpe(), best = p;
      const double ks = kGrid[best / G], kt = kGrid[best % G];
      auto name = [](double x) { return std::isfinite(x) ? std::to_string(x).substr(0, 3) : std::string("none"); };
      ++pairUse[name(ks) + " / " + name(kt)];
      // The trades, against day d's bar.
      double rPlain = 0, rLearnt = 0, rOpt = 0, rGross = 0;
      std::vector<double> rPair(G * G, 0.0);
      for (std::size_t i : order) {
        barAt(d, i, &b);
        const double sig = std::sqrt(std::max(0.0, vol[i].s2 / vol[i].n - (vol[i].s / vol[i].n) * (vol[i].s / vol[i].n)));
        int e;
        rPlain += tradeReturn(b, sig, kInf, kInf, buy, sell, true, &e);
        rGross += b.c / b.o - 1;
        rLearnt += tradeReturn(b, sig, ks, kt, buy, sell, true, &e);
        stops += e == 1, takes += e == 2;
        rOpt += tradeReturn(b, sig, ks, kt, buy, sell, false, &e);
        if (std::isfinite(ks) && std::isfinite(kt) && b.l <= b.o * std::exp(-ks * sig) && b.h >= b.o * std::exp(kt * sig)) ++both;
        lowIsOpen += b.l >= b.o;
        for (std::size_t p = 0; p < G * G; ++p) rPair[p] += tradeReturn(b, sig, kGrid[p / G], kGrid[p % G], buy, sell, true, &e);
        ++trades;
      }
      const double n = static_cast<double>(K);  // empty slots stay in cash
      plain.push_back(rPlain / n), learnt.push_back(rLearnt / n), optimistic.push_back(rOpt / n), gross.push_back(rGross / n);
      for (std::size_t p = 0; p < G * G; ++p) pairRecord[p].add(rPair[p] / n);
      model.push_back(bt.model[k]), market.push_back(bt.market[k]);
      ++days;
    }

    const std::size_t n = days, h = n / 2;
    std::printf("Daily top 10 against the real bars: %zu stocks, %s .. %s; costs %.2f%% on purchases, %.2f%% on sales\n", N,
                m.dates[bt.start].c_str(), m.dates[bt.start + n - 1].c_str(), 100 * buy, 100 * sell);
    std::printf("%zu days, %zu trades (%.2f a day; 10 on %.1f%% of days)\n\n", n, trades, double(trades) / n, 100.0 * full / n);
    const char* head = "  strategy (compounded daily)                    return/yr      vol  Sharpe   max DD    growth\n";
    const std::pair<const char*, std::pair<std::size_t, std::size_t>> parts[] = {{"whole period", {0, n}}, {"first half", {0, h}}, {"second half", {h, n}}};
    for (const auto& [pname, range] : parts) {
      std::printf("%s: %s .. %s\n%s", pname, m.dates[bt.start + range.first].c_str(), m.dates[bt.start + range.second - 1].c_str(), head);
      row("top 10, open to close, before costs", gross, range.first, range.second);
      row("top 10, no stops, after costs", plain, range.first, range.second);
      row("top 10, learnt stop/take, worst-case fills", learnt, range.first, range.second);
      row("top 10, learnt stop/take, optimistic fills", optimistic, range.first, range.second);
      row("model portfolio (multi-day), after costs", model, range.first, range.second);
      row("market (bought and held)", market, range.first, range.second);
      std::printf("\n");
    }
    std::printf("Daily difference against the model portfolio (paired, same days):\n");
    compare("top 10 before costs", gross, model);
    compare("top 10, no stops, after costs", plain, model);
    compare("top 10, learnt stop/take, worst-case", learnt, model);
    std::printf("\nExits (learnt levels, worst case): %zu at the stop, %zu at the take-profit, %zu at the close\n", stops, takes,
                trades - stops - takes);
    std::printf("Trades that touched both levels (order unknown on daily bars): %zu (%.1f%%); the low was the open: %.1f%%\n", both,
                100.0 * both / std::max<std::size_t>(1, trades), 100.0 * lowIsOpen / std::max<std::size_t>(1, trades));
    std::printf("Costs paid by the daily book: %.2f%% a day (every trade pays %.2f%% on a tenth of the capital)\n",
                100 * (buy + sell) * double(trades) / n / K, 100 * (buy + sell));
    std::printf("Learnt pairs used (stop / take, multiples of the volatility; days):");
    for (const auto& [pname, c] : pairUse) std::printf(" %s %d;", pname.c_str(), c);
    std::printf("\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
