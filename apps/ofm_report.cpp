// Runs the model and the backtest and prints the report (results only, no prices).
//
//   ofm_report [--csv stocks.csv] [--series futures.csv] [--engine reference|emulated]
//              [--cost-pct 0.10] [--stamp-pct 0] [--futures-bps 1] [--day-cost-pct 0.05] [--json out.json]
//
// --cost-pct: transaction cost per side in percent of the value traded (commission and half the
// spread; market standard 0.10). --stamp-pct: stamp duty on purchases in percent (0: CFDs, spread
// bets and AIM shares; 0.50 for UK shares bought directly). --buy-bps / --sell-bps set the two sides directly instead.
// --day-cost-pct: the daily book's cost per side (daybook.hpp; CFD-like dealing, slippage included).
//
// --intraday paths.csv (date,ticker,time,high,low,close: 5-minute bars of the regular session)
// gives the daily trades exact fills on those stock-days. --candidates-out file --candidates-days n
// writes each of the last n days' strongest candidates (date,ticker), for fetching their paths.
// --forecast-cache file keeps the model's forecast between runs on the same data (binary).
// Risk limits of the daily trades (daybook.hpp, DayRisk) adapt to the Bollinger bands; for comparison,
// fixed settings: --max-stop-pct (the widest stop below the open), --max-risk-reward (a stop at most
// this many times as far as the take-profit), --max-day-loss-pct (the book's loss if every stop is
// hit), --cap-books (the budget scales the compared books too), --vol-limits (stop limits in the day's
// median volatility); --min-t 2 (no orders unless the backtest's gain has this t-statistic).
//
// Without --csv it runs on the synthetic sample market.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "ofm/intrabook.hpp"
#include "ofm/liquidity.hpp"
#include "ofm/report.hpp"

namespace {
std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
// Intraday paths: each stock-day's session bars in time order; kept only when the session's last
// close lies within the day's bar (the same prices: not before an adjusted split), each bar
// clipped to the day's range.
ofm::day::Paths readPaths(const std::string& text, const ofm::Market& m) {
  std::unordered_map<std::string, std::size_t> di, ti;
  for (std::size_t d = 0; d < m.dates.size(); ++d) di[m.dates[d]] = d;
  for (std::size_t i = 0; i < m.tickers.size(); ++i) ti[m.tickers[i]] = i;
  struct Row { std::string time; double h, l, c, v; };
  std::unordered_map<std::size_t, std::vector<Row>> rows;
  std::istringstream in(text);
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    std::string f[7];
    std::size_t k = 0, p = 0;
    for (; k < 7; ++k) {
      const std::size_t q = line.find(',', p);
      f[k] = line.substr(p, q == std::string::npos ? std::string::npos : q - p);
      if (q == std::string::npos) break;
      p = q + 1;
    }
    if (k < 5) continue;
    const auto d = di.find(f[0]), i = ti.find(f[1]);
    if (d == di.end() || i == ti.end()) continue;
    rows[d->second * m.N() + i->second].push_back({f[2], std::stod(f[3]), std::stod(f[4]), std::stod(f[5]), k >= 6 && !f[6].empty() ? std::stod(f[6]) : 0.0});
  }
  ofm::day::Paths out;
  for (auto& [key, r] : rows) {
    const std::size_t d = key / m.N(), i = key % m.N();
    const double h = m.high(d, i), l = m.low(d, i);
    std::sort(r.begin(), r.end(), [](const Row& a, const Row& b) { return a.time < b.time; });
    if (!(h > 0) || !(l > 0) || r.empty() || r.back().c < l || r.back().c > h) continue;
    auto& path = out[key];
    for (const auto& x : r) path.push_back({std::clamp(x.h, l, h), std::clamp(x.l, l, h), std::clamp(x.c, l, h), x.v});
  }
  return out;
}

// The forecast, saved after a run and reused while the market's dates and tickers are the same.
void put(std::ostream& o, const std::vector<double>& v) {
  const std::uint64_t n = v.size();
  o.write(reinterpret_cast<const char*>(&n), 8), o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(8 * n));
}
bool get(std::istream& in, std::vector<double>& v) {
  std::uint64_t n = 0;
  if (!in.read(reinterpret_cast<char*>(&n), 8) || n > (1ull << 32)) return false;
  v.resize(n);
  return bool(in.read(reinterpret_cast<char*>(v.data()), std::streamsize(8 * n)));
}
std::uint64_t marketKey(const ofm::Market& m) {
  std::size_t h = 0;
  for (const auto& s : m.dates) h = h * 31 + std::hash<std::string>{}(s);
  for (const auto& s : m.tickers) h = h * 31 + std::hash<std::string>{}(s);
  for (double c : m.close.v) h = h * 31 + std::hash<double>{}(c);
  return h;
}
void saveForecast(const std::string& path, const ofm::Market& m, const ofm::Forecast& f) {
  std::ofstream o(path, std::ios::binary);
  const std::uint64_t key = marketKey(m), T = f.E.T, N = f.E.N, recs = f.records;
  for (auto x : {key, T, N, recs}) o.write(reinterpret_cast<const char*>(&x), 8);
  put(o, f.E.v), put(o, f.mean), put(o, f.tstat), put(o, f.premium);
  put(o, std::vector<double>(f.horizons.begin(), f.horizons.end()));
  put(o, std::vector<double>{double(f.familyReturns.size())});
  for (const auto& v : f.familyReturns) put(o, v);
  o << f.engine;
}
bool loadForecast(const std::string& path, const ofm::Market& m, const std::string& engine, ofm::Forecast* f) {
  std::ifstream in(path, std::ios::binary);
  std::uint64_t key, T, N, recs;
  for (auto* x : {&key, &T, &N, &recs})
    if (!in.read(reinterpret_cast<char*>(x), 8)) return false;
  if (key != marketKey(m) || T != m.T() || N != m.N()) return false;
  f->E = ofm::Panel(T, N), f->records = recs;
  std::vector<double> hz, nf;
  if (!get(in, f->E.v) || !get(in, f->mean) || !get(in, f->tstat) || !get(in, f->premium) || !get(in, hz) || !get(in, nf) || nf.size() != 1) return false;
  f->horizons.assign(hz.begin(), hz.end());
  f->familyReturns.resize(std::size_t(nf[0]));
  for (auto& v : f->familyReturns)
    if (!get(in, v)) return false;
  std::getline(in, f->engine);
  return f->engine == engine;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string csv, series, engine = "reference", json, intraday, candOut, fcache, universeOut;
    std::size_t universeDays = 0, intraUniverse = 0;
    bool intraStrategy = false;
    std::size_t candDays = 0;
    ofm::DayRisk risk;
    double costPct = ofm::kDealingPct, stampPct = ofm::kStampPct, buyBps = -1, sellBps = -1, futBps = 1, dayPct = ofm::kDayDealingPct;
    for (int k = 1; k < argc; ++k) {
      const std::string a = argv[k];
      auto next = [&]() -> std::string {
        if (k + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++k];
      };
      if (a == "--csv") csv = next();
      else if (a == "--series") series = next();
      else if (a == "--engine") engine = next();
      else if (a == "--json") json = next();
      else if (a == "--cost-pct") costPct = std::stod(next());
      else if (a == "--stamp-pct") stampPct = std::stod(next());
      else if (a == "--buy-bps") buyBps = std::stod(next());
      else if (a == "--sell-bps") sellBps = std::stod(next());
      else if (a == "--futures-bps") futBps = std::stod(next());
      else if (a == "--day-cost-pct") dayPct = std::stod(next());
      else if (a == "--intraday") intraday = next();
      else if (a == "--candidates-out") candOut = next();
      else if (a == "--candidates-days") candDays = std::stoul(next());
      else if (a == "--forecast-cache") fcache = next();
      else if (a == "--max-stop-pct") risk.maxStop = std::stod(next()) / 100;
      else if (a == "--max-risk-reward") risk.maxRiskReward = std::stod(next());
      else if (a == "--max-day-loss-pct") risk.maxDayLoss = std::stod(next()) / 100, risk.worstDayCap = true;
      else if (a == "--min-t") risk.minT = std::stod(next());
      else if (a == "--cap-books") risk.capSizingOnly = false;
      else if (a == "--worst-day-cap") risk.worstDayCap = true;
      else if (a == "--vol-limits") risk.bandLimits = false;
      else if (a == "--daily-budget") risk.dailyBudget = true;
      else if (a == "--spread-costs") risk.spreadCosts = true;
      else if (a == "--universe") risk.universe = std::stoul(next());
      else if (a == "--universe-out") universeOut = next();
      else if (a == "--universe-days") universeDays = std::stoul(next());
      else if (a == "--intraday-universe") intraUniverse = std::stoul(next());
      else if (a == "--intraday-strategy") intraStrategy = true;
      else throw std::invalid_argument("unknown option " + a);
    }
    ofm::Costs costs = ofm::Costs::fromPercent(costPct, stampPct, futBps);
    if (buyBps >= 0) costs.buyBps = buyBps;
    if (sellBps >= 0) costs.sellBps = sellBps;
    costs.dayBps = 100 * dayPct;
    costs.spreadCosts = risk.spreadCosts, costs.universe = risk.universe;  // the same honest options for the model portfolio
    ofm::Market m;
    std::map<std::string, std::vector<double>> s;
    if (csv.empty()) {
      auto syn = ofm::syntheticMarket();
      m = std::move(syn.market), s = std::move(syn.series);
    } else {
      m = ofm::parseMarketCsv(readFile(csv));
      if (!series.empty()) s = ofm::parseSeriesCsv(readFile(series), m.dates);
    }
    // Each step's time, on stderr.
    auto t0 = std::chrono::steady_clock::now();
    auto lap = [&](const char* what) {
      const auto t1 = std::chrono::steady_clock::now();
      std::fprintf(stderr, "time: %s %.1f s\n", what, std::chrono::duration<double>(t1 - t0).count());
      t0 = t1;
    };
    lap("reading the market");
    ofm::Forecast f;
    if (fcache.empty() || !loadForecast(fcache, m, engine, &f)) {
      f = ofm::runModel(m, engine);
      if (!fcache.empty()) saveForecast(fcache, m, f);
    } else {
      std::cerr << "forecast: from the cache\n";
    }
    // Each of the last n days' universe (the intraday strategy's stocks), for fetching their paths.
    if (!universeOut.empty()) {
      const ofm::Liquidity liq(m, f.horizons.empty() ? 16.0 : double(f.horizons[f.horizons.size() / 2]));
      std::ofstream o(universeOut);
      o << "date,ticker\n";
      for (std::size_t d = m.T() > universeDays ? m.T() - universeDays : 1; d < m.T(); ++d) {
        const auto in = liq.universeAt(d - 1, intraUniverse);
        for (std::size_t i = 0; i < m.N(); ++i)
          if (in[i]) o << m.dates[d] << ',' << m.tickers[i] << '\n';
      }
      if (intraday.empty()) return 0;
    }
    lap("the model");
    ofm::day::Paths paths;
    if (!intraday.empty()) paths = readPaths(readFile(intraday), m);
    lap("reading the intraday paths");
    const ofm::MarketStructure ms = ofm::marketStructure(m, f.horizons, f.familyReturns);
    lap("the market structure");
    const ofm::Backtest bt = ofm::backtest(m, f, costs, s, &ms);
    lap("the model portfolio's backtest");
    const std::size_t candFrom = candOut.empty() ? std::size_t(-1) : m.T() > candDays ? m.T() - candDays : 0;
    const ofm::DayBook day = ofm::dayBook(m, f, bt.start, 1e-4 * costs.dayBps, 1e-4 * costs.dayBps, intraday.empty() ? nullptr : &paths, candFrom, risk);
    lap("the daily trades");
    if (!candOut.empty()) {
      std::ofstream o(candOut);
      o << "date,ticker\n";
      for (const auto& [d, i] : day.candidates) o << m.dates[d] << ',' << m.tickers[i] << '\n';
    }
    std::cout << ofm::reportText(m, f, bt, costs, &ms, &day);
    std::string intraJ;
    if (intraStrategy) {
      const auto ib = ofm::intraBook(m, f, paths, 1e-4 * costs.dayBps, 1e-4 * costs.dayBps, intraUniverse);
      std::cout << ofm::intraText(m, ib, 1e-4 * costs.dayBps);
      intraJ = ofm::intraJson(m, ib);
      lap("the US intraday strategy");
    }
    if (!json.empty()) {
      std::string j = ofm::reportJson(m, f, bt, costs, &ms, &day);
      // The intraday strategy's results join the scan's object.
      if (!intraJ.empty() && intraJ != "{}") j.insert(j.rfind('}'), ",\"intraday\":" + intraJ);
      std::ofstream(json) << j;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
