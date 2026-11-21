// ofm_strategy: the daily scan's engine for injectable strategies (strategies/*.cpp) on injectable compute
// back ends (backends/*.cpp), with fused kernels (include/strategy/kernels.hpp).
//
//   ofm_strategy --csv stocks.csv --costs UK --strategy momentum [--param name=value ...] [--backend threads]
//                [--capital 5000] [--slots 5] [--stamp-free aim.txt] [--venues tools/venues.txt] [--override ibkr.csv]
//                [--pricing fixed|tiered] [--min-value V] [--min-price P] [--json page.json] [--orders orders.json]
//                [--trace orders.trace]
//   ofm_strategy --list                      the strategies and back ends built in
//   ofm_strategy --synthetic out.csv [--seed n]   a generated market for the checks
//
// What it does, the same for every strategy:
//   universe  each day's tradable shares: a real bar, a price of at least --min-price, a median traded value
//             (20 bars) of at least --min-value (markets with volumes), a year of history and a volatility
//   book      `slots` positions (--slots, or X_NPICK), each 1/slots of the capital. Every `rebalance` sessions
//             the strategy scores the day's shares and the highest `slots` scores are the target. A position no
//             longer in the target is sold in the next opening auction; a target share not held is bought in
//             the next opening auction, at most `capsd` daily volatilities (at most `capmax`) over the close,
//             else tried again the next day. A position is held at most `maxdays` sessions.
//   backtest  walk-forward, day by day, on the days known then only, and BEFORE ANY COST (no commission, no tax,
//             no spread): the strategy is learnt and judged on what the market did, not on the broker.
//   expected  each order's expected return before costs: the mean result of the strategy's own past trades
//             at the same place on the list (shrunk to the mean of all of them, as if 50 such trades had made it, and that to 0).
//   orders    the next session's: the target in order of expected return, highest first. Then, at the very end,
//             the broker's costs (IBKR: commission both ways, the US regulators' fees, stamp duty or the exchange's
//             tax) come off each order's expected return; an order its costs make a loss is dropped, no other
//             takes its place, and the order of the list does not change.
// Outputs: --json the page's report (no prices: the data licence), --orders the bridge's book (with prices; the
// private store only), X_TRADELOG every order of the backtest and its .days the daily results (the budget's
// split), --trace each day's orders (the look-ahead check: a run on fewer days must give the same lines).
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef __EMSCRIPTEN__
#include <unistd.h>
#endif

#include "strategy/backend.hpp"
#include "strategy/kernels.hpp"
#include "strategy/montecarlo.hpp"
#include "strategy/neural.hpp"
#include "strategy/resample.hpp"
#include "strategy/strategy.hpp"

using strat::kNaN;

namespace {

struct Options {
  std::string csv, market = "UK", strategy, backend, stampFree, venues, overrideCsv, jsonOut, ordersOut, traceOut, synthetic;
  std::map<std::string, double> params;
  double capital = 5000, minValue = -1, minPrice = -1, advShare = 0.01, minOrder = 500;
  long top = -1;
  std::size_t slots = 5, roll = 5;
  bool tiered = false, list = false;
  unsigned seed = 7;
  // --resample SEED: the run on a synthetic market drawn from the real one (strategy/resample.hpp) instead of it;
  // --resample-runs N: also N such markets (seeds 1..N, each a run of its own), their results' spread in the report;
  // --resample-block B: the drawn blocks' length in days (default 21).
  unsigned resample = 0, resampleRuns = 0, resampleBlock = 21;
};

[[noreturn]] void usage(const char* why) {
  std::fprintf(stderr, "ofm_strategy: %s\nusage: ofm_strategy --csv stocks.csv --costs UK --strategy NAME [--param k=v] [--backend NAME] [--json f] [--orders f]\n", why);
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  if (const char* k = std::getenv("X_NPICK")) o.slots = std::size_t(std::max(1, std::atoi(k)));
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    const auto next = [&]() -> std::string {
      if (k + 1 >= argc) usage(("no value after " + a).c_str());
      return argv[++k];
    };
    if (a == "--csv") o.csv = next();
    else if (a == "--costs") o.market = next();
    else if (a == "--strategy") o.strategy = next();
    else if (a == "--backend") o.backend = next();
    else if (a == "--param") {
      const std::string v = next();
      const auto e = v.find('=');
      if (e == std::string::npos) usage("--param takes name=value");
      o.params[v.substr(0, e)] = std::atof(v.c_str() + e + 1);
    } else if (a == "--capital") o.capital = std::atof(next().c_str());
    else if (a == "--slots") o.slots = std::size_t(std::max(1, std::atoi(next().c_str())));
    else if (a == "--stamp-free") o.stampFree = next();
    else if (a == "--venues") o.venues = next();
    else if (a == "--override") o.overrideCsv = next();
    else if (a == "--pricing") o.tiered = next() == "tiered";
    else if (a == "--min-value") o.minValue = std::atof(next().c_str());
    else if (a == "--min-price") o.minPrice = std::atof(next().c_str());
    else if (a == "--adv-share") o.advShare = std::atof(next().c_str());
    else if (a == "--roll") o.roll = std::size_t(std::atoi(next().c_str()));
    else if (a == "--json") o.jsonOut = next();
    else if (a == "--orders") o.ordersOut = next();
    else if (a == "--trace") o.traceOut = next();
    else if (a == "--synthetic") o.synthetic = next();
    else if (a == "--seed") o.seed = unsigned(std::atoi(next().c_str()));
    else if (a == "--resample") o.resample = unsigned(std::atoi(next().c_str()));
    else if (a == "--resample-runs") o.resampleRuns = unsigned(std::atoi(next().c_str()));
    else if (a == "--resample-block") o.resampleBlock = unsigned(std::max(1, std::atoi(next().c_str())));
    else if (a == "--list") o.list = true;
    // the scan's arguments for the earlier models, not used here
    else if (a == "--top") o.top = std::atol(next().c_str());
    else if (a == "--lag" || a == "--exit" || a == "--segments" || a == "--hold" || a == "--bars") next();
    else if (a == "--scan") {
    } else usage(("unknown argument " + a).c_str());
  }
  return o;
}

// ---------------------------------------------------------------------------------------------------------
// The data: date,ticker,open,high,low,close,volume (a header naming the columns, any order).

struct Bar { std::uint32_t d, i; double o, h, l, c, v; };

struct Raw {
  std::vector<std::string> days, tickers;
  std::vector<Bar> bars;
};

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) usage(("cannot read " + path).c_str());
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

// Every row of a CSV of bars, in the file's order; days and tickers in the order first seen (sorted after). b: the
// back end the rows are parsed on (the text cut at line ends into chunks, each parsed with its own names, then
// merged in order: the same bars as one thread's); null: one thread.
Raw readBars(const std::string& path, const strat::Backend* b = nullptr) {
  std::string text = slurp(path);
  text.push_back('\n');
  int col[7] = {0, 1, 2, 3, 4, 5, 6};
  std::size_t body = 0;
  {  // the header: which column is which
    const std::size_t e = text.find('\n');
    std::string h = text.substr(0, e);
    if (!h.empty() && h.back() == '\r') h.pop_back();
    std::vector<std::string> names;
    for (std::size_t a = 0;;) {
      const std::size_t c = h.find(',', a);
      names.push_back(h.substr(a, c == std::string::npos ? std::string::npos : c - a));
      if (c == std::string::npos) break;
      a = c + 1;
    }
    const char* want[7] = {"date", "ticker", "open", "high", "low", "close", "volume"};
    for (int k = 0; k < 7; ++k)
      for (std::size_t j = 0; j < names.size(); ++j)
        if (names[j] == want[k]) col[k] = int(j);
    body = e + 1;
  }
  // The chunks: about 8 a thread, each ending at a line's end.
  const std::size_t parts = b ? 64 : 1, len = text.size() - body;
  std::vector<std::size_t> cut{body};
  for (std::size_t q = 1; q < parts; ++q) {
    std::size_t at = body + len * q / parts;
    at = text.find('\n', std::max(at, cut.back()));
    if (at == std::string::npos || at + 1 >= text.size()) break;
    if (at + 1 > cut.back()) cut.push_back(at + 1);
  }
  cut.push_back(text.size());
  struct Part {
    std::vector<std::string> days, tickers;
    std::vector<Bar> bars;
  };
  std::vector<Part> part(cut.size() - 1);
  const auto num = [](const char* a, const char* e) {  // the same value strtod gives (both rounded exactly); 0 if none
    while (a < e && (*a == ' ' || *a == '+')) ++a;
#if defined(__EMSCRIPTEN__)  // the Lab's toolchain: strtod on the field (from_chars for doubles may be missing)
    char buf[64];
    const std::size_t n = std::min<std::size_t>(sizeof buf - 1, std::size_t(e - a));
    std::memcpy(buf, a, n), buf[n] = '\0';
    return std::strtod(buf, nullptr);
#else
    double x = 0;
    const auto r = std::from_chars(a, e, x);
    return r.ec == std::errc() ? x : 0.0;
#endif
  };
  const auto parse = [&](std::size_t x, std::size_t y) {
    for (std::size_t q = x; q < y; ++q) {
      Part& out = part[q];
      std::unordered_map<std::string, std::uint32_t> dIx, tIx;
      const char* p = text.data() + cut[q];
      const char* const stop = text.data() + cut[q + 1];
      const char* f[16];
      const char* fe[16];
      while (p < stop) {
        const char* e = static_cast<const char*>(std::memchr(p, '\n', std::size_t(stop - p)));
        if (!e) e = stop;
        const char* le = e > p && e[-1] == '\r' ? e - 1 : e;
        std::size_t nf = 0;
        f[0] = p;
        for (const char* c = p; c < le && nf < 15; ++c)
          if (*c == ',') fe[nf] = c, f[++nf] = c + 1;
        fe[nf++] = le;
        p = e + 1;
        if (nf < 7 || std::max({col[0], col[1], col[2], col[3], col[4], col[5], col[6]}) >= int(nf)) continue;
        const std::string day(f[col[0]], std::min<std::size_t>(10, std::size_t(fe[col[0]] - f[col[0]])));
        const std::string tk(f[col[1]], std::size_t(fe[col[1]] - f[col[1]]));
        auto di = dIx.emplace(day, std::uint32_t(out.days.size()));
        if (di.second) out.days.push_back(day);
        auto ti = tIx.emplace(tk, std::uint32_t(out.tickers.size()));
        if (ti.second) out.tickers.push_back(tk);
        Bar bar{di.first->second, ti.first->second, num(f[col[2]], fe[col[2]]), num(f[col[3]], fe[col[3]]), num(f[col[4]], fe[col[4]]),
                num(f[col[5]], fe[col[5]]), num(f[col[6]], fe[col[6]])};
        if (bar.o > 0 && bar.h > 0 && bar.l > 0 && bar.c > 0 && bar.h >= bar.l) out.bars.push_back(bar);
      }
    }
  };
  if (b) b->parallelFor(part.size(), parse);
  else parse(0, part.size());
  // Merged in the file's order: each chunk's names mapped to the whole file's.
  Raw r;
  std::unordered_map<std::string, std::uint32_t> dIx, tIx;
  std::size_t total = 0;
  for (const auto& pt : part) total += pt.bars.size();
  r.bars.reserve(total);
  for (const auto& pt : part) {
    std::vector<std::uint32_t> dm(pt.days.size()), tm(pt.tickers.size());
    for (std::size_t k = 0; k < pt.days.size(); ++k) {
      auto it = dIx.emplace(pt.days[k], std::uint32_t(r.days.size()));
      if (it.second) r.days.push_back(pt.days[k]);
      dm[k] = it.first->second;
    }
    for (std::size_t k = 0; k < pt.tickers.size(); ++k) {
      auto it = tIx.emplace(pt.tickers[k], std::uint32_t(r.tickers.size()));
      if (it.second) r.tickers.push_back(pt.tickers[k]);
      tm[k] = it.first->second;
    }
    for (Bar bar : pt.bars) bar.d = dm[bar.d], bar.i = tm[bar.i], r.bars.push_back(bar);
  }
  return r;
}

void toPanel(const Raw& r, strat::Data& d) {
  std::vector<std::uint32_t> dOrd(r.days.size()), tOrd(r.tickers.size());
  for (std::uint32_t k = 0; k < dOrd.size(); ++k) dOrd[k] = k;
  for (std::uint32_t k = 0; k < tOrd.size(); ++k) tOrd[k] = k;
  std::sort(dOrd.begin(), dOrd.end(), [&](auto a, auto b) { return r.days[a] < r.days[b]; });
  std::sort(tOrd.begin(), tOrd.end(), [&](auto a, auto b) { return r.tickers[a] < r.tickers[b]; });
  std::vector<std::uint32_t> dAt(dOrd.size()), tAt(tOrd.size());
  for (std::uint32_t k = 0; k < dOrd.size(); ++k) dAt[dOrd[k]] = k, d.days.push_back(r.days[dOrd[k]]);
  for (std::uint32_t k = 0; k < tOrd.size(); ++k) tAt[tOrd[k]] = k, d.tickers.push_back(r.tickers[tOrd[k]]);
  d.T = d.days.size(), d.N = d.tickers.size();
  const std::size_t n = d.T * d.N;
  d.O.assign(n, kNaN), d.H.assign(n, kNaN), d.L.assign(n, kNaN), d.C.assign(n, kNaN), d.V.assign(n, kNaN);
  for (const auto& b : r.bars) {
    const std::size_t k = std::size_t(dAt[b.d]) * d.N + tAt[b.i];
    d.O[k] = b.o, d.H[k] = b.h, d.L[k] = b.l, d.C[k] = b.c, d.V[k] = b.v;
  }
}

// IBKR's own bars (their open the opening auction's price) in place of the store's where it has them, each row
// in the store's price unit (a London share can be quoted in pence or pounds: the nearest of 1, 1/100, 100).
std::string overrideBars(strat::Data& d, const std::string& path) {
  const Raw r = readBars(path);
  std::unordered_map<std::string, std::size_t> dIx, tIx;
  for (std::size_t t = 0; t < d.T; ++t) dIx[d.days[t]] = t;
  for (std::size_t i = 0; i < d.N; ++i) tIx[d.tickers[i]] = i;
  std::size_t n = 0, scaled = 0;
  for (const auto& b : r.bars) {
    const auto x = dIx.find(r.days[b.d]), y = tIx.find(r.tickers[b.i]);
    if (x == dIx.end() || y == tIx.end()) continue;
    const std::size_t k = x->second * d.N + y->second;
    double s = 1.0;
    if (d.C[k] > 0) {
      const double q = d.C[k] / b.c;
      for (double c : {1.0, 0.01, 100.0})
        if (std::fabs(std::log(q / c)) < std::fabs(std::log(q / s))) s = c;
      if (std::fabs(std::log(q / s)) > 0.2) continue;  // not the same share's price
    } else {
      continue;
    }
    scaled += s != 1.0;
    d.O[k] = b.o * s, d.H[k] = b.h * s, d.L[k] = b.l * s, d.C[k] = b.c * s, ++n;
  }
  return std::to_string(n) + " share-days from IBKR in place of the store's (" + std::to_string(scaled) + " put in the store's price unit)";
}

// A stretch of a share's bars quoted in another unit (pence and pounds: a jump of about 100 times in a day,
// which no share makes) put in the unit of its latest bar.
std::size_t repairUnits(strat::Data& d) {
  std::size_t fixed = 0;
  for (std::size_t i = 0; i < d.N; ++i) {
    double scale = 1.0, next = kNaN;
    for (std::size_t t = d.T; t-- > 0;) {
      const std::size_t k = t * d.N + i;
      if (!(d.C[k] > 0)) continue;
      if (std::isfinite(next)) {
        const double q = d.C[k] * scale / next;
        if (q > 50 && q < 200) scale /= 100;
        else if (q < 0.02 && q > 0.005) scale *= 100;
      }
      if (scale != 1.0) d.O[k] *= scale, d.H[k] *= scale, d.L[k] *= scale, d.C[k] *= scale, ++fixed;
      next = d.C[k];
    }
  }
  return fixed;
}

// Bad prices rolled forward: a bar whose low or high is impossible for its open and close (the low under half of
// the lower, the high over twice the higher, or the low above the high) is dropped: the share has no bar that day,
// and its last real close carries over it (the strategies' signals, the book's marks; no trade on it). Judged on the
// day's own bar only: nothing later decides it (no look-ahead).
std::size_t rollOutliers(strat::Data& d) {
  std::size_t n = 0;
  const auto drop = [&](std::size_t k) { d.O[k] = d.H[k] = d.L[k] = d.C[k] = d.V[k] = kNaN, ++n; };
  for (std::size_t i = 0; i < d.N; ++i) {
    for (std::size_t t = 0; t < d.T; ++t) {
      const std::size_t k = t * d.N + i;
      if (!(d.C[k] > 0)) continue;
      const double o = d.O[k], h = d.H[k], l = d.L[k], c = d.C[k], lo = std::min(o, c), hi = std::max(o, c);
      if (!(l > 0 && h > 0 && o > 0) || l > h || l < 0.5 * lo || h > 2.0 * hi) drop(k);
    }
  }
  return n;
}

// A generated market for the checks: random walks with persistent drifts, volumes, and a few gaps.
void writeSynthetic(const std::string& path, unsigned seed) {
  std::mt19937_64 g(seed);
  std::normal_distribution<double> z(0, 1);
  std::uniform_real_distribution<double> u(0, 1);
  std::ofstream f(path);
  f << "date,ticker,open,high,low,close,volume\n";
  const int N = 80, T = 1400;  // long enough for the longest warm-up (ltrev: 3 years)
  std::vector<double> c(N, 100), mu(N, 0);
  for (int t = 0; t < T; ++t) {
    const int y = 2019 + t / 252, m = 1 + (t % 252) / 21, dd = 1 + (t % 21);
    char day[16];
    std::snprintf(day, sizeof day, "%04d-%02d-%02d", y, m, dd);
    const double mk = 0.008 * z(g);
    for (int i = 0; i < N; ++i) {
      if (t % 63 == 0) mu[i] = 0.0008 * z(g);
      const double o = c[i] * std::exp(0.004 * z(g)), cl = o * std::exp(mu[i] + mk + 0.015 * z(g));
      if (u(g) < 0.02 && t > 0) {  // a missing bar
        c[i] = cl;
        continue;
      }
      char b[160];
      std::snprintf(b, sizeof b, "%s,S%02d,%.4f,%.4f,%.4f,%.4f,%d\n", day, i, o, std::max(o, cl) * 1.005, std::min(o, cl) * 0.995, cl, 20000 + int(80000 * u(g)));
      f << b;
      c[i] = cl;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// The broker's costs (IBKR), applied only to the final orders.

struct Costs {
  std::string market;
  bool tiered = false;
  std::set<std::string> stampFree;
  std::map<std::string, std::vector<double>> venue;  // amount, rate, minimum, fee, tax on buying, tax both sides, tax on buying held

  const std::vector<double>* venueOf(const std::string& t) const {
    const auto k = t.find(':');
    if (k == std::string::npos) return nullptr;
    const auto v = venue.find(t.substr(0, k));
    return v == venue.end() ? nullptr : &v->second;
  }
  double commission(double value, const std::string& t, double price) const {
    if (!(value > 0)) return 0;
    if (market == "FOREX") return std::max(2.0, 0.00002 * value);
    if (market == "US") {
      const double sh = price > 0 ? std::max(1.0, std::floor(value / price)) : 1.0;
      return tiered ? std::max(0.35, 0.0035 * sh) + 0.0005 * sh : std::min(0.01 * value, std::max(1.0, 0.005 * sh));
    }
    if (market == "UK") return tiered ? std::max(1.0, 0.0005 * value) + std::max(0.10, 0.000045 * value) + 0.06 : std::max(3.0, 0.0005 * value);
    const auto* v = venueOf(t);
    return v ? std::max((*v)[2], (*v)[1] * value) + ((*v)[3] + (*v)[5]) * value : std::max(3.0, 0.0005 * value);
  }
  double buyTax(const std::string& t) const {
    if (market == "UK") return stampFree.count(t) ? 0.0 : 0.005;
    if (market == "US" || market == "FOREX") return 0.0;
    const auto* v = venueOf(t);
    return v ? (*v)[6] : 0.0;
  }
  double saleFees(double value, double price) const {
    if (market != "US" || !(value > 0)) return 0;
    return 0.0000278 * value + std::min(8.30, 0.000166 * (price > 0 ? std::floor(value / price) : 0.0));
  }
  // The round trip as a share of the order's value; price in the currency (not pence).
  double roundTrip(double value, const std::string& t, double price) const {
    return value > 0 ? (2 * commission(value, t, price) + saleFees(value, price)) / value + buyTax(t) : 0.0;
  }
};

// ---------------------------------------------------------------------------------------------------------

struct Row {  // an order of the backtest
  std::size_t t = 0, i = 0, rank = 0;  // decided after the close of t; its place on the list
  double limit = 0, expected = 0, entry = kNaN, exit = kNaN, ret = 0;
  double turbulence = kNaN, ahead = kNaN, marketVol = kNaN, shareVol = kNaN;  // mcregime: the market's state when it was decided (regimeState)
  bool filled = false, closed = false, stopped = false;
  std::size_t since = 0, days = 0;
};

// A position: its row, the price it is marked at, and its stop: a fixed level, or the Parabolic SAR (Wilder 1978)
// from the day it was bought: the level, the highest high since (ep) and the step (af).
// step, mx: its SAR's step and largest step (psaradapt: chosen for the share by the simulation at the reset)
// vstop (the volatility stop): sar the catastrophic level, ep the close's level, af the distance, hic the highest
// close, exitNext a close under ep (sold at the next open).
// band: psarvol, the SAR's volatility band (a fraction of the price, set at the buy).
struct Pos { std::size_t row; double mark, sar = 0, ep = 0, af = 0, step = 0.02, mx = 0.2, hic = 0, band = 0; bool exitNext = false; };
// A share whose position was stopped: its SAR on the way down (with psar: the highest high since as the level,
// the lowest low as ep), and whether it has turned up again (a close above it).
struct Stopped { double level = 0, sar = 0, ep = 0, af = 0, step = 0.02, mx = 0.2, trigger = 0; std::size_t at = 0; bool turned = false; };

struct Learnt {  // the strategy's past trades by their place on the list, known on the day
  std::vector<double> sum, n;
  double all = 0, nAll = 0, days = 0, nDays = 0;
  explicit Learnt(std::size_t k) : sum(k, 0.0), n(k, 0.0) {}
  void add(const Row& r) {
    if (r.rank < sum.size()) sum[r.rank] += r.ret, n[r.rank] += 1;
    all += r.ret, nAll += 1, days += double(r.days), nDays += 1;
  }
  double expected(std::size_t rank) const {
    const double mu = all / (nAll + 50.0);
    return rank < sum.size() ? (sum[rank] + 50.0 * mu) / (n[rank] + 50.0) : mu;
  }
  double holdDays(double fallback) const { return nDays > 0 ? days / nDays : fallback; }
};

std::string jnum(double x, const char* f = "%.6g") {
  if (!std::isfinite(x)) return "null";
  char b[64];
  std::snprintf(b, sizeof b, f, x);
  return b;
}

std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += c;
    else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
    else o += c;
  }
  return o + "\"";
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse(argc, argv);
  if (o.list) {
    std::printf("strategies:\n");
    for (const auto& [name, f] : strat::registry()) std::printf("  %-10s %s\n", name.c_str(), f()->about().c_str());
    std::printf("back ends:\n");
    for (const auto& [name, f] : strat::backends()) std::printf("  %-10s %s\n", name.c_str(), f()->about().c_str());
    return 0;
  }
  if (!o.synthetic.empty()) {
    writeSynthetic(o.synthetic, o.seed);
    return 0;
  }
  if (o.csv.empty()) usage("no --csv");
  // --resample-runs: the same run on N synthetic markets first (each its own process, its record from its JSON)
  struct Resampled { double annual = kNaN, sharpe = kNaN, dd = kNaN, annualNet = kNaN, sharpeNet = kNaN; };
  std::vector<Resampled> resampled;
#ifndef __EMSCRIPTEN__
  if (o.resampleRuns > 0 && o.resample == 0) {
    const auto quote = [](const std::string& x) {
      std::string q = "'";
      for (char c : x) q += c == '\'' ? std::string("'\\''") : std::string(1, c);
      return q + "'";
    };
    const auto number = [](const std::string& js, const std::string& key, std::size_t from = 0) {
      const auto at = js.find("\"" + key + "\":", from);
      return at == std::string::npos ? kNaN : std::atof(js.c_str() + at + key.size() + 3);
    };
    std::string self = argv[0];  // the engine's own path (resolved: /proc/self/exe would name the shell's child)
    {
      char buf[4096];
      const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
      if (n > 0) self.assign(buf, std::size_t(n));
    }
    for (unsigned sd = 1; sd <= o.resampleRuns; ++sd) {
      const std::string tmp = (std::getenv("TMPDIR") ? std::string(std::getenv("TMPDIR")) : std::string("/tmp")) + "/ofm-resample-" + std::to_string(getpid()) + "-" + std::to_string(sd) + ".json";
      std::string cmd = "env -u X_TRADELOG -u X_PROFILE " + quote(self);
      for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--json" || arg == "--orders" || arg == "--trace" || arg == "--resample-runs") { ++a; continue; }
        cmd += " " + quote(arg);
      }
      cmd += " --resample " + std::to_string(sd) + " --json " + quote(tmp) + " > /dev/null 2>&1";
      Resampled r;
      if (std::system(cmd.c_str()) == 0) {
        const std::string js = slurp(tmp);
        const auto bt = js.find("\"backtest\":");
        r.annual = number(js, "annualReturn", bt), r.sharpe = number(js, "sharpe", bt), r.dd = number(js, "maxDrawdown", bt);
        const auto net = js.find("\"afterRebuyCosts\":", bt);
        if (net != std::string::npos) r.annualNet = number(js, "annualReturn", net), r.sharpeNet = number(js, "sharpe", net);
        std::remove(tmp.c_str());
      }
      resampled.push_back(r);
    }
  }
#endif
  const auto sf = strat::registry().find(o.strategy);
  if (sf == strat::registry().end()) usage(("no strategy '" + o.strategy + "' (see --list)").c_str());
  const auto strategy = sf->second();
#ifdef __EMSCRIPTEN__
  const std::string backendName = o.backend.empty() ? "serial" : o.backend;
#else
  const std::string backendName = o.backend.empty() ? "threads" : o.backend;
#endif
  const auto bf = strat::backends().find(backendName);
  if (bf == strat::backends().end()) usage(("no back end '" + backendName + "' (see --list)").c_str());
  const auto backend = bf->second();
  strat::Params P = strategy->defaults();
  // The engine's own, for every strategy: a stop-loss (stop: a position sold once the price falls this far under
  // the price paid; 0: none) and the buy-back after one (rebuy: bought again, while the strategy still wants it,
  // once the close is this far above the level it was stopped at; negative: never).
  // psar: 1 makes the stop the Parabolic SAR (from the lowest low of the 5 sessions before the buy, rising each day
  // by psarstep of the gap to the highest high, the step growing by psarstep at each new high to at most psarmax);
  // its buy-back: once the share's SAR on the way down turns up again (a close above it).
  P.emplace("stop", 0.0), P.emplace("rebuy", -1.0), P.emplace("psar", 0.0), P.emplace("psarstep", 0.02), P.emplace("psarmax", 0.2);
  // vstop: the volatility stop (a protective stop that leaves a rising share alone): vstop times the share's daily
  // volatility (60 sessions, at the buy) under the price paid, then under the highest close since (a chandelier,
  // never lowered); a close under it sells at the next open, and the price touching 1.5 times that distance under the
  // price sells at once (the catastrophic stop at the broker, good till cancelled); bought back, while the strategy
  // still wants it, once a close is rebuy over the level it left at. vstopadapt: 1 learns each share's vstop at every
  // reset (2, 3, 4 or 6), the one with the most expected log growth on its futures in the nested Monte Carlo.
  // refill: 1 gives a slot freed by a stop (or an order not filled) to the next share on the list (the simulation's
  // next choices, as many as the slots), so the money stays invested; the reserves are bought only into free slots.
  P.emplace("vstop", 0.0), P.emplace("vstopadapt", 0.0), P.emplace("refill", 0.0);
  // psarvol: the volatility-adjusted SAR (with psar): it starts psarvol times the share's daily volatility (60
  // sessions, at the buy) under the price paid, instead of the lows before it, and rises as the SAR does but never
  // closer than that under the close: a calm share's stop close, a volatile one's further, and a rising share not sold
  // because the SAR sped up. psarvoladapt: 1 learns each share's multiple at every reset (2, 3, 4 or 6), the one with
  // the most expected log growth on its futures in the nested Monte Carlo (the stop and the buy-back included).
  P.emplace("psarvol", 0.0), P.emplace("psarvoladapt", 0.0);
  // psaradapt: 1 makes the SAR's step and largest step each share's own, learnt at every reset: the nested Monte Carlo
  // (with the network's shapes, mcnn) runs the share's futures under each pair of a grid (steps 0.005-0.03, largest
  // 0.1 or 0.2) with the stop and its buy-back, and the pair with the most expected log growth is the share's until
  // the next reset (a position keeps the pair it was bought with).
  P.emplace("psaradapt", 0.0);
  // rebuystop: the buy-back as an order at the broker attached to the stop (psar or a fixed stop): once the stop sells at
  // L, a buy stop at L * (1 + rebuystop) works while the strategy still wants the share; bought back the day the price
  // comes back through it (the stop's own day only when the close is at or over it), at that level or the open through
  // a gap, its own stop at once. It replaces rebuy's buy-back on the SAR's turn. 0: none. rebuystopadapt: 1 learns each
  // share's distance at every reset (0.5%, 1%, 2%, 3% or 5%), the one with the most expected log growth on its futures
  // in the nested Monte Carlo (its stop's own setting included).
  // stopmedian: 1 buys only shares whose stop would start no wider than the median of the day's candidates (the
  // book's list and its reserves), in the backtest and in the orders alike: the threshold is the day's own, nothing
  // fixed; a far stop is left out, not just ranked last.
  P.emplace("stopmedian", 0.0);
  P.emplace("rebuystop", 0.0), P.emplace("rebuystopadapt", 0.0), P.emplace("stopjoint", 0.0), P.emplace("mcprob", 0.0);
  // mc: 1 models with the nested Monte Carlo (include/strategy/montecarlo.hpp): resets on the real market at each
  // rebalance, mcpaths futures of mchorizon sessions (0: the rebalance's) from blocks of mcblock real days of the
  // mcwindow sessions before; each order's expected return, chance of a loss and of a stop from those futures.
  P.emplace("mc", 0.0), P.emplace("mcpaths", 500.0), P.emplace("mchorizon", 0.0), P.emplace("mcblock", 5.0), P.emplace("mcwindow", 750.0), P.emplace("mcseed", 1.0), P.emplace("mcdrift", 1.0);
  // mcselect: 1 lets the nested Monte Carlo choose the book at every reset (the backtest's and the live one): the
  // strategy's mccand best (0: mccandk a slot) simulated on the same futures, the k with the most expected
  // log growth together kept (include/strategy/montecarlo.hpp: mcChoose). mcskip: the window ends that many
  // sessions before the reset (the futures drawn from [t0 - mcwindow, t0 - mcskip]).
  P.emplace("mcselect", 0.0), P.emplace("mccand", 0.0), P.emplace("mccandk", 4.0), P.emplace("mcskip", 0.0);  // mccandk: candidates a slot when mccand is 0
  // mcnn: 1 overlays the differential neural network (include/strategy/neural.hpp) on the nested Monte Carlo: at
  // every reset each share's futures shaped (a drift and a volatility scale on its drawn real days) by what the
  // network learnt, walk-forward, from the resets whose outcome was known by then.
  P.emplace("mcnn", 0.0);
  // mcobj: the simulation's choice by 0 the expected log growth of the book together, 1 each share's expected return.
  P.emplace("mcobj", 0.0);
  // mcpool: 1 injects every strategy (strategies/*.cpp) into the network's Monte Carlo: each one's score of every
  // share is an input to the network, and each one's best mccandk a slot join the candidates the simulation chooses
  // from. mcgate: 1 makes it autonomous: at each reset the book held is the arm (the simulation's choice, or one
  // strategy's own top picks) with the best real record over the last mcgaten resets known by then (shrunk by its
  // t statistic); the simulation's until enough are known.
  P.emplace("mcpool", 0.0), P.emplace("mcgate", 0.0), P.emplace("mcgaten", 12.0);
  // mcnnstep: the network learns every that many sessions (0: at the resets only): each day's realised bars become
  // known after its close, and with 1 the network takes them in the same evening (new samples every session, a pass
  // over the newest), not up to a month later at the next reset.
  P.emplace("mcnnstep", 0.0), P.emplace("mcnnsample", 0.0);
  // mcregime: K > 0 reverse-engineers the market's K hidden states at every reset (include/strategy/regime.hpp): a
  // hidden Markov model of the market's daily move refitted to the window up to the reset (warm from the last), the
  // futures drawn in states (each block from the past days in its state, the states moving by the fitted chances),
  // and every volatility stop sized by the share's volatility in the states expected over the horizon. 0: none.
  // mcregime -1: the number of states learnt at every reset, 1 to 4, the fit with the lowest BIC (1: one state, the
  // plain Monte Carlo's draws). mcregimestop: 1 sizes the volatility stops by the states expected; 0 leaves them on the
  // last 60 sessions (the states shape the futures only).
  P.emplace("mcregime", 0.0), P.emplace("mcregimestop", 1.0);
  // mccost: 1 charges every buy-back after a stop its round trip at the broker (the share's own costs at the reset:
  // commissions, taxes, fees for the order's amount) in the nested Monte Carlo, so the stop's band and the buy-back's
  // distance are learnt net of the whipsaws they cause. 0: the futures without costs.
  P.emplace("mccost", 0.0), P.emplace("adaptfine", 0.0);
  // invert: 1 holds the opposite picks (the lowest scores): the same list, the same book, the strategy's choice
  // reversed. The scan runs it beside the live one: the gap between the two is the strategy's own edge; what the
  // opposite picks also make is the share list's (shares that survived to today, chosen with hindsight).
  P.emplace("invert", 0.0);
  for (const auto& [k, v] : o.params) {
    if (!P.count(k)) usage(("the strategy has no parameter " + k).c_str());
    P[k] = v;
  }

  const std::string& M = o.market;
  const bool uk = M == "UK", noVolume = M == "FOREX" || M == "INDEX";
  const double unit = uk ? 100.0 : 1.0;  // London's prices in pence
  const double minPrice = o.minPrice >= 0 ? o.minPrice : uk ? 20 : (M == "US" || M == "EU") ? 1 : M == "APAC" ? 0.1 : 0;
  const double minValue = o.minValue >= 0 ? o.minValue : noVolume ? 0 : M == "US" ? 1e6 : 250000;
  // Each day's most traded only, as of that day (--top; 0 for all): UK 300, US 1000, EU 500, APAC 400.
  const std::size_t top = o.top >= 0 ? std::size_t(o.top) : M == "UK" ? 300 : M == "US" ? 1000 : M == "EU" ? 500 : M == "APAC" ? 400 : 0;

  strat::Data d;
  const auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  const double ld0 = now();
  Raw raw = readBars(o.csv, backend.get());
  const double ld1 = now();
  toPanel(raw, d);
  raw = Raw();
  const double ld2 = now();
  std::string note;
  if (!o.overrideCsv.empty()) note = overrideBars(d, o.overrideCsv);
  const std::size_t units = repairUnits(d);
  const std::size_t outliers = rollOutliers(d);
  strat::ResampleInfo rsInfo;
  if (o.resample > 0) rsInfo = strat::resampleMarket(d, o.resample, o.resampleBlock);
  if (std::getenv("X_PROFILE")) std::fprintf(stderr, "profile: reading %.2fs, the panel %.2fs, repairs %.2fs\n", ld1 - ld0, ld2 - ld1, now() - ld2);
  const std::size_t T = d.T, N = d.N, k = o.slots;
  const std::size_t reb = std::max<std::size_t>(1, std::size_t(P["rebalance"])), warm = std::size_t(P["warmup"]), maxDays = std::size_t(P["maxdays"]);
  if (T < warm + 30) usage("too few days for the strategy's warm-up");

  strat::PrepareSpec ps;
  ps.roll = o.roll, ps.unit = unit;
  strat::UniverseSpec us;
  us.minPrice = minPrice, us.minValue = minValue, us.warmup = warm, us.needValue = !noVolume, us.top = noVolume ? 0 : top;
  std::vector<char> in;
  strat::prepare(d, ps, us, in, *backend);

  Costs costs;
  costs.market = M, costs.tiered = o.tiered;
  if (!o.stampFree.empty()) {
    std::ifstream f(o.stampFree);
    for (std::string l; std::getline(f, l);) {
      const auto c = l.find_first_of(", \t\r");
      if (!l.empty() && l[0] != '#') costs.stampFree.insert(l.substr(0, c));
    }
  }
  if (!o.venues.empty()) {
    std::ifstream f(o.venues);
    for (std::string l; std::getline(f, l);) {
      if (l.empty() || l[0] == '#') continue;
      std::istringstream s(l);
      std::string ex;
      s >> ex;
      std::vector<double> v;
      for (double x; s >> x;) v.push_back(x);
      if (v.size() < 6) continue;
      if (v.size() < 7) v.push_back(v[4]);
      costs.venue[ex] = v;
    }
  }

  // ------------------------------------------------------------------ the walk-forward backtest, before costs
  const std::size_t start = warm;
  const double capSd = P["capsd"], capMax = P["capmax"], stopLoss = P["stop"], rebuy = P["rebuy"];
  const bool psar = P["psar"] > 0, vstop = !psar && P["vstop"] > 0, stops = psar || stopLoss > 0 || vstop;
  const bool psarVol = psar && P["psarvol"] > 0;
  const bool refill = stops && P["refill"] > 0;
  std::map<std::size_t, double> vPick;  // vstopadapt: each share's multiple
  const auto vOf = [&](std::size_t i) {
    const auto it = vPick.find(i);
    return it != vPick.end() ? it->second : psarVol ? P["psarvol"] : P["vstop"];
  };
  strat::RegimeFits fits;  // mcregime: the market's states fitted at each reset
  std::size_t regK = 0;  // mcregime: the states (SIZE_MAX: learnt by BIC)
  bool regStop = false;
  std::map<std::size_t, std::size_t> regKChosen;  // mcregime -1: the states chosen, and how often
  std::map<std::size_t, strat::RegimeFit> regLast;  // each number of states' last fit (the next cycle's warm start)
  // A share's volatility for its stop at day t: in the states expected (the last fit by t) with mcregime, else its
  // last 60 sessions'.
  const auto sigmaAt = [&](std::size_t t, std::size_t i) {
    if (regK && regStop && !fits.empty()) {
      auto it = fits.upper_bound(t);
      if (it != fits.begin() && std::isfinite((--it)->second.sigma[i])) return it->second.sigma[i];
    }
    return d.sigma[t * N + i];
  };
  // mcregime: the market's state at day t (and share i's volatility in it) from the last fit by t; none without.
  std::size_t stateHorizon = 0;
  const auto stateAt = [&](std::size_t t, std::size_t i) {
    if (!regK || fits.empty()) return strat::RegimeState{};
    auto it = fits.upper_bound(t);
    if (it == fits.begin()) return strat::RegimeState{};
    return strat::regimeState(d, (--it)->second, t, i, stateHorizon);
  };
  const auto stamp = [&](Row& r) {
    if (!regK) return;
    const auto st = stateAt(r.t, r.i);
    r.turbulence = st.turbulence, r.ahead = st.ahead, r.marketVol = st.marketVol, r.shareVol = st.shareVol;
  };
  const auto vDist = [&](std::size_t t, std::size_t i) {  // the stop's distance under the price (a fraction)
    const double sg = sigmaAt(t, i);
    return std::clamp(vOf(i) * (std::isfinite(sg) ? sg : 0.02), 0.02, 0.5);
  };
  const double afStep = P["psarstep"], afMax = P["psarmax"];
  std::map<std::size_t, std::pair<double, double>> sarPick;  // psaradapt: each share's SAR step and largest step
  const auto sarOf = [&](std::size_t i) {
    const auto it = sarPick.find(i);
    return it != sarPick.end() ? it->second : std::make_pair(afStep, afMax);
  };
  const bool rbStop = stops && !vstop && P["rebuystop"] > 0;
  std::map<std::size_t, double> rbPick;  // rebuystopadapt: each share's buy-back distance over its stop
  const auto rbOf = [&](std::size_t i) {
    const auto it = rbPick.find(i);
    return it != rbPick.end() ? it->second : P["rebuystop"];
  };
  std::map<std::size_t, Stopped> stoppedAt;  // share -> its stop (until bought back or dropped by the strategy)
  std::size_t nStops = 0, nRebuys = 0;
  const auto lowOf = [&](std::size_t a, std::size_t b, std::size_t i) {  // the lowest low of days a..b (real bars)
    double m = kNaN;
    for (std::size_t q = a; q <= b; ++q)
      if (d.L[q * N + i] > 0) m = std::isfinite(m) ? std::min(m, d.L[q * N + i]) : d.L[q * N + i];
    return m;
  };
  // A new position's stop, bought at `entry` on day u: a fixed level, or the SAR from the 5 sessions' lowest low before.
  const auto openStop = [&](Pos& p, std::size_t u, std::size_t i, double entry) {
    if (vstop) {
      p.af = vDist(u - 1, i), p.ep = entry * (1 - p.af), p.sar = entry * (1 - 1.5 * p.af), p.hic = entry, p.exitNext = false;
      return;
    }
    if (psar) {
      const double lo = lowOf(u >= 5 ? u - 5 : 0, u - 1, i);
      const auto sp = sarOf(i);
      p.step = sp.first, p.mx = sp.second;
      p.sar = std::isfinite(lo) && lo < entry ? lo : entry * 0.95, p.ep = entry, p.af = p.step;
      if (psarVol) p.band = vDist(u - 1, i), p.sar = entry * (1 - p.band);  // the volatility-adjusted SAR: its band under the price paid
    } else {
      p.sar = stopLoss > 0 ? entry * (1 - stopLoss) : 0.0;
    }
  };
  // Day u's bar against the stop: hit if the low reached it (sold there, or at the open through a gap); else the
  // SAR moves up for the next day (never above the last two lows).
  const auto stopDay = [&](Pos& p, std::size_t u, std::size_t i, double& px) {
    if (!stops) return false;
    const double lo = d.L[u * N + i], hi = d.H[u * N + i], op = d.O[u * N + i];
    if (lo > 0 && p.sar > 0 && lo <= p.sar) {
      px = op > 0 ? std::min(op, p.sar) : p.sar;
      return true;
    }
    if (vstop) {  // the close against its level: under it, sold at the next open; else the levels rise with the highest close
      const double C = d.C[u * N + i];
      if (C > 0) {
        if (C < p.ep) {
          p.exitNext = true;
        } else {
          p.hic = std::max(p.hic, C), p.ep = std::max(p.ep, p.hic * (1 - p.af)), p.sar = std::max(p.sar, p.ep * (1 - 0.5 * p.af));
        }
      }
      return false;
    }
    if (psar && hi > 0) {
      if (hi > p.ep) p.ep = hi, p.af = std::min(p.mx, p.af + p.step);
      double next = p.sar + p.af * (p.ep - p.sar);
      const double l2 = lowOf(u >= 1 ? u - 1 : 0, u, i);
      if (std::isfinite(l2)) next = std::min(next, l2);
      if (p.band > 0 && d.C[u * N + i] > 0) next = std::min(next, d.C[u * N + i] * (1 - p.band));  // never closer than its band
      p.sar = std::max(p.sar, next);
    }
    return false;
  };
  // A stopped share's state: with psar, its SAR on the way down from the highest high it reached while held.
  const auto stoppedFrom = [&](const Pos& p, double level, std::size_t i, std::size_t u) {
    Stopped x;
    x.at = u, x.trigger = rbStop ? level * (1 + rbOf(i)) : 0.0;  // rebuystop: its buy-back order's level
    x.level = vstop ? p.ep : level;  // vstop: the buy-back measured from the close's level
    if (psar) x.sar = std::max(p.ep, level), x.ep = level, x.af = p.step, x.step = p.step, x.mx = p.mx;
    return x;
  };
  // A stopped share through day t (at its close): with psar, turned once the close is above its falling SAR; else
  // the SAR falls toward the lowest low (never under the last two highs).
  const auto stoppedDay = [&](Stopped& x, std::size_t t, std::size_t i) {
    if (!psar || x.turned) return;
    const double c = d.C[t * N + i], lo = d.L[t * N + i];
    if (!(c > 0)) return;
    if (c > x.sar) {
      x.turned = true;
      return;
    }
    if (lo > 0 && lo < x.ep) x.ep = lo, x.af = std::min(x.mx, x.af + x.step);
    double next = x.sar + x.af * (x.ep - x.sar);
    double h2 = 0;
    for (std::size_t q = t >= 1 ? t - 1 : 0; q <= t; ++q) h2 = std::max(h2, d.H[q * N + i] > 0 ? d.H[q * N + i] : 0.0);
    x.sar = std::min(x.sar, std::max(next, h2));
  };
  // May a stopped share be bought again after the close of t: never (rebuy < 0); with psar once its SAR has turned
  // up; with a fixed stop once the close is rebuy above the level it was stopped at.
  const auto mayBuy = [&](std::size_t t, std::size_t i) {
    const auto s_ = stoppedAt.find(i);
    if (s_ == stoppedAt.end()) return true;
    if (rebuy < 0 || rbStop) return false;  // rebuystop: bought back by its own order only
    return psar ? s_->second.turned : d.Cf[t * N + i] >= s_->second.level * (1 + rebuy);
  };
  const auto capOf = [&](std::size_t t, std::size_t i) {
    const double s = d.sigma[t * N + i];
    return std::min(capMax, capSd * (std::isfinite(s) ? s : 0.02));
  };
  const bool mcOn = P["mc"] > 0 || P["mcselect"] > 0, mcSelect = P["mcselect"] > 0;
  strat::McSpec mc;
  mc.paths = std::max<std::size_t>(20, std::size_t(P["mcpaths"])), mc.horizon = P["mchorizon"] > 0 ? std::size_t(P["mchorizon"]) : reb;
  mc.block = std::max<std::size_t>(1, std::size_t(P["mcblock"])), mc.window = std::max<std::size_t>(60, std::size_t(P["mcwindow"]));
  mc.seed = std::uint64_t(P["mcseed"]), mc.drift = P["mcdrift"] > 0 ? 1 : 0, mc.psar = psar, mc.stopLoss = stopLoss, mc.rebuy = rebuy, mc.afStep = afStep, mc.afMax = afMax;
  mc.vstop = vstop ? P["vstop"] : 0.0;
  mc.psarVol = psarVol ? P["psarvol"] : 0.0;
  mc.rebuyStop = rbStop ? P["rebuystop"] : 0.0;
  mc.skip = std::size_t(std::max(0.0, P["mcskip"]));
  // A share's round trip at the broker for its order at day t, as a fraction of the order (the final orders' costs).
  const auto tripOf = [&](std::size_t t, std::size_t i) {
    const double v = d.value[t * N + i], c = d.Cf[t * N + i];
    const double amount = noVolume || !(std::isfinite(v) && v > 0) ? o.capital : std::min(o.capital, o.advShare * v);
    return c > 0 ? costs.roundTrip(amount, d.tickers[i], c / unit) : 0.0;
  };
  std::vector<double> tripV;  // mccost: each share's round trip at the last reset (the Monte Carlo's)
  if (mcOn && P["mccost"] > 0) tripV.assign(N, 0.0), mc.trip = tripV.data();
  regK = !mcOn || P["mcregime"] == 0 ? 0 : P["mcregime"] < 0 ? SIZE_MAX : std::size_t(P["mcregime"]);
  regStop = regK && P["mcregimestop"] > 0;
  stateHorizon = mc.horizon;
  if (regK) mc.regime = &fits;
  const bool sarAdapt = psar && mcOn && P["psaradapt"] > 0;
  // stopjoint: 1 learns each share's stop band (psarvol: 2, 3, 4 or 6) and its buy-back order's distance together, as one
  // coupled setting (instead of psarvoladapt and rebuystopadapt each alone); mcprob: 1 picks the setting best on the
  // most simulated paths (the most probable best) instead of the most expected log growth.
  const bool joint = rbStop && psarVol && mcOn && P["stopjoint"] > 0, probable = P["mcprob"] > 0;
  const bool rbAdapt = rbStop && mcOn && (P["rebuystopadapt"] > 0 || joint);
  const std::vector<double> rbGrid{0.005, 0.01, 0.02, 0.03, 0.05};
  std::map<double, std::size_t> rbChosen;
  const bool vAdapt = mcOn && ((vstop && P["vstopadapt"] > 0) || (psarVol && P["psarvoladapt"] > 0));
  const std::vector<double> vGrid{2, 3, 4, 6};
  std::map<double, std::size_t> vChosen;
  // adaptfine: 1 learns the stop's multiple and the buy-back's distance as any value (fractional), not from the short
  // lists: each share's futures on a fine grid evenly spaced in log (the multiple 1 to 10 times the daily volatility,
  // the distance 0.25% to 10%), and the peak between the best point and its neighbours (fineArgmax).
  const bool fine = P["adaptfine"] > 0;
  const auto logGrid = [](double lo, double hi, std::size_t n) {
    std::vector<double> g(n);
    for (std::size_t q = 0; q < n; ++q) g[q] = lo * std::pow(hi / lo, double(q) / double(n - 1));
    return g;
  };
  const std::vector<double> vFine = logGrid(1.0, 10.0, 13), rbFine = logGrid(0.0025, 0.10, 13);
  std::vector<double> vPicks, rbPicks;  // adaptfine: every value learnt (the report's spread)
  const std::vector<std::pair<double, double>> sarGrid{{0.005, 0.1}, {0.01, 0.1}, {0.02, 0.1}, {0.03, 0.1}, {0.005, 0.2}, {0.01, 0.2}, {0.02, 0.2}, {0.03, 0.2}};
  std::map<std::pair<double, double>, std::size_t> sarChosen;
  const std::size_t nCand = P["mccand"] > 0 ? std::size_t(P["mccand"]) : std::max<std::size_t>(k + 1, std::size_t(std::max(1.0, P["mccandk"]) * double(k)));
  std::size_t mcChosen = 0, mcFromTop = 0;  // the simulation's picks, and how many of them the strategy's own top k held too
  const bool nnOn = P["mcnn"] > 0;
  strat::DiffNet net;
  std::vector<strat::NnSample> nnSamples;
  std::size_t nnFirstTrained = 0;  // the first reset the network had learnt at (its held-out check starts there)
  std::vector<double> nnF;
  std::vector<char> nnOk;
  std::size_t nnAt = SIZE_MAX;
  std::vector<std::vector<double>> resetMu, resetSc;  // each reset's shapes (for its check on the real future)
  // The injected strategies (mcpool): every registered one, each with its own defaults.
  const bool pool = mcSelect && P["mcpool"] > 0, gate = mcSelect && P["mcgate"] > 0;
  std::vector<std::string> poolNames;
  std::vector<std::unique_ptr<strat::Strategy>> poolS;
  std::vector<strat::Params> poolP;
  if (pool)
    for (const auto& [name, make] : strat::registry()) {
      if (poolS.size() >= strat::kNS) break;
      poolNames.push_back(name), poolS.push_back(make()), poolP.push_back(poolS.back()->defaults());
      poolP.back()["invert"] = P["invert"];
    }
  // The gate's arms: 0 the simulation's choice, 1.. each pooled strategy's own top k (or the live one's alone).
  struct Arm {
    std::vector<std::pair<std::size_t, std::vector<std::size_t>>> lists;  // each reset's day and the book it would hold
    std::size_t used = 0;
  };
  std::vector<Arm> arms(1 + (pool ? poolS.size() : 1));
  std::vector<std::string> armNames{"the simulation"};
  if (pool) for (const auto& n : poolNames) armNames.push_back(n);
  else armNames.push_back(o.strategy);
  std::vector<Row> rows;
  std::map<std::size_t, Pos> book;
  std::vector<std::size_t> target, reserve;  // reserve (refill): the next shares for the slots stops free
  std::vector<std::size_t> simReserve;
  std::vector<double> score, daily, dailyNet;  // dailyNet: after the buy-backs' round trips
  std::vector<std::string> dailyDays;
  std::map<std::size_t, std::size_t> lastHeld;  // share -> the last day it was in the book
  Learnt learnt(k);
  FILE* trace = o.traceOut.empty() ? nullptr : std::fopen(o.traceOut.c_str(), "w");
  std::size_t of = 0;  // the day's tradable shares with a score
  std::size_t decided = start;  // the last rebalance: the target's day
  std::vector<std::pair<std::size_t, std::vector<std::size_t>>> resets;  // each rebalance: its day and its list (the Monte Carlo's resets)

  // Every pooled strategy's ranked list at day t: (score, share) best first (the live one's alone without mcpool).
  const auto ranked = [&](const strat::Strategy& st, const strat::Params& pp, std::size_t t, std::vector<std::pair<double, std::size_t>>& c) {
    const strat::View v(d, t, &in[t * N], *backend);
    std::vector<double> sc_;
    st.score(v, pp, sc_);
    if (&st == strategy.get()) score = sc_;  // the live strategy's scores (the report's ranks and reasons)
    c.clear();
    for (std::size_t i = 0; i < N; ++i)
      if (in[t * N + i] && std::isfinite(sc_[i])) c.emplace_back(P["invert"] > 0 ? -sc_[i] : sc_[i], i);
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
  };
  std::vector<std::vector<std::pair<double, std::size_t>>> poolRanked;
  // X_PROFILE: each stage's wall time, to stderr (the work's own clock; never in the results)
  struct Prof {
    double feat = 0, rank = 0, train = 0, sample = 0, select = 0, adapt = 0, check = 0, regime = 0;
  } prof;
  const auto clock = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
  // The network's inputs at day t: the share's own and the market's, and with mcpool each strategy's rank of it.
  const auto features = [&](std::size_t t) {
    if (nnAt == t) return;
    const double c0 = clock();
    strat::nnFeatures(d, t, &in[t * N], nnF, nnOk, backend.get()), nnAt = t;
    const double c1 = clock();
    prof.feat += c1 - c0;
    if (!pool) return;
    poolRanked.assign(poolS.size(), {});
    for (std::size_t q = 0; q < poolS.size(); ++q) {
      ranked(*poolS[q], poolP[q], t, poolRanked[q]);
      const auto& c = poolRanked[q];
      for (std::size_t r = 0; r < c.size(); ++r) nnF[c[r].second * strat::kNF + 9 + q] = c.size() > 1 ? 0.5 - double(r) / double(c.size() - 1) : 0.0;
    }
    prof.rank += clock() - c1;
  };
  // The network's shapes for a list of shares at day t (its features then): mu and scale each (0 and 1 without it).
  const auto shapesFor = [&](std::size_t t, const std::vector<std::size_t>& list, std::vector<double>& mu, std::vector<double>& sc) {
    mu.assign(list.size(), 0.0), sc.assign(list.size(), 1.0);
    if (!nnOn || !net.trained) return;
    features(t);
    for (std::size_t j = 0; j < list.size(); ++j)
      if (nnOk[list[j]]) {
        const strat::NnShape x = net.shape(&nnF[list[j] * strat::kNF]);
        mu[j] = x.mu, sc[j] = x.scale;
      }
  };
  // The gate's record of an arm at t: its books' real returns over the horizon, the last mcgaten known by t's close,
  // shrunk by their t statistic (nothing known: NaN).
  const std::size_t gateN = std::max<std::size_t>(3, std::size_t(P["mcgaten"]));
  const auto armRecord = [&](const Arm& a, std::size_t t) {
    std::vector<double> x;
    for (auto it = a.lists.rbegin(); it != a.lists.rend() && x.size() < gateN; ++it) {
      if (it->first + mc.horizon > t || it->second.empty()) continue;  // its future not over by t's close
      std::vector<double> caps;
      for (std::size_t i : it->second) caps.push_back(capOf(it->first, i));
      x.push_back(strat::mcReal(d, it->first, it->second, caps, k, mc));
    }
    if (x.size() < 3) return kNaN;
    double m = 0, v2 = 0;
    for (double y : x) m += y;
    m /= double(x.size());
    for (double y : x) v2 += (y - m) * (y - m);
    const double sdv = std::sqrt(v2 / double(x.size())), tt = sdv > 0 ? m / sdv * std::sqrt(double(x.size())) : 0;
    return tt != 0 ? m * std::max(0.0, 1 - 1 / (tt * tt)) : 0.0;
  };
  // The network learns at t (once a day at most): from every sample whose outcome is known by t's close, then the
  // day's own samples kept (their outcome known only later).
  std::size_t learntAt = SIZE_MAX;
  const auto learn = [&](std::size_t t, int epochs) {
    if (!nnOn || learntAt == t) return;
    learntAt = t;
    const double c0 = clock();
    net.train(nnSamples, t, mc.horizon, 40000, epochs, backend.get());
    prof.train += clock() - c0;
    if (net.trained && !nnFirstTrained) nnFirstTrained = t;
    // New samples only at the resets (P["mcnnsample"] 1: every learning day): a day's samples overlap the last
    // day's almost wholly, and taken daily they crowd the window with the last months (UK: CRPS +2.8%, 17 Nov).
    if (P["mcnnsample"] <= 0 && (t - start) % reb != 0) return;
    features(t);
    const double c1 = clock();
    {  // each share's sample in parallel, then kept in the shares' order
      std::vector<strat::NnSample> got(N);
      std::vector<char> has(N, 0);
      backend->parallelFor(N, [&](std::size_t a, std::size_t e) {
        for (std::size_t i = a; i < e; ++i) has[i] = nnOk[i] && strat::nnSample(d, t, i, &nnF[i * strat::kNF], mc, got[i]);
      });
      for (std::size_t i = 0; i < N; ++i)
        if (has[i]) nnSamples.push_back(got[i]);
    }
    prof.sample += clock() - c1;
  };
  const std::size_t nnStep = std::size_t(std::max(0.0, P["mcnnstep"]));
  const auto decide = [&](std::size_t t) {
    if ((t - start) % reb != 0 && t != start) return;
    std::vector<std::pair<double, std::size_t>> c;
    ranked(*strategy, P, t, c);
    of = c.size(), decided = t;
    target.clear();
    if (!tripV.empty())
      for (std::size_t i = 0; i < N; ++i) tripV[i] = tripOf(t, i);
    if (regK) {  // this cycle's reverse engineering: the market's states refitted up to t (warm from the last cycle)
      const double cr = clock();
      std::size_t best = 0;
      const std::size_t k0 = regK == SIZE_MAX ? 1 : regK, k1 = regK == SIZE_MAX ? 4 : regK;
      for (std::size_t kk = k0; kk <= k1; ++kk) {  // each number of states (one with mcregime K), warm from its own last fit
        const auto w = regLast.find(kk);
        strat::RegimeFit f;
        strat::fitRegime(d, t, mc.window, mc.skip, mc.horizon, mc.block, kk, w != regLast.end() ? &w->second : nullptr, f);
        regLast[kk] = std::move(f);
        if (!best || regLast[kk].bic < regLast[best].bic) best = kk;
      }
      ++regKChosen[best];
      strat::RegimeFit f = regLast[best];
      strat::regimeSigma(d, *backend, f, regStop);  // the stops' volatility with mcregimestop; each share's variance in each state always
      f.gamma.clear();
      fits[t] = std::move(f);
      prof.regime += clock() - cr;
    }
    learn(t, 2);
    if (pool) features(t);
    std::vector<double> mu, sc;
    if (mcSelect && !c.empty()) {  // the simulation chooses: the candidates on the same futures
      std::vector<std::size_t> cand;
      std::vector<double> caps;
      const auto add = [&](std::size_t i) {
        if (std::find(cand.begin(), cand.end(), i) == cand.end()) cand.push_back(i), caps.push_back(capOf(t, i));
      };
      if (pool) {  // each strategy's best mccandk a slot, in turn (the first of each, then the second, ...)
        const std::size_t each = std::max<std::size_t>(1, std::size_t(std::max(1.0, P["mccandk"]) * double(k)));
        for (std::size_t r = 0; r < each; ++r)
          for (const auto& pr : poolRanked)
            if (r < pr.size()) add(pr[r].second);
      } else {
        for (std::size_t r = 0; r < std::min(nCand, c.size()); ++r) add(c[r].second);
      }
      strat::McReset rr;
      shapesFor(t, cand, mu, sc);
      const double cs = clock();
      strat::mcRun(d, t, cand, caps, std::vector<strat::McHeld>(cand.size()), k, mc, t, *backend, rr, mu, sc);
      prof.select += clock() - cs;
      std::vector<std::size_t> simBook;
      simReserve.clear();
      {
        const auto ch = strat::mcChoose(rr, cand.size(), mc.paths, refill ? 2 * k : k, P["mcobj"] > 0);  // greedy: the first k are the book
        for (std::size_t q = 0; q < ch.size(); ++q) (q < k ? simBook : simReserve).push_back(cand[ch[q]]);
      }
      for (std::size_t i : simBook) {
        const auto at = std::find_if(c.begin(), c.end(), [&](const auto& x) { return x.second == i; });
        mcFromTop += at != c.end() && std::size_t(at - c.begin()) < k;
      }
      mcChosen += simBook.size();
      // The arms' books at t: the simulation's, and each strategy's own top k.
      arms[0].lists.emplace_back(t, simBook);
      for (std::size_t q = 1; q < arms.size(); ++q) {
        const auto& pr = pool ? poolRanked[q - 1] : c;
        std::vector<std::size_t> b;
        for (std::size_t r = 0; r < std::min(k, pr.size()); ++r) b.push_back(pr[r].second);
        arms[q].lists.emplace_back(t, b);
      }
      std::size_t pick = 0;
      if (gate) {  // autonomous: the arm with the best real record known by now (the simulation's until one is known)
        double best = armRecord(arms[0], t);
        for (std::size_t q = 1; q < arms.size(); ++q) {
          const double r = armRecord(arms[q], t);
          if (std::isfinite(r) && (!std::isfinite(best) || r > best + 1e-12)) best = r, pick = q;
        }
      }
      ++arms[pick].used;
      target = arms[pick].lists.back().second;
    } else {
      for (std::size_t r = 0; r < std::min(k, c.size()); ++r) target.push_back(c[r].second);
      simReserve.clear();
      for (std::size_t r = k; r < std::min(2 * k, c.size()); ++r) simReserve.push_back(c[r].second);
    }
    reserve.clear();
    if (refill)
      for (std::size_t i : simReserve)
        if (std::find(target.begin(), target.end(), i) == target.end()) reserve.push_back(i);
    resets.emplace_back(t, target);
    shapesFor(t, target, mu, sc);
    const double ca = clock();
    // A share's best stop setting of `specs`: every setting on the same futures in one pass a path (its days drawn and
    // its bars scaled once), in parallel over the paths; the setting with the most expected log growth (the first of
    // equals), from each setting's sum over the paths in their order.
    // probable: the setting that is best on the most paths instead (the most probable best; a path where several are best
    // shares its vote among them, so a path no setting changes favours none), the most expected log growth among equals.
    const auto bestSetting = [&](std::size_t i, double adj, double cap, double scale, const std::vector<strat::McSpec>& specs, bool byVotes = false,
                                 std::vector<double>* meansOut = nullptr) {
      const std::size_t G = specs.size(), Pn = mc.paths;
      std::vector<double> g(G * Pn, 0.0);
      backend->parallelFor(Pn, [&](std::size_t a, std::size_t e) {
        std::vector<std::size_t> days;
        std::vector<double> pre;
        for (std::size_t q = a; q < e; ++q) {
          strat::mcDays(t, d.T, mc, strat::mcMix(mc.seed ^ strat::mcMix(t * 1000003ULL + q)), false, days);
          strat::mcRatios(d, i, days, scale, pre);
          for (std::size_t s2 = 0; s2 < G; ++s2)
            g[s2 * Pn + q] = std::log(std::max(1e-6, 1 + strat::mcSlot(d, t, i, days, cap, specs[s2], strat::McHeld{}, adj, scale, pre.data()).ret));
        }
      });
      std::vector<double> votes(G, 0.0), mean(G, 0.0);
      for (std::size_t s2 = 0; s2 < G; ++s2)
        for (std::size_t q = 0; q < Pn; ++q) mean[s2] += g[s2 * Pn + q];
      if (byVotes)
        for (std::size_t q = 0; q < Pn; ++q) {
          double hiQ = -1e300;
          for (std::size_t s2 = 0; s2 < G; ++s2) hiQ = std::max(hiQ, g[s2 * Pn + q]);
          std::size_t ties = 0;
          for (std::size_t s2 = 0; s2 < G; ++s2) ties += g[s2 * Pn + q] >= hiQ - 1e-12;
          for (std::size_t s2 = 0; s2 < G; ++s2)
            if (g[s2 * Pn + q] >= hiQ - 1e-12) votes[s2] += 1.0 / double(ties);
        }
      std::size_t pick = 0;
      for (std::size_t s2 = 1; s2 < G; ++s2)
        if (votes[s2] > votes[pick] + 1e-9 || (std::fabs(votes[s2] - votes[pick]) <= 1e-9 && mean[s2] > mean[pick] + 1e-12)) pick = s2;
      if (meansOut) *meansOut = mean;
      return pick;
    };
    // adaptfine: the value with the most expected log growth between the fine grid's points: a parabola through the best
    // point and its two neighbours (in log of the value), its peak (inside them); the best point itself at an end.
    // The curve is smoothed first (each point with its neighbours, 1-2-1), so a single noisy setting is not taken for a peak.
    const auto fineArgmax = [](const std::vector<double>& grid, const std::vector<double>& curve) {
      std::vector<double> mean(curve.size());
      for (std::size_t q = 0; q < curve.size(); ++q) {
        const double l = curve[q > 0 ? q - 1 : q], r = curve[q + 1 < curve.size() ? q + 1 : q];
        mean[q] = (l + 2 * curve[q] + r) / 4;
      }
      const std::size_t b = std::size_t(std::max_element(mean.begin(), mean.end()) - mean.begin());
      if (b == 0 || b + 1 >= grid.size()) return grid[b];
      const double x0 = std::log(grid[b - 1]), x1 = std::log(grid[b]), x2 = std::log(grid[b + 1]);
      const double y0 = mean[b - 1], y1 = mean[b], y2 = mean[b + 1];
      const double den = (x0 - x1) * (x0 - x2) * (x1 - x2);
      const double A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den, B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den;
      if (!(A < 0)) return grid[b];
      return std::exp(std::clamp(-B / (2 * A), x0, x2));
    };
    if (sarAdapt) {  // each target share's SAR from its simulated futures: the pair with the most expected log growth
      std::vector<strat::McSpec> specs;
      for (const auto& [st, mx] : sarGrid) {
        strat::McSpec sv = mc;
        sv.afStep = st, sv.afMax = mx;
        specs.push_back(sv);
      }
      for (std::size_t j = 0; j < target.size(); ++j) {
        const std::size_t i = target[j];
        const double adj = strat::mcAdjust(d, t, i, mc) * std::exp(mu[j]), cap = capOf(t, i);
        const auto pick = sarGrid[bestSetting(i, adj, cap, sc[j], specs)];
        sarPick[i] = pick;
        ++sarChosen[pick];
      }
    }
    if (vAdapt && !joint) {  // each share's volatility stop from its simulated futures: the multiple with the most expected log growth
      std::vector<std::size_t> all(target);
      all.insert(all.end(), reserve.begin(), reserve.end());
      std::vector<double> mu2, sc2;
      shapesFor(t, all, mu2, sc2);
      std::vector<strat::McSpec> specs;
      const auto& vg = fine ? vFine : vGrid;
      for (double m : vg) {
        strat::McSpec sv = mc;
        (psarVol ? sv.psarVol : sv.vstop) = m;
        specs.push_back(sv);
      }
      for (std::size_t j = 0; j < all.size(); ++j) {
        const std::size_t i = all[j];
        const double adj = strat::mcAdjust(d, t, i, mc) * std::exp(mu2[j]), cap = capOf(t, i);
        std::vector<double> means;
        const std::size_t b = bestSetting(i, adj, cap, sc2[j], specs, false, &means);
        const double pick = fine ? fineArgmax(vg, means) : vg[b];
        vPick[i] = pick;
        if (fine) vPicks.push_back(pick);
        else ++vChosen[pick];
      }
    }
    if (rbAdapt) {  // each share's buy-back distance from its simulated futures (with its own stop): the most expected log growth
      std::vector<std::size_t> all(target);
      all.insert(all.end(), reserve.begin(), reserve.end());
      std::vector<double> mu2, sc2;
      shapesFor(t, all, mu2, sc2);
      for (std::size_t j = 0; j < all.size(); ++j) {
        const std::size_t i = all[j];
        std::vector<strat::McSpec> specs;
        std::vector<std::pair<double, double>> pairs;  // (the stop's band, the buy-back's distance)
        const auto& rg = fine && !joint ? rbFine : rbGrid;
        for (double v : joint ? vGrid : std::vector<double>{psarVol ? vOf(i) : 0.0})
          for (double m : rg) {
            strat::McSpec sv = mc;
            sv.rebuyStop = m, sv.afStep = sarOf(i).first, sv.afMax = sarOf(i).second;
            if (psarVol) sv.psarVol = v;
            specs.push_back(sv), pairs.emplace_back(v, m);
          }
        const double adj = strat::mcAdjust(d, t, i, mc) * std::exp(mu2[j]), cap = capOf(t, i);
        std::vector<double> means;
        const auto pick = pairs[bestSetting(i, adj, cap, sc2[j], specs, probable, &means)];
        if (joint) vPick[i] = pick.first, ++vChosen[pick.first];
        if (fine && !joint && !probable) {
          rbPick[i] = fineArgmax(rg, means);
          rbPicks.push_back(rbPick[i]);
        } else {
          rbPick[i] = pick.second;
          ++rbChosen[pick.second];
        }
      }
    }
    prof.adapt += clock() - ca;
    resetMu.push_back(mu), resetSc.push_back(sc);
  };
  // stopmedian: the distance a new position's stop would start under the last close of t (as openStop), and the
  // median of it over the day's candidates (the target and its reserves); without stopmedian no limit.
  const auto startDist = [&](std::size_t t, std::size_t i) {
    const double c = d.Cf[t * N + i];
    if (!(c > 0) || !stops) return 0.0;
    Pos q{};
    openStop(q, t + 1, i, c);
    return q.sar > 0 ? 1 - q.sar / c : 0.0;
  };
  const auto stopMedianOf = [&](std::size_t t) {
    if (!(P["stopmedian"] > 0) || !stops) return 1e300;
    std::vector<double> v;
    for (std::size_t i : target) v.push_back(startDist(t, i));
    for (std::size_t i : reserve) v.push_back(startDist(t, i));
    if (v.empty()) return 1e300;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const auto inTarget = [&](std::size_t i) {
    return std::find(target.begin(), target.end(), i) != target.end() || (refill && std::find(reserve.begin(), reserve.end(), i) != reserve.end());
  };

  for (std::size_t t = start; t + 1 < T; ++t) {
    if (nnStep && (t - start) % nnStep == 0 && (t - start) % reb != 0) learn(t, 1);  // between the resets: the day's bars
    decide(t);
    const std::size_t u = t + 1;
    double day = 0, dayCost = 0;  // dayCost: the day's buy-backs' round trips (the record after them)
    std::vector<std::size_t> done, sameDayBack;
    // rebuystop: share i bought back at px on day u by its buy-back order, its stop placed at once. sameDay: the stop's own
    // day (its close over the level): no second stop that day; intraday: bought at the level during the day (a stop
    // the same day sells at its level, taken as after the buy).
    const auto buyBack = [&](std::size_t i, double px, bool sameDay, bool intraday = false) {
      const double C = d.Cf[u * N + i];
      const auto at = std::find(target.begin(), target.end(), i);
      Row r;
      r.t = t, r.i = i, r.rank = at != target.end() ? std::size_t(at - target.begin()) : target.size();
      r.expected = learnt.expected(r.rank), r.limit = px, r.filled = true, r.entry = px, r.since = u;
      stoppedAt.erase(i), ++nRebuys;
      dayCost += tripOf(t, i);
      stamp(r);
      Pos p{rows.size(), C > 0 ? C : px};
      openStop(p, u, i, px);
      const double level = p.sar;
      double sp = 0;
      if (!sameDay && stopDay(p, u, i, sp)) {
        if (intraday) sp = level;
        day += sp / px - 1;
        r.exit = sp, r.ret = sp / px - 1, r.closed = true, r.stopped = true, r.days = 0;
        stoppedAt[i] = stoppedFrom(p, sp, i, u), ++nStops;  // its next buy-back order (from the next day)
        done.push_back(rows.size());
      } else {
        day += p.mark / px - 1;
        book[i] = p;
        lastHeld[i] = u;
      }
      rows.push_back(r);
    };
    for (auto it = book.begin(); it != book.end();) {  // the book: sold at the open when no longer wanted, else marked to the close
      const std::size_t i = it->first;
      Row& r = rows[it->second.row];
      const double O = d.O[u * N + i], C = d.Cf[u * N + i];
      if (it->second.exitNext && O > 0) {  // vstop: yesterday's close under its level: sold at this open
        day += O / it->second.mark - 1;
        r.exit = O, r.ret = O / r.entry - 1, r.closed = true, r.stopped = true, r.days = u - r.since;
        stoppedAt[i] = stoppedFrom(it->second, O, i, u), ++nStops;
        done.push_back(it->second.row);
        it = book.erase(it);
        continue;
      }
      if (!inTarget(i) && O > 0) {
        day += O / it->second.mark - 1;
        r.exit = O, r.ret = O / r.entry - 1, r.closed = true, r.days = u - r.since;
        done.push_back(it->second.row);
        it = book.erase(it);
        continue;
      }
      double px = 0;
      if (stopDay(it->second, u, i, px)) {  // the stop-loss: sold at its level (or the open, through a gap)
        day += px / it->second.mark - 1;
        r.exit = px, r.ret = px / r.entry - 1, r.closed = true, r.stopped = true, r.days = u - r.since;
        stoppedAt[i] = stoppedFrom(it->second, px, i, u), ++nStops;
        done.push_back(it->second.row);
        it = book.erase(it);
        if (rbStop && C >= stoppedAt[i].trigger) buyBack(i, stoppedAt[i].trigger, true);  // back through its buy stop after the stop
        continue;
      }
      if (C > 0) day += C / it->second.mark - 1, it->second.mark = C;
      if (u - r.since + 1 >= maxDays) {  // its last day: sold at the close
        r.exit = it->second.mark, r.ret = r.exit / r.entry - 1, r.closed = true, r.days = u - r.since;
        done.push_back(it->second.row);
        it = book.erase(it);
        continue;
      }
      lastHeld[i] = u;
      ++it;
    }
    for (auto it = stoppedAt.begin(); it != stoppedAt.end();) it = inTarget(it->first) ? std::next(it) : stoppedAt.erase(it);  // dropped: forgotten
    if (rbStop) {  // the buy-back orders working since an earlier day's stop (a slot each)
      std::vector<std::pair<std::size_t, double>> due;
      for (const auto& [i, x] : stoppedAt)
        if (x.at < u && x.trigger > 0) due.emplace_back(i, x.trigger);
      for (const auto& [i, trig] : due) {
        if (book.size() >= k) break;
        const double O = d.O[u * N + i], H = d.H[u * N + i];
        if (O > 0 && O >= trig) buyBack(i, O, false);
        else if (H > 0 && H >= trig) buyBack(i, trig, false, true);
      }
    }
    std::size_t free = k > book.size() ? k - book.size() : 0;
    std::vector<std::size_t> wants(target);  // the book's shares in order, then (refill) the reserves for free slots
    const double stopCut = stopMedianOf(t);
    if (refill) wants.insert(wants.end(), reserve.begin(), reserve.end());
    for (std::size_t rank = 0; rank < wants.size() && free > 0; ++rank) {
      const std::size_t i = wants[rank];
      if (book.count(i) || !mayBuy(t, i)) continue;
      if (startDist(t, i) > stopCut) continue;  // stopmedian: its stop wider than the day's median
      const bool again = stoppedAt.count(i) > 0;
      Row r;
      r.t = t, r.i = i, r.rank = rank, r.expected = learnt.expected(rank);
      r.limit = d.Cf[t * N + i] * (1 + capOf(t, i));
      stamp(r);
      if (trace) std::fprintf(trace, "%s,%s,%.6g,%.8g,%zu\n", d.days[t].c_str(), d.tickers[i].c_str(), r.limit / d.Cf[t * N + i] - 1, r.expected, rank);
      const double O = d.O[u * N + i], C = d.Cf[u * N + i];
      if (O > 0 && O <= r.limit) {
        r.filled = true, r.entry = O, r.since = u;
        if (again) stoppedAt.erase(i), ++nRebuys, dayCost += tripOf(t, i);
        --free;
        double px = 0;
        Pos p{rows.size(), C > 0 ? C : O};
        openStop(p, u, i, O);
        if (stopDay(p, u, i, px)) {  // stopped the day it was bought
          day += px / O - 1;
          r.exit = px, r.ret = px / O - 1, r.closed = true, r.stopped = true, r.days = 0;
          stoppedAt[i] = stoppedFrom(p, px, i, u), ++nStops;
          done.push_back(rows.size());
          if (rbStop && C >= stoppedAt[i].trigger) sameDayBack.push_back(i);  // after this row is in
        } else {
          day += p.mark / O - 1;
          book[i] = p;
          lastHeld[i] = u;
        }
      }
      rows.push_back(r);
    }
    for (std::size_t i : sameDayBack)
      if (stoppedAt.count(i) && !book.count(i)) buyBack(i, stoppedAt[i].trigger, true);
    for (auto& [i, x] : stoppedAt) stoppedDay(x, u, i);  // the stopped shares through day u's close
    daily.push_back(day / double(k));
    dailyNet.push_back((day - dayCost) / double(k));
    dailyDays.push_back(d.days[u]);
    for (std::size_t x : done) learnt.add(rows[x]);
  }
  decide(T - 1);  // the next session's target
  if (trace) std::fclose(trace);
  for (auto& [i, p] : book) {  // still held: marked at the last close (the trade log only)
    Row& r = rows[p.row];
    r.exit = p.mark, r.ret = p.mark / r.entry - 1, r.days = T - 1 - r.since;
  }

  // ------------------------------------------------------------------ the backtest's record
  double w = 0, s1 = 0, s2 = 0, peak = 0, dd = 0;
  for (double x : daily) {
    w += std::log1p(x), s1 += x, s2 += x * x;
    peak = std::max(peak, w), dd = std::max(dd, 1 - std::exp(w - peak));
  }
  const double nD = double(daily.size()), mean = nD ? s1 / nD : 0, sd = nD ? std::sqrt(std::max(0.0, s2 / nD - mean * mean)) : 0;
  const double annual = nD ? std::expm1(w * 252.0 / nD) : 0, sharpe = sd > 0 ? mean / sd * std::sqrt(252.0) : 0;
  // The same record after the buy-backs' own round trips (each buy-back after a stop is one more round trip than holding).
  double annualNet = 0, sharpeNet = 0;
  {
    double wn = 0, a1 = 0, a2 = 0;
    for (double x : dailyNet) wn += std::log1p(x), a1 += x, a2 += x * x;
    const double m = nD ? a1 / nD : 0, sdn = nD ? std::sqrt(std::max(0.0, a2 / nD - m * m)) : 0;
    annualNet = nD ? std::expm1(wn * 252.0 / nD) : 0, sharpeNet = sdn > 0 ? m / sdn * std::sqrt(252.0) : 0;
  }
  double nFilled = 0, sumTrade = 0, nWin = 0;
  for (const auto& r : rows)
    if (r.filled) nFilled += 1, sumTrade += r.ret, nWin += r.ret > 0;

  if (const char* tl = std::getenv("X_TRADELOG")) {
    std::ofstream f(tl);
    f << "date,ticker,rank,limit,expected,filled,entry,exit,days,ret" << (regK ? ",turbulence,turbulence_ahead,market_vol,share_vol" : "") << "\n";
    for (const auto& r : rows) {
      f << d.days[r.t + 1] << ',' << d.tickers[r.i] << ',' << r.rank << ',' << jnum(r.limit) << ',' << jnum(r.expected) << ',' << (r.filled ? 1 : 0) << ','
        << jnum(r.entry) << ',' << jnum(r.exit) << ',' << r.days << ',' << jnum(r.filled ? r.ret : 0.0);
      if (regK) f << ',' << jnum(r.turbulence) << ',' << jnum(r.ahead) << ',' << jnum(r.marketVol) << ',' << jnum(r.shareVol);
      f << '\n';
    }
    std::ofstream g(std::string(tl) + ".days");
    g << "date,ret\n";
    for (std::size_t x = 0; x < daily.size(); ++x) g << dailyDays[x] << ',' << jnum(daily[x], "%.8g") << '\n';
  }

  // mcregime: the bought trades split into thirds by the market's turbulence expected over the horizon when decided
  // (continuous: today's chances carried through the matrix; today's alone are mostly 0 or 1) (the thirds' bounds from
  // the trades' own turbulence, nothing fixed): each third's count, mean trade and share gained.
  struct Third { double lo = 0, hi = 0, mean = 0, win = 0; std::size_t n = 0; };
  std::vector<Third> turbThird;
  if (regK) {
    std::vector<std::pair<double, double>> tr;  // (turbulence, the trade's return)
    for (const auto& r : rows)
      if (r.filled && r.closed && std::isfinite(r.ahead)) tr.emplace_back(r.ahead, r.ret);
    std::sort(tr.begin(), tr.end());
    if (tr.size() >= 30)
      for (std::size_t q = 0; q < 3; ++q) {
        const std::size_t a = tr.size() * q / 3, e = tr.size() * (q + 1) / 3;
        Third th;
        th.lo = tr[a].first, th.hi = tr[e - 1].first, th.n = e - a;
        for (std::size_t j = a; j < e; ++j) th.mean += tr[j].second, th.win += tr[j].second > 0;
        th.mean /= double(th.n), th.win /= double(th.n);
        turbThird.push_back(th);
      }
  }
  const double cCheck = clock();
  // ------------------------------------------------------------------ the nested Monte Carlo
  // The resets on the real market: at each rebalance with a full horizon after it, the simulated futures of the book
  // and the real one through the same simulator.
  double mcCover = 0, mcPit = 0, mcSim = 0, mcReal = 0, mcSxy = 0, mcSxx = 0, mcSyy = 0, mcN = 0;
  if (mcOn) {
    strat::McReset rr;
    for (std::size_t x = 0; x < resets.size(); ++x) {
      const std::size_t t0 = resets[x].first;
      const auto& list = resets[x].second;
      if (t0 + mc.horizon >= T || list.empty()) continue;
      std::vector<double> caps;
      for (std::size_t i : list) caps.push_back(capOf(t0, i));
      strat::mcRun(d, t0, list, caps, std::vector<strat::McHeld>(list.size()), k, mc, t0, *backend, rr, resetMu[x], resetSc[x]);
      const double real = strat::mcReal(d, t0, list, caps, k, mc);
      double below = 0, m = 0;
      for (double b : rr.book) below += b < real ? 1.0 : b == real ? 0.5 : 0.0, m += b;
      m /= double(rr.book.size());
      const double p5 = strat::mcQuantile(rr.book, 0.05), p95 = strat::mcQuantile(rr.book, 0.95);
      mcCover += real >= p5 && real <= p95, mcPit += below / double(rr.book.size()), mcSim += m, mcReal += real, mcN += 1;
      mcSxy += m * real, mcSxx += m * m, mcSyy += real * real;
    }
  }
  const double mcCorr = mcN > 2 ? (mcSxy / mcN - mcSim / mcN * mcReal / mcN) /
                                       std::sqrt(std::max(1e-30, (mcSxx / mcN - (mcSim / mcN) * (mcSim / mcN)) * (mcSyy / mcN - (mcReal / mcN) * (mcReal / mcN))))
                                 : kNaN;

  // ------------------------------------------------------------------ the next session's orders
  struct Order { std::size_t i, rank; double close, limit, value, gross, fill, cost = 0, net = 0; bool held; double p5 = kNaN, p95 = kNaN, pLoss = kNaN, pStop = kNaN; };
  const std::size_t L = T - 1;
  std::vector<Order> orders;
  std::vector<std::size_t> targetAll(target);  // the book's shares, then (refill) the reserves for slots stops free
  if (refill) targetAll.insert(targetAll.end(), reserve.begin(), reserve.end());
  for (std::size_t rank = 0; rank < targetAll.size(); ++rank) {
    const std::size_t i = targetAll[rank];
    if (!book.count(i) && !mayBuy(L, i)) continue;  // stopped and not recovered: not bought back yet
    Order x{i, rank, d.Cf[L * N + i], 0, 0, learnt.expected(rank), 0, 0, 0, book.count(i) > 0};
    x.limit = x.close * (1 + capOf(L, i));
    x.value = noVolume ? o.capital : std::min(o.capital, o.advShare * d.value[L * N + i]);
    if (!(x.value >= o.minOrder)) continue;
    double n = 0, f = 0;
    for (std::size_t q = L > 251 ? L - 251 : 0; q < L; ++q) {
      const double c = d.Cf[q * N + i], O = d.O[(q + 1) * N + i];
      if (c > 0 && O > 0) n += 1, f += O <= c * (1 + capOf(q, i));
    }
    x.fill = n > 0 ? f / n : kNaN;
    orders.push_back(x);
  }
  // The Monte Carlo from the last close (a reset on today's real market): each order's expected return over the
  // horizon (its slot's mean over the paths, with the stop and the buy-back), its 5% and 95%, its chances of a loss
  // and of a stop; and the book's.
  double mcNext[5] = {kNaN, kNaN, kNaN, kNaN, kNaN};  // mean, 5%, 50%, 95%, chance of a loss
  if (mcOn && !target.empty()) {
    std::vector<double> caps;
    std::vector<strat::McHeld> held;
    for (std::size_t i : targetAll) {
      caps.push_back(capOf(L, i));
      strat::McHeld h;
      const auto b = book.find(i);
      if (b != book.end())
        h.held = true, h.mark = d.Cf[L * N + i], h.sar = b->second.sar, h.ep = b->second.ep, h.af = b->second.af, h.exitNext = b->second.exitNext;
      held.push_back(h);
    }
    strat::McReset rr;
    std::vector<double> mu, sc;
    shapesFor(L, targetAll, mu, sc);
    strat::mcRun(d, L, targetAll, caps, held, k, mc, L, *backend, rr, mu, sc);
    for (auto& x : orders) {
      const std::size_t j = std::size_t(std::find(targetAll.begin(), targetAll.end(), x.i) - targetAll.begin());
      std::vector<double> v(rr.share.begin() + long(j * mc.paths), rr.share.begin() + long((j + 1) * mc.paths));
      double m = 0, loss = 0, st = 0;
      for (std::size_t p = 0; p < mc.paths; ++p) m += v[p], loss += v[p] < 0, st += rr.stop[j * mc.paths + p];
      x.gross = m / double(mc.paths), x.pLoss = loss / double(mc.paths), x.pStop = st / double(mc.paths);
      x.p5 = strat::mcQuantile(v, 0.05), x.p95 = strat::mcQuantile(v, 0.95);
    }
    double m = 0, loss = 0;
    for (double b : rr.book) m += b, loss += b < 0;
    mcNext[0] = m / double(mc.paths), mcNext[1] = strat::mcQuantile(rr.book, 0.05), mcNext[2] = strat::mcQuantile(rr.book, 0.5),
    mcNext[3] = strat::mcQuantile(rr.book, 0.95), mcNext[4] = loss / double(mc.paths);
  }
  // The list in order of expected return before costs, highest first.
  // (refill: the book's shares first, then the reserves, each in that order: a reserve only fills a slot a stop freed)
  const std::size_t nBook = target.size();
  std::stable_sort(orders.begin(), orders.end(), [&](const Order& a, const Order& b) {
    const bool ra = a.rank >= nBook, rb = b.rank >= nBook;
    return ra != rb ? rb : a.gross > b.gross;
  });
  // Then, at the end, the broker's costs: off each order's expected return; an order they make a loss is dropped.
  std::vector<Order> final;
  for (auto x : orders) {
    x.cost = costs.roundTrip(x.value, d.tickers[x.i], x.close / unit);
    x.net = x.gross - x.cost;
    if (x.net > 0) final.push_back(x);
  }

  // Each share's stop for the next session (absolute, the store's price unit): a position held, its stop as it
  // stands after the last close; one to be bought, the stop it starts with (psar: the lowest low of the last 5
  // sessions; a fixed stop: from the price paid, so the bridge sets it from the fill: stopLoss).
  prof.check = clock() - cCheck;
  if (std::getenv("X_PROFILE"))
    std::fprintf(stderr, "profile: features %.2fs, strategies' ranks %.2fs, network training %.2fs, samples %.2fs, selection %.2fs, learnt stops %.2fs, the check and orders %.2fs, the market's states %.2fs\n",
                 prof.feat, prof.rank, prof.train, prof.sample, prof.select, prof.adapt, prof.check, prof.regime);
  const auto stopPrice = [&](std::size_t i) {
    if (!stops) return kNaN;
    const auto b = book.find(i);
    if (b != book.end()) return b->second.sar;
    if (vstop) return d.Cf[L * N + i] * (1 - 1.5 * vDist(L, i));
    if (psarVol) return d.Cf[L * N + i] * (1 - vDist(L, i));  // the volatility-adjusted SAR's start (the bridge: under the fill)  // the catastrophic stop, from the last close (the bridge: under the fill)
    if (psar) {
      const double lo = lowOf(L >= 4 ? L - 4 : 0, L, i);
      return std::isfinite(lo) && lo < d.Cf[L * N + i] ? lo : d.Cf[L * N + i] * 0.95;
    }
    return d.Cf[L * N + i] * (1 - stopLoss);
  };
  // The close a stopped share must beat to be bought again, were it stopped at the next session (as stoppedFrom and
  // canBuy): with psar its falling SAR's start (the highest price while held, at least the stop); with a fixed or a
  // volatility stop rebuy above the level it left at. A share not held: as bought at the last close.
  const auto rebuyPrice = [&](std::size_t i) {
    if (!stops || (rebuy < 0 && !rbStop)) return kNaN;
    const auto b = book.find(i);
    Pos p{};
    if (b != book.end()) p = b->second;
    else openStop(p, L + 1, i, d.Cf[L * N + i]);
    if (rbStop) return p.sar * (1 + rbOf(i));  // the buy-back order's level, over the stop's
    if (psar) return std::max(p.ep, p.sar);
    return (vstop ? p.ep : p.sar) * (1 + rebuy);
  };
  // rebuystop: a bought-back position's stop from the price it is bought back at (the same as a new position's: openStop).
  const auto rebuyStopRel = [&](std::size_t i) {
    if (!rbStop) return kNaN;
    const double at = rebuyPrice(i);
    if (!(at > 0)) return kNaN;
    Pos q{};
    openStop(q, L + 1, i, at);
    return q.sar / at - 1;
  };
  // stopmedian: the orders to buy (not the shares held) whose stop would start wider than the day's median left out.
  if (P["stopmedian"] > 0 && stops) {
    const double cut = stopMedianOf(L);
    final.erase(std::remove_if(final.begin(), final.end(), [&](const Order& x) { return !x.held && startDist(L, x.i) > cut; }), final.end());
  }
  // The orders by expected return per unit of stop (stops): the expected return after costs (times the chance to
  // fill) over the stop's distance under the last close (at least 0.5%), highest first, so a far stop must earn its
  // place; the bridge fills the free slots in this order. The order of equals (and without stops) kept.
  if (stops)
    std::stable_sort(final.begin(), final.end(), [&](const Order& a, const Order& b) {
      const auto perStop = [&](const Order& x) {
        const double dist = std::max(0.005, 1 - stopPrice(x.i) / x.close), fill = std::isfinite(x.fill) ? x.fill : 1.0;
        return std::isfinite(dist) ? fill * x.net / dist : -1e300;
      };
      return perStop(a) > perStop(b);
    });
  const std::string stopKind = psarVol ? "psar-vol" : psar ? "psar" : vstop ? "vol" : stopLoss > 0 ? "fixed" : "none";

  // ------------------------------------------------------------------ report
  const std::string cur = uk ? "GBP" : M == "US" || M == "FOREX" ? "USD" : M == "APAC" ? "AUD" : "EUR";
  std::printf("strategy: %s (%s) on %s, back end %s; %zu shares, %s .. %s%s%s\n", o.strategy.c_str(), strategy->about().c_str(), M.c_str(), backend->about().c_str(), N,
              d.days.front().c_str(), d.days.back().c_str(), note.empty() ? "" : "; ", note.c_str());
  if (units) std::printf("strategy: %zu bars put in their share's latest price unit (pence and pounds)\n", units);
  if (outliers) std::printf("strategy: %zu bad bars dropped (an impossible low or high for the day), the last real close rolled over them\n", outliers);
  if (top) std::printf("strategy: each day the %zu most traded shares as of that day (of %zu in the data), not today's list applied backwards\n", top, N);
  std::printf("strategy: the account (%s .. %s, %zu slots of %s %.0f, rebalanced every %zu sessions): %+.1f%% a year, Sharpe %.2f, max drawdown %.1f%%, before costs\n",
              dailyDays.empty() ? "-" : dailyDays.front().c_str(), dailyDays.empty() ? "-" : dailyDays.back().c_str(), k, cur.c_str(), o.capital, reb, 100 * annual, sharpe, 100 * dd);
  if (o.resample > 0)
    std::printf("strategy: on a synthetic market (seed %u) drawn from the real one: blocks of %zu real days in the market's %zu states, each share on its real calendar and volume\n",
                o.resample, rsInfo.block, rsInfo.states);
  // --resample-runs: the same strategy on N synthetic markets drawn from the real one: the spread of its results
  struct Spread { double mean = kNaN, lo = kNaN, mid = kNaN, hi = kNaN; std::size_t n = 0; };
  const auto spreadOf = [&](double Resampled::*m) {
    std::vector<double> v;
    for (const auto& r : resampled)
      if (std::isfinite(r.*m)) v.push_back(r.*m);
    Spread sp;
    if (v.empty()) return sp;
    std::sort(v.begin(), v.end());
    sp.n = v.size(), sp.mean = 0;
    for (double x : v) sp.mean += x / double(v.size());
    sp.lo = v.front(), sp.mid = v[v.size() / 2], sp.hi = v.back();
    return sp;
  };
  const Spread rsA = spreadOf(&Resampled::annual), rsS = spreadOf(&Resampled::sharpe), rsD = spreadOf(&Resampled::dd), rsAN = spreadOf(&Resampled::annualNet),
               rsSN = spreadOf(&Resampled::sharpeNet);
  if (rsS.n)
    std::printf("strategy: on %zu synthetic markets drawn from the real one (the nested Monte Carlo's draw, blocks of %u days in the market's states): "
                "%+.1f%% a year on average (%+.1f%% .. %+.1f%%), Sharpe %.2f on average (%.2f .. %.2f, median %.2f), max drawdown %.1f%% on average%s\n",
                rsS.n, o.resampleBlock, 100 * rsA.mean, 100 * rsA.lo, 100 * rsA.hi, rsS.mean, rsS.lo, rsS.hi, rsS.mid, 100 * rsD.mean,
                rsSN.n ? (", after the buy-backs' round trips Sharpe " + jnum(rsSN.mean, "%.2f") + " on average (" + jnum(rsSN.lo, "%.2f") + " .. " + jnum(rsSN.hi, "%.2f") + ")").c_str() : "");
  if (stops)
    std::printf("strategy:   %s: %zu positions stopped; %s\n",
                psarVol ? (std::string("a volatility-adjusted Parabolic SAR stop (") + (vAdapt ? "a multiple learnt for each share of" : jnum(P["psarvol"]) + " times") + " the daily volatility under the price paid, never closer than that under the close)").c_str()
                : psar ? "a Parabolic SAR stop (trailing, from the lows before the buy)"
                : vstop ? (std::string("a volatility stop (") + (vAdapt ? "a multiple learnt for each share of" : jnum(P["vstop"]) + " times") + " the daily volatility under the price paid, then under the highest close; a close under it sells at the next open)").c_str()
                        : ("a stop-loss " + std::to_string(int(std::lround(100 * stopLoss))) + "% under the price paid").c_str(), nStops,
                rbStop ? ("bought back by the buy stop over the level it sold at: " + std::to_string(nRebuys) + " times").c_str()
                : rebuy < 0 ? "never bought back"
                          : (std::string(psar ? "bought back once its SAR turned up again: " : ("bought back once the close was " + std::to_string(int(std::lround(100 * rebuy))) + "% over the stop level: ").c_str()) +
                             std::to_string(nRebuys) + " times").c_str());
  if (stops && nRebuys)
    std::printf("strategy:   after the buy-backs' own round trips at the broker (%zu buy-backs, each one more round trip than holding): %+.1f%% a year, Sharpe %.2f%s\n",
                nRebuys, 100 * annualNet, sharpeNet, P["mccost"] > 0 ? " (the stops and buy-backs learnt net of them: mccost)" : "");
  std::printf("strategy:   %zu orders, %.0f bought (%.0f%%); a trade on average %+.2f%% before costs, %.0f%% gained, held %.1f sessions\n", rows.size(), nFilled,
              rows.empty() ? 0.0 : 100 * nFilled / double(rows.size()), nFilled ? 100 * sumTrade / nFilled : 0.0, nFilled ? 100 * nWin / nFilled : 0.0, learnt.holdDays(double(reb)));
  if (mcOn) {
    std::printf("strategy: nested Monte Carlo: %.0f resets on the real market, %zu futures of %zu sessions each (blocks of %zu real days from the %zu before); "
                "the real book fell in the simulated 5-95%% band %.0f%% of the time (90%% if honest), at percentile %.0f on average (50 if honest); "
                "the simulated mean %+.2f%% a period against the real %+.2f%% (correlation %.2f)\n",
                mcN, mc.paths, mc.horizon, mc.block, mc.window, mcN ? 100 * mcCover / mcN : 0.0, mcN ? 100 * mcPit / mcN : 0.0, mcN ? 100 * mcSim / mcN : 0.0,
                mcN ? 100 * mcReal / mcN : 0.0, mcCorr);
    if (nnOn) {
      const auto [a, b] = net.score(nnSamples, nnFirstTrained, T, mc.horizon);
      std::printf("strategy: the differential neural network on the Monte Carlo: trained on %zu days%s on %zu share-days (walk-forward); its futures' CRPS "
                  "%.4f against the plain Monte Carlo's %.4f (lower is better: %+.1f%%)\n",
                  net.trained, nnStep ? (" (every " + std::to_string(nnStep) + " session" + (nnStep > 1 ? "s" : "") + ", the day's bars taken in)").c_str() : " (the resets)",
                  nnSamples.size(), a, b, b > 0 ? 100 * (a / b - 1) : 0.0);
    }
    if (regK && !turbThird.empty()) {
      std::string by;
      const char* names[3] = {"the calmest third", "the middle third", "the most turbulent third"};
      for (std::size_t q = 0; q < turbThird.size(); ++q)
        by += (q ? "; " : "") + std::string(names[q]) + " (turbulence " + jnum(turbThird[q].lo, "%.2f") + "-" + jnum(turbThird[q].hi, "%.2f") + "): " +
              std::to_string(turbThird[q].n) + " trades, " + jnum(100 * turbThird[q].mean, "%+.2f") + "% on average, " + jnum(100 * turbThird[q].win, "%.0f") + "% gained";
      std::printf("strategy: the trades by the market's turbulence expected over the horizon when decided (0 calm .. 1 turbulent, from the states' chances that day): %s\n", by.c_str());
    }
    if (regK && !fits.empty()) {
      const auto& f = fits.rbegin()->second;
      std::string st;
      for (std::size_t q = 0; q < f.K; ++q)
        st += (q ? "; " : "") + std::string("state ") + std::to_string(q + 1) + " (" + (q == 0 ? "calmest" : q + 1 == f.K ? "most turbulent" : "between") + "): the market " +
              jnum(100 * 252 * f.mean[q], "%+.0f") + "% a year at " + jnum(100 * std::sqrt(252 * f.var[q]), "%.0f") + "% volatility, " +
              (f.A[q * f.K + q] < 0.9999 ? "staying " + jnum(1 / (1 - f.A[q * f.K + q]), "%.0f") + " sessions on average, " : std::string("never left, ")) + jnum(100 * f.now[q], "%.0f") + "% today, " +
              jnum(100 * f.ahead[q], "%.0f") + "% over the next " + std::to_string(mc.horizon) + " sessions";
      std::string kLearnt;
      if (regK == SIZE_MAX)
        for (const auto& [kk, n] : regKChosen) kLearnt += (kLearnt.empty() ? "" : ", ") + std::to_string(kk) + " states " + std::to_string(n) + " times";
      std::printf("strategy: the market's %zu hidden states, refitted at each of %zu resets (walk-forward) and the futures drawn in them%s%s; the last fit (%s): %s\n", f.K,
                  fits.size(), kLearnt.empty() ? "" : (" (the number learnt at each reset by BIC: " + kLearnt + ")").c_str(),
                  regStop ? ", the stops sized by the states expected" : ", the stops on the last 60 sessions", d.days[f.hi].c_str(), st.c_str());
    }
    if (gate) {
      std::string used;
      for (std::size_t q = 0; q < arms.size(); ++q) used += (q ? ", " : "") + armNames[q] + " " + std::to_string(arms[q].used);
      std::printf("strategy: autonomous: at each reset the book held is the one (the simulation's or a strategy's own top %zu) with the best real record over "
                  "the last %zu resets known by then; held: %s (of %zu resets)\n", k, gateN, used.c_str(), resets.size());
    }
    // adaptfine: the learnt values' spread (10%, 25%, median, 75%, 90%)
    const auto spread = [](std::vector<double> v, double scale, const char* fmt) {
      if (v.empty()) return std::string("none");
      std::sort(v.begin(), v.end());
      std::string out;
      const double qs[5] = {0.1, 0.25, 0.5, 0.75, 0.9};
      const char* names[5] = {"10%", "25%", "median", "75%", "90%"};
      for (int q = 0; q < 5; ++q) out += (q ? ", " : "") + std::string(names[q]) + " " + jnum(scale * v[std::size_t(qs[q] * double(v.size() - 1))], fmt);
      return out + " (of " + std::to_string(v.size()) + ")";
    };
    if (vAdapt) {
      std::string used;
      for (const auto& [m, n] : vChosen) used += (used.empty() ? "" : ", ") + jnum(m) + " " + std::to_string(n);
      if (fine) used = "learnt as any value from 1 to 10: " + spread(vPicks, 1.0, "%.2f");
      std::printf(psarVol ? "strategy: the SAR's volatility band learnt for each share at each reset (the multiple with the most expected log growth on its simulated "
                  "futures, the stop and the buy-back included): chosen (multiple, times): %s\n" : "strategy: the volatility stop learnt for each share at each reset (the multiple with the most expected log growth on its simulated "
                  "futures, the stop and the buy-back included): chosen (multiple, times): %s\n", used.c_str());
    }
    if (rbStop) {
      std::string used;
      for (const auto& [m, n] : rbChosen) used += (used.empty() ? "" : ", ") + jnum(m) + " " + std::to_string(n);
      if (fine && !rbPicks.empty()) used = "learnt as any value from 0.25% to 10% (in %): " + spread(rbPicks, 100.0, "%.2f");
      std::printf("strategy: the buy-back an order attached to the stop (a buy stop over the level it sold at): %s; %zu buy-backs\n",
                  rbAdapt ? ("its distance learnt for each share at each reset, chosen (distance, times): " + used).c_str() : ("at " + jnum(P["rebuystop"]) + " over it").c_str(), nRebuys);
    }
    if (refill) std::printf("strategy: a slot freed by a stop goes to the next share the simulation chose (%zu reserves at the last reset)\n", reserve.size());
    if (sarAdapt) {
      std::string used;
      for (const auto& [k2, n] : sarChosen) used += (used.empty() ? "" : ", ") + std::to_string(k2.first).substr(0, 5) + "/" + std::to_string(k2.second).substr(0, 3) + " " + std::to_string(n);
      std::printf("strategy: the SAR learnt for each share at each reset (its step and largest step with the most expected log growth on its simulated futures, "
                  "the stop and the buy-back included): chosen (step/largest, times): %s\n", used.c_str());
    }
    if (pool) {
      std::string names;
      for (std::size_t q = 0; q < poolNames.size(); ++q) names += (q ? ", " : "") + poolNames[q];
      std::printf("strategy: the strategies injected into the network's Monte Carlo: %s (each one's score of every share an input to the network, each "
                  "one's best %.0f a slot among the candidates)\n", names.c_str(), std::max(1.0, P["mccandk"]));
    }
    if (mcSelect)
      std::printf("strategy: the book chosen by the simulation at every reset: of the strategy's %zu best, the %zu with the most expected log growth together on "
                  "the same futures (the days drawn from the %zu sessions up to %zu before, %s); %.0f%% of its picks were in the strategy's own top %zu\n",
                  nCand, k, mc.window, mc.skip, mc.drift ? "each share's own trend taken out" : "each share's own moves kept", mcChosen ? 100.0 * double(mcFromTop) / double(mcChosen) : 0.0, k);
    if (std::isfinite(mcNext[0]))
      std::printf("strategy: the book over the next %zu sessions (from the last close): mean %+.2f%%, 5%% %+.2f%%, median %+.2f%%, 95%% %+.2f%%, a loss on %.0f%% of the futures\n",
                  mc.horizon, 100 * mcNext[0], 100 * mcNext[1], 100 * mcNext[2], 100 * mcNext[3], 100 * mcNext[4]);
  }
  std::printf("strategy: next session (after the close of %s; the list from the rebalance after the close of %s): %zu on the list, %zu after costs\n", d.days[L].c_str(),
              d.days[decided].c_str(), orders.size(), final.size());
  for (const auto& x : orders)
    std::printf("strategy:   %-12s rank %zu: expected %+.2f%% before costs, costs %.2f%%: %+.2f%% after%s%s\n", d.tickers[x.i].c_str(), x.rank + 1, 100 * x.gross,
                100 * costs.roundTrip(x.value, d.tickers[x.i], x.close / unit), 100 * (x.gross - costs.roundTrip(x.value, d.tickers[x.i], x.close / unit)),
                x.gross - costs.roundTrip(x.value, d.tickers[x.i], x.close / unit) > 0 ? "" : " (dropped)", x.held ? " (held already)" : "");

  const double holdDays = mcOn ? double(mc.horizon) : learnt.holdDays(double(reb));
  const strat::View last(d, decided, &in[decided * N], *backend);  // the target's day: its ranks and reasons
  if (!o.jsonOut.empty()) {
    std::ofstream f(o.jsonOut);
    std::size_t stocks = 0;
    for (std::size_t i = 0; i < N; ++i) stocks += in[L * N + i] != 0;
    f << "{\"model\":\"strategy\",\"strategy\":" << jstr(o.strategy) << ",\"about\":" << jstr(strategy->about()) << ",\"backend\":" << jstr(backendName)
      << ",\"asOf\":" << jstr(d.days[L]) << ",\"decided\":" << jstr(d.days[decided]) << ",\"stocks\":" << stocks << ",\"book\":\"hold\",\"slots\":" << k << ",\"rebalance\":" << reb << ",\"days\":" << maxDays
      << ",\"costs\":{\"broker\":\"IBKR\",\"market\":" << jstr(M) << ",\"net\":false,\"atEnd\":true,\"capital\":" << jnum(o.capital) << ",\"stamp\":" << (uk ? "0.005" : "0")
      << ",\"stampFree\":" << costs.stampFree.size() << ",\"venues\":" << (costs.venue.empty() ? "false" : "true") << "}";
    f << ",\"backtest\":{\"from\":" << jstr(dailyDays.empty() ? "" : dailyDays.front()) << ",\"to\":" << jstr(dailyDays.empty() ? "" : dailyDays.back())
      << ",\"annualReturn\":" << jnum(annual) << ",\"sharpe\":" << jnum(sharpe) << ",\"maxDrawdown\":" << jnum(dd) << ",\"orders\":" << rows.size()
      << ",\"trades\":" << nFilled << ",\"avgTrade\":" << jnum(nFilled ? sumTrade / nFilled : 0.0) << ",\"winRate\":" << jnum(nFilled ? nWin / nFilled : 0.0)
      << ",\"holdDays\":" << jnum(holdDays) << ",\"stopKind\":" << jstr(stopKind) << ",\"stopLoss\":" << (stopLoss > 0 && !psar ? jnum(stopLoss) : "null") << ",\"stops\":" << nStops
      << ",\"rebuy\":" << (stops && (rebuy >= 0 || rbStop) ? "true" : "false") << ",\"rebuyStop\":" << (rbStop ? "true" : "false") << ",\"rebuys\":" << nRebuys
      << ",\"afterRebuyCosts\":{\"annualReturn\":" << jnum(annualNet) << ",\"sharpe\":" << jnum(sharpeNet) << "}";
    {  // what the stops learnt over the backtest: the volatility band's and the buy-back's spread (10%, 25%, median, 75%, 90%)
      const auto quant = [](std::vector<double> v, const std::map<double, std::size_t>& chosen) {
        for (const auto& [m, n] : chosen) v.insert(v.end(), n, m);
        if (v.empty()) return std::string("null");
        std::sort(v.begin(), v.end());
        std::string a = "[";
        for (double q : {0.1, 0.25, 0.5, 0.75, 0.9}) a += (a.size() > 1 ? "," : "") + jnum(v[std::size_t(q * double(v.size() - 1))]);
        return a + "]";
      };
      f << ",\"learnt\":{\"fine\":" << (fine ? "true" : "false") << ",\"tripCosts\":" << (P["mccost"] > 0 ? "true" : "false")
        << ",\"band\":" << (vAdapt ? quant(vPicks, vChosen) : "null") << ",\"rebuy\":" << (rbAdapt ? quant(rbPicks, rbChosen) : "null") << "}";
    }
    if (o.resample > 0) f << ",\"resample\":{\"seed\":" << o.resample << ",\"block\":" << rsInfo.block << ",\"states\":" << rsInfo.states << "}";
    if (rsS.n)
      f << ",\"resampled\":{\"markets\":" << rsS.n << ",\"block\":" << o.resampleBlock << ",\"annualReturn\":" << jnum(rsA.mean) << ",\"sharpe\":" << jnum(rsS.mean)
        << ",\"sharpeMin\":" << jnum(rsS.lo) << ",\"sharpeMax\":" << jnum(rsS.hi) << ",\"maxDrawdown\":" << jnum(rsD.mean) << ",\"afterRebuyCosts\":{\"annualReturn\":"
        << jnum(rsAN.mean) << ",\"sharpe\":" << jnum(rsSN.mean) << "}}";
    if (mcOn)
      f << ",\"mc\":{\"resets\":" << mcN << ",\"paths\":" << mc.paths << ",\"horizon\":" << mc.horizon << ",\"block\":" << mc.block << ",\"window\":" << mc.window
        << ",\"coverage\":" << jnum(mcN ? mcCover / mcN : kNaN) << ",\"pit\":" << jnum(mcN ? mcPit / mcN : kNaN) << ",\"simMean\":" << jnum(mcN ? mcSim / mcN : kNaN)
        << ",\"realMean\":" << jnum(mcN ? mcReal / mcN : kNaN) << ",\"corr\":" << jnum(mcCorr) << ",\"next\":{\"mean\":" << jnum(mcNext[0]) << ",\"p5\":" << jnum(mcNext[1])
        << ",\"p50\":" << jnum(mcNext[2]) << ",\"p95\":" << jnum(mcNext[3]) << ",\"pLoss\":" << jnum(mcNext[4]) << "}";
    if (mcOn && regK && !fits.empty()) {  // the last fit of the market's states: each state's yearly move and volatility, its stay, its chances
      const auto& rf = fits.rbegin()->second;
      const auto arr = [&](auto g) {
        std::string a = "[";
        for (std::size_t q = 0; q < rf.K; ++q) a += (q ? "," : "") + jnum(g(q));
        return a + "]";
      };
      f << ",\"regime\":{\"states\":" << rf.K << ",\"fits\":" << fits.size() << ",\"day\":" << jstr(d.days[rf.hi])
        << ",\"annualMean\":" << arr([&](std::size_t q) { return 252 * rf.mean[q]; }) << ",\"annualVol\":" << arr([&](std::size_t q) { return std::sqrt(252 * rf.var[q]); })
        << ",\"stay\":" << arr([&](std::size_t q) { return rf.A[q * rf.K + q]; }) << ",\"now\":" << arr([&](std::size_t q) { return rf.now[q]; })
        << ",\"ahead\":" << arr([&](std::size_t q) { return rf.ahead[q]; });
      const auto st = stateAt(L, 0);
      f << ",\"today\":{\"turbulence\":" << jnum(st.turbulence) << ",\"ahead\":" << jnum(st.ahead) << ",\"marketVol\":" << jnum(st.marketVol)
        << ",\"marketDrift\":" << jnum(st.marketDrift) << "},\"byTurbulence\":[";
      for (std::size_t q = 0; q < turbThird.size(); ++q)
        f << (q ? "," : "") << "{\"lo\":" << jnum(turbThird[q].lo) << ",\"hi\":" << jnum(turbThird[q].hi) << ",\"trades\":" << turbThird[q].n
          << ",\"avgTrade\":" << jnum(turbThird[q].mean) << ",\"winRate\":" << jnum(turbThird[q].win) << "}";
      f << "]}";
    }
    if (mcOn) f << "}";
    f << "}";  // the backtest
    f << ",\"orders\":[";
    for (std::size_t q = 0; q < final.size(); ++q) {
      const auto& x = final[q];
      const double fill = std::isfinite(x.fill) ? x.fill : 1.0;
      f << (q ? "," : "") << "{\"ticker\":" << jstr(d.tickers[x.i]) << ",\"rank\":" << x.rank + 1 << ",\"of\":" << of << ",\"entry\":\"open\",\"limit\":" << jnum(x.limit / x.close - 1)
        << ",\"stop\":" << (stops ? jnum(stopPrice(x.i) / x.close - 1) : "null") << ",\"stopKind\":" << jstr(stopKind) << ",\"rebuy\":" << (stops && (rebuy >= 0 || rbStop) ? "true" : "false")
        << ",\"rebuyAt\":" << (stops && (rebuy >= 0 || rbStop) ? jnum(rebuyPrice(x.i) / x.close - 1) : "null")
        << ",\"rebuyStop\":" << (rbStop ? jnum(rbOf(x.i)) : "null")
        << ",\"take\":null,\"exitAbove\":null,\"days\":" << maxDays << ",\"holdDays\":" << jnum(holdDays) << ",\"gross\":" << jnum(x.gross)
        << ",\"cost\":" << jnum(x.cost) << ",\"ifFilled\":" << jnum(x.net) << ",\"fill\":" << jnum(x.fill) << ",\"expected\":" << jnum(fill * x.net)
        << ",\"amount\":" << jnum(x.value, "%.0f") << ",\"held\":" << (x.held ? "true" : "false") << ",\"sigma\":" << jnum(d.sigma[L * N + x.i]);
      if (regK) {  // the market's state now (continuous): its turbulence 0..1 now and over the horizon, the market's and the share's volatility in it
        const auto st = stateAt(L, x.i);
        f << ",\"state\":{\"turbulence\":" << jnum(st.turbulence) << ",\"ahead\":" << jnum(st.ahead) << ",\"marketVol\":" << jnum(st.marketVol)
          << ",\"marketDrift\":" << jnum(st.marketDrift) << ",\"shareVol\":" << jnum(st.shareVol) << "}";
      }
      f
        << ",\"score\":" << jnum(score[x.i]) << ",\"reason\":" << jstr(strategy->reason(last, P, x.i));
      if (mcOn) f << ",\"mc\":{\"p5\":" << jnum(x.p5) << ",\"p95\":" << jnum(x.p95) << ",\"pLoss\":" << jnum(x.pLoss) << ",\"pStop\":" << jnum(x.pStop) << ",\"horizon\":" << mc.horizon << "}";
      f << "}";
    }
    f << "],\"recent\":[";
    for (std::size_t x = daily.size() > 20 ? daily.size() - 20 : 0, q = 0; x < daily.size(); ++x, ++q)
      f << (q ? "," : "") << "{\"date\":" << jstr(dailyDays[x]) << ",\"day\":" << jnum(daily[x]) << "}";
    f << "]";
    // The account's growth (log scale) for the page's chart: at most 800 points.
    if (!daily.empty()) {
      std::vector<double> lw(1, 0.0);
      for (double x : daily) lw.push_back(lw.back() + std::log1p(x));
      const double lo = *std::min_element(lw.begin(), lw.end()), hi = std::max(*std::max_element(lw.begin(), lw.end()), lo + 1e-9);
      const std::size_t step = std::max<std::size_t>(1, lw.size() / 800);
      f << ",\"chart\":{\"series\":[{\"name\":" << jstr(o.strategy) << ",\"final\":" << jstr(jnum(std::exp(lw.back()), "%.2f") + "×") << ",\"y\":[";
      for (std::size_t x = 0, q = 0; x < lw.size(); x += step, ++q) f << (q ? "," : "") << jnum((lw[x] - lo) / (hi - lo), "%.4f");
      f << "]}],\"ticks\":[";
      std::size_t q = 0;
      for (double m : {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0})
        if (std::log(m) >= lo && std::log(m) <= hi) f << (q++ ? "," : "") << "{\"label\":" << jstr(jnum(m, "%g") + "×") << ",\"y\":" << jnum((std::log(m) - lo) / (hi - lo), "%.4f") << "}";
      f << "],\"years\":[";
      q = 0;
      for (std::size_t x = 0; x < dailyDays.size(); ++x)
        if (x == 0 || dailyDays[x].substr(0, 4) != dailyDays[x - 1].substr(0, 4))
          f << (q++ ? "," : "") << "{\"label\":" << jstr(dailyDays[x].substr(0, 4)) << ",\"x\":" << jnum(double(x) / double(std::max<std::size_t>(1, dailyDays.size() - 1)), "%.4f") << "}";
      f << "]}";
    }
    f << "}\n";
  }
  if (!o.ordersOut.empty()) {  // the bridge's multi-day book (prices: the private store only)
    std::ofstream f(o.ordersOut);
    f << "{\"model\":\"strategy\",\"strategy\":" << jstr(o.strategy) << ",\"asOf\":" << jstr(d.days[L]) << ",\"slots\":" << k << ",\"book\":\"hold\",\"days\":" << maxDays << ",\"orders\":[";
    for (std::size_t q = 0; q < final.size(); ++q) {
      const auto& x = final[q];
      const double fill = std::isfinite(x.fill) ? x.fill : 1.0;
      f << (q ? "," : "") << "{\"ticker\":" << jstr(d.tickers[x.i]) << ",\"close\":" << jnum(x.close) << ",\"entry\":\"open\",\"open\":true,\"limit\":" << jnum(x.limit)
        << ",\"stop\":null,\"stopKind\":" << jstr(stopKind) << ",\"stopPrice\":" << jnum(stopPrice(x.i)) << ",\"rebuyStop\":" << (rbStop ? jnum(rbOf(x.i)) : "null") << ",\"rebuyStopRel\":" << (rbStop ? jnum(rebuyStopRel(x.i)) : "null") << ",\"psarStep\":" << (psar ? jnum(sarOf(x.i).first) : "null") << ",\"psarMax\":" << (psar ? jnum(sarOf(x.i).second) : "null") << ",\"stopLoss\":" << (stopLoss > 0 && !psar ? jnum(stopLoss) : "null") << ",\"take\":null,\"exitAbove\":null,\"days\":" << maxDays << ",\"holdDays\":" << jnum(holdDays) << ",\"expected\":" << jnum(fill * x.net)
        << ",\"ifFilled\":" << jnum(x.net) << ",\"gross\":" << jnum(x.gross) << ",\"fill\":" << jnum(std::isfinite(x.fill) ? x.fill : 0.0) << ",\"capital\":" << jnum(x.value, "%.2f")
        << ",\"maxCapital\":" << jnum(noVolume ? o.capital : o.advShare * d.value[L * N + x.i], "%.2f") << ",\"rebuy\":0,\"cost\":" << jnum(x.cost)
        << ",\"costAmount\":" << jnum(x.cost * x.value, "%.4g") << ",\"stampFree\":" << (costs.stampFree.count(d.tickers[x.i]) ? "true" : "false") << ",\"rank\":" << x.rank + 1
        << ",\"reserve\":" << (x.rank >= nBook ? "true" : "false") << ",\"vstop\":" << (vstop ? jnum(vOf(x.i)) : "null") << ",\"psarVol\":" << (psarVol ? jnum(vOf(x.i)) : "null") << "}";
    }
    // The daily update: each share the book held lately, kept while it is in the target, else sold at the open.
    f << "],\"updates\":[";
    std::size_t q = 0;
    std::set<std::size_t> named;
    for (std::size_t i : targetAll) named.insert(i);
    for (const auto& [i, day] : lastHeld)
      if (L - std::min(L, day) <= 260) named.insert(i);
    for (std::size_t i : named) f << (q++ ? "," : "") << "{\"ticker\":" << jstr(d.tickers[i]) << ",\"keep\":" << (inTarget(i) && !(book.count(i) && book.at(i).exitNext) ? "true" : "false") << ",\"exitAbove\":null,\"stopPrice\":" << (book.count(i) ? jnum(stopPrice(i)) : "null") << ",\"rebuyStop\":" << (rbStop ? jnum(rbOf(i)) : "null") << ",\"rebuyStopRel\":" << (rbStop ? jnum(rebuyStopRel(i)) : "null") << "}";
    f << "]}\n";
  }
  return 0;
}
