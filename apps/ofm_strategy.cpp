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

#include "strategy/backend.hpp"
#include "strategy/kernels.hpp"
#include "strategy/strategy.hpp"

using strat::kNaN;

namespace {

struct Options {
  std::string csv, market = "UK", strategy, backend, stampFree, venues, overrideCsv, jsonOut, ordersOut, traceOut, synthetic;
  std::map<std::string, double> params;
  double capital = 5000, minValue = -1, minPrice = -1, advShare = 0.01, minOrder = 500;
  std::size_t slots = 5, roll = 5;
  bool tiered = false, list = false;
  unsigned seed = 7;
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
    else if (a == "--list") o.list = true;
    // the scan's arguments for the earlier models, not used here
    else if (a == "--lag" || a == "--exit" || a == "--segments" || a == "--top" || a == "--hold" || a == "--bars") next();
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

// Every row of a CSV of bars; days and tickers in the order first seen (sorted after).
Raw readBars(const std::string& path) {
  std::string text = slurp(path);
  text.push_back('\n');
  Raw r;
  std::unordered_map<std::string, std::uint32_t> dIx, tIx;
  std::size_t p = 0;
  std::vector<char*> f;
  int col[7] = {0, 1, 2, 3, 4, 5, 6};
  bool header = true;
  while (p < text.size()) {
    const std::size_t e = text.find('\n', p);
    char* line = &text[p];
    text[e] = '\0';
    if (e > p && text[e - 1] == '\r') text[e - 1] = '\0';
    f.clear();
    f.push_back(line);
    for (char* c = line; *c; ++c)
      if (*c == ',') *c = '\0', f.push_back(c + 1);
    p = e + 1;
    if (header) {
      header = false;
      const char* names[7] = {"date", "ticker", "open", "high", "low", "close", "volume"};
      for (int k = 0; k < 7; ++k)
        for (std::size_t j = 0; j < f.size(); ++j)
          if (std::strcmp(f[j], names[k]) == 0) col[k] = int(j);
      continue;
    }
    if (f.size() < 7) continue;
    std::string day(f[col[0]], std::min<std::size_t>(10, std::strlen(f[col[0]])));
    std::string tk(f[col[1]]);
    auto d = dIx.emplace(day, std::uint32_t(r.days.size()));
    if (d.second) r.days.push_back(day);
    auto t = tIx.emplace(tk, std::uint32_t(r.tickers.size()));
    if (t.second) r.tickers.push_back(tk);
    Bar b{d.first->second, t.first->second, std::strtod(f[col[2]], nullptr), std::strtod(f[col[3]], nullptr), std::strtod(f[col[4]], nullptr),
          std::strtod(f[col[5]], nullptr), std::strtod(f[col[6]], nullptr)};
    if (b.o > 0 && b.h > 0 && b.l > 0 && b.c > 0 && b.h >= b.l) r.bars.push_back(b);
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

// A generated market for the checks: random walks with persistent drifts, volumes, and a few gaps.
void writeSynthetic(const std::string& path, unsigned seed) {
  std::mt19937_64 g(seed);
  std::normal_distribution<double> z(0, 1);
  std::uniform_real_distribution<double> u(0, 1);
  std::ofstream f(path);
  f << "date,ticker,open,high,low,close,volume\n";
  const int N = 80, T = 900;
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
  bool filled = false, closed = false;
  std::size_t since = 0, days = 0;
};

struct Pos { std::size_t row; double mark; };

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
  for (const auto& [k, v] : o.params) {
    if (!P.count(k)) usage(("the strategy has no parameter " + k).c_str());
    P[k] = v;
  }

  const std::string& M = o.market;
  const bool uk = M == "UK", noVolume = M == "FOREX" || M == "INDEX";
  const double unit = uk ? 100.0 : 1.0;  // London's prices in pence
  const double minPrice = o.minPrice >= 0 ? o.minPrice : uk ? 20 : (M == "US" || M == "EU") ? 1 : M == "APAC" ? 0.1 : 0;
  const double minValue = o.minValue >= 0 ? o.minValue : noVolume ? 0 : M == "US" ? 1e6 : 250000;

  strat::Data d;
  toPanel(readBars(o.csv), d);
  std::string note;
  if (!o.overrideCsv.empty()) note = overrideBars(d, o.overrideCsv);
  const std::size_t units = repairUnits(d);
  const std::size_t T = d.T, N = d.N, k = o.slots;
  const std::size_t reb = std::max<std::size_t>(1, std::size_t(P["rebalance"])), warm = std::size_t(P["warmup"]), maxDays = std::size_t(P["maxdays"]);
  if (T < warm + 30) usage("too few days for the strategy's warm-up");

  strat::PrepareSpec ps;
  ps.roll = o.roll, ps.unit = unit;
  strat::UniverseSpec us;
  us.minPrice = minPrice, us.minValue = minValue, us.warmup = warm, us.needValue = !noVolume;
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
  const double capSd = P["capsd"], capMax = P["capmax"];
  const auto capOf = [&](std::size_t t, std::size_t i) {
    const double s = d.sigma[t * N + i];
    return std::min(capMax, capSd * (std::isfinite(s) ? s : 0.02));
  };
  std::vector<Row> rows;
  std::map<std::size_t, Pos> book;
  std::vector<std::size_t> target;
  std::vector<double> score, daily;
  std::vector<std::string> dailyDays;
  std::map<std::size_t, std::size_t> lastHeld;  // share -> the last day it was in the book
  Learnt learnt(k);
  FILE* trace = o.traceOut.empty() ? nullptr : std::fopen(o.traceOut.c_str(), "w");
  std::size_t of = 0;  // the day's tradable shares with a score
  std::size_t decided = start;  // the last rebalance: the target's day

  const auto decide = [&](std::size_t t) {
    if ((t - start) % reb != 0 && t != start) return;
    const strat::View v(d, t, &in[t * N], *backend);
    strategy->score(v, P, score);
    std::vector<std::pair<double, std::size_t>> c;
    for (std::size_t i = 0; i < N; ++i)
      if (in[t * N + i] && std::isfinite(score[i])) c.emplace_back(score[i], i);
    std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    of = c.size(), decided = t;
    target.clear();
    for (std::size_t r = 0; r < std::min(k, c.size()); ++r) target.push_back(c[r].second);
  };
  const auto inTarget = [&](std::size_t i) { return std::find(target.begin(), target.end(), i) != target.end(); };

  for (std::size_t t = start; t + 1 < T; ++t) {
    decide(t);
    const std::size_t u = t + 1;
    double day = 0;
    std::vector<std::size_t> done;
    for (auto it = book.begin(); it != book.end();) {  // the book: sold at the open when no longer wanted, else marked to the close
      const std::size_t i = it->first;
      Row& r = rows[it->second.row];
      const double O = d.O[u * N + i], C = d.Cf[u * N + i];
      if (!inTarget(i) && O > 0) {
        day += O / it->second.mark - 1;
        r.exit = O, r.ret = O / r.entry - 1, r.closed = true, r.days = u - r.since;
        done.push_back(it->second.row);
        it = book.erase(it);
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
    std::size_t free = k > book.size() ? k - book.size() : 0;
    for (std::size_t rank = 0; rank < target.size() && free > 0; ++rank) {
      const std::size_t i = target[rank];
      if (book.count(i)) continue;
      Row r;
      r.t = t, r.i = i, r.rank = rank, r.expected = learnt.expected(rank);
      r.limit = d.Cf[t * N + i] * (1 + capOf(t, i));
      if (trace) std::fprintf(trace, "%s,%s,%.6g,%.8g,%zu\n", d.days[t].c_str(), d.tickers[i].c_str(), r.limit / d.Cf[t * N + i] - 1, r.expected, rank);
      const double O = d.O[u * N + i], C = d.Cf[u * N + i];
      if (O > 0 && O <= r.limit) {
        r.filled = true, r.entry = O, r.since = u;
        const double mark = C > 0 ? C : O;
        day += mark / O - 1;
        book[i] = Pos{rows.size(), mark};
        lastHeld[i] = u;
        --free;
      }
      rows.push_back(r);
    }
    daily.push_back(day / double(k));
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
  double nFilled = 0, sumTrade = 0, nWin = 0;
  for (const auto& r : rows)
    if (r.filled) nFilled += 1, sumTrade += r.ret, nWin += r.ret > 0;

  if (const char* tl = std::getenv("X_TRADELOG")) {
    std::ofstream f(tl);
    f << "date,ticker,rank,limit,expected,filled,entry,exit,days,ret\n";
    for (const auto& r : rows)
      f << d.days[r.t + 1] << ',' << d.tickers[r.i] << ',' << r.rank << ',' << jnum(r.limit) << ',' << jnum(r.expected) << ',' << (r.filled ? 1 : 0) << ','
        << jnum(r.entry) << ',' << jnum(r.exit) << ',' << r.days << ',' << jnum(r.filled ? r.ret : 0.0) << '\n';
    std::ofstream g(std::string(tl) + ".days");
    g << "date,ret\n";
    for (std::size_t x = 0; x < daily.size(); ++x) g << dailyDays[x] << ',' << jnum(daily[x], "%.8g") << '\n';
  }

  // ------------------------------------------------------------------ the next session's orders
  struct Order { std::size_t i, rank; double close, limit, value, gross, fill, cost = 0, net = 0; bool held; };
  const std::size_t L = T - 1;
  std::vector<Order> orders;
  for (std::size_t rank = 0; rank < target.size(); ++rank) {
    const std::size_t i = target[rank];
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
  // The list in order of expected return before costs, highest first.
  std::stable_sort(orders.begin(), orders.end(), [](const Order& a, const Order& b) { return a.gross > b.gross; });
  // Then, at the end, the broker's costs: off each order's expected return; an order they make a loss is dropped.
  std::vector<Order> final;
  for (auto x : orders) {
    x.cost = costs.roundTrip(x.value, d.tickers[x.i], x.close / unit);
    x.net = x.gross - x.cost;
    if (x.net > 0) final.push_back(x);
  }

  // ------------------------------------------------------------------ report
  const std::string cur = uk ? "GBP" : M == "US" || M == "FOREX" ? "USD" : M == "APAC" ? "AUD" : "EUR";
  std::printf("strategy: %s (%s) on %s, back end %s; %zu shares, %s .. %s%s%s\n", o.strategy.c_str(), strategy->about().c_str(), M.c_str(), backend->about().c_str(), N,
              d.days.front().c_str(), d.days.back().c_str(), note.empty() ? "" : "; ", note.c_str());
  if (units) std::printf("strategy: %zu bars put in their share's latest price unit (pence and pounds)\n", units);
  std::printf("strategy: the account (%s .. %s, %zu slots of %s %.0f, rebalanced every %zu sessions): %+.1f%% a year, Sharpe %.2f, max drawdown %.1f%%, before costs\n",
              dailyDays.empty() ? "-" : dailyDays.front().c_str(), dailyDays.empty() ? "-" : dailyDays.back().c_str(), k, cur.c_str(), o.capital, reb, 100 * annual, sharpe, 100 * dd);
  std::printf("strategy:   %zu orders, %.0f bought (%.0f%%); a trade on average %+.2f%% before costs, %.0f%% gained, held %.1f sessions\n", rows.size(), nFilled,
              rows.empty() ? 0.0 : 100 * nFilled / double(rows.size()), nFilled ? 100 * sumTrade / nFilled : 0.0, nFilled ? 100 * nWin / nFilled : 0.0, learnt.holdDays(double(reb)));
  std::printf("strategy: next session (after the close of %s; the list from the rebalance after the close of %s): %zu on the list, %zu after costs\n", d.days[L].c_str(),
              d.days[decided].c_str(), orders.size(), final.size());
  for (const auto& x : orders)
    std::printf("strategy:   %-12s rank %zu: expected %+.2f%% before costs, costs %.2f%%: %+.2f%% after%s%s\n", d.tickers[x.i].c_str(), x.rank + 1, 100 * x.gross,
                100 * costs.roundTrip(x.value, d.tickers[x.i], x.close / unit), 100 * (x.gross - costs.roundTrip(x.value, d.tickers[x.i], x.close / unit)),
                x.gross - costs.roundTrip(x.value, d.tickers[x.i], x.close / unit) > 0 ? "" : " (dropped)", x.held ? " (held already)" : "");

  const double holdDays = learnt.holdDays(double(reb));
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
      << ",\"holdDays\":" << jnum(holdDays) << "}";
    f << ",\"orders\":[";
    for (std::size_t q = 0; q < final.size(); ++q) {
      const auto& x = final[q];
      const double fill = std::isfinite(x.fill) ? x.fill : 1.0;
      f << (q ? "," : "") << "{\"ticker\":" << jstr(d.tickers[x.i]) << ",\"rank\":" << x.rank + 1 << ",\"of\":" << of << ",\"entry\":\"open\",\"limit\":" << jnum(x.limit / x.close - 1)
        << ",\"stop\":null,\"take\":null,\"exitAbove\":null,\"days\":" << maxDays << ",\"holdDays\":" << jnum(holdDays) << ",\"gross\":" << jnum(x.gross)
        << ",\"cost\":" << jnum(x.cost) << ",\"ifFilled\":" << jnum(x.net) << ",\"fill\":" << jnum(x.fill) << ",\"expected\":" << jnum(fill * x.net)
        << ",\"amount\":" << jnum(x.value, "%.0f") << ",\"held\":" << (x.held ? "true" : "false") << ",\"sigma\":" << jnum(d.sigma[L * N + x.i])
        << ",\"score\":" << jnum(score[x.i]) << ",\"reason\":" << jstr(strategy->reason(last, P, x.i)) << "}";
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
        << ",\"stop\":null,\"take\":null,\"exitAbove\":null,\"days\":" << maxDays << ",\"holdDays\":" << jnum(holdDays) << ",\"expected\":" << jnum(fill * x.net)
        << ",\"ifFilled\":" << jnum(x.net) << ",\"gross\":" << jnum(x.gross) << ",\"fill\":" << jnum(std::isfinite(x.fill) ? x.fill : 0.0) << ",\"capital\":" << jnum(x.value, "%.2f")
        << ",\"maxCapital\":" << jnum(noVolume ? o.capital : o.advShare * d.value[L * N + x.i], "%.2f") << ",\"rebuy\":0,\"cost\":" << jnum(x.cost)
        << ",\"costAmount\":" << jnum(x.cost * x.value, "%.4g") << ",\"stampFree\":" << (costs.stampFree.count(d.tickers[x.i]) ? "true" : "false") << ",\"rank\":" << x.rank + 1 << "}";
    }
    // The daily update: each share the book held lately, kept while it is in the target, else sold at the open.
    f << "],\"updates\":[";
    std::size_t q = 0;
    std::set<std::size_t> named;
    for (std::size_t i : target) named.insert(i);
    for (const auto& [i, day] : lastHeld)
      if (L - std::min(L, day) <= 260) named.insert(i);
    for (std::size_t i : named) f << (q++ ? "," : "") << "{\"ticker\":" << jstr(d.tickers[i]) << ",\"keep\":" << (inTarget(i) ? "true" : "false") << ",\"exitAbove\":null}";
    f << "]}\n";
  }
  return 0;
}
