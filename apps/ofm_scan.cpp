// ofm_scan: the daily scan's orchestration, in C++ (it replaces tools/scan/strategies.py and tools/report/daily_email.py).
//
//   ofm_scan run   --market UK [--config config/markets.json] [--publish-orders] [--fx data/futures.csv] [--engine ofm_strategy]
//                  -- <the engine's arguments: --csv data/stocks.csv --costs UK ...>
//       The market's live strategy (config/markets.json) on the strategy engine. With a budget: the strategy run once for
//       each book size k (T/k each, in waves the memory holds), each day the k whose year before was best (walk-forward),
//       the market's share of the account's limit (the split that maximises the account's risk-adjusted return over the
//       markets' records before costs and their outlook), a book no larger than the money allows, the final orders run at
//       the amount the bridge places (costs at the end), and the opposite picks (the strategy's own edge). Writes
//       report/ofm.json, report/ofm.txt, report/budget.json, report/strategies.json and, with --publish-orders, the
//       bridge's orders to data/eoddata/orders/<MARKET>.json.
//   ofm_scan track report/ofm.json web/data/track.json         the scan's last day added to the live record (once a date)
//   ofm_scan mail  report/ofm.json web/data/track.json [--print] the daily report by SMTP (SMTP_SERVER, SMTP_PORT,
//                  SMTP_USERNAME, SMTP_PASSWORD, MAIL_TO, MAIL_FROM); without SMTP_SERVER printed, nothing sent
//   ofm_scan index report/strategies.json web/data/strategies.json   the market's entry in the page's index
//   ofm_scan test  the money split's checks (was tools/scan/test_allocate.py)
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <random>
#include <regex>
#include <set>

#include "eod/common.hpp"
#include "eod/http.hpp"

extern char** environ;

using namespace eod;
using ojson = nlohmann::ordered_json;

namespace {

const double MIN_ORDER = 500.0;  // the strategy engine's smallest order, in the market's currency
const std::size_t SPLIT_WINDOW = 20;  // the split's memory: four 5-day holds
const double HALF_LIFE = 10;          // a day this many sessions old counts half as much as the last
const std::map<std::string, std::string> CURRENCY{{"UK", "GBP"}, {"EU", "EUR"}, {"US", "USD"}, {"APAC", "AUD"}, {"INDEX", "EUR"}, {"FOREX", "USD"}};

[[noreturn]] void die(const std::string& why, int code = 1) {
  std::cout.flush();
  std::cerr << "strategies: " << why << "\n";
  std::exit(code);
}

std::string g(double x) {  // Python's {:g}
  char b[64];
  std::snprintf(b, sizeof b, "%g", x);
  return b;
}
std::string thousands(double x, int decimals = 0) {  // Python's {:,.Nf}
  char b[64];
  std::snprintf(b, sizeof b, "%.*f", decimals, std::fabs(x));
  std::string s = b, intPart = s.substr(0, s.find('.')), frac = s.find('.') == std::string::npos ? "" : s.substr(s.find('.'));
  std::string out;
  for (std::size_t k = 0; k < intPart.size(); ++k) {
    if (k && (intPart.size() - k) % 3 == 0) out += ',';
    out += intPart[k];
  }
  return (x < 0 ? "-" : "") + out + frac;
}
std::string f(const char* fmt, double x) {
  char b[64];
  std::snprintf(b, sizeof b, fmt, x);
  return b;
}

// ------------------------------------------------------------------ the money split (pure functions)

std::vector<double> ewWeights(std::size_t n, double halfLife) {
  std::vector<double> w(n, 1.0);
  if (halfLife > 0)
    for (std::size_t i = 0; i < n; ++i) w[i] = std::pow(0.5, double(n - 1 - i) / halfLife);
  return w;
}

// The mean shrunk by its t statistic (a mean of no significance to 0); with a half-life, exponentially weighted.
double shrunk(const std::vector<double>& x, double halfLife = 0) {
  const std::size_t n = x.size();
  if (n < 2) return 0.0;
  const auto w = ewWeights(n, halfLife);
  double sw = 0, sw2 = 0, m = 0;
  for (std::size_t i = 0; i < n; ++i) sw += w[i], sw2 += w[i] * w[i], m += w[i] * x[i];
  m /= sw;
  double v = 0;
  for (std::size_t i = 0; i < n; ++i) v += w[i] * (x[i] - m) * (x[i] - m);
  v /= sw;
  const double neff = sw * sw / sw2, t = v > 0 ? m / std::sqrt(v) * std::sqrt(neff) : 0.0;
  return t != 0 ? m * std::max(0.0, 1 - 1 / (t * t)) : 0.0;
}

std::vector<double> solve(std::vector<std::vector<double>> m, std::vector<double> b) {  // Gauss-Jordan with pivoting
  const std::size_t n = b.size();
  for (std::size_t r = 0; r < n; ++r) m[r].push_back(b[r]);
  for (std::size_t c = 0; c < n; ++c) {
    std::size_t p = c;
    for (std::size_t r = c; r < n; ++r)
      if (std::fabs(m[r][c]) > std::fabs(m[p][c])) p = r;
    std::swap(m[c], m[p]);
    if (std::fabs(m[c][c]) < 1e-18) return std::vector<double>(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
      if (r == c) continue;
      const double fct = m[r][c] / m[c][c];
      for (std::size_t j = 0; j <= n; ++j) m[r][j] -= fct * m[c][j];
    }
  }
  std::vector<double> x(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = m[i][n] / m[i][i];
  return x;
}

using Record = std::map<std::string, double>;  // date -> the day's result

// The account's limit split over the markets to maximise its risk-adjusted return: each market's record gives its
// shrunk mean (half of it the outlook, where given) and the covariance, shrunk halfway to its diagonal; the
// mean-variance weights, none negative, adding up to the whole, none over the cap; on paper each at least the floor
// (floors: a market's own floor, where given, in place of floor_).
std::map<std::string, double> allocate(const std::map<std::string, Record>& records, double cap = 0.5, double floor_ = 0.0, const std::map<std::string, double>* forward = nullptr,
                                       double blend = 0.5, double halfLife = HALF_LIFE, std::size_t window = SPLIT_WINDOW,
                                       const std::map<std::string, double>* floors = nullptr) {
  const auto floorOf = [&](const std::string& m) { return floors && floors->count(m) ? floors->at(m) : floor_; };
  std::vector<std::string> names;
  for (const auto& [m, r] : records) names.push_back(m);
  if (names.empty()) return {};
  std::set<std::string> all;
  for (const auto& [m, r] : records)
    for (const auto& [d, v] : r) all.insert(d);
  std::vector<std::string> dates(all.begin(), all.end());
  if (dates.size() > window) dates.erase(dates.begin(), dates.end() - long(window));
  std::map<std::string, std::vector<double>> x;
  for (const auto& m : names) {
    for (const auto& d : dates) {
      const auto it = records.at(m).find(d);
      x[m].push_back(it == records.at(m).end() ? 0.0 : it->second);
    }
  }
  const std::size_t n = dates.size();
  std::map<std::string, double> mu, mean, w;
  for (const auto& m : names) {
    mu[m] = shrunk(x[m], halfLife);
    if (forward && forward->count(m)) mu[m] = (1 - blend) * mu[m] + blend * forward->at(m);
  }
  const auto ew = ewWeights(n, halfLife);
  double sw = 0;
  for (double e : ew) sw += e;
  if (sw == 0) sw = 1;
  for (const auto& m : names) {
    double s = 0;
    for (std::size_t i = 0; i < n; ++i) s += ew[i] * x[m][i];
    mean[m] = n ? s / sw : 0.0;
  }
  const auto cov = [&](const std::string& a, const std::string& b) {
    if (n < 2) return 0.0;
    double s = 0;
    for (std::size_t i = 0; i < n; ++i) s += ew[i] * (x[a][i] - mean[a]) * (x[b][i] - mean[b]);
    return s / sw;
  };
  std::vector<std::string> active;
  for (const auto& m : names) {
    w[m] = 0.0;
    if (mu[m] > 0 && cov(m, m) > 0) active.push_back(m);
  }
  while (!active.empty()) {  // the positive solution: a market given a negative weight dropped, solved again
    std::vector<std::vector<double>> a(active.size(), std::vector<double>(active.size()));
    std::vector<double> b;
    for (std::size_t p = 0; p < active.size(); ++p) {
      for (std::size_t q = 0; q < active.size(); ++q) a[p][q] = cov(active[p], active[q]) * (p == q ? 1.0 : 0.5);
      b.push_back(mu[active[p]]);
    }
    const auto sol = solve(a, b);
    std::vector<std::string> keep;
    for (std::size_t k = 0; k < active.size(); ++k)
      if (sol[k] > 0) keep.push_back(active[k]);
    if (keep.size() == active.size()) {
      for (std::size_t k = 0; k < active.size(); ++k) w[active[k]] = sol[k];
      break;
    }
    active = keep;
  }
  double total = 0;
  for (const auto& [m, v] : w) total += v;
  if (total > 0)
    for (auto& [m, v] : w) v /= total;
  for (std::size_t it = 0; it < names.size(); ++it) {  // none over the cap: the excess to the others in proportion
    std::map<std::string, double> over, rest;
    for (const auto& [m, v] : w) {
      if (v > cap + 1e-12) over[m] = v - cap;
      else if (v < cap - 1e-12 && v > 0) rest[m] = v;
    }
    if (over.empty()) break;
    double spare = 0;
    for (const auto& [m, v] : over) w[m] = cap, spare += v;
    if (rest.empty()) break;
    double t = 0;
    for (const auto& [m, v] : rest) t += v;
    for (const auto& [m, v] : rest) w[m] = v + spare * v / t;
  }
  bool anyFloor = floor_ > 0;
  if (floors)
    for (const auto& [m, f] : *floors) anyFloor = anyFloor || f > 0;
  if (anyFloor) {  // the floors from the limit left unused first, then from the others
    double lift = 0, sumW = 0, big = 0;
    for (const auto& [m, v] : w) lift += std::max(0.0, floorOf(m) - v), sumW += v, big += v > floorOf(m) ? v : 0.0;
    const double need = std::max(0.0, lift - std::max(0.0, 1.0 - sumW));
    for (auto& [m, v] : w) v = floorOf(m) - v > 0 ? floorOf(m) : v - need * (big > 0 ? v / big : 0);
  }
  return w;
}

// ------------------------------------------------------------------ the scan's files

std::map<std::string, double> fxRates(const std::string& path) {  // each currency's units for one EUR
  std::map<std::string, std::pair<std::string, double>> last;
  std::string text;
  if (tryRead(path, text)) {
    std::istringstream in(text);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto c = split(line, ',');
      if (c.size() >= 6 && c[1].rfind("FOREX:", 0) == 0 && std::atof(c[5].c_str()) > 0 && c[0] >= last[c[1]].first) last[c[1]] = {c[0], std::atof(c[5].c_str())};
    }
  }
  std::map<std::string, double> r{{"EUR", 1.0}};
  if (last.count("FOREX:EURGBP") && last["FOREX:EURGBP"].second > 0) {
    r["GBP"] = last["FOREX:EURGBP"].second;
    if (last.count("FOREX:GBPUSD") && last["FOREX:GBPUSD"].second > 0) r["USD"] = r["GBP"] * last["FOREX:GBPUSD"].second;
  }
  if (last.count("FOREX:EURAUD") && last["FOREX:EURAUD"].second > 0) r["AUD"] = last["FOREX:EURAUD"].second;
  return r;
}

Record daysOf(const std::string& path) {  // a run's daily results (X_TRADELOG.days: date,ret)
  Record out;
  std::istringstream in(readFile(path));
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    const auto c = split(line, ',');
    if (c.size() >= 2) out[c[0]] = std::atof(c[1].c_str());
  }
  return out;
}

struct Choice {
  std::vector<std::string> days;
  std::vector<int> picks;
  std::vector<double> meta;
  int next = 0;
};

// Walk-forward: each day the k with the best shrunk mean over the window before it (the most trades before `least` days).
Choice choose(const std::map<int, Record>& series, std::size_t window = 250, std::size_t least = 60) {
  Choice c;
  std::vector<int> ks;
  for (const auto& [k, r] : series) ks.push_back(k);
  std::set<std::string> common;
  bool first = true;
  for (const auto& [k, r] : series) {
    std::set<std::string> ds;
    for (const auto& [d, v] : r) ds.insert(d);
    if (first) common = ds, first = false;
    else {
      std::set<std::string> both;
      std::set_intersection(common.begin(), common.end(), ds.begin(), ds.end(), std::inserter(both, both.begin()));
      common = both;
    }
  }
  c.days.assign(common.begin(), common.end());
  const auto best = [&](std::size_t upto) {
    const std::size_t lo = upto > window ? upto - window : 0;
    if (upto - lo < least) return ks.back();
    int bk = ks.back();
    double bv = -1e300;
    for (auto it = ks.rbegin(); it != ks.rend(); ++it) {  // ties to the most trades (Python's max over reversed ks)
      std::vector<double> x;
      for (std::size_t i = lo; i < upto; ++i) x.push_back(series.at(*it).at(c.days[i]));
      const double v = shrunk(x);
      if (v > bv) bv = v, bk = *it;
    }
    return bk;
  };
  for (std::size_t i = 0; i < c.days.size(); ++i) {
    const int k = best(i);
    c.picks.push_back(k), c.meta.push_back(series.at(k).at(c.days[i]));
  }
  c.next = best(c.days.size());
  return c;
}

std::pair<double, double> stats(const std::vector<double>& x) {
  const std::size_t n = x.size();
  if (n < 2) return {0.0, 0.0};
  double m = 0;
  for (double y : x) m += y;
  m /= double(n);
  double v = 0;
  for (double y : x) v += (y - m) * (y - m);
  v /= double(n);
  return {252 * m, v > 0 ? m / std::sqrt(v) * std::sqrt(252.0) : 0.0};
}

ojson readOJson(const std::string& p) { return ojson::parse(readFile(p)); }
bool tryOJson(const std::string& p, ojson& out) {
  std::string t;
  if (!tryRead(p, t)) return false;
  try {
    out = ojson::parse(t);
    return true;
  } catch (...) {
    return false;
  }
}

std::map<std::string, ojson> publishedBudgets(const ojson& config) {
  std::map<std::string, ojson> out;
  for (auto it = config["markets"].begin(); it != config["markets"].end(); ++it) {
    ojson b;
    if (tryOJson("web/data/budget-" + lower(it.key()) + ".json", b)) out[it.key()] = b;
  }
  return out;
}
bool fresh(const ojson& b, int days = 10) {
  const std::string d = b.value("date", std::string());
  return d.size() >= 10 && isDay(d.substr(0, 10)) && daysBetween(d.substr(0, 10), today()) <= days;
}

// The market's outlook: its final orders' expected daily return before costs (times their chance to fill, per day held,
// over the book's slots), discounted by what the backtest delivered of its expectation lately (0 to 1).
std::tuple<double, double, double> forwardOf(const std::string& ordersPath, const std::string& tradelogPath, const std::string& since) {
  ojson book;
  if (!tryOJson(ordersPath, book)) return {0.0, 0.0, 0.0};
  const auto& orders = book.contains("orders") ? book["orders"] : ojson::array();
  const double slots = std::max(1.0, double(book.value("slots", 0) ? book.value("slots", 0) : (orders.empty() ? 1 : int(orders.size()))));
  double exp = 0;
  for (const auto& o : orders) {
    const auto gv = [&](const char* k) { return o.contains(k) && o[k].is_number() ? o[k].get<double>() : 0.0; };
    exp += (gv("expected") + gv("fill") * gv("cost")) / std::max(1.0, gv("holdDays") > 0 ? gv("holdDays") : 1.0);
  }
  exp /= slots;
  std::vector<std::tuple<std::string, double, double>> rows;
  std::string text;
  if (tryRead(tradelogPath, text)) {
    std::istringstream in(text);
    std::string line;
    std::getline(in, line);
    const auto head = split(line, ',');
    const auto col = [&](const char* k) { return long(std::find(head.begin(), head.end(), k) - head.begin()); };
    const long cd = col("date"), ce = col("expected"), cf = col("filled"), cr = col("ret");
    while (std::getline(in, line)) {
      const auto c = split(line, ',');
      if (long(c.size()) > std::max({cd, ce, cf, cr}) && c[size_t(cd)] >= since && std::atof(c[size_t(cf)].c_str()) > 0)
        rows.emplace_back(c[size_t(cd)], std::atof(c[size_t(cr)].c_str()), std::atof(c[size_t(ce)].c_str()));
    }
  }
  std::set<std::string> ds;
  for (const auto& r : rows) ds.insert(std::get<0>(r));
  std::map<std::string, double> age;
  std::size_t k = 0;
  for (const auto& d : ds) age[d] = double(ds.size() - 1 - k++);
  double got = 0, want = 0;
  for (const auto& [d, r, e] : rows) got += std::pow(0.5, age[d] / HALF_LIFE) * r, want += std::pow(0.5, age[d] / HALF_LIFE) * e;
  const double delivered = want > 0 ? std::min(1.0, std::max(0.0, got / want)) : 0.0;
  return {exp * delivered, exp, delivered};
}

std::map<std::string, std::vector<double>> ordersOf(const std::string& path) {  // each day's orders' results (unfilled 0)
  std::map<std::string, std::vector<double>> out;
  std::string text;
  if (!tryRead(path, text)) return out;
  std::istringstream in(text);
  std::string line;
  std::getline(in, line);
  const auto head = split(line, ',');
  const auto col = [&](const char* k) { return std::size_t(std::find(head.begin(), head.end(), k) - head.begin()); };
  const std::size_t cd = col("date"), cr = col("ret"), cf = col("filled");
  while (std::getline(in, line)) {
    const auto c = split(line, ',');
    if (c.size() > std::max({cd, cr, cf})) out[c[cd]].push_back(std::atof(c[cf].c_str()) > 0 ? std::atof(c[cr].c_str()) : 0.0);
  }
  return out;
}

std::tuple<double, double, double, double> expectedOf(const std::vector<std::string>& days, const std::vector<int>& picks, std::map<int, std::map<std::string, std::vector<double>>>& orders) {
  std::vector<double> per;
  double n = 0, filled = 0;
  for (std::size_t i = 0; i < days.size(); ++i) {
    const auto& o = orders[picks[i]][days[i]];
    double s = 0;
    for (double x : o) s += x, filled += x != 0;
    per.push_back(o.empty() ? 0.0 : s / double(o.size()));
    n += double(o.size());
  }
  if (per.empty()) return {0, 0, 0, 0};
  double daily = 0;
  for (double x : per) daily += x;
  daily /= double(per.size());
  return {daily, std::pow(1 + daily, 252) - 1, n / double(per.size()), n > 0 ? filled / n : 0.0};
}

// ------------------------------------------------------------------ running the engine

struct Proc {
  pid_t pid = -1;
};

// The command with the environment (this one's, less X_*, plus `env`), its output to `log`.
Proc spawn(const std::vector<std::string>& cmd, const std::map<std::string, std::string>& env, const std::string& log) {
  std::vector<std::string> envs;
  for (char** e = environ; *e; ++e)
    if (std::string(*e).rfind("X_", 0) != 0) envs.push_back(*e);
  for (const auto& [k, v] : env) envs.push_back(k + "=" + v);
  std::vector<char*> argv, envp;
  for (const auto& c : cmd) argv.push_back(const_cast<char*>(c.c_str()));
  argv.push_back(nullptr);
  for (auto& e : envs) envp.push_back(const_cast<char*>(e.c_str()));
  envp.push_back(nullptr);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  posix_spawn_file_actions_adddup2(&fa, 1, 2);
  Proc p;
  if (posix_spawnp(&p.pid, argv[0], &fa, nullptr, argv.data(), envp.data()) != 0) p.pid = -1;
  posix_spawn_file_actions_destroy(&fa);
  return p;
}
int waitFor(Proc p) {
  if (p.pid < 0) return 127;
  int st = 0;
  waitpid(p.pid, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 128;
}

// The engine's arguments with --capital at amount, and --venues' amounts scaled to it.
std::vector<std::string> withCapital(std::vector<std::string> a, double amount, const std::string& tag) {
  double base = 0;
  const auto it = std::find(a.begin(), a.end(), "--capital");
  if (it != a.end() && it + 1 != a.end()) base = std::atof((it + 1)->c_str()), *(it + 1) = g(amount);
  else a.push_back("--capital"), a.push_back(g(amount));
  const auto v = std::find(a.begin(), a.end(), "--venues");
  if (v != a.end() && v + 1 != a.end() && base > 0) {
    const std::string out = "data/budget/venues-" + tag + ".txt";
    std::istringstream in(readFile(*(v + 1)));
    std::string s, line;
    while (std::getline(in, line)) {
      std::istringstream w(line);
      std::vector<std::string> p;
      for (std::string x; w >> x;) p.push_back(x);
      if (line.rfind('#', 0) == 0 || p.size() < 4) {
        s += line + "\n";
        continue;
      }
      p[1] = g(std::atof(p[1].c_str()) * amount / base);
      for (std::size_t k = 0; k < p.size(); ++k) s += (k ? " " : "") + p[k];
      s += "\n";
    }
    writeFile(out, s);
    *(v + 1) = out;
  }
  return a;
}

int cmdRun(const std::vector<std::string>& args) {
  std::vector<std::string> mine, model;
  bool after = false;
  for (const auto& x : args) {
    if (!after && x == "--") {
      after = true;
      continue;
    }
    (after ? model : mine).push_back(x);
  }
  const std::string market = opt(mine, "market"), configPath = opt(mine, "config", "config/markets.json"), fxPath = opt(mine, "fx", "data/futures.csv"),
                    engineBin = opt(mine, "engine", "ofm_strategy");
  const bool publish = flag(mine, "publish-orders");
  if (market.empty()) die("--market");
  const ojson config = readOJson(configPath);
  if (!config["markets"].contains(market)) die(market + ": not in " + configPath);
  const ojson& mk = config["markets"][market];
  ojson s = mk["live"];
  const std::regex NAME("^[a-z0-9][a-z0-9-]{0,40}$");
  if (!std::regex_match(s.value("name", std::string()), NAME)) die(market + ": a bad name");
  if (!std::regex_match(s.value("strategy", std::string()), NAME)) die(market + " " + s.value("name", std::string()) + ": no strategy (one of ofm_strategy --list)");
  if (s.contains("only") && s["only"] != "aim") die(market + ": an unknown share filter");
  for (const auto& x : s.value("args", ojson::array()))
    if (!x.is_string() || !std::regex_match(x.get<std::string>(), std::regex("^[A-Za-z0-9_.,:/=-]{1,64}$"))) die(market + ": bad args");
  if (mk.contains("shadows") && !mk["shadows"].empty()) std::cout << "strategies: " << market << ": shadows are not run (live only)\n";
  // The engine's command for the strategy: its name, back end and parameters.
  std::vector<std::string> engine{engineBin, "--strategy", s["strategy"].get<std::string>()};
  if (s.contains("backend")) engine.push_back("--backend"), engine.push_back(s["backend"].get<std::string>());
  std::string params;
  if (s.contains("params"))
    for (auto it = s["params"].begin(); it != s["params"].end(); ++it) {
      if (!it.value().is_number() || !std::regex_match(it.key(), std::regex("^[a-z][a-z0-9_]{0,30}$"))) die(market + ": bad params");
      engine.push_back("--param"), engine.push_back(it.key() + "=" + g(it.value().get<double>()));
      params += (params.empty() ? "" : " ") + it.key() + "=" + g(it.value().get<double>());
    }
  const std::string describe = s["strategy"].get<std::string>() + (params.empty() ? "" : " (" + params + ")") + (s.contains("backend") ? " on " + s["backend"].get<std::string>() : "");
  std::map<std::string, std::string> senv;
  if (s.contains("env"))
    for (auto it = s["env"].begin(); it != s["env"].end(); ++it) {
      if (!std::regex_match(it.key(), std::regex("^X_[A-Z0-9_]+$"))) die(market + ": only X_* switches may be set");
      senv[it.key()] = it.value().is_string() ? it.value().get<std::string>() : it.value().dump();
    }
  std::vector<std::string> argsM = model;
  if (s.value("only", std::string()) == "aim") {  // the UK's AIM shares only (no stamp duty)
    const auto c = std::find(argsM.begin(), argsM.end(), "--csv");
    if (c != argsM.end() && c + 1 != argsM.end()) {
      const std::string in = *(c + 1), out = in + ".aim.csv";
      if (!fs::exists(out)) {
        std::set<std::string> aim;
        std::istringstream al(readFile("tools/eoddata/aim.txt"));
        for (std::string l; std::getline(al, l);)
          if (!l.empty()) aim.insert(split(l, ',')[0]);
        std::istringstream csv(readFile(in));
        std::string o, line;
        std::getline(csv, line);
        o += line + "\n";
        while (std::getline(csv, line)) {
          const auto p = split(line, ',');
          if (p.size() > 1 && aim.count(p[1])) o += line + "\n";
        }
        writeFile(out, o);
      }
      *(c + 1) = out;
    }
  }
  for (const auto& x : s.value("args", ojson::array())) argsM.push_back(x.get<std::string>());
  fs::create_directories("report");
  ojson index{{"market", market}, {"strategies", ojson::array()}};
  const auto entry = [&](const ojson& extra) {
    ojson e{{"name", s["name"]}, {"role", "live"}, {"env", s.value("env", ojson::object())}, {"strategy", s["strategy"]}, {"params", s.value("params", ojson::object())},
            {"only", s.contains("only") ? s["only"] : ojson()}, {"report", "ofm.json"}};
    for (auto it = extra.begin(); it != extra.end(); ++it) e[it.key()] = it.value();
    return e;
  };
  if (!mk.contains("budget")) {  // no budget (FOREX: research only): the live strategy once
    std::vector<std::string> cmd = engine;
    cmd.insert(cmd.end(), argsM.begin(), argsM.end());
    cmd.push_back("--json"), cmd.push_back("report/ofm.json");
    if (publish) {
      fs::create_directories("data/eoddata/orders");
      cmd.push_back("--orders"), cmd.push_back("data/eoddata/orders/" + market + ".json");
    }
    auto env = senv;
    env["X_VERIFY"] = "1";
    const int code = waitFor(spawn(cmd, env, "report/ofm.txt"));
    std::cout << "strategies: " << market << " live " << s["name"].get<std::string>() << ": " << describe << "\n" << readFile("report/ofm.txt");
    if (code) die("the live strategy " + s["name"].get<std::string>() + " failed (" + std::to_string(code) + ")");
    index["strategies"].push_back(entry(ojson::object()));
    writeFile("report/strategies.json", index.dump(1));
    return 0;
  }
  // ------------------------------------------------------------------ the budget
  const ojson& b = mk["budget"];
  double total = b.value("total", 0.0);
  std::vector<int> slots;
  for (const auto& k : b.value("slots", ojson::array({2, 3, 5, 10})))
    if (k.get<int>() > 0) slots.push_back(k.get<int>());
  std::sort(slots.begin(), slots.end());
  slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
  if (total <= 0 || slots.empty()) die(market + ": a bad budget");
  fs::create_directories("data/budget");
  const ojson acc = config.value("account", ojson::object());
  const double daily = std::max(0.0, acc.value("daily_max", 0.0) - acc.value("reserve", 0.0));
  const std::string cur = acc.value("currency", std::string("EUR"));
  const bool paper = acc.value("paper", false);
  const auto rates = fxRates(fxPath);
  const std::string mcur = CURRENCY.at(market);
  if (daily > 0) {  // the runs at the whole limit (the record then shares it out)
    if (cur != "EUR" || !rates.count(mcur)) die(market + ": no rate for " + mcur + " (or the limit's currency " + cur + " is not EUR)");
    total = daily * rates.at(mcur);
  }
  const std::vector<std::string> finalArgs = argsM;
  // The runs the memory allows at once (a large market's run holds about 1 GB per million share-days).
  std::size_t rows = 0;
  const auto c = std::find(model.begin(), model.end(), "--csv");
  if (c != model.end() && c + 1 != model.end()) {
    std::ifstream in(*(c + 1));
    std::string l;
    while (std::getline(in, l)) ++rows;
  }
  double avail = 0;
  {
    std::ifstream mi("/proc/meminfo");
    for (std::string l; std::getline(mi, l);)
      if (l.rfind("MemAvailable", 0) == 0) avail = std::atof(l.substr(l.find(':') + 1).c_str()) * 1024;
  }
  const double per = double(std::max<std::size_t>(1, rows)) * 1000;
  const std::size_t width = avail > 0 ? std::max<std::size_t>(1, std::min<std::size_t>(slots.size(), std::size_t(0.7 * avail / per))) : slots.size();
  std::cout << "strategies: " << market << ": " << slots.size() << " budget runs, " << width << " at a time (" << rows << " share-days, " << f("%.1f", avail / 1073741824.0) << " GB free)\n";
  std::cout.flush();
  {
    std::vector<std::pair<int, Proc>> running;
    std::size_t next = 0;
    while (next < slots.size() || !running.empty()) {
      while (next < slots.size() && running.size() < width) {
        const int k = slots[next++];
        auto env = senv;
        env["X_VERIFY"] = "1", env["X_NPICK"] = std::to_string(k), env["X_TRADELOG"] = "data/budget/trades-" + std::to_string(k) + ".csv";
        std::vector<std::string> cmd = engine;
        const auto wa = withCapital(argsM, total / k, std::to_string(k));
        cmd.insert(cmd.end(), wa.begin(), wa.end());
        for (const char* x : {"--json", "--orders"}) cmd.push_back(x), cmd.push_back(std::string("data/budget/") + (x[2] == 'j' ? "report-" : "orders-") + std::to_string(k) + ".json");
        running.emplace_back(k, spawn(cmd, env, "data/budget/report-" + std::to_string(k) + ".txt"));
      }
      const auto [k, p] = running.front();
      running.erase(running.begin());
      const int code = waitFor(p);
      if (code) {
        std::cout << readFile("data/budget/report-" + std::to_string(k) + ".txt");
        for (auto& r : running) kill(r.second.pid, SIGKILL), waitFor(r.second);
        die("the live strategy " + s["name"].get<std::string>() + " with " + std::to_string(k) + " trades failed (" + std::to_string(code) + ")");
      }
    }
  }
  std::map<int, Record> series;
  for (int k : slots) series[k] = daysOf("data/budget/trades-" + std::to_string(k) + ".csv.days");
  Choice ch = choose(series);
  const auto& days = ch.days;
  int nxt = ch.next;
  const double runTotal = total;
  Record record;  // the last year's daily results of the choice, before costs (the split's record)
  for (std::size_t i = days.size() > 250 ? days.size() - 250 : 0; i < days.size(); ++i) record[days[i]] = ch.meta[i];
  const std::vector<double> lastMeta(ch.meta.end() - long(std::min<std::size_t>(SPLIT_WINDOW, ch.meta.size())), ch.meta.end());
  const double score = shrunk(lastMeta, HALF_LIFE);
  const auto [outlook, fexp, fdel] = forwardOf("data/budget/orders-" + std::to_string(nxt) + ".json", "data/budget/trades-" + std::to_string(nxt) + ".csv",
                                               days.size() >= SPLIT_WINDOW ? days[days.size() - SPLIT_WINDOW] : "");
  double share = 1.0, floor_ = 0.0;
  std::map<std::string, double> alloc;
  if (daily > 0) {  // this market's share of the account's limit
    const auto pub = publishedBudgets(config);
    std::map<std::string, Record> recs;
    std::map<std::string, double> held, fwd;
    for (const auto& [m, bb] : pub) {
      if (m == market || !fresh(bb)) continue;
      if (bb.contains("record") && bb["record"].is_object() && !bb["record"].empty()) {
        for (auto it = bb["record"].begin(); it != bb["record"].end(); ++it) recs[m][it.key()] = it.value().get<double>();
        if (bb.contains("forward") && bb["forward"].is_object()) fwd[m] = bb["forward"].value("outlook", 0.0);
      } else {
        held[m] = bb.value("share", 0.0);
      }
    }
    recs[market] = record, fwd[market] = outlook;
    double room = 1.0;
    for (const auto& [m, v] : held) room -= v;
    room = std::max(0.0, room);
    floor_ = paper ? acc.value("paper_floor", 0.05) : 0.0;
    std::map<std::string, double> floors;  // paper_floors: a market's own floor on paper ({"US": 0.3, "UK": 0.1})
    if (paper && acc.contains("paper_floors") && acc["paper_floors"].is_object())
      for (auto it = acc["paper_floors"].begin(); it != acc["paper_floors"].end(); ++it) floors[it.key()] = it.value().get<double>();
    for (const auto& [m, v] : allocate(recs, acc.value("market_cap", 0.5), floor_, &fwd, 0.5, HALF_LIFE, SPLIT_WINDOW, &floors)) alloc[m] = v * room;
    share = alloc.count(market) ? alloc[market] : 0.0;
    if (!paper) {  // growing only into its part of what is free
      std::vector<std::string> withBudget;
      for (auto it = config["markets"].begin(); it != config["markets"].end(); ++it)
        if (it.value().contains("budget")) withBudget.push_back(it.key());
      double others = 0;
      std::size_t growers = 1;
      for (const auto& m : withBudget) {
        if (m == market) continue;
        double sh = 1.0 / double(withBudget.size());
        if (pub.count(m) && fresh(pub.at(m))) sh = pub.at(m).value("share", 0.0);
        others += sh;
        const bool grows = !(pub.count(m) && fresh(pub.at(m)) && pub.at(m).contains("score")) || pub.at(m)["score"].get<double>() > 0;
        growers += grows;
      }
      double own = 1.0 / double(std::max<std::size_t>(1, withBudget.size()));
      if (pub.count(market) && pub.at(market).value("share", 0.0) > 0) own = pub.at(market)["share"].get<double>();
      const double freeRoom = std::max(0.0, 1.0 - own - others);
      share = std::min({share, own + freeRoom / double(growers), std::max(0.0, 1.0 - others)});
    }
    total = daily * share * rates.at(mcur);
  }
  std::vector<std::string> lines;
  {
    std::string each;
    for (std::size_t i = 0; i < slots.size(); ++i) each += (i ? ", " : "") + std::to_string(slots[i]) + ": " + thousands(runTotal / slots[i]);
    lines.push_back("budget: " + market + " " + thousands(runTotal) + " a day run over k trades (each " + each + "); each day the k whose year before was best (walk-forward)");
  }
  ojson annual = ojson::object(), chosen = ojson::object();
  for (int k : slots) {
    std::vector<double> x;
    for (const auto& d : days) x.push_back(series[k][d]);
    const auto [an, sh] = stats(x);
    const long cnt = std::count(ch.picks.begin(), ch.picks.end(), k);
    char line[256];
    std::snprintf(line, sizeof line, "budget:   %2d trades of %s: %+.1f%% a year, Sharpe %.2f; chosen on %ld of %zu days", k, thousands(runTotal / k).c_str(), 100 * an, sh, cnt, days.size());
    lines.push_back(line);
    annual[std::to_string(k)] = an, chosen[std::to_string(k)] = cnt;
  }
  const auto [aAll, shAll] = stats(ch.meta);
  {
    const std::size_t h = ch.meta.size() / 2;
    const auto [a1, s1] = stats(std::vector<double>(ch.meta.begin(), ch.meta.begin() + long(h)));
    const auto [a2, s2] = stats(std::vector<double>(ch.meta.begin() + long(h), ch.meta.end()));
    char line[256];
    std::snprintf(line, sizeof line, "budget:   chosen day by day: %+.1f%% a year, Sharpe %.2f | first half %+.1f%% %.2f | second half %+.1f%% %.2f", 100 * aAll, shAll, 100 * a1, s1, 100 * a2, s2);
    lines.push_back(line);
  }
  std::map<int, std::map<std::string, std::vector<double>>> orders;
  for (int k : slots) orders[k] = ordersOf("data/budget/trades-" + std::to_string(k) + ".csv");
  const auto [ed, ey, opd, fr] = expectedOf(days, ch.picks, orders);
  const std::size_t y0 = days.size() > 252 ? days.size() - 252 : 0;
  const auto [ed1, ey1, opd1, fr1] = expectedOf(std::vector<std::string>(days.begin() + long(y0), days.end()), std::vector<int>(ch.picks.begin() + long(y0), ch.picks.end()), orders);
  (void)opd1, (void)fr1;
  {
    char line[400];
    std::snprintf(line, sizeof line, "budget:   expected: a day's recommended orders %+.3f%% on average (an unfilled order 0; %.1f orders a day, %.0f%% filled), compounded %+.1f%% a year | the last year: %+.3f%% a day, %+.1f%% a year",
                  100 * ed, opd, 100 * fr, 100 * ey, 100 * ed1, 100 * ey1);
    lines.push_back(line);
  }
  if (daily > 0) {
    char line[400];
    std::snprintf(line, sizeof line, "budget:   the account's limit %s %s a day across the markets; this market's share %.0f%% (%s %s; its record before costs %+.3f%% a day shrunk)", g(daily).c_str(),
                  cur.c_str(), 100 * share, thousands(total).c_str(), mcur.c_str(), 100 * score);
    lines.push_back(line);
    std::snprintf(line, sizeof line, "budget:   the outlook: the next session's orders %+.3f%% a day expected (before costs, times their chance to fill), of which the backtest delivered %.0f%% over the last %zu sessions (recent trades counting most): %+.3f%% a day",
                  100 * fexp, 100 * fdel, SPLIT_WINDOW, 100 * outlook);
    lines.push_back(line);
    std::vector<std::pair<std::string, double>> sorted(alloc.begin(), alloc.end());
    std::stable_sort(sorted.begin(), sorted.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
    std::string split_;
    for (std::size_t i = 0; i < sorted.size(); ++i) split_ += (i ? ", " : "") + sorted[i].first + " " + f("%.0f", 100 * sorted[i].second) + "%";
    lines.push_back("budget:   the split that maximises the account's risk-adjusted return (half the records before costs, half the outlook, over the last " + std::to_string(SPLIT_WINDOW) +
                    " sessions, the most recent counting most (half-life " + g(HALF_LIFE) + "); the markets' covariance; at most " + f("%.0f", 100 * acc.value("market_cap", 0.5)) + "% a market" +
                    (paper ? ", at least " + f("%.0f", 100 * floor_) + "% each on paper" : "") + "): " + split_);
  }
  // A book no larger than the money allows: each order at least the smallest, else fewer and larger positions.
  if (total > 0 && total / nxt < MIN_ORDER) {
    const int fit = int(total / MIN_ORDER);
    lines.push_back("budget:   " + std::to_string(nxt) + " trades of " + thousands(total / nxt) + " would each be under the smallest order (" + thousands(MIN_ORDER) + "): " +
                    (fit ? std::to_string(fit) + " of " + thousands(total / fit) + " instead" : std::string("no orders")));
    nxt = std::max(1, fit);
    if (!fit) total = 0.0;
  }
  // The final orders at the amount the bridge places: the same strategy and book again, costs at the end. Their size
  // fitted to the costs (budget.fit_costs, default on): the broker's fixed minimums weigh less on a larger order, so
  // the market's money is also tried over fewer, larger trades (k, k/2, ... 1). The number kept is the one whose
  // final orders (those still expected to gain after their costs) carry the most expected profit after costs; more
  // trades among equals. Nothing before the final order list sees a cost: the backtests are the same.
  const auto finalRun = [&](int kk) {  // the expected profit after costs of the final orders at total/kk each (-1: failed)
    const std::string Kk = std::to_string(kk);
    auto env = senv;
    env["X_NPICK"] = Kk;
    std::vector<std::string> cmd = engine;
    const auto wa = withCapital(finalArgs, total / kk, "final");
    cmd.insert(cmd.end(), wa.begin(), wa.end());
    cmd.insert(cmd.end(), {"--json", "data/budget/final-" + Kk + ".json", "--orders", "data/budget/final-orders-" + Kk + ".json"});
    if (waitFor(spawn(cmd, env, "data/budget/final-" + Kk + ".txt"))) return -1.0;
    ojson r;
    double v = 0;
    if (tryOJson("data/budget/final-" + Kk + ".json", r) && r.contains("orders"))
      for (const auto& o : r["orders"]) v += o.value("expected", 0.0) * o.value("amount", 0.0);
    return v;
  };
  if (total > 0) {
    const bool fit = b.value("fit_costs", true);
    std::vector<int> tries{nxt};
    for (int kk = nxt; fit && kk > 1;) tries.push_back(kk = (kk + 1) / 2 == kk ? kk - 1 : (kk + 1) / 2);
    int best = nxt;
    double bestV = -1;
    std::string tried;
    for (int kk : tries) {
      const double v = finalRun(kk);
      if (v < 0) {
        std::cout << readFile("data/budget/final-" + std::to_string(kk) + ".txt");
        die("the live strategy " + s["name"].get<std::string>() + "'s final run at " + thousands(total / kk) + " a trade failed");
      }
      ojson r;
      std::size_t n = 0;
      if (tryOJson("data/budget/final-" + std::to_string(kk) + ".json", r) && r.contains("orders")) n = r["orders"].size();
      tried += (tried.empty() ? "" : ", ") + std::to_string(kk) + " of " + thousands(total / kk) + ": " + std::to_string(n) + " order" + (n == 1 ? "" : "s") + " " + f("%+.2f", v);
      if (v > bestV + 1e-9) bestV = v, best = kk;
    }
    if (tries.size() > 1)
      lines.push_back("budget:   the orders' size fitted to the costs (the expected profit after costs of the final orders, in " + mcur + "): " + tried + "; kept " + std::to_string(best) +
                      (bestV > 0 ? "" : " (none expected to gain after costs at any size)"));
    if (bestV <= 0) best = nxt;
    nxt = best;
    const std::string Kb = std::to_string(nxt);
    fs::copy_file("data/budget/final-" + Kb + ".json", "data/budget/report-" + Kb + ".json", fs::copy_options::overwrite_existing);
    fs::copy_file("data/budget/final-orders-" + Kb + ".json", "data/budget/orders-" + Kb + ".json", fs::copy_options::overwrite_existing);
    fs::copy_file("data/budget/final-" + Kb + ".txt", "data/budget/report-" + Kb + ".txt", fs::copy_options::overwrite_existing);
  }
  lines.push_back("budget:   next session: " + std::to_string(nxt) + " trades of " + thousands(total / nxt) + (total <= 0 ? " (no orders: no positive record)" : "") + (paper ? " (paper account)" : ""));
  const std::string K = std::to_string(nxt);
  // The opposite picks on the same list and book (invert=1): the gap is the strategy's own edge.
  ojson opposite;
  {
    auto env = senv;
    env["X_NPICK"] = K;
    std::vector<std::string> cmd = engine;
    cmd.insert(cmd.end(), {"--param", "invert=1"});
    const auto wa = withCapital(finalArgs, std::max(total, 1.0) / nxt, "opposite");
    cmd.insert(cmd.end(), wa.begin(), wa.end());
    cmd.insert(cmd.end(), {"--json", "data/budget/opposite.json"});
    ojson ob;
    if (waitFor(spawn(cmd, env, "data/budget/opposite.txt")) == 0 && tryOJson("data/budget/opposite.json", ob) && ob.contains("backtest"))
      opposite = {{"annualReturn", ob["backtest"]["annualReturn"]}, {"sharpe", ob["backtest"]["sharpe"]}, {"maxDrawdown", ob["backtest"]["maxDrawdown"]}};
  }
  fs::copy_file("data/budget/report-" + K + ".json", "report/ofm.json", fs::copy_options::overwrite_existing);
  {
    std::string t;
    for (const auto& l : lines) t += l + "\n";
    writeFile("report/ofm.txt", t + readFile("data/budget/report-" + K + ".txt"));
  }
  ojson book = readOJson("data/budget/orders-" + K + ".json");  // each order's amount: the market's share over the trades
  if (book.contains("orders"))
    for (auto& o : book["orders"]) {
      const double cap = o.contains("maxCapital") && o["maxCapital"].is_number() && o["maxCapital"].get<double>() > 0 ? o["maxCapital"].get<double>() : 1e300;
      o["capital"] = std::min(total / nxt, cap);
    }
  if (total <= 0) book["orders"] = ojson::array();
  book["budget"] = {{"total", total}, {"trades", nxt}, {"amount", total / nxt}, {"share", share}};
  writeFile("data/budget/orders-" + K + ".json", book.dump());
  if (publish) writeFile("data/eoddata/orders/" + market + ".json", book.dump());
  ojson rec = ojson::object();
  for (const auto& [d, v] : record) rec[d] = std::round(v * 1e8) / 1e8;
  ojson al = ojson::object();
  for (const auto& [m, v] : alloc) al[m] = v;
  ojson slotsJ = ojson::array();
  for (int k : slots) slotsJ.push_back(k);
  const ojson expectedJ{{"daily", ed}, {"yearly", ey}, {"ordersPerDay", opd}, {"filled", fr}, {"lastYearDaily", ed1}, {"lastYearYearly", ey1}};
  writeFile("report/budget.json", ojson{{"market", market}, {"date", days.empty() ? "" : days.back()}, {"score", score}, {"share", share}, {"allocation", al}, {"record", rec},
                                        {"forward", {{"outlook", outlook}, {"expectedDaily", fexp}, {"delivered", fdel}}}, {"total", total}, {"currency", mcur}, {"slots", slotsJ},
                                        {"next", nxt}, {"amount", total / nxt}, {"chosen", chosen}, {"annual", annual}, {"auto", {{"annual", aAll}, {"sharpe", shAll}}},
                                        {"expected", expectedJ}}
                                      .dump(1));
  ojson rj;
  if (tryOJson("report/ofm.json", rj)) {  // the page's report carries the expected return, the budget and the edge
    rj["expected"] = expectedJ;
    rj["budget"] = {{"total", total}, {"trades", nxt}, {"amount", total / nxt}, {"share", share}, {"currency", mcur}};
    if (!opposite.is_null()) rj["backtest"]["opposite"] = opposite;
    writeFile("report/ofm.json", rj.dump());
  }
  std::cout << "strategies: " << market << " live " << s["name"].get<std::string>() << " (budget " << g(total) << ", " << nxt << " trades of " << g(total / nxt) << "): " << describe << "\n";
  std::cout << readFile("report/ofm.txt");
  index["strategies"].push_back(entry({{"budget", {{"total", total}, {"trades", nxt}, {"amount", total / nxt}}}}));
  writeFile("report/strategies.json", index.dump(1));
  return 0;
}

// ------------------------------------------------------------------ track, mail, index

std::string pct(const ojson& x, int d = 2) {
  if (!x.is_number()) return "–";
  char b[32];
  std::snprintf(b, sizeof b, "%+.*f%%", d, 100 * x.get<double>());
  return b;
}

ojson loadTrack(const std::string& p) {
  ojson t;
  if (!tryOJson(p, t)) t = {{"about", "Live record, only the days orders were made for: ret the model portfolio, day the daily trades (results only)."}, {"days", ojson::array()}};
  return t;
}

// The live record keeps only the days orders were made for: a scan with orders marks its date ("ordered": the
// close they were made after), and the next scan's day after that date is added (its result); a day without
// orders is not recorded.
int cmdTrack(const std::vector<std::string>& a) {
  if (a.size() < 3) die("track report/ofm.json web/data/track.json");
  const ojson scan = readOJson(a[1]);
  ojson track = loadTrack(a[2]);
  const std::string before = track.dump();
  const auto& recent = scan.contains("recent") ? scan["recent"] : ojson::array();
  const std::string asOf = scan.value("asOf", std::string()), ordered = track.value("ordered", std::string());
  std::string added;
  if (!recent.empty() && !ordered.empty()) {
    const ojson& day = recent.back();
    const std::string date = day.value("date", std::string());
    bool have = false;
    for (const auto& d : track["days"]) have = have || d["date"] == day["date"];
    if (!have && date > ordered) {
      ojson e{{"date", day["date"]}, {"ret", day.contains("ret") ? day["ret"] : day.value("day", ojson())}, {"ordersAfter", ordered}};
      if (day.contains("day") && !day["day"].is_null()) e["day"] = day["day"];
      track["days"].push_back(e);
      std::vector<ojson> v(track["days"].begin(), track["days"].end());
      std::stable_sort(v.begin(), v.end(), [](const ojson& x, const ojson& y) { return x["date"].get<std::string>() < y["date"].get<std::string>(); });
      track["days"] = v;
      added = date;
      track.erase("ordered");
    }
  }
  const auto& orders = scan.contains("orders") ? scan["orders"] : ojson::array();
  if (!orders.empty() && !asOf.empty()) track["ordered"] = asOf;
  else if (track.value("ordered", std::string()) == asOf) track.erase("ordered");  // a later scan of the same close dropped them
  if (track.dump() == before) {
    std::cout << "track: unchanged\n";
    return 0;
  }
  writeFile(a[2], track.dump(1));
  std::cout << "track: " << (added.empty() ? "no day added" : "added " + added) << (track.contains("ordered") ? "; orders after the close of " + track["ordered"].get<std::string>() : "; no orders pending") << "\n";
  return 0;
}

int cmdMail(const std::vector<std::string>& a) {
  if (a.size() < 3) die("mail report/ofm.json web/data/track.json");
  const ojson scan = readOJson(a[1]);
  const ojson track = loadTrack(a[2]);
  const auto& recent = scan.contains("recent") ? scan["recent"] : ojson::array();
  const ojson day = recent.empty() ? ojson::object() : recent.back();
  const std::string date = day.value("date", scan.value("asOf", std::string()));
  double w = 1;
  std::size_t n = 0;
  for (const auto& d : track["days"])
    if (d.contains("day") && d["day"].is_number()) w *= 1 + d["day"].get<double>(), ++n;
  const auto& orders = scan.contains("orders") ? scan["orders"] : ojson::array();
  const ojson& B = scan["backtest"];
  const char* m = std::getenv("MARKET");
  const std::string subject = std::string(m ? m : "UK") + " daily trades · " + date + ": " + pct(day.value("day", ojson())) + " · next open: " + std::to_string(orders.size()) + " trade" +
                              (orders.size() == 1 ? "" : "s");
  std::string text = "Day " + date + ": daily trades " + pct(day.value("day", ojson())) + "\nLive record: " + pct(w - 1) + " over " + std::to_string(n) + " days\nBacktest " +
                     B.value("from", std::string()) + " – " + B.value("to", std::string()) + ": " + pct(B["annualReturn"], 1) + " a year, Sharpe " + f("%.2f", B.value("sharpe", 0.0)) +
                     ", max drawdown " + f("%.1f", 100 * B.value("maxDrawdown", 0.0)) + "%\n\nTrades for the next open (after the close of " + scan.value("asOf", std::string()) + ")\n";
  if (orders.empty()) text += "  none: stay in cash\n";
  for (const auto& o : orders) {
    char t[64];
    std::snprintf(t, sizeof t, "  BUY %-6s", o.value("ticker", std::string()).c_str());
    text += std::string(t) + " expected " + pct(o.value("expected", ojson())) +
            (o.contains("ifFilled") && o.contains("fill") && o["fill"].is_number() ? " (if it buys " + pct(o["ifFilled"]) + ", chance " + f("%.0f", 100 * o["fill"].get<double>()) + "%)" : "") + "\n";
  }
  if (!text.empty() && text.back() == '\n') text.pop_back();
  const char* server = std::getenv("SMTP_SERVER");
  if (flag(a, "print") || !server || !*server) {
    if (!flag(a, "print")) std::cout << "SMTP_SERVER not set: nothing sent\n\n";
    std::cout << text << "\n";
    return 0;
  }
  Mail mail;
  mail.server = server, mail.port = std::getenv("SMTP_PORT") && *std::getenv("SMTP_PORT") ? std::atol(std::getenv("SMTP_PORT")) : 465;
  mail.user = std::getenv("SMTP_USERNAME") ? std::getenv("SMTP_USERNAME") : "", mail.password = std::getenv("SMTP_PASSWORD") ? std::getenv("SMTP_PASSWORD") : "";
  mail.from = std::getenv("MAIL_FROM") && *std::getenv("MAIL_FROM") ? std::getenv("MAIL_FROM") : mail.user;
  for (auto t : split(std::getenv("MAIL_TO") ? std::getenv("MAIL_TO") : "", ','))
    if (!t.empty()) mail.to.push_back(t);
  mail.subject = subject, mail.text = text, mail.html = "<pre>" + text + "</pre>";
  const std::string err = sendMail(mail);
  if (!err.empty()) die("mail: " + err);
  std::cout << "sent: " << subject << "\n";
  return 0;
}

int cmdIndex(const std::vector<std::string>& a) {  // the market's entry in the page's index (results only)
  if (a.size() < 3) die("index report/strategies.json web/data/strategies.json");
  const ojson s = readOJson(a[1]);
  ojson d;
  if (!tryOJson(a[2], d)) d = {{"about", "Each market's strategy (config/markets.json); results only."}, {"markets", ojson::object()}};
  ojson list = ojson::array();
  for (const auto& x : s["strategies"]) list.push_back({{"name", x["name"]}, {"role", x["role"]}, {"strategy", x.value("strategy", ojson())}, {"params", x.value("params", ojson::object())}});
  d["markets"][lower(s["market"].get<std::string>())] = list;
  writeFile(a[2], d.dump(1));
  return 0;
}

// ------------------------------------------------------------------ test (was tools/scan/test_allocate.py)

int cmdTest() {
  std::mt19937_64 rng(3);
  std::normal_distribution<double> z(0, 1);
  std::vector<std::string> days;
  for (int k = 0; k < 250; ++k) {
    char b[16];
    std::snprintf(b, sizeof b, "2026-%02d-%02d", 1 + k / 28, 1 + k % 28);
    days.push_back(b);
  }
  std::map<std::string, Record> rec;
  for (const auto& d : days) {
    const double common = 0.006 * z(rng);
    rec["UK"][d] = 0.002 + common + 0.003 * z(rng);   // an edge, moving with EU
    rec["EU"][d] = 0.002 + common + 0.003 * z(rng);   // the same edge, moving with UK
    rec["US"][d] = 0.002 + 0.0067 * z(rng);           // the same edge and risk, independent
    rec["APAC"][d] = -0.001 + 0.01 * z(rng);          // no edge
  }
  const auto check = [](bool ok, const std::string& what) {
    if (!ok) die("test: " + what);
  };
  auto w = allocate(rec, 0.5, 0, nullptr, 0.5, 60, 250);
  double sum = 0, mx = 0;
  for (const auto& [m, v] : w) sum += v, mx = std::max(mx, v);
  check(w["APAC"] == 0, "a market with no edge got money");
  check(std::fabs(sum - 1) < 1e-9, "the shares do not add up to the limit");
  check(mx <= 0.5 + 1e-9, "a market over the cap");
  check(w["US"] > w["UK"] && w["US"] > w["EU"], "the independent market did not get more than the two that move together");
  auto wp = allocate(rec, 0.5, 0.05, nullptr, 0.5, 60, 250);
  double mn = 1, sp = 0;
  for (const auto& [m, v] : wp) mn = std::min(mn, v), sp += v;
  check(mn >= 0.05 - 1e-9 && std::fabs(sp - 1) < 1e-9, "the paper floor");
  Record neg;
  for (const auto& d : days) neg[d] = -0.001;
  check(allocate({{"UK", neg}}, 0.5, 0, nullptr, 0.5, 60, 250)["UK"] == 0.0, "no edge anywhere: nothing");
  auto one = allocate({{"INDEX", rec["US"]}, {"UK", rec["APAC"]}, {"EU", rec["APAC"]}}, 0.5, 0.05, nullptr, 0.5, 60, 250);
  check(std::fabs(one["INDEX"] - 0.5) < 1e-9 && std::fabs(one["UK"] - 0.05) < 1e-9, "the floors came out of the capped market, not the unused limit");
  const std::map<std::string, double> fw{{"UK", 0.004}, {"EU", 0.0}};
  auto w0 = allocate({{"UK", rec["US"]}, {"EU", rec["US"]}}, 1.0, 0, nullptr, 0.5, 60, 250);
  auto w1 = allocate({{"UK", rec["US"]}, {"EU", rec["US"]}}, 1.0, 0, &fw, 0.5, 60, 250);
  check(std::fabs(w0["UK"] - w0["EU"]) < 1e-9 && w1["UK"] > w1["EU"], "the outlook does not shift the split");
  // the outlook's arithmetic (forwardOf)
  const fs::path d = fs::temp_directory_path() / "ofm_scan_test";
  fs::create_directories(d);
  writeFile(d / "o.json", R"({"slots": 2, "orders": [{"expected": 0.006, "holdDays": 3}, {"expected": 0.002, "holdDays": 2}]})");
  writeFile(d / "t.csv", "date,ticker,expected,filled,ret\n2026-01-02,A,0.01,1,0.004\n2026-01-03,B,0.01,1,0.001\n2026-01-03,C,0.02,0,0\n2025-01-01,D,0.01,1,0.05\n");
  const auto [out, exp, dl] = forwardOf((d / "o.json").string(), (d / "t.csv").string(), "2026-01-01");
  const double wa = std::pow(0.5, 1 / HALF_LIFE), want = (wa * 0.004 + 0.001) / (wa * 0.01 + 0.01);
  check(std::fabs(exp - (0.002 + 0.001) / 2) < 1e-12 && std::fabs(dl - want) < 1e-12 && std::fabs(out - exp * want) < 1e-12, "the outlook's arithmetic");
  // the most recent days count most
  Record late, early;
  for (std::size_t k = 0; k < days.size(); ++k) late[days[k]] = (k >= 190 ? 0.004 : -0.001) + 0.004 * z(rng);
  for (std::size_t k = 0; k < days.size(); ++k) early[days[k]] = late[days[249 - k]];
  auto wl = allocate({{"UK", late}, {"EU", early}}, 1.0, 0, nullptr, 0.5, 60, 250);
  check(wl["UK"] > wl["EU"], "recent gains did not count more");
  // the live split looks at the last SPLIT_WINDOW sessions only
  Record turn, fade;
  for (std::size_t k = 0; k < days.size(); ++k) {
    turn[days[k]] = (k < 250 - SPLIT_WINDOW ? -0.003 : 0.006) + 0.001 * z(rng);
    fade[days[k]] = (k < 250 - SPLIT_WINDOW ? 0.003 : -0.002) + 0.001 * z(rng);
  }
  auto ws = allocate({{"UK", turn}, {"EU", fade}}, 1.0);
  check(SPLIT_WINDOW == 20 && ws["UK"] > ws["EU"], "the split's window");
  // the walk-forward choice and the shrunk mean
  std::map<int, Record> series{{2, {}}, {5, {}}};
  for (std::size_t k = 0; k < days.size(); ++k) series[2][days[k]] = -0.001 + 0.001 * z(rng), series[5][days[k]] = 0.002 + 0.001 * z(rng);
  const Choice c = choose(series);
  check(c.next == 5 && c.days.size() == days.size(), "the walk-forward choice");
  check(shrunk({0.01, -0.01, 0.01, -0.01}) == 0.0, "a mean of no significance not shrunk to 0");
  std::cout << "allocate: the split maximises the risk-adjusted return, no money without an edge, capped, floored on paper; the outlook shifts it, the recent days count most, the last "
            << SPLIT_WINDOW << " sessions only; the walk-forward choice; OK\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> a(argv + 1, argv + argc);
  if (a.empty()) die("usage: ofm_scan run|track|mail|index|test ...", 2);
  try {
    if (a[0] == "run") return cmdRun(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "track") return cmdTrack(a);
    if (a[0] == "mail") return cmdMail(a);
    if (a[0] == "index") return cmdIndex(a);
    if (a[0] == "test") return cmdTest();
  } catch (const std::exception& e) {
    die(a[0] + ": " + e.what());
  }
  die("unknown command " + a[0], 2);
}
