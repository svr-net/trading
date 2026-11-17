// ofm_eod: the loaders, in C++ (they replace tools/eoddata/*.mjs and series_list.py). The store is the clone of the
// private svr-net/eod at data/eoddata (EOD_STORE), kept and pushed through libgit2; EODData's API through libcurl
// (the key from EODDATA_API_KEY only: never written or printed).
//
//   ofm_eod sync     [--dir data/eoddata] [--url https://github.com/svr-net/eod.git]   the clone up to date (libgit2)
//   ofm_eod push     [--dir data/eoddata] --message M                                   the run's new data committed and pushed
//   ofm_eod shares   --exchange LSE [--pool store|top] [--universe N] [--years 9] [--min-days 750] [--currency GBP]
//                    [--out data/LSE.csv] [--cached] [--refresh] [--verbose]
//                    an exchange's shares: the store completed with the days it lacks, split-adjusted, as a CSV
//   ofm_eod series   --symbols EX:CODE,... [--years 20] [--out data/futures.csv] [--cached] [--refresh]
//   ofm_eod market   --list tools/eoddata/index.txt --out data/stocks.csv [--years 9] [--inverse] [--cached]
//                    a market of named series, each from the first of its exchanges with bars (INDEX, FOREX)
//   ofm_eod profiles --exchanges LSE,... --out data/eoddata/profiles [--universe data/stocks.csv] [--fundamentals DIR] [--probe]
//                    [--max-age 7] [--refresh]   the profiles once a week, the fundamentals once a day (kept in the store)
//   ofm_eod aim      [tools/eoddata/aim.txt] [--cache data/eoddata/lists/aim.txt] [--refresh]
//                    the AIM shares (no stamp duty) from stockanalysis.com, once a day (kept in the store)
//   ofm_eod symbols  [--exchanges] [--find EX:PRE,PRE;EX:...]
//   ofm_eod snapshot --symbols EX:CODE,... [--from 2019-01]  5-minute bars by month into the store
#include <zlib.h>

#include <iostream>
#include <map>
#include <regex>
#include <set>

#include "eod/common.hpp"
#include "eod/git.hpp"
#include "eod/http.hpp"
#include "eod/store.hpp"

using namespace eod;
using store::Bar;
using store::Bars;

namespace {

[[noreturn]] void die(const std::string& why, int code = 2) {
  std::cerr << "ofm_eod: " << why << "\n";
  std::exit(code);
}

bool keyOrCached(const std::vector<std::string>& a) {
  if (std::getenv("EODDATA_API_KEY") || flag(a, "cached")) return true;
  std::cerr << "Set EODDATA_API_KEY in the environment (never pass it on the command line).\n";
  std::exit(2);
}

std::time_t yearsAgo(double years) { return std::time(nullptr) - std::time_t(years * 365.25 * 86400); }

// ------------------------------------------------------------------ sync, push (libgit2)

int cmdSync(const std::vector<std::string>& a) {
  const std::string dir = opt(a, "dir", "data/eoddata"), url = opt(a, "url", "https://github.com/svr-net/eod.git");
  std::cout << "EOD store: " << git::sync(dir, url) << "\n";
  return 0;
}

int cmdPush(const std::vector<std::string>& a) {
  const std::string dir = opt(a, "dir", "data/eoddata"), msg = opt(a, "message", "data " + today());
  std::cout << "EOD store: " << git::push(dir, msg) << "\n";
  return 0;
}

// ------------------------------------------------------------------ shares (was fetch.mjs)

struct Cache {
  fs::path dir;
  bool cached = false;
  // An API list, kept as a file in the exchange's folder (once a day); in cached mode the newest kept copy.
  json get(const std::string& route, const std::map<std::string, std::string>& params, const std::string& name) {
    const fs::path f = dir / name;
    std::string text;
    if (tryRead(f, text)) return json::parse(text);
    if (cached) {
      const std::string stem = std::regex_replace(name, std::regex("(-\\d{4}-\\d{2}-\\d{2})+\\.json$"), "");
      std::vector<std::string> hits;
      if (fs::exists(dir))
        for (const auto& e : fs::directory_iterator(dir)) {
          const std::string n = e.path().filename().string();
          if (n.rfind(stem + "-", 0) == 0 && std::regex_search(n, std::regex("(-\\d{4}-\\d{2}-\\d{2})+\\.json$"))) hits.push_back(n);
        }
      if (hits.empty()) throw std::runtime_error(route + ": not in the cache");
      std::sort(hits.begin(), hits.end());
      return json::parse(readFile(dir / hits.back()));
    }
    json body = api(route, params);
    writeFile(f, body.dump());
    return body;
  }
};

const std::regex FUND_FAMILY(
    "\\b(ishares|spdr|proshares|direxion|vanguard|invesco|wisdomtree|global x|vaneck|first trust|graniteshares|tidal|defiance|roundhill|yieldmax|grayscale|ark|amplify|pacer|schwab|"
    "fidelity|jpmorgan(?! chase)|franklin|janus|dimensional|avantis|t-rowe|xtrackers|simplify|rex|tuttle|leverage shares|microsectors|kraneshares|pgim|x-links|eaton vance|nuveen|"
    "blackrock|gabelli|templeton|voya|pimco|john hancock|legg mason|columbia|lazard|abrdn|aberdeen|cohen & steers|calamos|western asset|doubleline|guggenheim|mfs|putnam|neuberger|"
    "tortoise|kayne|clearbridge|flaherty|rivernorth|duff & phelps|babson|northern lights|virtus|allianz|liberty all-star|tri-continental|saba|trust|index|bitcoin|ether|ultra|2x|3x|bull|"
    "bear|daily|leveraged|inverse|portfolio|treasury|strategy shares|municipal|muni|tax-advantaged|tax-advantage|tax advantaged|tax-managed|buffer|laddered|definedterm|term|income fund|"
    "income fun|income f|income opportunity|income t|dividend opportunity|global dividend|dividend achievers|enhanced|managed duration|total return|stocksplus|precious metals|"
    "commodity strategy|source capital|income shares)\\b");
const std::regex NOT_ORDINARY(
    "(\\d+(\\.\\d+)?\\s*%|\\b\\d+\\.\\d{2,3}\\b|\\bnotes?\\b|\\bdue\\b|\\bpfd\\b|\\bpref(erred)?\\b|\\bdepositary\\b|\\bdebentures?\\b|\\bsenior\\b|\\bjunior\\b|\\bsubordinated?\\b|"
    "\\bfixed[- ]to[- ]floating\\b|\\bfloating rate\\b|\\bperpetual\\b|\\b(non-)?cumula|\\bcum\\b|\\bseries [a-z]\\b|\\bunits?\\b|\\brights?\\b|\\bwarrants?\\b|\\bwt\\b|\\babs\\b|\\bstrats\\b|\\bsecurit)",
    std::regex::icase);

int cmdShares(const std::vector<std::string>& a) {
  keyOrCached(a);
  const std::string exchange = opt(a, "exchange", "LSE"), out = opt(a, "out", "data/" + exchange + ".csv"), pool = opt(a, "pool", "top");
  const std::size_t universe = std::size_t(std::atol(opt(a, "universe", "120").c_str()));
  const double years = std::atof(opt(a, "years", "6").c_str());
  const std::size_t minDays = std::size_t(std::atol(opt(a, "min-days", "750").c_str()));
  const bool cached = flag(a, "cached"), refresh = flag(a, "refresh"), verbose = flag(a, "verbose");
  Cache cache{store::root() / exchange, cached};
  fs::create_directories(cache.dir);
  const std::string from = ymd(yearsAgo(years)), stamp = today();
  std::cout << "exchange " << exchange << ": listing symbols and the latest quotes\n";
  json exchangeInfo = json::object();
  try {
    exchangeInfo = cache.get("/Exchange/Get/" + exchange, {}, "exchange.json");
  } catch (...) {
  }
  const json symbols = asList(cache.get("/Symbol/List/" + exchange, {}, "symbols-" + stamp + ".json"), {"data", "items", "quotes", "symbols"});
  std::map<std::string, std::string> typeOf, nameOf, currencyOf;
  for (const auto& s : symbols) {
    const std::string code = str(field(s, {"code", "symbolCode", "symbol"}));
    typeOf[code] = str(field(s, {"type", "symbolType", "typeCode"})), nameOf[code] = str(field(s, {"name", "description"})), currencyOf[code] = str(field(s, {"currency", "currencyCode"}));
  }
  std::string currency = opt(a, "currency", exchange == "LSE" ? "GBP" : "");
  const json quotes = asList(cache.get("/Quote/List/" + exchange, {}, "quotes-" + stamp + ".json"));
  const std::regex codeOk(exchange == "ASX" ? "^[A-Z0-9]{3}$" : exchange == "SGX" ? "^[A-Z0-9]{3,4}$" : "^[A-Z]{1,5}(\\.[A-Z])?$");
  const std::regex typeOk("(share|stock|equity|ord|common)"), notShare(
      "\\b(etf|etc|etn|fund|trust plc|investment trust|ucits|tracker|swap|s&p|msci|ftse|warrant|bond|gilt|pref)\\b"),
      companyRe("\\b(plc|ltd|limited|s\\.a\\.|sa|ag|n\\.v\\.|nv|se|inc|corp|group|holdings|ord)\\b"), fraHome("\\b(ag|se|kgaa)\\b"),
      fraForeign("\\b(inc|corp|plc|ltd|limited|n\\.v\\.|nv|s\\.a\\.)\\b");
  const auto isShare = [&](const std::string& code) {
    if (!std::regex_match(code, codeOk)) return false;
    if (!currency.empty() && !currencyOf[code].empty() && lower(currencyOf[code]) != lower(currency)) return false;
    const std::string t = lower(typeOf[code]);
    if (!t.empty() && !std::regex_search(t, typeOk)) return false;
    const std::string n = lower(nameOf[code]);
    if (std::regex_search(n, notShare)) return false;
    const bool company = std::regex_search(n, companyRe);
    if (exchange == "LSE") return company;
    if (exchange == "FRA") return std::regex_search(n, fraHome) && !std::regex_search(n, fraForeign);
    return company || !std::regex_search(n, FUND_FAMILY);
  };
  const std::vector<std::string> US{"NYSE", "NASDAQ", "AMEX"};  // in this order: a share's history joined from the first first
  const auto isOrdinary = [&](const std::string& code) {
    if (exchange == "LSE") return true;
    if (std::regex_search(nameOf[code], NOT_ORDINARY)) return false;
    return !(exchange == "NASDAQ" && code.size() == 5 && std::string("LMNOPQRTUVWZ").find(code[4]) != std::string::npos);
  };
  struct Ranked { std::string code; double value; };
  std::vector<Ranked> ranked;
  for (const auto& q : quotes) {
    const std::string code = str(field(q, {"symbolCode", "code", "symbol"}));
    const double v = num(field(q, {"close"})) * num(field(q, {"volume"}));
    if (!code.empty() && std::isfinite(v) && v > 0 && isShare(code)) ranked.push_back({code, v});
  }
  std::stable_sort(ranked.begin(), ranked.end(), [](const Ranked& x, const Ranked& y) { return x.value > y.value; });
  // --pool store: every share the store keeps, not only today's most traded (today's first)
  if (pool == "store") {
    std::set<std::string> have;
    for (const auto& r : ranked) have.insert(r.code);
    std::vector<std::string> dir;
    const fs::path sd = store::root() / "store" / exchange;
    if (fs::exists(sd))
      for (const auto& e : fs::directory_iterator(sd))
        if (e.path().extension() == ".json" && e.path().filename().string()[0] != '.') dir.push_back(e.path().stem().string());
    std::sort(dir.begin(), dir.end());
    for (const auto& code : dir)
      if (!have.count(code) && isShare(code)) ranked.push_back({code, 0}), have.insert(code);
  }
  std::map<std::string, int> currencies;
  for (const auto& [c, cur] : currencyOf) ++currencies[cur];
  std::cout << "  " << symbols.size() << " symbols, " << quotes.size() << " quotes, " << ranked.size() << " candidate shares; keeping the " << universe << " most traded with " << minDays
            << "+ days since " << from << "\n";
  // Splits: prices before a split divided by its ratio, so returns are not distorted.
  std::map<std::string, std::vector<std::pair<std::string, double>>> splits;
  try {
    const json list = asList(cache.get("/Splits/List/" + exchange, {}, "splits-" + stamp + ".json"));
    std::cout << "  splits: " << list.size() << "\n";
    for (const auto& sp : list) {
      const std::string code = str(field(sp, {"symbolCode", "code", "symbol"}));
      const std::string date = str(field(sp, {"dateStamp", "date", "exDate"})).substr(0, 10);
      const json* rv = field(sp, {"ratio", "splitRatio"});
      double ratio = NAN;
      if (rv && rv->is_string() && std::regex_search(rv->get<std::string>(), std::regex("[-:/]"))) {
        const std::string s = rv->get<std::string>();
        const auto k = s.find_first_of("-:/");
        ratio = std::atof(s.substr(0, k).c_str()) / std::atof(s.substr(k + 1).c_str());  // "2-1": two new for one old
      } else if (!rv) {
        ratio = num(field(sp, {"numerator", "to", "newShares"})) / num(field(sp, {"denominator", "from", "oldShares"}));
      } else {
        ratio = num(rv);
      }
      if (!code.empty() && !date.empty() && std::isfinite(ratio) && ratio > 0 && ratio != 1) splits[code].push_back({date, ratio});
    }
  } catch (const std::exception& e) {
    std::cerr << "  splits unavailable: " << e.what() << "\n";
  }
  // Each share's bars: from the store, completed with the days it is missing.
  std::map<std::string, std::pair<std::string, Bar>> quoteBar;
  std::string latest;
  for (const auto& q : quotes) {
    std::string d;
    Bar b;
    if (store::bar(q, d, b)) {
      quoteBar[str(field(q, {"symbolCode", "code", "symbol"}))] = {d, b};
      latest = std::max(latest, d);
    }
  }
  if (latest.empty()) latest = stamp;
  int nFull = 0, nGap = 0, nQuote = 0, nCurrent = 0, nKnown = 0;
  // What EODData could not give, kept in the store (store/<EX>/.unavailable.json) so that no run asks again: a code
  // with no history (HTTP 404: a warrant, a unit, a code gone) for 30 days; a gap the API had nothing for until the
  // exchange has a newer day.
  const fs::path memoFile = store::root() / "store" / store::safe(exchange) / ".unavailable.json";
  json memo = json::object();
  {
    std::string t;
    if (!refresh && tryRead(memoFile, t)) try {
        memo = json::parse(t);
      } catch (...) {
        memo = json::object();
      }
  }
  bool memoChanged = false;
  const auto known = [&](const std::string& code) {
    if (!memo.contains(code)) return false;
    const json& m = memo[code];
    const std::string why = m.value("why", std::string()), on = m.value("on", std::string());
    if (why == "404") return isDay(on) && daysBetween(on, stamp) < 30;
    if (why == "nothing") return m.value("latest", std::string()) == latest;
    return false;
  };
  const auto remember = [&](const std::string& code, const char* why) {
    memo[code] = {{"why", why}, {"on", stamp}, {"latest", latest}};
    memoChanged = true;
  };
  // A missing day of every share at once: the exchange's quote list for that day (one request), used only when every
  // bar in it carries that day (an API that ignored the date and sent the latest day instead is caught there); the
  // shares are then filled from it, not asked one by one (16 Nov: a failed push left two days of gaps, 5,500 shares
  // asked one at a time, over an hour). A day list that does not check out: the shares one by one, as before.
  std::map<std::string, std::map<std::string, Bar>> dayBars;
  std::set<std::string> dayBad;
  int nDay = 0, dayRequests = 0;
  const auto dayList = [&](const std::string& d) -> const std::map<std::string, Bar>* {
    if (dayBad.count(d)) return nullptr;
    const auto hit = dayBars.find(d);
    if (hit != dayBars.end()) return &hit->second;
    std::map<std::string, Bar> m;
    bool good = true;
    try {
      ++dayRequests;
      const json list = asList(cache.get("/Quote/List/" + exchange, {{"DateStamp", d}}, "dayquotes-" + d + ".json"));
      for (const auto& q : list) {
        std::string qd;
        Bar b;
        if (!store::bar(q, qd, b)) continue;
        if (qd != d) {
          good = false;
          break;
        }
        m[str(field(q, {"symbolCode", "code", "symbol"}))] = b;
      }
    } catch (const std::exception& e) {
      good = false;
    }
    if (!good || m.size() < 20) {
      dayBad.insert(d);
      std::cerr << "  " << d << ": the exchange's quote list for that day was not that day's; the shares one by one\n";
      return nullptr;
    }
    return &(dayBars[d] = std::move(m));
  };
  // A share's gap after L filled from the day lists (at most 10 weekdays): false when a day's list is not usable.
  const auto fromDays = [&](const std::string& code, const std::string& L, Bars& bars) {
    std::vector<std::string> days;
    for (std::string d = addDays(L, 1); d <= latest && days.size() <= 10; d = addDays(d, 1)) {
      const std::time_t t = parseDay(d);
      const int wd = std::gmtime(&t)->tm_wday;
      if (wd != 0 && wd != 6) days.push_back(d);
    }
    if (days.empty() || days.size() > 10) return false;
    std::vector<const std::map<std::string, Bar>*> lists;
    for (const auto& d : days) {
      const auto* m = dayList(d);
      if (!m) return false;
      lists.push_back(m);
    }
    for (std::size_t k = 0; k < days.size(); ++k) {
      const auto it = lists[k]->find(code);
      if (it != lists[k]->end()) bars[days[k]] = it->second;  // not in the day's list: no trade that day
    }
    return true;
  };
  const auto barsOf = [&](const std::string& code, bool& ok) {
    ok = true;
    Bars bars = store::load(exchange, code);
    if (cached) {
      if (!store::has(exchange, code)) ok = false;
      return bars;
    }
    const auto qb = quoteBar.find(code);
    if (qb == quoteBar.end() && pool == "store" && !bars.empty()) {  // not quoted today: its stored bars as they are
      ++nCurrent;
      return bars;
    }
    const std::string L = store::lastDate(bars);
    const bool fullNeeded = refresh || L.empty() || L < from;
    const bool gapNeeded = !fullNeeded && !(qb != quoteBar.end() && qb->second.first >= L && previousWeekday(qb->second.first) <= L) &&
                           (L < previousWeekday(latest) || (qb != quoteBar.end() && qb->second.first > L));
    if ((fullNeeded || gapNeeded) && known(code)) {  // asked before: EODData had nothing
      ++nKnown;
      if (bars.empty()) ok = false;
      return bars;
    }
    try {
      if (fullNeeded) {
        if (!store::merge(bars, asList(api("/Quote/List/" + exchange + "/" + escape(code), {{"Interval", "d"}, {"FromDateStamp", from}, {"ToDateStamp", stamp}}))) && bars.empty())
          remember(code, "404");
        ++nFull;
      } else if (qb != quoteBar.end() && qb->second.first >= L && previousWeekday(qb->second.first) <= L) {
        bars[qb->second.first] = qb->second.second;  // the latest day, from the quote list
        ++nQuote;
      } else if (L < previousWeekday(latest) || (qb != quoteBar.end() && qb->second.first > L)) {
        if (fromDays(code, L, bars)) {
          ++nDay;
        } else {
          if (!store::merge(bars, asList(api("/Quote/List/" + exchange + "/" + escape(code), {{"Interval", "d"}, {"FromDateStamp", addDays(L, 1)}, {"ToDateStamp", stamp}}))))
            remember(code, "nothing");
          ++nGap;
        }
      } else {
        ++nCurrent;
      }
    } catch (const std::exception& e) {
      const std::string why = e.what();
      if (why.find("HTTP 404") != std::string::npos) remember(code, fullNeeded ? "404" : "nothing");
      else std::cerr << "  skip " << code << ": " << why << "\n";
      ok = false;
      return Bars{};
    }
    store::save(exchange, code, bars);
    return bars;
  };
  struct Row { std::string d; double o, h, l, c, v; };
  std::string csv = "date,ticker,open,high,low,close,volume\n";
  std::size_t nRows = 0;
  int inferred = 0, moved = 0, notOrdinary = 0;
  json kept = json::array();
  for (std::size_t i = 0; i < ranked.size(); ++i) {
    if (kept.size() + std::size_t(notOrdinary) >= universe) break;
    const std::string& code = ranked[i].code;
    bool ok = true;
    const Bars stored = barsOf(code, ok);
    if (!ok && cached && pool != "store") die("cached: " + code + ": not in the store", 3);
    std::vector<Row> bars;
    for (const auto& [d, b] : stored)
      if (d >= from && b.c > 0 && b.o > 0 && b.h > 0 && b.l > 0) bars.push_back({d, b.o, b.h, b.l, b.c, b.v});
    if (i == 0) std::cout << "  " << code << ": " << bars.size() << " bars\n";
    // A share that moved between US exchanges: its earlier days from the other exchange.
    if (bars.size() < minDays && std::find(US.begin(), US.end(), exchange) != US.end()) {
      for (const auto& other : US) {
        if (other == exchange) continue;
        Bars older = store::load(other, code);
        if (older.empty() && !cached && !store::has(other, code)) {
          try {
            store::merge(older, asList(api("/Quote/List/" + other + "/" + escape(code), {{"Interval", "d"}, {"FromDateStamp", from}, {"ToDateStamp", stamp}})));
          } catch (...) {
          }
          store::save(other, code, older);
        }
        const std::string first = bars.empty() ? "9999" : bars.front().d;
        std::vector<Row> add;
        for (const auto& [d, b] : older)
          if (d >= from && d < first && b.c > 0 && b.o > 0 && b.h > 0 && b.l > 0) add.push_back({d, b.o, b.h, b.l, b.c, b.v});
        if (!add.empty()) bars.insert(bars.begin(), add.begin(), add.end()), ++moved;
      }
    }
    if (bars.size() < minDays && pool != "store") {
      std::cout << "  " << code << ": " << bars.size() << " days, dropped\n";
      continue;
    }
    if (bars.size() < 60) continue;  // too short for any strategy
    for (const auto& [date, ratio] : splits[code])
      for (auto& b : bars)
        if (b.d < date) b.o /= ratio, b.h /= ratio, b.l /= ratio, b.c /= ratio, b.v *= ratio;
    // Splits the list lacks, read from the prices (both the close and the open moved by a split's ratio, and the
    // volume the other way).
    const auto med = [](std::vector<double> xs) {
      xs.erase(std::remove_if(xs.begin(), xs.end(), [](double x) { return !(x > 0); }), xs.end());
      if (xs.empty()) return 0.0;
      std::sort(xs.begin(), xs.end());
      return xs[xs.size() >> 1];
    };
    for (std::size_t k = 1; k < bars.size(); ++k) {
      const double r = bars[k - 1].c / bars[k].c, ro = bars[k - 1].c / bars[k].o;
      for (double n : {2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0})
        for (double q : {n, 1 / n}) {
          if (!(std::fabs(std::log(r / q)) < (n >= 5 ? 0.1 : 0.03) && std::fabs(std::log(ro / q)) < (n >= 5 ? 0.15 : 0.1))) continue;
          std::vector<double> before, after;
          for (std::size_t j = k >= 20 ? k - 20 : 0; j < k; ++j) before.push_back(bars[j].v);
          for (std::size_t j = k; j < std::min(bars.size(), k + 20); ++j) after.push_back(bars[j].v);
          const double vb = med(before), va = med(after);
          if (!(vb > 0 && va > 0 && std::log(va / vb) * (std::log(q) > 0 ? 1 : -1) > 0.5 * std::fabs(std::log(q)))) continue;
          for (std::size_t j = 0; j < k; ++j) bars[j].o /= q, bars[j].h /= q, bars[j].l /= q, bars[j].c /= q, bars[j].v *= q;
          ++inferred;
          if (verbose) std::cout << "  " << code << ": split " << (q >= 1 ? shortest(q) : "1/" + shortest(std::round(1 / q))) << " read on " << bars[k].d << "\n";
        }
    }
    // A move of over 60% in a day that stays (a split the list lacks): the share dropped; one that comes back within 5
    // days is a bad price: kept (its impossible bar is rolled over by the strategy engine).
    long jump = -1;
    for (std::size_t k = 1; k < bars.size() && jump < 0; ++k) {
      if (!(std::fabs(std::log(bars[k].c / bars[k - 1].c)) > std::log(1.6))) continue;
      bool back = false;
      for (std::size_t j = k + 1; j < std::min(bars.size(), k + 6); ++j) back = back || std::fabs(std::log(bars[j].c / bars[k - 1].c)) < std::log(1.25);
      if (!back) jump = long(k);
    }
    if (jump > 0) {
      std::cout << "  " << code << ": unexplained move on " << bars[std::size_t(jump)].d << ", dropped\n";
      continue;
    }
    if (!isOrdinary(code)) {
      ++notOrdinary;
      continue;
    }
    for (const auto& b : bars) {
      csv += b.d + "," + code + "," + shortest(b.o) + "," + shortest(b.h) + "," + shortest(b.l) + "," + shortest(b.c) + "," + shortest(std::isfinite(b.v) ? b.v : 0) + "\n";
      ++nRows;
    }
    kept.push_back({{"code", code}, {"name", nameOf[code]}, {"days", bars.size()}});
    if (kept.size() % 20 == 0) std::cout << "  kept " << kept.size() << "\n";
  }
  if (memoChanged) writeFile(memoFile, memo.dump(1));
  std::cout << "  store: " << nDay << " gaps filled from " << dayBars.size() << " day lists (" << dayRequests << " requests), " << nFull << " full histories downloaded, " << nGap << " gaps filled one by one, " << nQuote << " days from the quote list, " << nCurrent << " up to date, " << nKnown
            << " not asked again (EODData had nothing before; " << memo.size() << " kept in .unavailable.json)\n";
  if (!cached) {
    for (const char* p : {"symbols-", "quotes-", "splits-"}) store::prune(cache.dir, p, 1);
    store::prune(cache.dir, "dayquotes-", 0);  // the day lists: this run's only
  }
  writeFile(out, csv);
  const std::string cur = str(field(exchangeInfo, {"currency", "currencyCode"}));
  writeFile(std::regex_replace(out, std::regex("\\.csv$"), ".symbols.json"),
            json{{"exchange", exchange}, {"currency", cur.empty() ? json() : json(cur)}, {"from", from}, {"to", stamp}, {"symbols", kept}}.dump(1));
  std::cout << "wrote " << out << ": " << kept.size() << " stocks (" << notOrdinary << " of the most traded left out: not ordinary shares; " << inferred << " splits read from the prices; "
            << moved << " histories joined across exchanges), " << nRows << " bars\n";
  return 0;
}

// ------------------------------------------------------------------ series (was fetch_series.mjs), market (was series_list.py)

struct SeriesOut {
  std::map<std::string, std::size_t> counts;  // EX:CODE -> bars
  std::vector<std::string> rows;              // CSV rows without the header
};

SeriesOut fetchSeries(const std::vector<std::string>& symbols, double years, bool cached, bool refresh, int& requests) {
  SeriesOut out;
  const std::string stamp = today();
  for (const auto& s : symbols) {
    const auto k = s.find(':');
    const std::string ex = s.substr(0, k), code = s.substr(k + 1);
    const std::string route = "/Quote/List/" + escape(ex) + "/" + escape(code);
    Bars bars = refresh ? Bars{} : store::load("series", s);
    const std::string L = store::lastDate(bars);
    try {
      if (!cached && L.empty()) {
        for (int y = 0; y < int(years); ++y) {  // year by year: a plan's limit on how far back one request reaches
          const std::time_t to = std::time(nullptr) - std::time_t(y * 365.25 * 86400), fr = to - std::time_t(365.25 * 86400);
          const json list = asList(api(route, {{"Interval", "d"}, {"FromDateStamp", ymd(fr)}, {"ToDateStamp", ymd(to)}}, 7, 5000));
          ++requests;
          if (list.empty() && y > 0) break;
          store::merge(bars, list);
        }
      } else if (!cached && L < previousWeekday(stamp)) {
        store::merge(bars, asList(api(route, {{"Interval", "d"}, {"FromDateStamp", L}, {"ToDateStamp", stamp}}, 7, 5000)));
        ++requests;
      }
    } catch (const std::exception& e) {
      std::cerr << "  " << s << ": " << e.what() << "\n";
    }
    if (!bars.empty()) store::save("series", s, bars);
    char line[160];
    std::snprintf(line, sizeof line, "%-14s %6zu bars  %s .. %s", s.c_str(), bars.size(), bars.empty() ? "-" : bars.begin()->first.c_str(), bars.empty() ? "-" : bars.rbegin()->first.c_str());
    std::cout << line << "\n";
    out.counts[s] = bars.size();
    for (const auto& [d, b] : bars)
      out.rows.push_back(d + "," + s + "," + shortest(b.o) + "," + shortest(b.h) + "," + shortest(b.l) + "," + shortest(b.c) + "," + shortest(b.v));
  }
  return out;
}

int cmdSeries(const std::vector<std::string>& a) {
  keyOrCached(a);
  const std::vector<std::string> symbols = [&] {
    std::vector<std::string> v;
    for (auto& s : split(opt(a, "symbols"), ','))
      if (!s.empty()) v.push_back(s);
    return v;
  }();
  int requests = 0;
  const SeriesOut r = fetchSeries(symbols, std::atof(opt(a, "years", "20").c_str()), flag(a, "cached"), flag(a, "refresh"), requests);
  std::string csv = "date,ticker,open,high,low,close,volume\n";
  for (const auto& row : r.rows) csv += row + "\n";
  const std::string out = opt(a, "out", "data/futures.csv");
  writeFile(out, csv);
  std::cout << "wrote " << out << ": " << r.rows.size() << " bars; " << requests << " requests\n";
  return 0;
}

int cmdMarket(const std::vector<std::string>& a) {
  keyOrCached(a);
  const std::string list = opt(a, "list"), out = opt(a, "out");
  if (list.empty() || out.empty()) die("market: --list and --out");
  std::vector<std::pair<std::string, std::vector<std::string>>> todo;  // code, its exchanges in order
  std::istringstream in(readFile(list));
  for (std::string line; std::getline(in, line);) {
    line = line.substr(0, line.find('#'));
    std::istringstream w(line);
    std::string code, ex;
    if (!(w >> code)) continue;
    w >> ex;
    todo.push_back({code, split(ex.empty() ? "AMEX" : ex, ',')});
  }
  std::size_t passes = 0;
  for (const auto& t : todo) passes = std::max(passes, t.second.size());
  std::map<std::string, std::string> found;  // code -> EX:CODE
  std::vector<std::string> rows;
  int requests = 0;
  for (std::size_t k = 0; k < passes; ++k) {  // each pass: the codes not found yet, on their next exchange
    std::vector<std::string> ask;
    std::map<std::string, std::string> codeOf;
    for (const auto& [code, exs] : todo)
      if (!found.count(code) && k < exs.size()) ask.push_back(exs[k] + ":" + code), codeOf[exs[k] + ":" + code] = code;
    if (ask.empty()) continue;
    const SeriesOut r = fetchSeries(ask, std::atof(opt(a, "years", "9").c_str()), flag(a, "cached"), false, requests);
    for (const auto& [s, n] : r.counts)
      if (n > 0) found[codeOf[s]] = s;
    for (const auto& row : r.rows) {
      const auto p = split(row, ',');
      if (p.size() > 1 && r.counts.count(p[1]) && r.counts.at(p[1]) > 0) rows.push_back(row);
    }
  }
  std::vector<std::string> missing;
  for (const auto& t : todo)
    if (!found.count(t.first)) missing.push_back(t.first);
  if (flag(a, "inverse")) {  // each series' inverse beside it (CODE-: 1/price, its high from the low)
    std::vector<std::string> inv;
    for (const auto& row : rows) {
      const auto p = split(row, ',');
      if (p.size() < 7) continue;
      const double o = std::atof(p[2].c_str()), h = std::atof(p[3].c_str()), l = std::atof(p[4].c_str()), c = std::atof(p[5].c_str());
      if (std::min({o, h, l, c}) > 0) {
        char b[256];
        std::snprintf(b, sizeof b, "%s,%s-,%.8g,%.8g,%.8g,%.8g,%s", p[0].c_str(), p[1].c_str(), 1 / o, 1 / l, 1 / h, 1 / c, p[6].c_str());
        inv.push_back(b);
      }
    }
    rows.insert(rows.end(), inv.begin(), inv.end());
  }
  std::string csv = "date,ticker,open,high,low,close,volume\n";
  for (const auto& r : rows) csv += r + "\n";
  writeFile(out, csv);
  std::string miss;
  for (const auto& m : missing) miss += (miss.empty() ? "" : ", ") + m;
  std::cout << fs::path(list).filename().string() << ": " << found.size() << " of " << todo.size() << " series" << (missing.empty() ? "" : " (none at EODData: " + miss + ")")
            << (flag(a, "inverse") ? ", each with its inverse" : "") << "; " << rows.size() << " rows in " << out << "\n";
  return 0;
}

// ------------------------------------------------------------------ profiles (was profiles.mjs)

// At most n bytes, never cutting a UTF-8 character in two (a cut one makes the JSON unwritable).
std::string utf8Cut(const std::string& t, std::size_t n) {
  if (t.size() <= n) return t;
  while (n > 0 && (static_cast<unsigned char>(t[n]) & 0xC0) == 0x80) --n;
  return t.substr(0, n);
}

int cmdProfiles(const std::vector<std::string>& a) {
  if (!std::getenv("EODDATA_API_KEY")) die("Set EODDATA_API_KEY in the environment.");
  const std::string out = opt(a, "out"), fundDir = opt(a, "fundamentals"), uni = opt(a, "universe");
  std::set<std::string> universe;
  if (!uni.empty()) {
    std::istringstream in(readFile(uni));
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto p = split(line, ',');
      if (p.size() > 1) universe.insert(std::regex_replace(p[1], std::regex("^[A-Z]+:"), ""));
    }
  }
  const std::vector<std::string> FUND{"marketCapitalization", "ebitda", "peg", "bookValue", "dividendPerShare", "dividendYield", "earningsPerShare", "revenuePerShare",
                                      "priceToSales", "priceToBook", "sharesOutstanding", "grossMargin", "profitMargin", "operatingMargin", "returnOnAssets", "returnOnEquity",
                                      "revenue", "grossProfit", "trailingPE", "forwardPE", "totalCash", "totalCashPerShare", "totalDebt", "bookValuePerShare"};
  const auto list = [](const std::string& route, long& status) {
    try {
      const json b = api(route, {}, 3);
      status = 200;
      return asList(b, {"data", "items"});
    } catch (const std::exception& e) {
      const std::string m = e.what();
      const auto k = m.find("HTTP ");
      status = k == std::string::npos ? 0 : std::atol(m.c_str() + k + 5);
      return json::array();
    }
  };
  const auto filled = [](const json& items, const char* name) {
    std::size_t n = 0;
    for (const auto& x : items) {
      const json* v = field(x, {name});
      n += v && !v->is_null() && !(v->is_string() && v->get<std::string>().empty()) && !(v->is_number() && v->get<double>() == 0);
    }
    return n;
  };
  const std::string stamp = today();
  // Incremental: the day's fundamentals once a day (fundamentals/<EX>/<date>.csv.gz kept), the profiles (sectors,
  // names, descriptions: they rarely change) once every --max-age days (7; the date kept in profiles/<EX>.fetched).
  const bool refresh = flag(a, "refresh"), probe = flag(a, "probe");
  const int maxAge = std::atoi(opt(a, "max-age", "7").c_str());
  for (const auto& ex : split(opt(a, "exchanges", "LSE"), ',')) {
    if (ex.empty()) continue;
    const fs::path stampFile = out.empty() ? fs::path() : fs::path(out) / (ex + ".fetched");
    std::string last;
    const bool profOld = !out.empty() && (refresh || !fs::exists(fs::path(out) / (ex + ".json")) || !tryRead(stampFile, last) || !isDay(last.substr(0, 10)) ||
                                          daysBetween(last.substr(0, 10), stamp) >= maxAge);
    const bool fundOld = !fundDir.empty() && (refresh || !fs::exists(fs::path(fundDir) / ex / (stamp + ".csv.gz")));
    if (!probe && !profOld && !fundOld) {
      std::cout << ex << ": profiles from " << last.substr(0, 10) << " and today's fundamentals already in the store (nothing downloaded)\n";
      continue;
    }
    long st = 0;
    const json p = (profOld || probe) ? list("/Profile/List/" + ex, st) : json::array();
    std::set<std::string> sectors, industries;
    for (const auto& x : p) {
      if (!str(field(x, {"sector"})).empty()) sectors.insert(str(field(x, {"sector"})));
      if (!str(field(x, {"industry"})).empty()) industries.insert(str(field(x, {"industry"})));
    }
    if (profOld || probe) std::cout << ex << " profiles: HTTP " << st << ", " << p.size() << " symbols, sector on " << filled(p, "sector") << ", industry on " << filled(p, "industry") << "; "
              << sectors.size() << " sectors, " << industries.size() << " industries\n";
    long fst = 0;
    const json f = (profOld || fundOld || probe) ? list("/Fundamental/List/" + ex, fst) : json::array();
    if (profOld && !p.empty()) {  // with the share's name (the store's symbol list), description and key fundamentals
      std::map<std::string, std::string> names;
      const fs::path dir = fs::path(out).parent_path() / ex;
      std::string newest;
      if (fs::exists(dir))
        for (const auto& e : fs::directory_iterator(dir)) {
          const std::string n = e.path().filename().string();
          if (std::regex_match(n, std::regex("^symbols-\\d{4}-\\d{2}-\\d{2}\\.json$"))) newest = std::max(newest, n);
        }
      if (!newest.empty())
        for (const auto& s : asList(json::parse(readFile(dir / newest)), {"data", "items", "symbols"})) names[str(field(s, {"code", "symbolCode", "symbol"}))] = str(field(s, {"name", "description"}));
      std::map<std::string, const json*> fund;
      for (const auto& x : f) fund[str(field(x, {"symbolCode", "code"}))] = &x;
      json rows = json::array();
      std::size_t nName = 0, nDesc = 0, nFund = 0;
      for (const auto& x : p) {
        const std::string code = str(field(x, {"symbolCode", "code"}));
        if (code.empty()) continue;
        json r{{"code", code}, {"name", names.count(code) ? names[code] : str(field(x, {"name"}))}, {"sector", str(field(x, {"sector"}))}, {"industry", str(field(x, {"industry"}))},
               {"description", utf8Cut(str(field(x, {"description", "summary", "businessSummary"})), 600)}};
        const auto fi = fund.find(code);
        if (fi != fund.end())
          for (const auto& [key, name] : std::vector<std::pair<const char*, const char*>>{{"marketCap", "marketCapitalization"}, {"pe", "trailingPE"}, {"dividendYield", "dividendYield"}}) {
            const double v = num(field(*fi->second, {name}));
            if (std::isfinite(v) && v != 0) r[key] = v;
          }
        nName += !r["name"].get<std::string>().empty(), nDesc += !r["description"].get<std::string>().empty(), nFund += r.contains("marketCap");
        rows.push_back(r);
      }
      writeFile(fs::path(out) / (ex + ".json"), rows.dump(-1, ' ', false, json::error_handler_t::replace));  // a bad byte from the API: U+FFFD
      writeFile(stampFile, stamp + "\n");
      std::cout << ex << " profiles kept: " << rows.size() << ", a name on " << nName << ", a description on " << nDesc << ", fundamentals on " << nFund << "\n";
    }
    if (fundOld && fst == 200) {  // the day's fundamentals of the scan's shares, a history building up
      std::string csv = "code";
      for (const auto& n : FUND) csv += "," + n;
      csv += "\n";
      std::size_t n = 0;
      for (const auto& x : f) {
        const std::string code = str(field(x, {"symbolCode", "code"}));
        if (!universe.empty() && !universe.count(code)) continue;
        csv += code;
        for (const auto& name : FUND) {
          const double v = num(field(x, {name.c_str()}));
          csv += "," + (std::isfinite(v) ? shortest(v) : std::string());
        }
        csv += "\n", ++n;
      }
      std::string gz(compressBound(uLong(csv.size())) + 64, '\0');
      z_stream zs{};
      deflateInit2(&zs, 9, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
      zs.next_in = reinterpret_cast<Bytef*>(csv.data()), zs.avail_in = uInt(csv.size());
      zs.next_out = reinterpret_cast<Bytef*>(gz.data()), zs.avail_out = uInt(gz.size());
      deflate(&zs, Z_FINISH);
      gz.resize(zs.total_out);
      deflateEnd(&zs);
      writeFile(fs::path(fundDir) / ex / (stamp + ".csv.gz"), gz);
      std::cout << ex << " fundamentals kept: HTTP " << fst << ", " << n << " of the model's shares\n";
    }
    if (flag(a, "probe")) {
      std::cout << ex << " fundamentals: HTTP " << fst << ", " << f.size() << " symbols;";
      for (const char* n : {"marketCapitalization", "priceToBook", "trailingPE", "returnOnEquity", "totalDebtToEquity", "beta", "sharesOutstanding"}) std::cout << " " << n << " on " << filled(f, n);
      std::cout << "\n";
    }
  }
  return 0;
}

// ------------------------------------------------------------------ aim (was aim.mjs)

int cmdAim(const std::vector<std::string>& a) {
  const std::string out = a.size() > 1 && a[1].rfind("--", 0) != 0 ? a[1] : "tools/eoddata/aim.txt";
  // Incremental: the list downloaded once a day and kept in the store (--cache, lists/aim.txt beside lists/aim.fetched);
  // a later run that day reads the kept copy.
  const std::string cache = opt(a, "cache");
  std::string when, kept;
  if (!cache.empty() && !flag(a, "refresh") && tryRead(cache + ".fetched", when) && when.substr(0, 10) == today() && tryRead(cache, kept) && !kept.empty()) {
    writeFile(out, kept);
    std::cout << "AIM list: today's from the store (nothing downloaded)\n";
    return 0;
  }
  std::set<std::string> tickers;
  const std::regex link("/quote/aim/([A-Za-z0-9.]+)/");
  for (int page = 1; page <= 5; ++page) {
    const Response r = get("https://stockanalysis.com/list/london-stock-exchange-aim/" + (page > 1 ? "?page=" + std::to_string(page) : std::string()));
    if (r.status != 200) break;
    const std::size_t before = tickers.size();
    for (std::sregex_iterator it(r.body.begin(), r.body.end(), link), end; it != end; ++it) tickers.insert((*it)[1]);
    if (tickers.size() == before) break;
  }
  std::vector<std::string> old;
  std::string text;
  if (tryRead(out, text)) {
    std::istringstream in(text);
    for (std::string w; in >> w;) old.push_back(w);
  }
  if (double(tickers.size()) < std::max(300.0, 0.8 * double(old.size()))) {  // far shorter: the page changed; the last kept
    if (!cache.empty() && tryRead(cache, kept) && !kept.empty()) writeFile(out, kept);  // the store's last good list
    std::cout << "AIM list: " << tickers.size() << " tickers found, kept the last (" << old.size() << ")\n";
  } else {
    std::string s;
    for (const auto& t : tickers) s += t + "\n";
    writeFile(out, s);
    if (!cache.empty()) writeFile(cache, s), writeFile(cache + ".fetched", today() + "\n");
    std::cout << "AIM list: " << tickers.size() << " tickers (was " << old.size() << ")\n";
  }
  return 0;
}

// ------------------------------------------------------------------ symbols, snapshot (were symbols.mjs, snapshot.mjs)

int cmdSymbols(const std::vector<std::string>& a) {
  if (!std::getenv("EODDATA_API_KEY")) die("Set EODDATA_API_KEY in the environment.");
  if (flag(a, "exchanges")) {
    json list = json::array();
    try {
      list = asList(api("/Exchange/List"), {"data", "items", "exchanges"});
    } catch (...) {
    }
    std::cout << "exchanges: " << list.size() << "\n";
    for (const auto& e : list) std::cout << "  " << str(field(e, {"code"})) << "\t" << str(field(e, {"name"})) << "\n";
  }
  for (const auto& part : split(opt(a, "find"), ';')) {
    if (part.empty()) continue;
    const auto k = part.find(':');
    const std::string ex = part.substr(0, k);
    std::vector<std::string> pre;
    if (k != std::string::npos)
      for (auto& p : split(part.substr(k + 1), ','))
        if (!p.empty()) pre.push_back(p);
    json list;
    try {
      list = asList(api("/Symbol/List/" + ex), {"data", "items", "symbols"});
    } catch (const std::exception& e) {
      std::cout << ex << ": " << e.what() << "\n";
      continue;
    }
    std::cout << ex << ": " << list.size() << " symbols\n";
    for (const auto& s : list) {
      const std::string code = str(field(s, {"code", "symbol"})), name = str(field(s, {"name", "description"}));
      bool hit = pre.empty();
      for (const auto& p : pre) hit = hit || code.rfind(p, 0) == 0;
      if (hit) std::cout << "  " << code << "\t" << name << "\n";
    }
  }
  return 0;
}

int cmdSnapshot(const std::vector<std::string>& a) {
  if (!std::getenv("EODDATA_API_KEY")) die("Set EODDATA_API_KEY in the environment (never pass it on the command line).");
  const std::string from = opt(a, "from", "2019-01"), now = today();
  std::vector<std::string> months;
  for (std::string m = from; m <= now.substr(0, 7);) {
    months.push_back(m);
    int y = std::atoi(m.substr(0, 4).c_str()), mo = std::atoi(m.substr(5, 2).c_str()) + 1;
    if (mo > 12) mo = 1, ++y;
    char b[32];
    std::snprintf(b, sizeof b, "%04d-%02d", y, mo);
    m = b;
  }
  const auto lastDay = [](const std::string& mo) {
    int y = std::atoi(mo.substr(0, 4).c_str()), m = std::atoi(mo.substr(5, 2).c_str()) + 1;
    if (m > 12) m = 1, ++y;
    char b[32];
    std::snprintf(b, sizeof b, "%04d-%02d-01", y, m);
    return addDays(b, -1);
  };
  for (const auto& sym : split(opt(a, "symbols", "EUREX:FX.C"), ',')) {
    if (sym.empty()) continue;
    const auto k = sym.find(':');
    const std::string ex = sym.substr(0, k), code = sym.substr(k + 1);
    const fs::path dir = store::root() / "intraday" / ex / store::safe(code);
    fs::create_directories(dir);
    int fetched = 0, kept = 0, empty = 0;
    std::string first;
    for (const auto& mo : months) {
      const fs::path f = dir / (mo + ".json");
      std::string text;
      if (tryRead(f, text) && json::parse(text).value("complete", false)) {
        ++kept;
        continue;
      }
      json rows;
      try {
        rows = asList(api("/Quote/List/" + ex + "/" + escape(code), {{"Interval", "5"}, {"FromDateStamp", mo + "-01"}, {"ToDateStamp", lastDay(mo)}}, 6, 3000));
      } catch (const std::exception& e) {
        if (std::string(e.what()).find("HTTP 404") == std::string::npos) {
          std::cerr << "  " << e.what() << "\n";
          continue;
        }
        rows = json::array();
      }
      json days = json::object();
      for (const auto& q : rows) {
        const std::string s = str(field(q, {"dateStamp", "date"}));
        const double o = num(field(q, {"open"})), h = num(field(q, {"high"})), l = num(field(q, {"low"})), c = num(field(q, {"close"})), v = num(field(q, {"volume"}));
        if (!(c > 0)) continue;
        days[s.substr(0, 10)].push_back(json::array({s, o, h, l, c, std::isfinite(v) ? v : 0}));
        if (first.empty()) first = s;
      }
      empty += rows.empty();
      writeFile(f, json{{"complete", lastDay(mo) < now}, {"days", days}}.dump());
      ++fetched;
    }
    std::vector<std::string> have;
    for (const auto& e : fs::directory_iterator(dir))
      if (e.path().extension() == ".json") have.push_back(e.path().filename().string());
    std::sort(have.begin(), have.end());
    std::size_t nd = 0;
    for (const auto& x : have) nd += json::parse(readFile(dir / x))["days"].size();
    std::cout << sym << ": " << have.size() << " months stored (" << fetched << " fetched, " << kept << " already complete, " << empty << " empty), " << nd << " days, first "
              << (have.empty() ? "-" : have.front()) << ", a stamp: " << (first.empty() ? "-" : first) << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> a(argv + 1, argv + argc);
  if (a.empty()) die("usage: ofm_eod sync|push|shares|series|market|profiles|aim|symbols|snapshot [options] (see the source's head)");
  const std::string cmd = a[0];
  try {
    if (cmd == "sync") return cmdSync(a);
    if (cmd == "push") return cmdPush(a);
    if (cmd == "shares") return cmdShares(a);
    if (cmd == "series") return cmdSeries(a);
    if (cmd == "market") return cmdMarket(a);
    if (cmd == "profiles") return cmdProfiles(a);
    if (cmd == "aim") return cmdAim(a);
    if (cmd == "symbols") return cmdSymbols(a);
    if (cmd == "snapshot") return cmdSnapshot(a);
  } catch (const std::exception& e) {
    die(cmd + ": " + e.what(), 1);
  }
  die("unknown command " + cmd);
}
