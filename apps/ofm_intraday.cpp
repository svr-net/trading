// Intraday momentum on 5-minute bars (research; results only): does the first half hour's move
// predict the last half hour's? (Gao, Han, Li and Zhou 2018, "Market intraday momentum", JFE.)
//
// From the store's 5-minute bars (tools/eoddata/intraday_export.mjs: date,ticker,time,high,low,close;
// a bar's time is its start, its close the price at its end) and the daily bars (for the previous
// close and the closing auction's price):
//   signal  the share's return from the previous close to 10:00 (the close of the 09:55 bar)
//   trade   bought at 15:30 (the close of the 15:25 bar, paying half a spread), sold in the closing auction (the day's close: one price, no spread)
// All on the same day: nothing is held overnight. Books, each day over the shares with both bars:
//   long top    the fifth with the highest signal, held long
//   long up     every share whose signal is positive
//   sign        long the shares whose signal is up, short those whose signal is down (gross of a
//               short's borrow), the paper's own rule
//   market      every share, long (what the last half hour gives without a signal)
// Costs: IBKR's commission on both sides (apps/swing.hpp, the US: fixed or tiered) on --capital a
// trade, and half a spread on the 15:30 buy, the spread taken as --spread-bp (default 2 bp: the most
// traded US shares). Reported for each book: the mean a year, the Sharpe ratio, the t statistic, and
// the first and the second half of the days apart.
//
//   ofm_intraday --intraday intraday.csv --csv stocks.csv [--capital 5000] [--spread-bp 2] [--tiered]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
struct Series {
  double sum = 0, sum2 = 0;
  long n = 0;
  void add(double x) { sum += x, sum2 += x * x, ++n; }
  double mean() const { return n ? sum / double(n) : 0; }
  double sd() const { return n > 1 ? std::sqrt(std::max(0.0, sum2 / double(n) - mean() * mean())) : 0; }
};
struct Stat {
  Series all, h1, h2;
  void add(double x, bool first) { all.add(x), (first ? h1 : h2).add(x); }
};
std::string fmt(const Series& s) {
  char b[96];
  const double sd = s.sd();
  std::snprintf(b, sizeof b, "%+7.1f%% %5.2f %+6.1f", 25200 * s.mean(), sd > 0 ? s.mean() / sd * std::sqrt(252.0) : 0.0,
                sd > 0 ? s.mean() / sd * std::sqrt(double(s.n)) : 0.0);
  return b;
}
std::vector<std::string> split(const std::string& l) {
  std::vector<std::string> out;
  std::string cell;
  std::istringstream in(l);
  while (std::getline(in, cell, ',')) out.push_back(cell);
  return out;
}
struct Day {
  double at10 = 0, at1530 = 0;  // the closes at 10:00 and at 15:30
};
}  // namespace

int main(int argc, char** argv) {
  std::string intraday, csv;
  double capital = 5000, spreadBp = 2;
  bool tiered = false;
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--intraday" && k + 1 < argc) intraday = argv[++k];
    else if (a == "--csv" && k + 1 < argc) csv = argv[++k];
    else if (a == "--capital" && k + 1 < argc) capital = std::atof(argv[++k]);
    else if (a == "--spread-bp" && k + 1 < argc) spreadBp = std::atof(argv[++k]);
    else if (a == "--tiered") tiered = true;
  }
  if (intraday.empty() || csv.empty()) {
    std::fprintf(stderr, "usage: ofm_intraday --intraday intraday.csv --csv stocks.csv [--capital 5000] [--spread-bp 2] [--tiered]\n");
    return 2;
  }
  // The 5-minute bars this needs: per (date, ticker), the closes at 10:00 and at 15:30.
  std::map<std::string, std::unordered_map<std::string, Day>> bars;
  {
    std::ifstream in(intraday);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto c = split(line);
      if (c.size() < 6) continue;
      const bool b10 = c[2] == "09:55", b1530 = c[2] == "15:25";
      if (!b10 && !b1530) continue;
      Day& d = bars[c[0]][c[1]];
      (b10 ? d.at10 : d.at1530) = std::atof(c[5].c_str());
    }
  }
  // The daily closes (the previous close and the closing auction's price), for the tickers above.
  std::unordered_map<std::string, std::map<std::string, double>> closes;
  {
    std::ifstream in(csv);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto c = split(line);
      if (c.size() < 7) continue;
      const double close = std::atof(c[5].c_str());
      if (close > 0) closes[c[1]][c[0]] = close;
    }
  }
  // IBKR's US commission on one side of a trade of `capital` at price px (fixed: USD 0.005 a share,
  // min 1, max 1% of the value; tiered: USD 0.0035 a share, min 0.35, plus exchange and clearing fees
  // of about USD 0.0005 a share), as a fraction of the trade.
  const auto side = [&](double px) {
    const double shares = std::floor(capital / px);
    if (shares < 1) return 1.0;
    const double value = shares * px;
    const double c = tiered ? std::max(0.35, 0.0035 * shares) + 0.0005 * shares : std::max(1.0, 0.005 * shares);
    return std::min(c, 0.01 * value) / value;
  };
  std::vector<std::string> days;
  for (const auto& kv : bars) days.push_back(kv.first);
  const std::size_t mid = days.size() / 2;
  Stat top, up, sign, mkt, topG, upG, signG, mktG;
  std::size_t used = 0;
  for (std::size_t q = 0; q < days.size(); ++q) {
    const std::string& day = days[q];
    struct Row { double sig, ret, cost; };
    std::vector<Row> rows;
    for (const auto& [ticker, d] : bars[day]) {
      const auto it = closes.find(ticker);
      if (it == closes.end() || !(d.at10 > 0) || !(d.at1530 > 0)) continue;
      const auto today = it->second.find(day);
      if (today == it->second.end() || today == it->second.begin()) continue;
      const double prev = std::prev(today)->second, close = today->second;
      const double sig = d.at10 / prev - 1, ret = close / d.at1530 - 1;
      if (!std::isfinite(sig) || !std::isfinite(ret) || std::fabs(sig) > 0.5 || std::fabs(ret) > 0.5) continue;
      const double cost = side(d.at1530) + side(close) + 0.5 * spreadBp * 1e-4;
      rows.push_back({sig, ret, cost});
    }
    if (rows.size() < 20) continue;
    ++used;
    const bool first = q < mid;
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.sig > b.sig; });
    const std::size_t k = rows.size() / 5;
    double g = 0, n = 0;
    for (std::size_t r = 0; r < k; ++r) g += rows[r].ret, n += rows[r].ret - rows[r].cost;
    topG.add(g / double(k), first), top.add(n / double(k), first);
    double gu = 0, nu = 0, cu = 0, gs = 0, ns = 0, gm = 0, nm = 0;
    for (const auto& r : rows) {
      gm += r.ret, nm += r.ret - r.cost;
      const double s = r.sig > 0 ? 1.0 : r.sig < 0 ? -1.0 : 0.0;
      gs += s * r.ret, ns += s * r.ret - std::fabs(s) * r.cost;
      if (r.sig > 0) gu += r.ret, nu += r.ret - r.cost, cu += 1;
    }
    const double m = double(rows.size());
    mktG.add(gm / m, first), mkt.add(nm / m, first), signG.add(gs / m, first), sign.add(ns / m, first);
    upG.add(cu > 0 ? gu / cu : 0.0, first), up.add(cu > 0 ? nu / cu : 0.0, first);  // no share up: cash
  }
  std::printf("intraday: %zu days (%s to %s) with 20+ shares; signal: previous close to 10:00; bought 15:30, sold in the closing auction; costs: IBKR %s on %.0f a trade, half of %.1f bp spread on the buy\n",
              used, days.empty() ? "" : days.front().c_str(), days.empty() ? "" : days.back().c_str(), tiered ? "tiered" : "fixed", capital, spreadBp);
  std::printf("intraday: %-24s %24s | %24s | %24s\n", "book", "all days: a year Sh t", "first half", "second half");
  const auto line = [&](const char* name, const Stat& s) { std::printf("intraday: %-24s %s | %s | %s\n", name, fmt(s.all).c_str(), fmt(s.h1).c_str(), fmt(s.h2).c_str()); };
  line("market, gross", mktG), line("market, net", mkt);
  line("long top fifth, gross", topG), line("long top fifth, net", top);
  line("long up, gross", upG), line("long up, net", up);
  line("sign (long/short), gross", signG), line("sign (long/short), net", sign);
  return 0;
}
