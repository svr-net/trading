// The backtest's fills against 5-minute bars (research; results only).
//
// The model's backtest decides each trade on daily bars (apps/ofm_research.cpp, trade()): a limit a
// fraction of the previous close; filled at the open when the open is at or under it, else during the
// day when the low is half a spread under it; then sold at the stop, the take-profit or the close.
// Daily bars cannot tell when the low came, nor whether the stop or the take-profit came first, so
// the backtest takes the cautious reading: the stop first when both were touched, and no take-profit
// for a fill during the day. This replays every filled trade of the backtest's log (X_TRADELOG) on
// the day's 5-minute bars and reports:
//   - fills the 5-minute bars confirm (a bar's low half a spread under the limit, or the open under it);
//   - each trade's result on the 5-minute path (the first of the stop and the take-profit in time; the
//     same bar: the stop) next to the same rules on the daily bar, and the log's own net result;
//   - the fills by the hour of the day and their results (a late fill has little time to recover).
// The daily bar's open, close and previous close are the ones the backtest used (--csv, --override);
// the 5-minute bars only order the day's moves. The buy-back after a stop is not replayed.
//
//   ofm_fills --csv stocks.csv [--override ibkr.csv] --log tl.csv --bars bars5.csv
//     log: the backtest's trade log (date,ticker,k,pred,sig,liq,spread,low/o-1,high/o-1,close/o-1,
//          stop,take,mkt,exit,ret,o/pc-1,limit,filled); bars: date,ticker,stamp,open,high,low,close
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "swing.hpp"

namespace {
std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> f;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, ',')) f.push_back(x);
  return f;
}
struct Bar { std::string stamp; double o, h, l, c; };
struct Sum {
  double n = 0, s = 0, ss = 0;
  void add(double x) { n += 1, s += x, ss += x * x; }
  double mean() const { return n > 0 ? s / n : 0.0; }
  double t() const { if (n < 2) return 0.0; const double m = mean(), v = (ss - n * m * m) / (n - 1); return v > 0 ? m / std::sqrt(v / n) : 0.0; }
};
int hourOf(const std::string& stamp) {  // the stamp's hour as EODData gives it (its own clock)
  const auto p = stamp.find_first_of("T ");
  return p != std::string::npos && p + 3 <= stamp.size() ? std::atoi(stamp.substr(p + 1, 2).c_str()) : -1;
}
}  // namespace

int main(int argc, char** argv) {
  std::string csv, overrideCsv, logFile, barsFile;
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    if (k + 1 >= argc) break;
    if (a == "--csv") csv = argv[++k];
    else if (a == "--override") overrideCsv = argv[++k];
    else if (a == "--log") logFile = argv[++k];
    else if (a == "--bars") barsFile = argv[++k];
  }
  if (csv.empty() || logFile.empty() || barsFile.empty()) {
    std::fprintf(stderr, "usage: ofm_fills --csv stocks.csv [--override ibkr.csv] --log tl.csv --bars bars5.csv\n");
    return 2;
  }
  swing::Panel p = swing::load(csv);
  if (!overrideCsv.empty()) std::printf("fills: IBKR's bars: %s\n", swing::override(p, overrideCsv).c_str());
  std::map<std::string, std::size_t> dayIx, tickIx;
  for (std::size_t t = 0; t < p.T; ++t) dayIx[p.days[t]] = t;
  for (std::size_t i = 0; i < p.N; ++i) tickIx[p.tickers[i]] = i;

  std::map<std::pair<std::string, std::string>, std::vector<Bar>> bars;  // (date, ticker) -> the day's bars in time
  {
    std::ifstream in(barsFile);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto f = split(line);
      if (f.size() < 7) continue;
      const double o = std::atof(f[3].c_str()), h = std::atof(f[4].c_str()), l = std::atof(f[5].c_str()), c = std::atof(f[6].c_str());
      if (!(c > 0 && h >= l && l > 0)) continue;
      bars[{f[0], f[1]}].push_back({f[2], o > 0 ? o : c, h, l, c});
    }
    for (auto& e : bars) std::sort(e.second.begin(), e.second.end(), [](const Bar& a, const Bar& b) { return a.stamp < b.stamp; });
  }

  Sum logNet, dailyG, fiveG, diff, early, late;
  long fills = 0, withBars = 0, confirmed = 0, notConfirmed = 0, noDay = 0;
  long exitD[3] = {0, 0, 0}, exit5[3] = {0, 0, 0}, changed = 0;
  std::map<int, Sum> byHour;
  std::ifstream in(logFile);
  std::string line;
  while (std::getline(in, line)) {
    const auto f = split(line);
    if (f.size() < 18) continue;
    const int filled = std::atoi(f[17].c_str());
    if (filled <= 0) continue;
    ++fills;
    const auto di = dayIx.find(f[0]);
    const auto ti = tickIx.find(f[1]);
    if (di == dayIx.end() || ti == tickIx.end() || di->second == 0) { ++noDay; continue; }
    const std::size_t t = di->second, i = ti->second;
    const double pc = p.C[(t - 1) * p.N + i], o = p.O[t * p.N + i], c = p.C[t * p.N + i];
    if (!(pc > 0 && o > 0 && c > 0)) { ++noDay; continue; }
    const double half = std::exp(0.5 * std::atof(f[6].c_str()));
    const double L = pc * (1 + std::atof(f[16].c_str()));
    const double entry = filled == 1 ? (L <= pc ? L : std::min(o * half, L)) : L;
    const double rs = std::atof(f[10].c_str()), rk = std::atof(f[11].c_str());
    const double stop = rs > -1 ? entry * (1 + rs) : 0.0, take = rk != 0 ? entry * (1 + rk) : 1e300;
    const double lo = o * (1 + std::atof(f[7].c_str())), hi = o * (1 + std::atof(f[8].c_str()));
    const bool atOpen = filled == 1;
    // The daily bar, the backtest's own rules.
    double pxD = c;
    int exD = 0;
    if (atOpen && o <= stop) pxD = o / half, exD = 1;
    else if (lo <= stop) pxD = stop / half, exD = 1;
    else if (atOpen && hi >= take * half) pxD = std::max(take, o), exD = 2;
    const double gD = pxD / entry - 1;
    const auto b = bars.find({f[0], f[1]});
    if (b == bars.end() || b->second.size() < 10) continue;  // no 5-minute day (or a thin one)
    ++withBars;
    const auto& v = b->second;
    // The 5-minute path: the fill, then the first of the stop and the take-profit.
    std::size_t k0 = 0;
    bool fillOk = atOpen;
    if (!atOpen)
      for (; k0 < v.size(); ++k0)
        if (v[k0].l <= L / half) { fillOk = true; break; }
    if (!fillOk) {  // the daily low reached the limit, the 5-minute bars did not: no fill
      ++notConfirmed, logNet.add(std::atof(f[14].c_str())), dailyG.add(gD), fiveG.add(0.0), diff.add(-gD), ++exitD[exD];
      continue;
    }
    ++confirmed;
    double px5 = c;
    int ex5 = 0;
    if (atOpen && o <= stop) px5 = o / half, ex5 = 1;
    else
      for (std::size_t k = k0; k < v.size(); ++k) {
        if (v[k].l <= stop) { px5 = stop / half, ex5 = 1; break; }  // the same bar as the take-profit: the stop
        if (k > k0 || atOpen) {  // the fill's own bar: its high may have come before the fill
          if (v[k].h >= take * half) { px5 = std::max(take, k == 0 && atOpen ? o : take), ex5 = 2; break; }
        }
      }
    const double g5 = px5 / entry - 1;
    logNet.add(std::atof(f[14].c_str())), dailyG.add(gD), fiveG.add(g5), diff.add(g5 - gD);
    ++exitD[exD], ++exit5[ex5], changed += ex5 != exD;
    const int h = atOpen ? hourOf(v[0].stamp) : hourOf(v[k0].stamp);
    byHour[h].add(g5);
    (atOpen || k0 < v.size() / 2 ? early : late).add(g5);
  }
  std::printf("fills: %ld filled trades in the log, %ld with a 5-minute day (%ld without the daily bar)\n", fills, withBars, noDay);
  if (withBars == 0) return 0;
  std::printf("fills: confirmed by the 5-minute bars %ld of %ld (%.1f%%); the daily low only: %ld\n", confirmed, withBars, 100.0 * double(confirmed) / double(withBars), notConfirmed);
  std::printf("fills: a trade's result before costs: daily rules %+.3f%%, 5-minute path %+.3f%% (difference %+.3f%%, t %.2f); the log's net %+.3f%%\n",
              100 * dailyG.mean(), 100 * fiveG.mean(), 100 * diff.mean(), diff.t(), 100 * logNet.mean());
  std::printf("fills: exits close/stop/take: daily %ld/%ld/%ld, 5-minute %ld/%ld/%ld, changed %ld\n", exitD[0], exitD[1], exitD[2], exit5[0], exit5[1], exit5[2], changed);
  std::printf("fills: the first half of the day %ld fills %+.3f%%, the second half %ld fills %+.3f%%\n", long(early.n), 100 * early.mean(), long(late.n), 100 * late.mean());
  for (const auto& [h, s] : byHour) std::printf("fills:   hour %02d (EODData's clock): %ld fills, %+.3f%% before costs\n", h, long(s.n), 100 * s.mean());
  return 0;
}
