// Delayed entry, measured on 5-minute bars (research; results only).
//
// The model's buy is a limit some daily standard deviations below the previous close, placed before
// the open and sold at the close. The first minutes after the open are the most volatile: does
// waiting before placing the limit (so the opening swings cannot fill it) buy better? For each
// share-day with 5-minute bars, a limit d sd below the previous close (sd: the last 20 daily
// close-to-close log returns, the past only) is placed at the open or only after a delay of D
// minutes. Placed at the open it fills at the open when the open is at or below it; placed later it
// fills at the price of that moment when the price is already below it, else at the limit when a
// later bar trades below it. Every fill is sold at the day's close. Reported for each (d, D): the
// orders, how many filled, the mean net result per fill and per order (an unfilled order: 0), the
// share of fills that gained, and the t statistic of the per-order mean.
//
//   ofm_delay --daily stocks.csv --bars intraday.csv [--cost 0.004]
//     daily: date,ticker,open,high,low,close,volume; bars: date,ticker,time,high,low,close[,volume]
//     (time: the bar's start, exchange time, 09:30 the first)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> f;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, ',')) f.push_back(x);
  return f;
}
int minutes(const std::string& t) { return t.size() >= 5 ? std::atoi(t.substr(0, 2).c_str()) * 60 + std::atoi(t.substr(3, 2).c_str()) : -1; }

struct Bar { int t; double h, l, c; };
struct Daily { std::string d; double o, c; };
struct Stat {
  long orders = 0, fills = 0, wins = 0;
  double sum = 0, sum2 = 0;  // per order (unfilled 0)
  void add(bool filled, double r) {
    ++orders;
    if (!filled) return;
    ++fills, wins += r > 0, sum += r, sum2 += r * r;
  }
};
}  // namespace

int main(int argc, char** argv) {
  std::string dailyFile, barsFile;
  double cost = 0.004;  // a round trip: IBKR's US minimum of $1 a side on a $500 order
  for (int k = 1; k + 1 < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--daily") dailyFile = argv[++k];
    else if (a == "--bars") barsFile = argv[++k];
    else if (a == "--cost") cost = std::atof(argv[++k]);
  }
  if (dailyFile.empty() || barsFile.empty()) {
    std::fprintf(stderr, "usage: ofm_delay --daily stocks.csv --bars intraday.csv [--cost 0.004]\n");
    return 2;
  }
  std::map<std::string, std::vector<Daily>> daily;  // ticker -> days in date order
  {
    std::ifstream in(dailyFile);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto f = split(line);
      if (f.size() < 6) continue;
      const double o = std::atof(f[2].c_str()), c = std::atof(f[5].c_str());
      if (o > 0 && c > 0) daily[f[1]].push_back({f[0], o, c});
    }
    for (auto& [t, v] : daily) std::sort(v.begin(), v.end(), [](const Daily& a, const Daily& b) { return a.d < b.d; });
  }
  std::map<std::string, std::vector<Bar>> bars;  // ticker|date -> the day's bars
  {
    std::ifstream in(barsFile);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto f = split(line);
      if (f.size() < 6) continue;
      const Bar b{minutes(f[2]), std::atof(f[3].c_str()), std::atof(f[4].c_str()), std::atof(f[5].c_str())};
      if (b.t >= 0 && b.h > 0 && b.l > 0 && b.c > 0) bars[f[1] + "|" + f[0]].push_back(b);
    }
    for (auto& [k, v] : bars) std::sort(v.begin(), v.end(), [](const Bar& a, const Bar& b) { return a.t < b.t; });
  }
  const std::vector<double> depth = {-1, 0, 0.5, 1, 1.5, 2};  // -1: no limit, bought at the price of the moment
  const std::vector<int> delay = {0, 15, 30, 60, 90, 120};
  const int open = 9 * 60 + 30;
  std::vector<std::vector<Stat>> st(depth.size(), std::vector<Stat>(delay.size()));
  long used = 0, skipped = 0;
  for (const auto& [key, path] : bars) {
    const auto bar = key.find('|');
    const std::string t = key.substr(0, bar), d = key.substr(bar + 1);
    const auto it = daily.find(t);
    if (it == daily.end()) { ++skipped; continue; }
    const auto& v = it->second;
    const auto at = std::lower_bound(v.begin(), v.end(), d, [](const Daily& x, const std::string& s) { return x.d < s; });
    if (at == v.end() || at->d != d || at - v.begin() < 21 || path.size() < 60 || path.front().t != open) { ++skipped; continue; }
    const long i = at - v.begin();
    double m = 0, m2 = 0;
    for (long j = i - 20; j < i; ++j) {
      const double r = std::log(v[j].c / v[j - 1].c);
      m += r, m2 += r * r;
    }
    m /= 20;
    const double sd = std::sqrt(std::max(m2 / 20 - m * m, 0.0)), prev = v[i - 1].c, o = at->o, close = at->c;
    if (!(sd > 0)) { ++skipped; continue; }
    ++used;
    for (size_t a = 0; a < depth.size(); ++a) {
      const double lim = depth[a] < 0 ? 1e300 : prev * std::exp(-depth[a] * sd);
      for (size_t b = 0; b < delay.size(); ++b) {
        const int start = open + delay[b];
        double px = 0;
        // The price when the order goes in: the open, else the close of the bar just before.
        double now = o;
        for (const auto& x : path) if (x.t + 5 <= start) now = x.c;
        if (now <= lim) px = now;
        else
          for (const auto& x : path)
            if (x.t >= start && x.l < lim) { px = lim; break; }
        st[a][b].add(px > 0, px > 0 ? close / px - 1 - cost : 0);
      }
    }
  }
  std::printf("delayed entry: %ld share-days with 5-minute bars (%ld without the daily history or a full session); "
              "a round trip's costs %.2f%%; sold at the close\n", used, skipped, 100 * cost);
  std::printf("delayed entry: limit (sd below the previous close) x the minutes after the open before it is placed\n");
  std::printf("delayed entry: %-6s %6s %8s %9s %9s %6s %6s\n", "limit", "delay", "filled", "per fill", "per order", "won", "t");
  for (size_t a = 0; a < depth.size(); ++a)
    for (size_t b = 0; b < delay.size(); ++b) {
      const Stat& s = st[a][b];
      const double perOrder = s.orders ? s.sum / s.orders : 0, perFill = s.fills ? s.sum / s.fills : 0;
      const double var = s.orders > 1 ? (s.sum2 / s.orders - perOrder * perOrder) : 0;
      const double tstat = var > 0 ? perOrder / std::sqrt(var / s.orders) : 0;
      char name[16];
      if (depth[a] < 0) std::snprintf(name, sizeof name, "none");
      else std::snprintf(name, sizeof name, "%.1f", depth[a]);
      std::printf("delayed entry: %-6s %4dm %7.1f%% %+8.3f%% %+8.4f%% %5.1f%% %+6.2f\n", name, delay[b],
                  s.orders ? 100.0 * s.fills / s.orders : 0, 100 * perFill, 100 * perOrder,
                  s.fills ? 100.0 * s.wins / s.fills : 0, tstat);
    }
  return 0;
}
