// The buy-back after a stop, measured exactly on intraday paths (research; results only).
//
// Daily bars cannot tell whether a day's high came before or after its low, so a buy-back after a
// stop (a buy-stop part of the way back up to the entry, sold at the entry and a round trip's costs,
// else at the close) can only be bounded on them: at worst bought whenever the high reached its
// level, at best only when the close shows the price came back after the stop. This replays the
// recorded paths (the bridge's paths/<MARKET>/<date>.csv, submit.py --paths: time,ticker,last,bid,
// ask,type; or 5-minute bars date,ticker,time,high,low,close with --bars) price by price: each share's
// day bought at its first price after the open, stopped at stop distances s below it, bought back at
// fractions f of the way up, and reports, for each (s, f), how often the buy-back happens, its net
// result after costs (--cost: a round trip's, default 0.6%: UK commissions on GBP 5,000 and the
// stamp duty) and where the truth falls between the two daily-bar bounds.
//
//   ofm_paths paths/UK/2026-11-02.csv paths/UK/2026-11-03.csv ...
//   ofm_paths --bars intraday.csv --cost 0.001
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
struct Day {
  std::vector<double> lo, hi, px;  // per sample: its low, high and price (equal for a quote)
};

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> f;
  std::stringstream ss(s);
  std::string x;
  while (std::getline(ss, x, ',')) f.push_back(x);
  return f;
}
}  // namespace

int main(int argc, char** argv) {
  bool bars = false;
  double cost = 0.006;
  std::vector<std::string> files;
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--bars") bars = true;
    else if (a == "--cost" && k + 1 < argc) cost = std::atof(argv[++k]);
    else files.push_back(a);
  }
  if (files.empty()) {
    std::fprintf(stderr, "usage: ofm_paths [--bars] [--cost 0.006] file...\n");
    return 2;
  }
  std::map<std::string, Day> days;  // date|ticker -> the day's path, in time order
  for (const auto& fn : files) {
    std::ifstream in(fn);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      const auto f = split(line);
      if (bars) {  // date,ticker,time,high,low,close
        if (f.size() < 6) continue;
        const double h = std::atof(f[3].c_str()), l = std::atof(f[4].c_str()), c = std::atof(f[5].c_str());
        if (!(h > 0 && l > 0 && c > 0)) continue;
        auto& d = days[f[0] + "|" + f[1]];
        d.hi.push_back(h), d.lo.push_back(l), d.px.push_back(c);
      } else {  // time,ticker,last,bid,ask,type: the last trade, else the mid
        if (f.size() < 5) continue;
        const double last = std::atof(f[2].c_str()), bid = std::atof(f[3].c_str()), ask = std::atof(f[4].c_str());
        const double p = last > 0 ? last : (bid > 0 && ask > 0 ? 0.5 * (bid + ask) : 0);
        if (!(p > 0)) continue;
        auto& d = days[f[0].substr(0, 10) + "|" + f[1]];
        d.hi.push_back(p), d.lo.push_back(p), d.px.push_back(p);
      }
    }
  }
  std::printf("%zu share-days; a round trip's costs %.2f%%\n", days.size(), 100 * cost);
  std::printf("%6s %5s %7s %9s | %10s | %10s %10s | %-28s | %s\n", "stop", "f", "stops", "rebought", "true/stop", "worst", "certain", "truth between", "unclear days: bought, true vs worst");
  for (const double s : {0.02, 0.03, 0.05, 0.08}) {
    for (const double f : {0.25, 0.5}) {
      long stops = 0, rebought = 0, amb = 0, ambBought = 0;
      double sT = 0, sW = 0, sC = 0, ambTrue = 0, ambWorst = 0;
      for (const auto& [key, d] : days) {
        if (d.px.size() < 20) continue;
        const double e = d.px[0], stop = e * (1 - s), R = stop + f * (e - stop), T = e * (1 + cost);
        std::size_t ks = 0;
        for (std::size_t k = 1; k < d.px.size() && !ks; ++k)
          if (d.lo[k] <= stop) ks = k;
        if (!ks) continue;
        ++stops;
        double hi = 0;
        for (std::size_t k = 1; k < d.px.size(); ++k) hi = std::max(hi, d.hi[k]);
        const double c = d.px.back();
        // The truth: bought at R at the first later touch, sold at T at the first touch after, else the close.
        std::size_t kr = 0;
        for (std::size_t k = ks + 1; k < d.px.size() && !kr; ++k)
          if (d.hi[k] >= R) kr = k;
        double t2 = 0;
        if (kr) {
          double px = c;
          for (std::size_t k = kr + 1; k < d.px.size(); ++k)
            if (d.hi[k] >= T) { px = T; break; }
          t2 = px / R - 1 - cost, ++rebought;
        }
        sT += t2;
        // The daily-bar bounds (sold at T when the close is above it, else at the close).
        const double daily = (c >= T ? T : c) / R - 1 - cost;
        if (hi >= R) sW += daily;
        if (c >= R) sC += daily;
        if (hi >= R && c < R) ++amb, ambBought += kr > 0, ambTrue += t2, ambWorst += daily;  // the days daily bars cannot settle
      }
      if (!stops) continue;
      const double t = sT / stops, w = sW / stops, cc = sC / stops;
      std::printf("%5.0f%% %5.2f %7ld %8.1f%% | %+9.2f%% | %+9.2f%% %+9.2f%% | %-28s | %ld, %.0f%% bought, %+.2f%% vs %+.2f%% each\n", 100 * s, f, stops,
                  100.0 * rebought / stops, 100 * t, 100 * w, 100 * cc,
                  cc != w ? (std::to_string(int(std::lround(100 * (t - w) / (cc - w)))) + "% of the way to certain").c_str() : "-", amb,
                  amb ? 100.0 * ambBought / amb : 0.0, amb ? 100 * ambTrue / amb : 0.0, amb ? 100 * ambWorst / amb : 0.0);
    }
  }
  return 0;
}
