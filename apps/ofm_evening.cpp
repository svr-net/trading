// The evening's news in daily bars (research; results only): index futures trade for hours after the
// cash market's close (EUREX's Euro Stoxx 50 future until 22:00 CET, the LSE and Euronext shut at
// 16:30 and 17:30), so on day t the future's close-to-close return less the cash market's holds what
// happened after the cash close (the US session, late news) and before the next open. Does it predict
// the next day's opening gap, and the open-to-close return after it?
//
// The cash market: each day the median close-to-close log return of the --top most traded shares
// (by their median traded value over the last 20 days, the past only). x_t = r_future(t) - b_t
// r_cash(t), b_t the future's beta on the cash market over the last 250 days (the past only). Reported:
// the slope and t statistic of the next day's median gap (log open / previous close) and median
// open-to-close on x_t, both halves of the days apart, and the cross-section: whether the shares with
// the highest beta to the market gap further (their gap on x_t, high-beta tenth less low-beta tenth).
//
// With --tradelog (ofm_research's X_TRADELOG): the model's trades grouped by the market's opening gap
// on their day (the median gap, in fifths of the days): the orders, the fills, the mean net result per
// fill and per order (an unfilled order: 0). The market's gap is known at the open, before most of
// the day's dips fill.
//
//   ofm_evening --csv stocks.csv --series futures.csv --future EUREX:FX.C [--top 300] [--tradelog t.csv]
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
struct Reg {  // y on x, with intercept
  double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
  void add(double x, double y) { n += 1, sx += x, sy += y, sxx += x * x, sxy += x * y, syy += y * y; }
  double slope() const { const double v = sxx - sx * sx / n; return v > 0 ? (sxy - sx * sy / n) / v : 0; }
  double t() const {
    if (n < 10) return 0;
    const double vx = sxx - sx * sx / n, b = slope(), a = (sy - b * sx) / n;
    const double sse = syy - 2 * a * sy - 2 * b * sxy + n * a * a + 2 * a * b * sx + b * b * sxx;
    const double se = std::sqrt(std::max(0.0, sse / (n - 2)) / vx);
    return se > 0 ? b / se : 0;
  }
  double corr() const {
    const double vx = sxx - sx * sx / n, vy = syy - sy * sy / n;
    return vx > 0 && vy > 0 ? (sxy - sx * sy / n) / std::sqrt(vx * vy) : 0;
  }
};
double median(std::vector<double> v) {
  if (v.empty()) return swing::kNaN;
  std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(v.size() / 2), v.end());
  return v[v.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
  std::string csv, series, future = "EUREX:FX.C", tradelog;
  std::size_t top = 300;
  for (int k = 1; k + 1 < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv") csv = argv[++k];
    else if (a == "--series") series = argv[++k];
    else if (a == "--future") future = argv[++k];
    else if (a == "--top") top = std::size_t(std::atol(argv[++k]));
    else if (a == "--tradelog") tradelog = argv[++k];
  }
  if (csv.empty() || series.empty()) {
    std::fprintf(stderr, "usage: ofm_evening --csv stocks.csv --series futures.csv [--future EUREX:FX.C] [--top 300]\n");
    return 2;
  }
  const swing::Panel p = swing::load(csv);
  const std::size_t T = p.T, N = p.N;
  // The future's closes on the shares' days (a day without a bar: none).
  std::map<std::string, double> fc;
  {
    std::ifstream in(series);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      std::vector<std::string> c;
      std::stringstream ss(line);
      for (std::string x; std::getline(ss, x, ',');) c.push_back(x);
      if (c.size() >= 6 && c[1] == future && std::atof(c[5].c_str()) > 0) fc[c[0]] = std::atof(c[5].c_str());
    }
  }
  const auto uni = swing::universe(p, top, 0);
  std::vector<double> rc(T, swing::kNaN), rf(T, swing::kNaN), gap(T, swing::kNaN), oc(T, swing::kNaN);
  for (std::size_t t = 1; t < T; ++t) {
    std::vector<double> a, g, o;
    for (std::size_t i : uni[t - 1]) {  // the shares tradable at the close before
      if (!p.ok(t, i) || !p.ok(t - 1, i) || !(p.O[t * N + i] > 0)) continue;
      a.push_back(std::log(p.C[t * N + i] / p.C[(t - 1) * N + i]));
      g.push_back(std::log(p.O[t * N + i] / p.C[(t - 1) * N + i]));
      o.push_back(std::log(p.C[t * N + i] / p.O[t * N + i]));
    }
    if (a.size() < 30) continue;
    rc[t] = median(a), gap[t] = median(g), oc[t] = median(o);
    const auto f1 = fc.find(p.days[t]), f0 = fc.find(p.days[t - 1]);
    if (f1 != fc.end() && f0 != fc.end()) rf[t] = std::log(f1->second / f0->second);
  }
  // x_t: the future's return less its beta times the cash market's (the beta from the last 250 days).
  std::vector<double> x(T, swing::kNaN);
  for (std::size_t t = 251; t < T; ++t) {
    Reg b;
    for (std::size_t u = t - 250; u < t; ++u)
      if (std::isfinite(rf[u]) && std::isfinite(rc[u])) b.add(rc[u], rf[u]);
    if (b.n < 150 || !std::isfinite(rf[t]) || !std::isfinite(rc[t])) continue;
    x[t] = rf[t] - b.slope() * rc[t];
  }
  std::size_t first = 0, last = 0;
  for (std::size_t t = 0; t + 1 < T; ++t)
    if (std::isfinite(x[t])) { if (!first) first = t; last = t; }
  const std::size_t mid = (first + last) / 2;
  Reg g[3], o[3], og[3];
  for (std::size_t t = first; t + 1 < T; ++t) {
    if (!std::isfinite(x[t]) || !std::isfinite(gap[t + 1])) continue;
    const int h = t < mid ? 1 : 2;
    for (int k : {0, h}) {
      g[k].add(x[t], gap[t + 1]);
      if (std::isfinite(oc[t + 1])) o[k].add(x[t], oc[t + 1]), og[k].add(gap[t + 1], oc[t + 1]);
    }
  }
  std::printf("evening: %s against the median of the %zu most traded shares, %s to %s (%.0f days)\n", future.c_str(), top,
              p.days[first].c_str(), p.days[last].c_str(), g[0].n);
  const char* part[3] = {"all days", "first half", "second half"};
  for (int k = 0; k < 3; ++k)
    std::printf("evening: %-11s next gap on x: slope %+.3f t %+6.1f corr %+.2f | next open-to-close on x: slope %+.3f t %+5.1f | open-to-close on the gap: slope %+.3f t %+5.1f\n",
                part[k], g[k].slope(), g[k].t(), g[k].corr(), o[k].slope(), o[k].t(), og[k].slope(), og[k].t());
  if (!tradelog.empty()) {
    std::map<std::string, double> gapOf;
    std::vector<double> all;
    for (std::size_t t = 0; t < T; ++t)
      if (std::isfinite(gap[t])) gapOf[p.days[t]] = gap[t], all.push_back(gap[t]);
    std::sort(all.begin(), all.end());
    const auto q = [&](double f) { return all[std::min(all.size() - 1, std::size_t(f * double(all.size())))]; };
    const double cut[4] = {q(0.2), q(0.4), q(0.6), q(0.8)};
    std::ifstream in(tradelog);
    std::string line;
    std::getline(in, line);
    std::vector<std::string> head;
    {
      std::stringstream ss(line);
      for (std::string w; std::getline(ss, w, ',');) head.push_back(w);
    }
    const auto col = [&](const char* n) { return std::size_t(std::find(head.begin(), head.end(), n) - head.begin()); };
    const std::size_t cd = col("date"), cr = col("ret"), cf = col("filled");
    double orders[5] = {}, fills[5] = {}, sum[5] = {}, sum2[5] = {};
    while (std::getline(in, line)) {
      std::vector<std::string> c;
      std::stringstream ss(line);
      for (std::string w; std::getline(ss, w, ',');) c.push_back(w);
      if (c.size() <= std::max({cd, cr, cf})) continue;
      const auto it = gapOf.find(c[cd]);
      if (it == gapOf.end()) continue;
      const int b = int(std::upper_bound(cut, cut + 4, it->second) - cut);
      const double r = std::atof(c[cr].c_str());
      orders[b] += 1;
      if (std::atof(c[cf].c_str()) > 0) fills[b] += 1, sum[b] += r, sum2[b] += r * r;
    }
    std::printf("evening: the model's trades (%s) by the market's opening gap on their day, lowest fifth first:\n", tradelog.c_str());
    for (int b = 0; b < 5; ++b) {
      const double m = fills[b] > 0 ? sum[b] / fills[b] : 0, sd = fills[b] > 1 ? std::sqrt(std::max(0.0, sum2[b] / fills[b] - m * m)) : 0;
      std::printf("evening:   gap fifth %d (median gap %s %+.2f%%): %5.0f orders, %5.0f filled (%4.1f%%), a fill %+.3f%% (t %+.1f), an order %+.4f%%\n", b + 1,
                  b == 0 ? "below" : "from ", 100 * (b == 0 ? cut[0] : cut[b - 1]), orders[b], fills[b], orders[b] > 0 ? 100 * fills[b] / orders[b] : 0.0,
                  100 * m, sd > 0 ? m / sd * std::sqrt(fills[b]) : 0.0, orders[b] > 0 ? 100 * sum[b] / orders[b] : 0.0);
    }
  }
  return 0;
}
