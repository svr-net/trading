// Where a market's returns are, and which kinds of strategy earn them (research; results only).
//
// On daily bars (date,ticker,open,high,low,close,volume), every day's tradable shares (a price of at
// least --min-price and among the --top most traded by their median daily traded value over the last
// 20 days, the past only) are ranked by each signal known at the close, and the top tenth is held
// over each window:
//   ON    bought at the close (auction), sold at the next open (auction)
//   ID    bought at the next open, sold at its close (what the daily model trades)
//   CC    bought at the close, sold at the next close
//   H5    bought at the next open, held 5 days (a fifth of the book renewed each day)
//   H20   the same, 20 days
// The returns are net of a round trip's costs: the commission (IBKR's US fixed rate: $0.005 a share,
// at least $1, at most 1%, on --capital a trade) on both sides; the auctions cross at one price, so a
// window traded only in auctions pays no spread; the others pay the spread (Abdi and Ranaldo's
// estimate from the bars, the past only) once. The book renews 1/h of itself a day over h days.
// Also: the market (every tradable share, equally weighted) over each window, and the top tenth less
// the bottom tenth (gross: a short's costs are not modelled). For each: the mean a year, the Sharpe
// ratio, the t statistic, and the first and the second half of the days apart (a pattern found in
// the first half must hold in the second to be worth anything).
//
//   ofm_anatomy --csv stocks.csv [--top 500] [--min-price 5] [--capital 5000]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {
const double kNaN = std::nan("");

struct Series {
  double sum = 0, sum2 = 0;
  long n = 0;
  void add(double x) { sum += x, sum2 += x * x, ++n; }
  double mean() const { return n ? sum / double(n) : 0; }
  double sd() const { return n > 1 ? std::sqrt(std::max(0.0, sum2 / double(n) - mean() * mean())) : 0; }
};

struct Stat {  // a daily series: all days, first half, second half
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
}  // namespace

int main(int argc, char** argv) {
  std::string csv;
  std::size_t top = 500;
  double minPrice = 5, capital = 5000;
  for (int k = 1; k + 1 < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv") csv = argv[++k];
    else if (a == "--top") top = std::size_t(std::atol(argv[++k]));
    else if (a == "--min-price") minPrice = std::atof(argv[++k]);
    else if (a == "--capital") capital = std::atof(argv[++k]);
  }
  if (csv.empty()) {
    std::fprintf(stderr, "usage: ofm_anatomy --csv stocks.csv [--top 500] [--min-price 5] [--capital 5000]\n");
    return 2;
  }
  // The panel: days x shares.
  std::map<std::string, std::size_t> dayIx, tickIx;
  std::vector<std::vector<std::string>> rows;
  {
    std::ifstream in(csv);
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      std::vector<std::string> f;
      std::stringstream ss(line);
      for (std::string x; std::getline(ss, x, ',');) f.push_back(x);
      if (f.size() < 7) continue;
      dayIx.emplace(f[0], 0), tickIx.emplace(f[1], 0);
      rows.push_back(std::move(f));
    }
  }
  std::size_t T = 0, N = 0;
  for (auto& [d, i] : dayIx) i = T++;
  for (auto& [t, i] : tickIx) i = N++;
  std::vector<double> O(T * N, kNaN), H = O, L = O, C = O, V = O;
  for (const auto& f : rows) {
    const std::size_t at = dayIx[f[0]] * N + tickIx[f[1]];
    const double o = std::atof(f[2].c_str()), h = std::atof(f[3].c_str()), l = std::atof(f[4].c_str()), c = std::atof(f[5].c_str());
    if (o > 0 && h > 0 && l > 0 && c > 0 && h >= l) O[at] = o, H[at] = h, L[at] = l, C[at] = c, V[at] = std::atof(f[6].c_str());
  }
  rows.clear();
  const auto ok = [&](std::size_t t, std::size_t i) { return t < T && C[t * N + i] > 0; };
  const auto ret = [&](std::size_t t, std::size_t i, std::size_t back) {  // close t-back to close t
    return t >= back && ok(t, i) && ok(t - back, i) ? C[t * N + i] / C[(t - back) * N + i] - 1 : kNaN;
  };
  // The spread (Abdi and Ranaldo, about 64 days, the past only) and the median traded value (20 days).
  std::vector<double> spread(T * N, 0.0), value(T * N, kNaN);
  for (std::size_t i = 0; i < N; ++i) {
    double e = 0, w = 0;
    std::vector<double> win;
    for (std::size_t t = 1; t < T; ++t) {
      if (ok(t, i) && ok(t - 1, i)) {
        const double c = std::log(C[(t - 1) * N + i]), a = c - 0.5 * std::log(H[(t - 1) * N + i] * L[(t - 1) * N + i]), b = c - 0.5 * std::log(H[t * N + i] * L[t * N + i]);
        e += (4 * a * b - e) / 64, w += (1 - w) / 64;
      }
      spread[t * N + i] = w > 0 ? std::sqrt(std::max(0.0, e / w)) : 0.0;
      if (ok(t, i)) win.push_back(C[t * N + i] * V[t * N + i]);
      if (win.size() > 20) win.erase(win.begin());
      if (win.size() == 20) {
        std::vector<double> s = win;
        std::nth_element(s.begin(), s.begin() + 10, s.end());
        value[t * N + i] = s[10];
      }
    }
  }
  // The spread where the estimate has none: that day's median among the shares with one.
  for (std::size_t t = 1; t < T; ++t) {
    std::vector<double> v;
    for (std::size_t i = 0; i < N; ++i)
      if (spread[t * N + i] > 0) v.push_back(spread[t * N + i]);
    if (v.empty()) continue;
    std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(v.size() / 2), v.end());
    for (std::size_t i = 0; i < N; ++i)
      if (!(spread[t * N + i] > 0)) spread[t * N + i] = v[v.size() / 2];
  }
  const auto commission = [&](double price) {  // one side, a fraction of the trade
    const double shares = std::floor(capital / price);
    return shares > 0 ? std::min(0.01, std::max(1.0, 0.005 * shares) / capital) : 0.01;
  };
  // The signals, known at the close t (higher: bought). NaN: none.
  struct Signal { const char* name; std::function<double(std::size_t, std::size_t)> f; };
  const auto vol = [&](std::size_t t, std::size_t i, std::size_t n) {
    Series s;
    for (std::size_t k = 0; k < n; ++k) {
      const double r = ret(t - k, i, 1);
      if (!std::isfinite(r)) return kNaN;
      s.add(r);
    }
    return s.sd();
  };
  const std::vector<Signal> sig = {
      {"loser 1d", [&](std::size_t t, std::size_t i) { return -ret(t, i, 1); }},
      {"loser 5d", [&](std::size_t t, std::size_t i) { return -ret(t, i, 5); }},
      {"loser 20d", [&](std::size_t t, std::size_t i) { return -ret(t, i, 20); }},
      {"momentum 12-1m", [&](std::size_t t, std::size_t i) { return t >= 252 && ok(t - 21, i) && ok(t - 252, i) ? C[(t - 21) * N + i] / C[(t - 252) * N + i] - 1 : kNaN; }},
      {"near 52w high", [&](std::size_t t, std::size_t i) {
         if (t < 252) return kNaN;
         double h = 0;
         for (std::size_t k = 0; k < 252; ++k) if (ok(t - k, i)) h = std::max(h, H[(t - k) * N + i]);
         return h > 0 ? C[t * N + i] / h : kNaN; }},
      {"low volatility", [&](std::size_t t, std::size_t i) { return t >= 21 ? -vol(t, i, 20) : kNaN; }},
      {"low max day", [&](std::size_t t, std::size_t i) {
         if (t < 22) return kNaN;
         double m = -1;
         for (std::size_t k = 0; k < 21; ++k) { const double r = ret(t - k, i, 1); if (std::isfinite(r)) m = std::max(m, r); }
         return -m; }},
      {"close near low", [&](std::size_t t, std::size_t i) {  // internal bar strength, low: bought
         const double h = H[t * N + i], l = L[t * N + i];
         return h > l ? -(C[t * N + i] - l) / (h - l) : kNaN; }},
      {"day fell open-close", [&](std::size_t t, std::size_t i) { return -(C[t * N + i] / O[t * N + i] - 1); }},
      {"gapped down", [&](std::size_t t, std::size_t i) { return t >= 1 && ok(t - 1, i) ? -(O[t * N + i] / C[(t - 1) * N + i] - 1) : kNaN; }},
      {"volume surge", [&](std::size_t t, std::size_t i) {
         if (t < 60) return kNaN;
         double a = 0, b = 0;
         for (std::size_t k = 0; k < 60; ++k) { if (!ok(t - k, i)) return kNaN; (k < 5 ? a : b) += V[(t - k) * N + i]; }
         return b > 0 ? (a / 5) / (b / 55) : kNaN; }},
      {"below 20d mean", [&](std::size_t t, std::size_t i) {
         if (t < 20) return kNaN;
         double s = 0;
         for (std::size_t k = 0; k < 20; ++k) { if (!ok(t - k, i)) return kNaN; s += C[(t - k) * N + i]; }
         return -(C[t * N + i] / (s / 20) - 1); }},
      {"loser 5d, low vol", [&](std::size_t t, std::size_t i) { const double v = t >= 21 ? vol(t, i, 20) : kNaN; return v > 0 ? -ret(t, i, 5) / v : kNaN; }},
  };
  const std::vector<std::pair<const char*, int>> windows = {{"ON", -1}, {"ID", 1}, {"CC", 0}, {"H5", 5}, {"H20", 20}};
  const std::size_t S = sig.size(), W = windows.size();
  // Each day's tradable shares and, for each signal, its top and bottom tenths (formed at the close t).
  std::vector<std::vector<std::size_t>> uni(T);
  std::vector<std::vector<std::vector<std::size_t>>> topSet(S, std::vector<std::vector<std::size_t>>(T)), botSet = topSet;
  for (std::size_t t = 0; t < T; ++t) {
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t i = 0; i < N; ++i)
      if (ok(t, i) && C[t * N + i] >= minPrice && std::isfinite(value[t * N + i])) v.emplace_back(value[t * N + i], i);
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (std::size_t r = 0; r < std::min(top, v.size()); ++r) uni[t].push_back(v[r].second);
    for (std::size_t s = 0; s < S; ++s) {
      std::vector<std::pair<double, std::size_t>> z;
      for (std::size_t i : uni[t]) {
        const double x = sig[s].f(t, i);
        if (std::isfinite(x)) z.emplace_back(x, i);
      }
      if (z.size() < 50) continue;
      std::sort(z.begin(), z.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
      const std::size_t k = z.size() / 10;
      for (std::size_t r = 0; r < k; ++r) topSet[s][t].push_back(z[r].second), botSet[s][t].push_back(z[z.size() - 1 - r].second);
    }
  }
  // A share's net return on day d for a cohort formed at the close f (the window w): the return from
  // its entry (or the last close) to its exit (or this close), less its costs on its entry and exit days.
  const auto cohortDay = [&](std::size_t f, std::size_t d, std::size_t i, int h, bool net) {
    // d: the day this return ends on (close d, or open d for ON).
    const double cf = C[f * N + i];
    if (h == -1) {  // ON: close f to open f+1
      if (!(O[d * N + i] > 0)) return 0.0;
      return O[d * N + i] / cf - 1 - (net ? 2 * commission(cf) : 0.0);
    }
    if (h == 0) {  // CC: close f to close f+1, both auctions
      if (!ok(d, i)) return 0.0;
      return C[d * N + i] / cf - 1 - (net ? 2 * commission(cf) : 0.0);
    }
    const std::size_t age = d - f;  // 1: the entry day (open to close)
    if (!ok(d, i) || !(O[(f + 1) * N + i] > 0)) return 0.0;
    double r = age == 1 ? C[d * N + i] / O[(f + 1) * N + i] - 1 : (ok(d - 1, i) ? C[d * N + i] / C[(d - 1) * N + i] - 1 : 0.0);
    if (net) {
      if (age == 1) r -= commission(O[(f + 1) * N + i]);
      if (age == std::size_t(h)) r -= commission(C[d * N + i]) + (h == 1 ? 0.0 : spread[f * N + i]);  // a spread once for a held trade (sold in the close's auction otherwise)
    }
    return r;
  };
  const auto bookDay = [&](const std::vector<std::vector<std::size_t>>& sets, std::size_t d, int h, bool net) {
    // The book's return on day d: the mean of its cohorts (h of them, each 1/h of the money).
    const int n = h <= 0 ? 1 : h;
    double sum = 0;
    int have = 0;
    for (int k = 1; k <= n; ++k) {
      if (d < std::size_t(k)) break;
      const std::size_t f = d - std::size_t(k);
      if (sets[f].empty()) continue;
      double s = 0;
      for (std::size_t i : sets[f]) s += cohortDay(f, d, i, h, net);
      sum += s / double(sets[f].size()), ++have;
    }
    return have ? sum / double(n) : kNaN;
  };
  const std::size_t start = 260, mid = start + (T - start) / 2;
  std::printf("anatomy: %zu shares, %zu days (%s to %s); each day the %zu most traded at %.0f or more; costs: IBKR US fixed on %.0f a trade, the spread outside the auctions\n",
              N, T, dayIx.begin()->first.c_str(), dayIx.rbegin()->first.c_str(), top, minPrice, capital);
  const auto first = [&](std::size_t d) { return d < mid; };
  // The market itself over each window (gross and net), every tradable share equally weighted.
  std::printf("anatomy: %-22s %-4s %24s | %24s | %24s\n", "the market (all held)", "win", "all days: a year Sh t", "first half", "second half");
  for (std::size_t w = 0; w < W; ++w)
    for (int net = 0; net < 2; ++net) {
      Stat st;
      for (std::size_t d = start; d < T; ++d) {
        const double x = bookDay(uni, d, windows[w].second, net);
        if (std::isfinite(x)) st.add(x, first(d));
      }
      std::printf("anatomy: %-22s %-4s %s | %s | %s\n", net ? "market, net" : "market, gross", windows[w].first, fmt(st.all).c_str(), fmt(st.h1).c_str(), fmt(st.h2).c_str());
    }
  std::printf("anatomy: %-22s %-4s %24s | %24s | %24s || %s\n", "signal: top tenth, net", "win", "all days: a year Sh t", "first half", "second half", "top less bottom, gross: all, 1st, 2nd");
  for (std::size_t s = 0; s < S; ++s)
    for (std::size_t w = 0; w < W; ++w) {
      Stat st, ls;
      for (std::size_t d = start; d < T; ++d) {
        const double a = bookDay(topSet[s], d, windows[w].second, true), b = bookDay(topSet[s], d, windows[w].second, false), c = bookDay(botSet[s], d, windows[w].second, false);
        if (std::isfinite(a)) st.add(a, first(d));
        if (std::isfinite(b) && std::isfinite(c)) ls.add(b - c, first(d));
      }
      std::printf("anatomy: %-22s %-4s %s | %s | %s || %s | %s | %s\n", sig[s].name, windows[w].first, fmt(st.all).c_str(), fmt(st.h1).c_str(), fmt(st.h2).c_str(),
                  fmt(ls.all).c_str(), fmt(ls.h1).c_str(), fmt(ls.h2).c_str());
    }
  return 0;
}
