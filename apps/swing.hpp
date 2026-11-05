// The daily bars as a days x shares panel, each day's tradable shares, the broker's costs and the
// signals known at a close: shared by ofm_anatomy (where a market's returns are) and ofm_swing (the
// model that trades them), so both measure the same things.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace swing {
inline const double kNaN = std::nan("");

struct Panel {
  std::vector<std::string> days, tickers;
  std::size_t T = 0, N = 0;
  std::vector<double> O, H, L, C, V;
  bool ok(std::size_t t, std::size_t i) const { return t < T && C[t * N + i] > 0; }
  bool hasOpen(std::size_t t, std::size_t i) const { return t < T && O[t * N + i] > 0; }
  double ret(std::size_t t, std::size_t i, std::size_t back) const {  // close t-back to close t
    return t >= back && ok(t, i) && ok(t - back, i) ? C[t * N + i] / C[(t - back) * N + i] - 1 : kNaN;
  }
};

// date,ticker,open,high,low,close,volume (a header line first).
inline Panel load(const std::string& csv) {
  Panel p;
  std::map<std::string, std::size_t> dayIx, tickIx;
  std::vector<std::vector<std::string>> rows;
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
  for (auto& [d, i] : dayIx) i = p.T++, p.days.push_back(d);
  for (auto& [t, i] : tickIx) i = p.N++, p.tickers.push_back(t);
  const std::size_t T = p.T, N = p.N;
  p.O.assign(T * N, kNaN), p.H = p.O, p.L = p.O, p.C = p.O, p.V = p.O;
  for (const auto& f : rows) {
    const std::size_t at = dayIx[f[0]] * N + tickIx[f[1]];
    const double o = std::atof(f[2].c_str()), h = std::atof(f[3].c_str()), l = std::atof(f[4].c_str()), c = std::atof(f[5].c_str());
    if (o > 0 && h > 0 && l > 0 && c > 0 && h >= l) p.O[at] = o, p.H[at] = h, p.L[at] = l, p.C[at] = c, p.V[at] = std::atof(f[6].c_str());
  }
  return p;
}

// The bars of another source where it has them (e.g. IBKR's own daily bars: their open is the
// opening auction's price), in place of the panel's; returns a summary of how far the panel's opens
// were from them (share-days compared, the median gap, the share of gaps over 1%).
inline std::string override(Panel& p, const std::string& csv) {
  std::map<std::string, std::size_t> dayIx, tickIx;
  for (std::size_t t = 0; t < p.T; ++t) dayIx[p.days[t]] = t;
  for (std::size_t i = 0; i < p.N; ++i) tickIx[p.tickers[i]] = i;
  std::ifstream in(csv);
  std::string line;
  std::getline(in, line);
  std::vector<double> gap;
  while (std::getline(in, line)) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    for (std::string x; std::getline(ss, x, ',');) f.push_back(x);
    if (f.size() < 7) continue;
    const auto d = dayIx.find(f[0].substr(0, 10)), k = tickIx.find(f[1]);
    if (d == dayIx.end() || k == tickIx.end()) continue;
    const std::size_t at = d->second * p.N + k->second;
    const double o = std::atof(f[2].c_str()), h = std::atof(f[3].c_str()), l = std::atof(f[4].c_str()), c = std::atof(f[5].c_str());
    if (!(o > 0 && h > 0 && l > 0 && c > 0 && h >= l)) continue;
    if (p.O[at] > 0) gap.push_back(std::fabs(std::log(p.O[at] / o)));
    p.O[at] = o, p.H[at] = h, p.L[at] = l, p.C[at] = c;
  }
  char b[200];
  if (gap.empty()) return "no share-day in common";
  std::sort(gap.begin(), gap.end());
  const double over = double(gap.end() - std::upper_bound(gap.begin(), gap.end(), 0.01)) / double(gap.size());
  std::snprintf(b, sizeof b, "%zu share-days replaced; the panel's open was off by a median %.2f%%, by over 1%% on %.1f%% of them", gap.size(), 100 * gap[gap.size() / 2], 100 * over);
  return b;
}

// Each day's tradable shares: a price of at least minPrice, 20 days of trading, and among the top
// most traded by their median daily traded value over those 20 days (the past only).
inline std::vector<std::vector<std::size_t>> universe(const Panel& p, std::size_t top, double minPrice) {
  const std::size_t T = p.T, N = p.N;
  std::vector<double> value(T * N, kNaN);
  for (std::size_t i = 0; i < N; ++i) {
    std::vector<double> win;
    for (std::size_t t = 0; t < T; ++t) {
      if (p.ok(t, i)) win.push_back(p.C[t * N + i] * p.V[t * N + i]);
      if (win.size() > 20) win.erase(win.begin());
      if (win.size() == 20 && p.ok(t, i)) {
        std::vector<double> s = win;
        std::nth_element(s.begin(), s.begin() + 10, s.end());
        value[t * N + i] = s[10];
      }
    }
  }
  std::vector<std::vector<std::size_t>> uni(T);
  for (std::size_t t = 0; t < T; ++t) {
    std::vector<std::pair<double, std::size_t>> v;
    for (std::size_t i = 0; i < N; ++i)
      if (std::isfinite(value[t * N + i]) && p.C[t * N + i] >= minPrice) v.emplace_back(value[t * N + i], i);
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (std::size_t r = 0; r < std::min(top, v.size()); ++r) uni[t].push_back(v[r].second);
  }
  return uni;
}

// The broker's costs on a trade of `capital` (IBKR fixed pricing), as fractions of its value:
// UK GBP 3 or 0.05%, the stamp duty 0.5% on buying (not AIM shares: --stamp-free); US USD 0.005 a
// share, at least USD 1, at most 1%; elsewhere each exchange's line of tools/venues.txt (EXCHANGE
// amount rate minimum fee buy-tax both-tax [buy-tax on a share held past the day]) for tickers
// EXCHANGE:CODE.
struct Costs {
  int kind = 0;  // 1 UK, 2 US, 3 by venue
  double capital = 5000;
  std::vector<std::array<double, 7>> venue;  // per share: amount rate minimum fee buy-tax both-tax overnight-buy-tax
  std::vector<char> stampFree;
  double side(std::size_t i, double price) const {  // the commission and fees of one side
    if (kind == 2) {
      const double shares = std::floor(capital / price);
      return shares > 0 ? std::min(0.01, std::max(1.0, 0.005 * shares) / capital) : 0.01;
    }
    if (kind == 1) return std::max(3.0, 0.0005 * capital) / capital;
    const auto& v = venue[i];
    return v[0] > 0 ? std::max(v[2], v[1] * v[0]) / v[0] + v[3] + v[5] : 0.0;
  }
  double buyTax(std::size_t i, bool overnight) const {  // UK stamp duty: on every purchase of a share
    if (kind == 1) return stampFree[i] ? 0.0 : 0.005;
    return kind == 3 ? venue[i][overnight ? 6 : 4] : 0.0;
  }
  double roundTrip(std::size_t i, double buy, double sell, bool overnight) const { return side(i, buy) + side(i, sell) + buyTax(i, overnight); }
};

inline Costs costs(const Panel& p, const std::string& market, double capital, const std::string& venuesFile, const std::string& stampFreeFile) {
  Costs c;
  c.kind = market == "UK" ? 1 : market == "US" ? 2 : 3;
  c.capital = capital;
  c.venue.assign(p.N, {0, 0, 0, 0, 0, 0, 0});
  c.stampFree.assign(p.N, 0);
  if (!venuesFile.empty()) {
    std::ifstream in(venuesFile);
    std::map<std::string, std::array<double, 7>> line;
    for (std::string s; std::getline(in, s);) {
      if (s.empty() || s[0] == '#') continue;
      std::istringstream ls(s);
      std::string ex;
      std::array<double, 7> v{};
      if (!(ls >> ex >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5])) continue;
      if (!(ls >> v[6])) v[6] = v[4];
      line[ex] = v;
    }
    for (std::size_t i = 0; i < p.N; ++i) {
      const auto k = p.tickers[i].find(':');
      const auto it = k == std::string::npos ? line.end() : line.find(p.tickers[i].substr(0, k));
      if (it != line.end()) c.venue[i] = it->second;
    }
  }
  if (!stampFreeFile.empty()) {
    std::ifstream in(stampFreeFile);
    std::set<std::string> free;
    for (std::string s; std::getline(in, s);) free.insert(s.substr(0, s.find_first_of(", \r")));
    for (std::size_t i = 0; i < p.N; ++i) c.stampFree[i] = free.count(p.tickers[i]) ? 1 : 0;
  }
  return c;
}

// The signals known at the close t (higher: bought); NaN where there is not enough history.
struct Signal {
  const char* name;
  std::function<double(std::size_t, std::size_t)> f;
};

inline std::vector<Signal> signals(const Panel& p) {
  const std::size_t N = p.N;
  const auto vol = [&p](std::size_t t, std::size_t i, std::size_t n) {
    if (t < n) return kNaN;
    double s = 0, s2 = 0;
    for (std::size_t k = 0; k < n; ++k) {
      const double r = p.ret(t - k, i, 1);
      if (!std::isfinite(r)) return kNaN;
      s += r, s2 += r * r;
    }
    s /= double(n);
    return std::sqrt(std::max(0.0, s2 / double(n) - s * s));
  };
  return {
      {"loser 1d", [&p](std::size_t t, std::size_t i) { return -p.ret(t, i, 1); }},
      {"loser 5d", [&p](std::size_t t, std::size_t i) { return -p.ret(t, i, 5); }},
      {"loser 20d", [&p](std::size_t t, std::size_t i) { return -p.ret(t, i, 20); }},
      {"momentum 12-1m", [&p, N](std::size_t t, std::size_t i) { return t >= 252 && p.ok(t - 21, i) && p.ok(t - 252, i) ? p.C[(t - 21) * N + i] / p.C[(t - 252) * N + i] - 1 : kNaN; }},
      {"near 52w high", [&p, N](std::size_t t, std::size_t i) {
         if (t < 252) return kNaN;
         double h = 0;
         for (std::size_t k = 0; k < 252; ++k) if (p.ok(t - k, i)) h = std::max(h, p.H[(t - k) * N + i]);
         return h > 0 ? p.C[t * N + i] / h : kNaN; }},
      {"low volatility", [vol](std::size_t t, std::size_t i) { return -vol(t, i, 20); }},
      {"low max day", [&p](std::size_t t, std::size_t i) {
         if (t < 22) return kNaN;
         double m = -1;
         for (std::size_t k = 0; k < 21; ++k) { const double r = p.ret(t - k, i, 1); if (std::isfinite(r)) m = std::max(m, r); }
         return -m; }},
      {"close near low", [&p, N](std::size_t t, std::size_t i) {  // internal bar strength, low: bought
         const double h = p.H[t * N + i], l = p.L[t * N + i];
         return h > l ? -(p.C[t * N + i] - l) / (h - l) : kNaN; }},
      {"day fell open-close", [&p, N](std::size_t t, std::size_t i) { return -(p.C[t * N + i] / p.O[t * N + i] - 1); }},
      {"gapped down", [&p, N](std::size_t t, std::size_t i) { return t >= 1 && p.ok(t - 1, i) ? -(p.O[t * N + i] / p.C[(t - 1) * N + i] - 1) : kNaN; }},
      {"volume surge", [&p, N](std::size_t t, std::size_t i) {
         if (t < 60) return kNaN;
         double a = 0, b = 0;
         for (std::size_t k = 0; k < 60; ++k) { if (!p.ok(t - k, i)) return kNaN; (k < 5 ? a : b) += p.V[(t - k) * N + i]; }
         return b > 0 ? (a / 5) / (b / 55) : kNaN; }},
      {"below 20d mean", [&p, N](std::size_t t, std::size_t i) {
         if (t < 20) return kNaN;
         double s = 0;
         for (std::size_t k = 0; k < 20; ++k) { if (!p.ok(t - k, i)) return kNaN; s += p.C[(t - k) * N + i]; }
         return -(p.C[t * N + i] / (s / 20) - 1); }},
      {"loser 5d, low vol", [&p, vol](std::size_t t, std::size_t i) { const double v = vol(t, i, 20); return v > 0 ? -p.ret(t, i, 5) / v : kNaN; }},
  };
}

// A holding window as orders can trade it, the signal known at the close t: bought at the next
// session's open (auction) or at its close (auction; the order in before the close, so the signal is
// a day old), held `days` closes after the entry (0: sold at the next open, overnight) and sold at
// that close (auction), or at the open after the entry for an overnight hold. The model trades only
// within the day (bought at the open, sold at the close); the others are kept for measurement.
struct Window {
  const char* name;
  bool entryAtClose;
  int days;  // closes held after the entry; 0 overnight (sold at the next open)
  bool overnight() const { return entryAtClose || days > 1; }
};

inline const std::vector<Window>& windows() {
  static const std::vector<Window> w = {{"open, sold at its close", false, 0}};
  return w;
}

// The window's trade of share i from the close f: its days (each the day its return ends on, its
// return that day gross and the costs charged that day). Empty when it cannot be traded. A share
// with no bar on a later day keeps its last price (no return).
struct Leg { std::size_t day; double gross, cost; };
inline std::vector<Leg> legs(const Panel& p, const Costs& c, const Window& w, std::size_t f, std::size_t i) {
  const std::size_t N = p.N, e = f + 1;
  std::vector<Leg> out;
  if (!w.entryAtClose) {
    if (!p.hasOpen(e, i) || !p.ok(e, i)) return out;
    const std::size_t last = e + std::size_t(std::max(0, w.days - 1));  // "sold at its close": 1 day; held h: h closes
    double prev = p.O[e * N + i];
    const double buy = prev;
    for (std::size_t d = e; d <= last && d < p.T; ++d) {
      const double px = p.ok(d, i) ? p.C[d * N + i] : prev;
      out.push_back({d, px / prev - 1, d == e ? c.side(i, buy) + c.buyTax(i, w.overnight()) : 0.0});
      prev = px;
    }
    if (out.size() < std::size_t(std::max(1, w.days))) return {};
    out.back().cost += c.side(i, prev);
    return out;
  }
  if (!p.ok(e, i)) return out;
  const double buy = p.C[e * N + i];
  if (w.days == 0) {
    if (!p.hasOpen(e + 1, i)) return out;
    const double sell = p.O[(e + 1) * N + i];
    out.push_back({e + 1, sell / buy - 1, c.roundTrip(i, buy, sell, true)});
    return out;
  }
  double prev = buy;
  for (std::size_t d = e + 1; d <= e + std::size_t(w.days) && d < p.T; ++d) {
    const double px = p.ok(d, i) ? p.C[d * N + i] : prev;
    out.push_back({d, px / prev - 1, d == e + 1 ? c.side(i, buy) + c.buyTax(i, true) : 0.0});
    prev = px;
  }
  if (out.size() < std::size_t(w.days)) return {};
  out.back().cost += c.side(i, prev);
  return out;
}

// The day the window's trade from the close f is fully known (its last leg).
inline std::size_t settles(const Window& w, std::size_t f) {
  return f + 1 + (w.entryAtClose ? std::size_t(std::max(1, w.days)) : std::size_t(std::max(0, w.days - 1)));
}
}  // namespace swing
