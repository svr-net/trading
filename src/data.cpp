#include "ofm/data.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ofm {

namespace {

std::vector<std::string> splitLine(const std::string& line) {
  std::vector<std::string> out;
  std::string cell;
  std::istringstream ss(line);
  while (std::getline(ss, cell, ',')) {
    while (!cell.empty() && (cell.back() == '\r' || cell.back() == ' ')) cell.pop_back();
    out.push_back(cell);
  }
  return out;
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

double num(const std::string& s) {
  if (s.empty()) return kNaN;
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  return end == s.c_str() ? kNaN : v;
}

struct Rows {
  std::vector<std::string> dates, tickers;
  std::vector<std::vector<std::string>> cells;
  std::unordered_map<std::string, int> col;
};

Rows readRows(const std::string& text) {
  Rows r;
  std::istringstream in(text);
  std::string line;
  if (!std::getline(in, line)) throw std::invalid_argument("empty CSV");
  const auto head = splitLine(line);
  for (std::size_t k = 0; k < head.size(); ++k) r.col[lower(head[k])] = static_cast<int>(k);
  for (const char* need : {"date", "ticker", "close"})
    if (!r.col.count(need)) throw std::invalid_argument(std::string("CSV needs a '") + need + "' column");
  while (std::getline(in, line)) {
    if (line.empty() || line == "\r") continue;
    r.cells.push_back(splitLine(line));
  }
  return r;
}

std::string cell(const Rows& r, const std::vector<std::string>& row, const char* name) {
  const auto it = r.col.find(name);
  if (it == r.col.end() || it->second >= static_cast<int>(row.size())) return {};
  return row[static_cast<std::size_t>(it->second)];
}

}  // namespace

Market parseMarketCsv(const std::string& text) {
  const Rows r = readRows(text);
  std::vector<std::string> dates, tickers;
  for (const auto& row : r.cells) dates.push_back(cell(r, row, "date")), tickers.push_back(cell(r, row, "ticker"));
  std::sort(dates.begin(), dates.end());
  dates.erase(std::unique(dates.begin(), dates.end()), dates.end());
  std::sort(tickers.begin(), tickers.end());
  tickers.erase(std::unique(tickers.begin(), tickers.end()), tickers.end());
  std::unordered_map<std::string, std::size_t> di, ti;
  for (std::size_t k = 0; k < dates.size(); ++k) di[dates[k]] = k;
  for (std::size_t k = 0; k < tickers.size(); ++k) ti[tickers[k]] = k;
  Market m;
  m.dates = dates, m.tickers = tickers;
  const std::size_t T = dates.size(), N = tickers.size();
  m.open = m.high = m.low = m.close = m.volume = Panel(T, N);
  for (const auto& row : r.cells) {
    const std::size_t t = di[cell(r, row, "date")], i = ti[cell(r, row, "ticker")];
    const double c = num(cell(r, row, "close"));
    if (!(c > 0)) continue;
    const double o = num(cell(r, row, "open")), h = num(cell(r, row, "high")), l = num(cell(r, row, "low"));
    m.close(t, i) = c;
    m.open(t, i) = o > 0 ? o : c;
    m.high(t, i) = h > 0 ? std::max(h, c) : c;
    m.low(t, i) = l > 0 ? std::min(l, c) : c;
    const double v = num(cell(r, row, "volume"));
    m.volume(t, i) = std::isfinite(v) && v >= 0 ? v : 0.0;
  }
  if (T < 3 || N < 2) throw std::invalid_argument("CSV has too few dates or tickers");
  return m;
}

std::map<std::string, std::vector<double>> parseSeriesCsv(const std::string& text, const std::vector<std::string>& dates) {
  const Rows r = readRows(text);
  std::map<std::string, std::map<std::string, double>> raw;
  for (const auto& row : r.cells) {
    const double c = num(cell(r, row, "close"));
    if (c > 0) raw[cell(r, row, "ticker")][cell(r, row, "date")] = c;
  }
  std::map<std::string, std::vector<double>> out;
  for (const auto& [name, byDate] : raw) {
    std::vector<double> v(dates.size(), kNaN);
    auto it = byDate.begin();
    double last = kNaN;
    const std::string& lastDate = byDate.rbegin()->first;
    for (std::size_t t = 0; t < dates.size() && dates[t] <= lastDate; ++t) {
      while (it != byDate.end() && it->first <= dates[t]) last = it->second, ++it;
      v[t] = last;
    }
    out[name] = std::move(v);
  }
  return out;
}

namespace {

// SplitMix64 + Box-Muller: the same numbers on every compiler and in WebAssembly.
struct Rng {
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }
  double normal() { return std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307179586 * uniform()); }
};

}  // namespace

Synthetic syntheticMarket(std::uint64_t seed, std::size_t N, std::size_t T) {
  Rng rng(seed);
  Synthetic s;
  Market& m = s.market;
  m.open = m.high = m.low = m.close = m.volume = Panel(T, N);
  for (std::size_t i = 0; i < N; ++i) {
    char b[32];
    std::snprintf(b, sizeof b, "SYN%03zu", i);
    m.tickers.push_back(b);
  }
  // Business days from 2016-01-04.
  int y = 2016, mo = 1, d = 4, wd = 1;
  static const int mdays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  while (m.dates.size() < T) {
    if (wd >= 1 && wd <= 5) {
      char b[48];
      std::snprintf(b, sizeof b, "%04d-%02d-%02d", y, mo, d);
      m.dates.push_back(b);
    }
    wd = (wd + 1) % 7;
    const int len = mdays[mo - 1] + (mo == 2 && y % 4 == 0 ? 1 : 0);
    if (++d > len) d = 1, ++mo;
    if (mo > 12) mo = 1, ++y;
  }
  std::vector<double> vol(N), size(N), logp(N, std::log(100.0)), beta(N);
  for (std::size_t i = 0; i < N; ++i) {
    vol[i] = 0.010 + 0.015 * rng.uniform();
    size[i] = 12.0 + 3.0 * rng.normal();
    beta[i] = 0.7 + 0.6 * rng.uniform();
  }
  std::vector<std::vector<double>> hist(T, std::vector<double>(N));
  std::vector<double> fut(T);
  double f = 1000.0;
  for (std::size_t t = 0; t < T; ++t) {
    const double mkt = 0.0003 + 0.009 * rng.normal();
    // Built-in effects, all cross-sectional: momentum over ~6 months, reversal over 5 days,
    // and a low-volatility premium.
    std::vector<double> mom(N, 0.0), rev(N, 0.0);
    if (t > 130) {
      std::vector<double> a(N), b(N);
      for (std::size_t i = 0; i < N; ++i) a[i] = hist[t - 1][i] - hist[t - 126][i], b[i] = hist[t - 1][i] - hist[t - 6][i];
      auto rankc = [&](const std::vector<double>& x, std::vector<double>& out) {
        std::vector<std::size_t> o(N);
        for (std::size_t i = 0; i < N; ++i) o[i] = i;
        std::sort(o.begin(), o.end(), [&](std::size_t p, std::size_t q) { return x[p] < x[q]; });
        for (std::size_t r = 0; r < N; ++r) out[o[r]] = static_cast<double>(r) / static_cast<double>(N - 1) - 0.5;
      };
      rankc(a, mom), rankc(b, rev);
    }
    for (std::size_t i = 0; i < N; ++i) {
      const double r = beta[i] * mkt + vol[i] * rng.normal() + 0.0006 * mom[i] - 0.0008 * rev[i] - 0.02 * (vol[i] - 0.0175);
      const double prev = std::exp(logp[i]);
      logp[i] += std::log1p(r);
      const double c = std::exp(logp[i]);
      const double o = prev * (1.0 + 0.4 * (c / prev - 1.0) + 0.002 * rng.normal());
      const double span = std::fabs(0.6 * vol[i] * rng.normal()) * c;
      m.close(t, i) = c;
      m.open(t, i) = o;
      m.high(t, i) = std::max({c, o}) + span;
      m.low(t, i) = std::max(0.01, std::min({c, o}) - span);
      m.volume(t, i) = std::exp(size[i] + 0.3 * rng.normal()) / c;
      hist[t][i] = logp[i];
    }
    f *= 1.0 + mkt;
    fut[t] = f;
  }
  s.series["SYN:INDEXFUT"] = fut;
  return s;
}

}  // namespace ofm
