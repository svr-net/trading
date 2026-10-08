#include "sat/data/market_data.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace sat {

Panel MarketData::returns() const {
  Panel r = close.like();
  for (std::size_t t = 1; t < close.dates(); ++t)
    for (std::size_t i = 0; i < close.assets(); ++i) r(t, i) = close(t, i) / close(t - 1, i) - 1.0;
  return r;
}

Panel MarketData::forwardReturns(std::size_t horizon) const {
  Panel r = close.like();
  for (std::size_t t = 0; t + horizon < close.dates(); ++t)
    for (std::size_t i = 0; i < close.assets(); ++i) r(t, i) = close(t + horizon, i) / close(t, i) - 1.0;
  return r;
}

void MarketData::validate() const {
  const std::size_t T = close.dates(), N = close.assets();
  if (T < 2 || N < 1) throw std::invalid_argument("market data needs at least 2 dates and 1 stock");
  for (const Panel* p : {&open, &high, &low, &volume, &vwap})
    if (p->dates() != T || p->assets() != N) throw std::invalid_argument("market data panels differ in shape");
  if (tickers.size() != N) throw std::invalid_argument("market data: one ticker per column expected");
  if (!dates.empty() && dates.size() != T) throw std::invalid_argument("market data: one date label per row expected");
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t i = 0; i < N; ++i)
      if (!(close(t, i) > 0.0) || !(open(t, i) > 0.0) || !(high(t, i) > 0.0) || !(low(t, i) > 0.0))
        throw std::invalid_argument("market data: prices must be positive (" + tickers[i] + ", row " + std::to_string(t) + ")");
}

MarketData MarketData::slice(std::size_t from, std::size_t to) const {
  to = std::min(to, numDates());
  if (from >= to) throw std::invalid_argument("market data slice is empty");
  MarketData out;
  out.tickers = tickers;
  out.regimeNames = regimeNames;
  auto cut = [&](const Panel& p) {
    Panel q(to - from, p.assets());
    std::copy(p.row(from), p.row(from) + (to - from) * p.assets(), q.row(0));
    return q;
  };
  out.open = cut(open);
  out.high = cut(high);
  out.low = cut(low);
  out.close = cut(close);
  out.volume = cut(volume);
  out.vwap = cut(vwap);
  if (!dates.empty()) out.dates.assign(dates.begin() + static_cast<std::ptrdiff_t>(from), dates.begin() + static_cast<std::ptrdiff_t>(to));
  if (!regime.empty()) out.regime.assign(regime.begin() + static_cast<std::ptrdiff_t>(from), regime.begin() + static_cast<std::ptrdiff_t>(to));
  return out;
}

namespace {

std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool quoted = false;
  for (char c : line) {
    if (c == '"') quoted = !quoted;
    else if (c == ',' && !quoted) {
      out.push_back(cur);
      cur.clear();
    } else if (c != '\r') cur += c;
  }
  out.push_back(cur);
  for (auto& s : out) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    std::size_t k = 0;
    while (k < s.size() && std::isspace(static_cast<unsigned char>(s[k]))) ++k;
    s.erase(0, k);
  }
  return out;
}

std::string lower(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

}  // namespace

MarketData parseCsv(const std::string& text) {
  std::istringstream in(text);
  std::string line;
  if (!std::getline(in, line)) throw std::invalid_argument("CSV is empty");
  const auto header = splitCsvLine(line);
  std::map<std::string, std::size_t> col;
  for (std::size_t k = 0; k < header.size(); ++k) col[lower(header[k])] = k;
  for (const char* need : {"date", "ticker", "open", "high", "low", "close", "volume"})
    if (!col.count(need)) throw std::invalid_argument(std::string("CSV header lacks column '") + need + "'");
  const bool hasVwap = col.count("vwap") > 0;

  struct Bar { double o, h, l, c, v, w; };
  std::map<std::string, std::map<std::string, Bar>> byTicker;  // ticker -> date -> bar
  std::map<std::string, int> dateSet;
  std::size_t lineNo = 1;
  while (std::getline(in, line)) {
    ++lineNo;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const auto f = splitCsvLine(line);
    auto get = [&](const char* name) -> double {
      const std::size_t k = col.at(name);
      if (k >= f.size()) throw std::invalid_argument("CSV line " + std::to_string(lineNo) + " has too few fields");
      try {
        return std::stod(f[k]);
      } catch (...) {
        throw std::invalid_argument("CSV line " + std::to_string(lineNo) + ": '" + f[k] + "' is not a number");
      }
    };
    Bar b{get("open"), get("high"), get("low"), get("close"), get("volume"), 0.0};
    b.w = hasVwap ? get("vwap") : (b.h + b.l + b.c) / 3.0;
    const std::string date = f.at(col.at("date")), ticker = f.at(col.at("ticker"));
    byTicker[ticker][date] = b;
    dateSet[date] = 0;
  }
  if (byTicker.empty()) throw std::invalid_argument("CSV has no rows");

  MarketData d;
  for (auto& kv : dateSet) d.dates.push_back(kv.first);
  for (auto& kv : byTicker) d.tickers.push_back(kv.first);
  const std::size_t T = d.dates.size(), N = d.tickers.size();
  d.open = d.high = d.low = d.close = d.volume = d.vwap = Panel(T, N);
  for (std::size_t i = 0; i < N; ++i) {
    const auto& bars = byTicker[d.tickers[i]];
    const Bar* last = &bars.begin()->second;  // back-fill before the first quote
    for (std::size_t t = 0; t < T; ++t) {
      const auto it = bars.find(d.dates[t]);
      double vol = 0.0;
      if (it != bars.end()) {
        last = &it->second;
        vol = last->v;
      }
      const Bar& b = *last;
      const bool fresh = it != bars.end();
      d.open(t, i) = fresh ? b.o : b.c;
      d.high(t, i) = fresh ? b.h : b.c;
      d.low(t, i) = fresh ? b.l : b.c;
      d.close(t, i) = b.c;
      d.vwap(t, i) = fresh ? b.w : b.c;
      d.volume(t, i) = vol;
    }
  }
  d.regime.assign(T, -1);
  d.validate();
  return d;
}

std::string toCsv(const MarketData& data) {
  std::ostringstream os;
  os << std::setprecision(10);
  os << "date,ticker,open,high,low,close,volume,vwap\n";
  for (std::size_t t = 0; t < data.numDates(); ++t)
    for (std::size_t i = 0; i < data.numAssets(); ++i)
      os << (data.dates.empty() ? std::to_string(t) : data.dates[t]) << ',' << data.tickers[i] << ',' << data.open(t, i) << ','
         << data.high(t, i) << ',' << data.low(t, i) << ',' << data.close(t, i) << ',' << data.volume(t, i) << ','
         << data.vwap(t, i) << '\n';
  return os.str();
}

}  // namespace sat
