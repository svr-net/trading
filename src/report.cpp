#include "ofm/report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>

namespace ofm {

namespace {

std::string num(double x, int digits = 6) {
  if (!std::isfinite(x)) return "null";
  char b[40];
  std::snprintf(b, sizeof b, "%.*g", digits, x);
  return b;
}

std::string str(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

struct Part {
  std::string name;
  std::size_t lo, hi;
};

std::vector<double> slice(const std::vector<double>& v, std::size_t lo, std::size_t hi) {
  return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(lo), v.begin() + static_cast<std::ptrdiff_t>(hi));
}

struct Facts {
  std::size_t n = 0, last = 0;
  std::vector<Part> parts;
  std::vector<std::pair<std::string, std::vector<double>>> series;
  bool pass = false, passHedged = false;
  double modelTurnoverYear = 0, meanHoldings = 0, hedgeShare = 0;
  std::size_t wins = 0, losses = 0;
  double meanWin = 0, meanLoss = 0;
};

Facts facts(const Market& m, const Backtest& bt) {
  Facts F;
  F.n = bt.model.size();
  F.last = m.T() - 1;
  F.parts = {{"whole period", 0, F.n}, {"first half", 0, F.n / 2}, {"second half", F.n / 2, F.n}};
  F.series = {{"model", bt.model}, {"model + futures hedge", bt.hedged}, {"market (bought and held)", bt.market}};
  F.pass = F.passHedged = F.n > 4;
  for (const auto& p : F.parts) {
    const double mk = metrics(slice(bt.market, p.lo, p.hi)).sharpe;
    F.pass = F.pass && metrics(slice(bt.model, p.lo, p.hi)).sharpe > mk;
    F.passHedged = F.passHedged && metrics(slice(bt.hedged, p.lo, p.hi)).sharpe > mk;
  }
  const double years = static_cast<double>(F.n) / 252.0;
  F.modelTurnoverYear = std::accumulate(bt.turnover.begin(), bt.turnover.end(), 0.0) / std::max(years, 1e-9);
  F.meanHoldings = bt.holdings.empty() ? 0 : std::accumulate(bt.holdings.begin(), bt.holdings.end(), 0.0) / static_cast<double>(bt.holdings.size());
  F.hedgeShare = bt.hedge.empty() ? 0 : static_cast<double>(std::count_if(bt.hedge.begin(), bt.hedge.end(), [](double x) { return x > 0; })) /
                                             static_cast<double>(bt.hedge.size());
  for (const auto& t : bt.trades) {
    if (t.ret > 0) ++F.wins, F.meanWin += t.ret;
    else ++F.losses, F.meanLoss += t.ret;
  }
  if (F.wins) F.meanWin /= static_cast<double>(F.wins);
  if (F.losses) F.meanLoss /= static_cast<double>(F.losses);
  return F;
}

// Stocks at the last close by expected return.
std::vector<std::size_t> byExpected(const Forecast& f, std::size_t t) {
  std::vector<std::size_t> o;
  for (std::size_t i = 0; i < f.E.N; ++i)
    if (std::isfinite(f.E(t, i))) o.push_back(i);
  std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return f.E(t, a) > f.E(t, b); });
  return o;
}

}  // namespace

std::string reportJson(const Market& m, const Forecast& f, const Backtest& bt, const Costs& costs, const MarketStructure* ms) {
  const Facts F = facts(m, bt);
  const Plan p = compilePlan(m);
  std::ostringstream o;
  o << "{\"engine\":" << str(f.engine) << ",\"kernelMs\":" << num(f.kernelMs) << ",\"stocks\":" << m.N() << ",\"days\":" << m.T()
    << ",\"from\":" << str(m.dates.front()) << ",\"to\":" << str(m.dates.back()) << ",\"records\":" << f.records
    << ",\"costs\":{\"buyBps\":" << num(costs.buyBps) << ",\"sellBps\":" << num(costs.sellBps) << ",\"futuresBps\":" << num(costs.futuresBps)
    << "},\"horizons\":[";
  for (std::size_t j = 0; j < p.H; ++j) o << (j ? "," : "") << p.horizons[j];
  o << "],\"hedgeSeries\":" << str(bt.hedgeSeries);
  const double life = bt.life.empty() ? 1.0 : bt.life.back();
  o << ",\"life\":" << num(life) << ",\"asOf\":" << str(m.dates.back()) << ",\"expected\":[";
  bool first = true;
  for (std::size_t i : byExpected(f, F.last)) {
    const double e = f.E(F.last, i);
    o << (first ? "" : ",") << "{\"ticker\":" << str(m.tickers[i]) << ",\"daily\":" << num(e) << ",\"overLife\":" << num(e * life)
      << ",\"held\":" << (bt.held.empty() ? 0 : bt.held[i]) << ",\"order\":" << (bt.orders.empty() ? 0 : bt.orders[i]) << "}";
    first = false;
  }
  o << "],\"factors\":[";
  for (std::size_t k = 0; k < p.K && k < f.mean.size(); ++k)
    o << (k ? "," : "") << "{\"name\":" << str(p.signalName(k)) << ",\"mean\":" << num(f.mean[k]) << ",\"t\":" << num(f.tstat[k])
      << ",\"premium\":" << num(f.premium[k]) << "}";
  o << "],\"dates\":[";
  // Cumulative curves, at most ~600 points.
  const std::size_t step = std::max<std::size_t>(1, F.n / 600);
  for (std::size_t k = 0; k < F.n; k += step) o << (k ? "," : "") << str(m.dates[bt.start + k]);
  o << "],\"curves\":{";
  for (std::size_t s = 0; s < F.series.size(); ++s) {
    o << (s ? "," : "") << str(F.series[s].first) << ":[";
    double w = 1;
    for (std::size_t k = 0; k < F.n; ++k) {
      w *= 1 + F.series[s].second[k];
      if (k % step == 0) o << (k ? "," : "") << num(w, 5);
    }
    o << "]";
  }
  o << "},\"parts\":[";
  for (std::size_t q = 0; q < F.parts.size(); ++q) {
    const auto& pt = F.parts[q];
    o << (q ? "," : "") << "{\"name\":" << str(pt.name) << ",\"from\":" << str(m.dates[bt.start + pt.lo]) << ",\"to\":"
      << str(m.dates[bt.start + pt.hi - 1]) << ",\"rows\":[";
    for (std::size_t s = 0; s < F.series.size(); ++s) {
      const Metrics mm = metrics(slice(F.series[s].second, pt.lo, pt.hi));
      o << (s ? "," : "") << "{\"name\":" << str(F.series[s].first) << ",\"annualReturn\":" << num(mm.annualReturn) << ",\"volatility\":"
        << num(mm.volatility) << ",\"sharpe\":" << num(mm.sharpe) << ",\"maxDrawdown\":" << num(mm.maxDrawdown) << "}";
    }
    o << "]}";
  }
  o << "],\"years\":[";
  std::string year;
  std::vector<double> acc(F.series.size(), 1.0);
  bool firstYear = true;
  for (std::size_t k = 0; k <= F.n; ++k) {
    const std::string y = k < F.n ? m.dates[bt.start + k].substr(0, 4) : "";
    if (k > 0 && y != year) {
      o << (firstYear ? "" : ",") << "{\"year\":" << str(year) << ",\"returns\":[";
      for (std::size_t s = 0; s < acc.size(); ++s) o << (s ? "," : "") << num(acc[s] - 1);
      o << "]}";
      firstYear = false;
      std::fill(acc.begin(), acc.end(), 1.0);
    }
    if (k == F.n) break;
    year = y;
    for (std::size_t s = 0; s < acc.size(); ++s) acc[s] *= 1 + F.series[s].second[k];
  }
  o << "],\"pass\":" << (F.pass ? "true" : "false") << ",\"passHedged\":" << (F.passHedged ? "true" : "false") << ",\"trades\":{\"count\":"
    << bt.trades.size() << ",\"wins\":" << F.wins << ",\"losses\":" << F.losses << ",\"meanWin\":" << num(F.meanWin) << ",\"meanLoss\":"
    << num(F.meanLoss) << ",\"turnoverPerYear\":" << num(F.modelTurnoverYear) << ",\"meanHoldings\":" << num(F.meanHoldings)
    << ",\"hedgeShare\":" << num(F.hedgeShare) << "}";
  if (ms && !ms->halfLives.empty()) {
    const std::size_t H = ms->halfLives.size(), mid = H / 2;
    o << ",\"structure\":{\"halfLives\":[";
    for (std::size_t j = 0; j < H; ++j) o << (j ? "," : "") << ms->halfLives[j];
    o << "],\"latest\":[";
    for (std::size_t j = 0; j < H; ++j)
      o << (j ? "," : "") << "{\"halfLife\":" << ms->halfLives[j] << ",\"treeLength\":" << num(ms->treeLength(F.last, j))
        << ",\"dimension\":" << num(ms->dimension(F.last, j)) << ",\"geodesic\":" << num(ms->geodesic(F.last, j)) << "}";
    o << "],\"series\":{\"halfLife\":" << ms->halfLives[mid] << ",\"treeLength\":[";
    for (std::size_t k = 0; k < F.n; k += step) o << (k ? "," : "") << num(ms->treeLength(bt.start + k, mid), 5);
    o << "],\"dimension\":[";
    for (std::size_t k = 0; k < F.n; k += step) o << (k ? "," : "") << num(ms->dimension(bt.start + k, mid), 5);
    o << "],\"geodesic\":[";
    for (std::size_t k = 0; k < F.n; k += step) o << (k ? "," : "") << num(ms->geodesic(bt.start + k, mid), 5);
    o << "]}}";
  }
  o << "}";
  return o.str();
}

std::string reportText(const Market& m, const Forecast& f, const Backtest& bt, const Costs& costs, const MarketStructure* ms) {
  const Facts F = facts(m, bt);
  const Plan p = compilePlan(m);
  std::ostringstream o;
  char b[256];
  std::snprintf(b, sizeof b, "data: %zu stocks, %zu days, %s .. %s; engine %s (%.0f ms)\n", m.N(), m.T(), m.dates.front().c_str(),
                m.dates.back().c_str(), f.engine.c_str(), f.kernelMs);
  o << b;
  o << "horizons (days):";
  for (auto h : p.horizons) o << " " << h;
  std::snprintf(b, sizeof b, "; %zu signals; %zu daily factor records\ncosts: %.0f bp on purchases, %.0f bp on sales, %.1f bp on futures; hedge series: %s\n",
                p.K, f.records, costs.buyBps, costs.sellBps, costs.futuresBps, bt.hedgeSeries.empty() ? "none" : bt.hedgeSeries.c_str());
  o << b;
  if (F.n == 0) return o.str() + "no trading days\n";
  for (const auto& pt : F.parts) {
    std::snprintf(b, sizeof b, "\n%s: %s .. %s (%.1f years)\n  %-28s %8s %7s %7s %7s\n", pt.name.c_str(), m.dates[bt.start + pt.lo].c_str(),
                  m.dates[bt.start + pt.hi - 1].c_str(), static_cast<double>(pt.hi - pt.lo) / 252.0, "strategy", "ann.ret", "vol", "Sharpe", "max DD");
    o << b;
    for (const auto& s : F.series) {
      const Metrics mm = metrics(slice(s.second, pt.lo, pt.hi));
      std::snprintf(b, sizeof b, "  %-28s %7.1f%% %6.1f%% %7.2f %6.1f%%\n", s.first.c_str(), 100 * mm.annualReturn, 100 * mm.volatility, mm.sharpe,
                    100 * mm.maxDrawdown);
      o << b;
    }
  }
  std::snprintf(b, sizeof b, "\ntrades %zu (won %zu, mean %+.2f%%; lost %zu, mean %+.2f%%); turnover %.1fx a year; %.1f stocks held on average; hedged %.0f%% of days\n",
                bt.trades.size(), F.wins, 100 * F.meanWin, F.losses, 100 * F.meanLoss, F.modelTurnoverYear, F.meanHoldings, 100 * F.hedgeShare);
  o << b;
  const double life = bt.life.empty() ? 1.0 : bt.life.back();
  std::snprintf(b, sizeof b, "\nExpected excess return over the market at %s (per day, and over a forecast's life of %.1f days)\n", m.dates.back().c_str(), life);
  o << b;
  const auto order = byExpected(f, F.last);
  auto line = [&](std::size_t i) {
    std::snprintf(b, sizeof b, "  %-10s %+8.2f bp  %+8.1f bp  %s%s\n", m.tickers[i].c_str(), 1e4 * f.E(F.last, i), 1e4 * f.E(F.last, i) * life,
                  bt.held.empty() || !bt.held[i] ? "" : "held", bt.orders.empty() ? "" : bt.orders[i] > 0 ? " buy at next open" : bt.orders[i] < 0 ? " sell at next open" : "");
    o << b;
  };
  for (std::size_t k = 0; k < std::min<std::size_t>(10, order.size()); ++k) line(order[k]);
  if (order.size() > 20) o << "  ...\n";
  for (std::size_t k = order.size() > 10 ? std::max(order.size() - 10, std::size_t{10}) : order.size(); k < order.size(); ++k) line(order[k]);
  o << "\nFactors with a non-zero shrunk premium (mean daily factor return, t, premium)\n";
  for (std::size_t k = 0; k < p.K && k < f.premium.size(); ++k)
    if (f.premium[k] != 0) {
      std::snprintf(b, sizeof b, "  %-22s %+.5f  t %+5.2f  %+.5f\n", p.signalName(k).c_str(), f.mean[k], f.tstat[k], f.premium[k]);
      o << b;
    }
  if (ms && !ms->halfLives.empty()) {
    o << "\nMarket topology and geometry at the last close (per half-life: MST tree length = H0 persistence, effective dimension, geodesic distance to the longest half-life)\n";
    for (std::size_t j = 0; j < ms->halfLives.size(); ++j) {
      std::snprintf(b, sizeof b, "  %4ud  tree %.4f  dimension %.4f  geodesic %s\n", ms->halfLives[j], ms->treeLength(F.last, j), ms->dimension(F.last, j),
                    std::isfinite(ms->geodesic(F.last, j)) ? num(ms->geodesic(F.last, j), 4).c_str() : "-");
      o << b;
    }
  }
  o << "\nCalendar years (net return)\n  year   model    hedged   market\n";
  std::string year;
  std::vector<double> acc(3, 1.0);
  for (std::size_t k = 0; k <= F.n; ++k) {
    const std::string y = k < F.n ? m.dates[bt.start + k].substr(0, 4) : "";
    if (k > 0 && y != year) {
      std::snprintf(b, sizeof b, "  %s %7.1f%% %7.1f%% %7.1f%%\n", year.c_str(), 100 * (acc[0] - 1), 100 * (acc[1] - 1), 100 * (acc[2] - 1));
      o << b;
      std::fill(acc.begin(), acc.end(), 1.0);
    }
    if (k == F.n) break;
    year = y;
    for (std::size_t s = 0; s < 3; ++s) acc[s] *= 1 + F.series[s].second[k];
  }
  o << "\nPass mark (beats the market on Sharpe ratio over the whole period and in both halves): model " << (F.pass ? "PASS" : "FAIL")
    << ", model + hedge " << (F.passHedged ? "PASS" : "FAIL") << "\n";
  o << "Caveats: universe of today's most traded surviving shares; prices exclude dividends; fills at the open with no slippage beyond the costs; past results do not predict future ones.\n";
  return o.str();
}

}  // namespace ofm
