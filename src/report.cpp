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

constexpr const char* kDayName = "daily trades";

Facts facts(const Market& m, const Backtest& bt, const DayBook* day) {
  Facts F;
  F.n = bt.model.size();
  F.last = m.T() - 1;
  F.parts = {{"whole period", 0, F.n}, {"first half", 0, F.n / 2}, {"second half", F.n / 2, F.n}};
  F.series = {{"model", bt.model}, {"model + futures hedge", bt.hedged}, {"market (bought and held)", bt.market}};
  if (day && day->start == bt.start && day->ret.size() == F.n) F.series.push_back({kDayName, day->ret});
  F.pass = F.passHedged = F.n > 4;
  for (const auto& p : F.parts) {
    const double mk = metrics(slice(bt.market, p.lo, p.hi)).sharpe;
    F.pass = F.pass && metrics(slice(bt.model, p.lo, p.hi)).sharpe > mk;
    F.passHedged = F.passHedged && metrics(slice(bt.hedged, p.lo, p.hi)).sharpe > mk;
  }
  const double years = static_cast<double>(F.n) / 252.0;
  F.modelTurnoverYear = std::accumulate(bt.turnover.begin(), bt.turnover.end(), 0.0) / std::max(years, 1e-9);
  F.meanHoldings = bt.holdings.empty() ? 0 : std::accumulate(bt.holdings.begin(), bt.holdings.end(), 0.0) / static_cast<double>(bt.holdings.size());
  // Average hedge, as a fraction of the book's value (beta times the hedge ratio).
  F.hedgeShare = bt.hedge.empty() ? 0 : std::accumulate(bt.hedge.begin(), bt.hedge.end(), 0.0) / static_cast<double>(bt.hedge.size());
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

// The daily book's mean daily lead over the model portfolio, its t-statistic and the share of days ahead.
struct Lead {
  double mean = 0, t = 0, ahead = 0;
};
Lead lead(const std::vector<double>& a, const std::vector<double>& b) {
  Lead L;
  const double n = static_cast<double>(a.size());
  if (n < 2) return L;
  double s = 0, s2 = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    const double d = a[k] - b[k];
    s += d, s2 += d * d, L.ahead += a[k] > b[k];
  }
  L.mean = s / n, L.ahead /= n;
  const double v = (s2 - n * L.mean * L.mean) / (n - 1);
  L.t = v > 0 ? L.mean / std::sqrt(v / n) : 0;
  return L;
}

bool hasDay(const Facts& F) { return F.series.size() > 3; }

}  // namespace

std::string reportJson(const Market& m, const Forecast& f, const Backtest& bt, const Costs& costs, const MarketStructure* ms, const DayBook* day) {
  const Facts F = facts(m, bt, day);
  const Plan p = compilePlan(m);
  std::ostringstream o;
  o << "{\"engine\":" << str(f.engine) << ",\"kernelMs\":" << num(f.kernelMs) << ",\"stocks\":" << m.N() << ",\"days\":" << m.T()
    << ",\"from\":" << str(m.dates.front()) << ",\"to\":" << str(m.dates.back()) << ",\"records\":" << f.records
    << ",\"costs\":{\"buyBps\":" << num(costs.buyBps) << ",\"sellBps\":" << num(costs.sellBps) << ",\"futuresBps\":" << num(costs.futuresBps)
    << ",\"dayBps\":" << num(costs.dayBps)  << "},\"horizons\":[";
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
  // The model portfolio's last 20 days (net), for the daily report and the live record.
  o << "],\"recent\":[";
  const std::size_t from = F.n > 20 ? F.n - 20 : 0;
  for (std::size_t k = from; k < F.n; ++k)
    o << (k > from ? "," : "") << "{\"date\":" << str(m.dates[bt.start + k]) << ",\"ret\":" << num(bt.model[k]) << ",\"market\":" << num(bt.market[k])
      << (hasDay(F) ? ",\"day\":" + num(F.series[3].second[k]) : std::string()) << "}";
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
  if (hasDay(F)) {
    const Lead L = lead(day->ret, bt.model);
    const double n = static_cast<double>(F.n);
    o << ",\"day\":{\"name\":" << str(kDayName) << ",\"buyBps\":" << num(1e4 * day->buy) << ",\"sellBps\":" << num(1e4 * day->sell)
      << ",\"trades\":" << day->trades << ",\"tradesPerDay\":" << num(static_cast<double>(day->trades) / n) << ",\"stops\":" << day->stops
      << ",\"takes\":" << day->takes << ",\"halfLife\":" << day->halfLife << ",\"stopLimit\":" << num(day->stopLimit) << ",\"bookSize\":" << day->bookSize << ",\"learner\":\"" << day->learner << "\"" << ",\"backtestLoses\":" << (day->backtestLoses ? "true" : "false") << ",\"optimisticAnnual\":" << num(metrics(day->optimistic).annualReturn) << ",\"exposure\":" << num(day->exposure.empty() ? 0.0 : std::accumulate(day->exposure.begin(), day->exposure.end(), 0.0) / static_cast<double>(day->exposure.size())) << ",\"lead\":{\"mean\":" << num(L.mean) << ",\"t\":" << num(L.t) << ",\"ahead\":" << num(L.ahead)
      << "},\"screen\":{\"stocks\":" << day->screen.stocks << ",\"bars\":" << day->screen.bars << ",\"flatBars\":" << day->screen.flatBars
      << ",\"candidates\":" << day->screen.candidates << ",\"forecastUp\":" << day->screen.forecastUp << ",\"gainExpected\":"
      << day->screen.gainExpected << ",\"recordUp\":" << day->screen.recordUp << ",\"all\":" << day->screen.all << "},\"orders\":[";
    for (std::size_t q = 0; q < day->orders.size(); ++q) {
      const DayOrder& d = day->orders[q];
      const double stop = std::isfinite(d.stop) ? std::exp(-d.stop * d.sigma) - 1 : kNaN, take = std::isfinite(d.take) ? std::exp(d.take * d.sigma) - 1 : kNaN;
      o << (q ? "," : "") << "{\"ticker\":" << str(m.tickers[d.asset]) << ",\"expected\":" << num(d.expected) << ",\"tradeReturn\":" << num(d.tradeReturn) << ",\"intraday\":" << num(d.intraday) << ",\"weight\":" << num(d.weight) 
        << ",\"sigma\":" << num(d.sigma)
        << ",\"stopMult\":" << num(d.stop) << ",\"takeMult\":" << num(d.take) << ",\"stop\":" << num(stop) << ",\"take\":" << num(take)
        << ",\"record\":" << num(d.record) << ",\"window\":" << d.window << "}";
    }
    o << "]}";
  }
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

std::string reportText(const Market& m, const Forecast& f, const Backtest& bt, const Costs& costs, const MarketStructure* ms, const DayBook* day) {
  const Facts F = facts(m, bt, day);
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
  std::snprintf(b, sizeof b, "\ntrades %zu (won %zu, mean %+.2f%%; lost %zu, mean %+.2f%%); turnover %.1fx a year; %.1f stocks held on average; average hedge %.0f%% of the book\n",
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
  if (hasDay(F)) {
    const Lead L = lead(day->ret, bt.model);
    std::snprintf(b, sizeof b, "\nDaily trades (%.2f%% per side; volatility half-life %u days): %.2f trades a day, %zu at the stop, %zu at the take-profit; ahead of the model portfolio by %+.3f%% a day (t %+.2f)\n",
                  100 * day->buy, day->halfLife, static_cast<double>(day->trades) / static_cast<double>(F.n), day->stops, day->takes, 100 * L.mean, L.t);
    o << b;
    const auto& sc = day->screen;
    std::snprintf(b, sizeof b, "Screen at the last close: %zu stocks, %zu with a bar (%zu flat: high = low), %zu candidates; forecast up %zu, gain expected %zu, record beats cash %zu; all three %zu\n",
                  sc.stocks, sc.bars, sc.flatBars, sc.candidates, sc.forecastUp, sc.gainExpected, sc.recordUp, sc.all);
    o << b;
    {
      const Metrics wc = metrics(day->ret), op = metrics(day->optimistic);
      std::snprintf(b, sizeof b, "Fill test: %+.1f%% a year with the stop first, %+.1f%% with the take-profit first when a bar touched both (%s)\n",
                    100 * wc.annualReturn, 100 * op.annualReturn,
                    std::fabs(op.annualReturn - wc.annualReturn) <= 0.25 * std::fabs(wc.annualReturn) ? "robust" : "FILL-DEPENDENT: do not trust");
      o << b;
      std::snprintf(b, sizeof b, "Stopped out: %.1f%% of trades\n", 100.0 * double(day->stops) / double(std::max<std::size_t>(1, day->trades)));
      o << b;
      o << "Stop and take-profit levels from: " << day->levelsFrom << "\n";
      if (day->backtestLoses) o << "No trades for the next open: the daily trades lost money over the backtest\n";
      if (day->pathDays > 0) {
        std::snprintf(b, sizeof b, "Intraday paths: %zu stock-days; %zu trades with exact fills, %zu where both levels were touched and the path decided\n",
                      day->pathDays, day->pathTrades, day->pathResolved);
        o << b;
      }
    }
    std::snprintf(b, sizeof b, "Learnt stop limit: %.2f%% from the open; book of the %zu strongest; stop/take-profit learner: %s (its record %+.4f%% a day)\n", 100 * day->stopLimit, day->bookSize, day->learner.c_str(), 100 * day->limitScore);
    o << b;
    std::snprintf(b, sizeof b, "Data check: largest open-to-close move %.1f volatilities; %zu outliers (Chauvenet) cut to the bound. Intraday forecast skill: correlation %+.4f over %.0f forecasts (t %+.1f)\n", day->maxTarget, day->bigTargets, day->skill.corr(), day->skill.n, day->skill.corr() * std::sqrt(std::max(0.0, day->skill.n - 2)));
    o << b << "Trades for the next open (buy at the open, sell at the stop, the take-profit or the close)\n";
    for (const DayOrder& d : day->orders) {
      std::snprintf(b, sizeof b, "  %-10s %4.1f%% of capital  trade %+.3f%%  stop %s  take %s  record %+.3f%% (%zu days)\n", m.tickers[d.asset].c_str(), 100 * d.weight, 100 * d.tradeReturn,
                    std::isfinite(d.stop) ? (num(100 * (std::exp(-d.stop * d.sigma) - 1), 3) + "%").c_str() : "none",
                    std::isfinite(d.take) ? (num(100 * (std::exp(d.take * d.sigma) - 1), 3) + "%").c_str() : "none", 100 * d.record, d.window);
      o << b;
    }
    if (day->orders.empty()) o << "  none: no trade is expected to make money\n";
  }
  o << "\nCalendar years (net return)\n  year   model    hedged   market" << (hasDay(F) ? "   daily" : "") << "\n";
  std::string year;
  std::vector<double> acc(F.series.size(), 1.0);
  for (std::size_t k = 0; k <= F.n; ++k) {
    const std::string y = k < F.n ? m.dates[bt.start + k].substr(0, 4) : "";
    if (k > 0 && y != year) {
      std::snprintf(b, sizeof b, "  %s %7.1f%% %7.1f%% %7.1f%%", year.c_str(), 100 * (acc[0] - 1), 100 * (acc[1] - 1), 100 * (acc[2] - 1));
      o << b;
      if (acc.size() > 3) std::snprintf(b, sizeof b, " %7.1f%%", 100 * (acc[3] - 1)), o << b;
      o << "\n";
      std::fill(acc.begin(), acc.end(), 1.0);
    }
    if (k == F.n) break;
    year = y;
    for (std::size_t s = 0; s < acc.size(); ++s) acc[s] *= 1 + F.series[s].second[k];
  }
  o << "\nPass mark (beats the market on Sharpe ratio over the whole period and in both halves): model " << (F.pass ? "PASS" : "FAIL")
    << ", model + hedge " << (F.passHedged ? "PASS" : "FAIL") << "\n";
  o << "Caveats: universe of today's most traded surviving shares; prices exclude dividends; fills at the open with no slippage beyond the costs; past results do not predict future ones.\n";
  return o.str();
}

}  // namespace ofm
