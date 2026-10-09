// A self-adaptive portfolio of a few stocks from daily bars (built for the London Stock
// Exchange via tools/eoddata/fetch.mjs, but any CSV in the library's format works):
//
//   uk_portfolio data/LSE.csv [--stocks 10] [--budget 100000] [--price-divisor 100]
//                [--stamp-duty-bps 50] [--cost-bps 10] [--rebalance 21] [--report report.json] [--robustness]
//
// Five ways to pick and weight the stocks compete, each rebalanced every --rebalance days
// with only the data available at that close:
//   ML ensemble + HRP      highest mean next-day probability of the library's default models
//                          (walk-forward, 23 alphas), hierarchical-risk-parity weights;
//   momentum 12-1          highest 12-month return skipping the last month, inverse-volatility weights;
//   minimum variance       lowest-volatility stocks, long-only minimum-variance weights;
//   trailing Sharpe + HRP  highest one-year Sharpe ratio, HRP weights;
//   low correlation + IVP  least correlated with the market, inverse-variance weights.
// The self-adaptive portfolio holds, at each rebalance, the construction whose own record
// over the past --lookback days has the best Sharpe ratio (the paper's selector, applied to
// portfolio constructions). Everything is net of costs: --cost-bps on every trade (spread
// and commission) plus UK stamp duty (SDRT) on purchases. Prices on the LSE are quoted in
// pence: --price-divisor 100 converts them to pounds for the trade list.
//
// --robustness re-runs the selector with slower settings fixed in advance (3-, 6- and
// 12-month look-backs, choosing monthly or quarterly, and switching only when a rival's
// Sharpe ratio beats the incumbent's by a margin), all compared on the same days. It is a
// sensitivity check: picking the best row after the fact would be overfitting.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "sat/sat.hpp"

using namespace sat;

namespace {

struct Options {
  std::string csv, report;
  std::size_t stocks = 10, rebalance = 21, lookback = 126, window = 252;
  bool robustness = false;
  double budget = 100000, divisor = 100, stampBps = 50, costBps = 10;
};

Options parse(int argc, char** argv) {
  Options o;
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    auto next = [&]() -> std::string {
      if (k + 1 >= argc) throw std::invalid_argument("missing value for " + a);
      return argv[++k];
    };
    if (a == "--stocks") o.stocks = std::stoul(next());
    else if (a == "--budget") o.budget = std::stod(next());
    else if (a == "--price-divisor") o.divisor = std::stod(next());
    else if (a == "--stamp-duty-bps") o.stampBps = std::stod(next());
    else if (a == "--cost-bps") o.costBps = std::stod(next());
    else if (a == "--rebalance") o.rebalance = std::stoul(next());
    else if (a == "--lookback") o.lookback = std::stoul(next());
    else if (a == "--report") o.report = next();
    else if (a == "--robustness") o.robustness = true;
    else if (!a.empty() && a[0] != '-') o.csv = a;
    else throw std::invalid_argument("unknown option " + a);
  }
  if (o.csv.empty()) throw std::invalid_argument("usage: uk_portfolio data.csv [options]");
  return o;
}

std::string readFile(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

// A construction maps (date t, eligible stocks) to target weights over all stocks.
using Construction = std::function<std::vector<double>(std::size_t t, const std::vector<std::size_t>& eligible)>;

struct Sleeve {
  std::string name;
  std::vector<double> returns;               ///< from `start`
  std::vector<std::vector<double>> targets;  ///< per rebalance (empty when not rebalanced)
};

struct Book {
  std::vector<double> returns, turnover;
  std::vector<double> weights;  ///< after the last rebalance, drifted to the last date
};

// Daily simulation: targets set at the close of a rebalance date t earn from t to t + 1.
// Weights drift with prices between rebalances; trades pay costBps, purchases also stamp duty.
Book simulate(const Panel& ret, std::size_t start, std::size_t end, std::size_t every,
              const std::function<std::vector<double>(std::size_t)>& target, double costBps, double stampBps) {
  const std::size_t N = ret.assets();
  Book b;
  std::vector<double> w(N, 0.0);
  for (std::size_t t = start; t + 1 < end; ++t) {
    double cost = 0, turn = 0;
    if ((t - start) % every == 0) {
      const auto tw = target(t);
      for (std::size_t i = 0; i < N; ++i) {
        const double d = tw[i] - w[i];
        turn += std::fabs(d);
        cost += std::fabs(d) * costBps * 1e-4 + std::max(0.0, d) * stampBps * 1e-4;
      }
      w = tw;
    }
    double r = 0, grow = 0, invested = 0;
    for (std::size_t i = 0; i < N; ++i) {
      const double x = std::isfinite(ret(t + 1, i)) ? ret(t + 1, i) : 0.0;
      r += w[i] * x;
      invested += w[i];
    }
    // Drift: each weight grows with its stock, cash (1 - invested) stays.
    grow = 1 + r;
    for (std::size_t i = 0; i < N; ++i) w[i] = w[i] * (1 + (std::isfinite(ret(t + 1, i)) ? ret(t + 1, i) : 0.0)) / grow;
    (void)invested;
    b.returns.push_back(r - cost);
    b.turnover.push_back(turn);
  }
  b.weights = w;
  return b;
}

Matrix windowReturns(const Panel& ret, std::size_t t, std::size_t window, const std::vector<std::size_t>& cols) {
  Matrix R(window, cols.size());
  for (std::size_t k = 0; k < window; ++k)
    for (std::size_t j = 0; j < cols.size(); ++j) {
      const double x = ret(t - window + 1 + k, cols[j]);
      R(k, j) = std::isfinite(x) ? x : 0.0;
    }
  return R;
}

// Spreads the weights of the selected stocks over all N, keeping every selected stock between
// half and twice its equal share (4% and 20% for ten), so a portfolio of ten holds ten.
std::vector<double> scatter(std::size_t N, const std::vector<std::size_t>& cols, const std::vector<double>& w) {
  std::vector<double> out(N, 0.0);
  const std::size_t n = cols.size();
  if (n == 0) return out;
  std::vector<double> v(n);
  double s = 0;
  for (double x : w) s += std::max(0.0, x);
  for (std::size_t j = 0; j < n; ++j) v[j] = s > 0 ? std::max(0.0, w[j]) / s : 1.0 / n;
  if (n > 2) {
    const double lo = 0.5 / n, hi = std::min(1.0, 2.0 / n);
    for (int it = 0; it < 100; ++it) {  // alternate clipping and renormalising until both hold
      double t = 0;
      for (double& x : v) t += (x = std::clamp(x, lo, hi));
      for (double& x : v) x /= t;
    }
  }
  for (std::size_t j = 0; j < n; ++j) out[cols[j]] = v[j];
  return out;
}

std::vector<std::size_t> topBy(const std::vector<std::size_t>& eligible, const std::vector<double>& score, std::size_t n) {
  std::vector<std::size_t> idx;
  for (std::size_t i : eligible)
    if (std::isfinite(score[i])) idx.push_back(i);
  std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return score[a] > score[b]; });
  idx.resize(std::min(n, idx.size()));
  return idx;
}

std::string json(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

}  // namespace

int main(int argc, char** argv) try {
  const Options o = parse(argc, argv);
  const MarketData d = parseCsv(readFile(o.csv));
  const std::size_t T = d.numDates(), N = d.numAssets();
  if (N < o.stocks + 5) throw std::invalid_argument("need more stocks than the portfolio size");
  const Panel ret = d.returns();
  std::printf("%zu stocks, %zu days (%s .. %s)\n", N, T, d.dates.front().c_str(), d.dates.back().c_str());

  // Machine-learning forecasts (walk-forward, out of sample).
  ExperimentSpec exp;
  const PredictionSet p = runPredictions(d, exp);
  std::size_t mlStart = 0;
  for (const auto& m : p.models) mlStart = std::max(mlStart, m.start);
  std::printf("walk-forward forecasts from %s:", d.dates[mlStart].c_str());
  for (const auto& m : p.models) std::printf(" %s AUC %.3f;", m.name.c_str(), m.oos.auc);
  std::printf("\n");

  // Eligible at t: a full trading year (>= 90% of days with volume) behind it.
  auto eligibleAt = [&](std::size_t t) {
    std::vector<std::size_t> e;
    for (std::size_t i = 0; i < N; ++i) {
      std::size_t traded = 0;
      for (std::size_t k = t + 1 - o.window; k <= t; ++k) traded += d.volume(k, i) > 0;
      if (traded >= 0.9 * o.window) e.push_back(i);
    }
    return e;
  };
  auto hrp = [&](std::size_t t, const std::vector<std::size_t>& cols) {
    const Matrix cov = afml::covarianceMatrix(windowReturns(ret, t, o.window, cols));
    return scatter(N, cols, afml::hierarchicalRiskParity(cov, afml::clusterAssets(afml::correlationFromCovariance(cov)).order));
  };
  auto vol = [&](std::size_t t, std::size_t i, std::size_t w) {
    std::vector<double> x;
    for (std::size_t k = t + 1 - w; k <= t; ++k) x.push_back(std::isfinite(ret(k, i)) ? ret(k, i) : 0.0);
    return stdev(x);
  };
  std::vector<double> mkt(T, 0.0);
  for (std::size_t t = 1; t < T; ++t) {
    double s = 0;
    std::size_t n = 0;
    for (std::size_t i = 0; i < N; ++i)
      if (std::isfinite(ret(t, i)) && d.volume(t, i) > 0) s += ret(t, i), ++n;
    mkt[t] = n ? s / n : 0.0;
  }

  const std::vector<std::pair<std::string, Construction>> constructions = {
      {"ML ensemble + HRP", [&](std::size_t t, const std::vector<std::size_t>& e) {
         std::vector<double> score(N, NAN);
         for (std::size_t i = 0; i < N; ++i) {
           double s = 0;
           std::size_t n = 0;
           for (const auto& m : p.models)  // mean probability over the last 5 days smooths daily noise
             for (std::size_t k = t - 4; k <= t; ++k)
               if (std::isfinite(m.probability(k, i))) s += m.probability(k, i), ++n;
           if (n) score[i] = s / n;
         }
         return hrp(t, topBy(e, score, o.stocks));
       }},
      {"momentum 12-1 + inverse vol", [&](std::size_t t, const std::vector<std::size_t>& e) {
         std::vector<double> score(N, NAN);
         for (std::size_t i : e) score[i] = d.close(t - 21, i) / d.close(t - 251, i) - 1;
         const auto top = topBy(e, score, o.stocks);
         std::vector<double> w;
         for (std::size_t i : top) w.push_back(1 / std::max(1e-6, vol(t, i, 63)));
         return scatter(N, top, w);
       }},
      {"minimum variance", [&](std::size_t t, const std::vector<std::size_t>& e) {
         std::vector<double> score(N, NAN);
         for (std::size_t i : e) score[i] = -vol(t, i, o.window);
         const auto top = topBy(e, score, o.stocks);
         return scatter(N, top, afml::longOnlyMinimumVarianceWeights(afml::covarianceMatrix(windowReturns(ret, t, o.window, top))));
       }},
      {"trailing Sharpe + HRP", [&](std::size_t t, const std::vector<std::size_t>& e) {
         std::vector<double> score(N, NAN);
         for (std::size_t i : e) {
           std::vector<double> x;
           for (std::size_t k = t + 1 - o.window; k <= t; ++k) x.push_back(std::isfinite(ret(k, i)) ? ret(k, i) : 0.0);
           const double s = stdev(x);
           score[i] = s > 0 ? mean(x) / s : NAN;
         }
         return hrp(t, topBy(e, score, o.stocks));
       }},
      {"low correlation + IVP", [&](std::size_t t, const std::vector<std::size_t>& e) {
         std::vector<double> m(mkt.begin() + static_cast<std::ptrdiff_t>(t + 1 - o.window), mkt.begin() + static_cast<std::ptrdiff_t>(t + 1));
         std::vector<double> score(N, NAN);
         for (std::size_t i : e) {
           std::vector<double> x;
           for (std::size_t k = t + 1 - o.window; k <= t; ++k) x.push_back(std::isfinite(ret(k, i)) ? ret(k, i) : 0.0);
           score[i] = -correlation(x, m);
         }
         const auto top = topBy(e, score, o.stocks);
         return scatter(N, top, afml::inverseVarianceWeights(afml::covarianceMatrix(windowReturns(ret, t, o.window, top))));
       }},
  };

  const std::size_t start = std::max(mlStart + 4, o.window + 1);
  if (start + o.lookback + 60 >= T) throw std::invalid_argument("not enough history after the model warm-up");
  // Each construction alone; targets cached by rebalance date for the adaptive portfolio.
  std::vector<std::vector<std::vector<double>>> targets(constructions.size());
  std::vector<Book> books;
  for (std::size_t c = 0; c < constructions.size(); ++c) {
    targets[c].assign(T, {});
    books.push_back(simulate(ret, start, T, o.rebalance, [&](std::size_t t) {
      targets[c][t] = constructions[c].second(t, eligibleAt(t));
      return targets[c][t];
    }, o.costBps, o.stampBps));
  }
  // Self-adaptive: the construction with the best trailing Sharpe ratio of its own net returns.
  std::vector<int> chosen;
  const std::size_t aStart = start + o.lookback;
  const Book adaptive = simulate(ret, aStart, T, o.rebalance, [&](std::size_t t) {
    std::size_t best = 0;
    double bestScore = -1e18;
    for (std::size_t c = 0; c < books.size(); ++c) {
      const auto& r = books[c].returns;  // r[k] earned from start + k to start + k + 1
      std::vector<double> past(r.begin() + static_cast<std::ptrdiff_t>(t - start - o.lookback), r.begin() + static_cast<std::ptrdiff_t>(t - start));
      const double s = stdev(past), score = s > 0 ? mean(past) / s : 0.0;
      if (score > bestScore) bestScore = score, best = c;
    }
    chosen.push_back(static_cast<int>(best));
    if (targets[best][t].empty()) targets[best][t] = constructions[best].second(t, eligibleAt(t));  // off its own schedule
    return targets[best][t];
  }, o.costBps, o.stampBps);
  const Book bench = simulate(ret, aStart, T, o.rebalance, [&](std::size_t t) {
    const auto e = eligibleAt(t);
    return scatter(N, e, std::vector<double>(e.size(), 1.0));
  }, o.costBps, o.stampBps);

  // Compare over the same days (from aStart).
  struct Row {
    std::string name;
    std::vector<double> r, turnover;
  };
  std::vector<Row> rows;
  rows.push_back({"self-adaptive portfolio", adaptive.returns, adaptive.turnover});
  for (std::size_t c = 0; c < books.size(); ++c)
    rows.push_back({constructions[c].first, std::vector<double>(books[c].returns.begin() + static_cast<std::ptrdiff_t>(o.lookback), books[c].returns.end()),
                    std::vector<double>(books[c].turnover.begin() + static_cast<std::ptrdiff_t>(o.lookback), books[c].turnover.end())});
  rows.push_back({"equal-weight universe", bench.returns, bench.turnover});
  std::printf("\nBacktest %s .. %s (%zu days), net of %.0f bp per trade and %.0f bp stamp duty on purchases, rebalanced every %zu days\n",
              d.dates[aStart].c_str(), d.dates[T - 1].c_str(), adaptive.returns.size(), o.costBps, o.stampBps, o.rebalance);
  std::printf("  %-30s %8s %8s %7s %8s %9s\n", "portfolio", "CAGR", "vol", "Sharpe", "max DD", "turnover/yr");
  for (const auto& r : rows) {
    const auto m = evaluatePerformance(r.r, r.turnover);
    std::printf("  %-30s %7.1f%% %7.1f%% %7.2f %7.1f%% %8.0f%%\n", r.name.c_str(), 100 * m.annualReturn, 100 * m.annualVolatility, m.sharpe,
                100 * m.maxDrawdown, 100 * m.averageTurnover * 252);
  }

  // The library's composite strategy (sat/algo/composite): exponential weights across the
  // self-adaptive selector on the composite forecast, the blended stock-selection book and the
  // momentum book. Stamp duty is charged on purchases only, so half of it is added to the cost
  // per unit of turnover.
  ExperimentSpec compExp;
  compExp.costBps = o.costBps + 0.5 * o.stampBps;
  algo::CompositeStrategySpec compSpec;
  for (auto& b : compSpec.books) b.costBps = compExp.costBps, b.holdings = o.stocks;
  const auto comp = algo::runCompositeStrategy(d, p, compExp, compSpec);
  const std::size_t compDays = comp.returns[0].size();
  std::printf("\nComposite strategy %s .. %s (%zu days), %.0f bp per unit of turnover\n", d.dates[comp.start].c_str(),
              d.dates[comp.start + compDays].c_str(), compDays, compExp.costBps);
  std::printf("  %-52s %8s %8s %7s %8s %9s\n", "series", "CAGR", "vol", "Sharpe", "max DD", "weight now");
  for (std::size_t k = 0; k < comp.names.size(); ++k) {
    const auto& m = comp.metrics[k];
    const std::string now = k < comp.sleeves ? std::to_string(static_cast<int>(std::lround(100 * comp.weights[k].back()))) + "%" : "";
    std::printf("  %-52s %7.1f%% %7.1f%% %7.2f %7.1f%% %9s\n", comp.names[k].c_str(), 100 * m.annualReturn, 100 * m.annualVolatility, m.sharpe,
                100 * m.maxDrawdown, now.c_str());
  }
  std::printf("  cash now %.0f%%\n", 100 * comp.cash.back());

  // Robustness of the selector: slower variants on common days (after the longest look-back).
  struct RobustRow {
    std::string name, kind;
    std::vector<double> r, turnover;
    std::size_t switches = 0;
    std::string holding;  ///< selectors: the construction held at the end
  };
  std::vector<RobustRow> robust;
  std::size_t rStart = 0;
  if (o.robustness) {
    struct Variant {
      std::string name;
      std::size_t lookback, every;  // every: choose on every n-th rebalance
      double margin;                // annualised Sharpe a rival needs over the incumbent
    };
    const std::vector<Variant> variants = {
        {"6m look-back, monthly (default)", 126, 1, 0.0}, {"3m look-back, monthly", 63, 1, 0.0},
        {"12m look-back, monthly", 252, 1, 0.0},          {"6m look-back, quarterly", 126, 3, 0.0},
        {"12m look-back, quarterly", 252, 3, 0.0},        {"6m look-back, monthly, margin 0.5", 126, 1, 0.5},
        {"12m look-back, monthly, margin 0.5", 252, 1, 0.5}, {"12m look-back, quarterly, margin 0.5", 252, 3, 0.5},
    };
    rStart = start + 252;
    if (rStart + 60 >= T) throw std::invalid_argument("not enough history for the robustness check");
    auto score = [&](std::size_t c, std::size_t t, std::size_t lb) {
      const auto& r = books[c].returns;
      std::vector<double> past(r.begin() + static_cast<std::ptrdiff_t>(t - start - lb), r.begin() + static_cast<std::ptrdiff_t>(t - start));
      const double sd = stdev(past);
      return sd > 0 ? mean(past) / sd * std::sqrt(252.0) : 0.0;
    };
    for (const auto& v : variants) {
      int held = -1;
      std::size_t k = 0, switches = 0;
      const Book b = simulate(ret, rStart, T, o.rebalance, [&](std::size_t t) {
        if (held < 0 || k % v.every == 0) {
          std::size_t best = 0;
          double bestScore = -1e18;
          std::vector<double> sc(books.size());
          for (std::size_t c = 0; c < books.size(); ++c)
            if ((sc[c] = score(c, t, v.lookback)) > bestScore) bestScore = sc[c], best = c;
          if (held < 0) held = static_cast<int>(best);
          else if (static_cast<int>(best) != held && bestScore > sc[static_cast<std::size_t>(held)] + v.margin) held = static_cast<int>(best), ++switches;
        }
        ++k;
        auto& tw = targets[static_cast<std::size_t>(held)][t];
        if (tw.empty()) tw = constructions[static_cast<std::size_t>(held)].second(t, eligibleAt(t));
        return tw;
      }, o.costBps, o.stampBps);
      robust.push_back({v.name, "selector", b.returns, b.turnover, switches, constructions[static_cast<std::size_t>(held)].first});
    }
    const std::size_t off = rStart - start;
    for (std::size_t c = 0; c < books.size(); ++c)
      robust.push_back({constructions[c].first, "construction", std::vector<double>(books[c].returns.begin() + static_cast<std::ptrdiff_t>(off), books[c].returns.end()),
                        std::vector<double>(books[c].turnover.begin() + static_cast<std::ptrdiff_t>(off), books[c].turnover.end()), 0, ""});
    const Book u = simulate(ret, rStart, T, o.rebalance, [&](std::size_t t) {
      const auto e = eligibleAt(t);
      return scatter(N, e, std::vector<double>(e.size(), 1.0));
    }, o.costBps, o.stampBps);
    robust.push_back({"equal-weight universe", "benchmark", u.returns, u.turnover, 0, ""});
    std::printf("\nRobustness of the selector, %s .. %s (%zu days, same costs)\n", d.dates[rStart].c_str(), d.dates[T - 1].c_str(), u.returns.size());
    std::printf("  %-38s %8s %8s %7s %8s %9s %8s  %s\n", "variant", "CAGR", "vol", "Sharpe", "max DD", "turnover/yr", "switches", "holding now");
    for (const auto& r : robust) {
      const auto m = evaluatePerformance(r.r, r.turnover);
      std::printf("  %-38s %7.1f%% %7.1f%% %7.2f %7.1f%% %8.0f%%", r.name.c_str(), 100 * m.annualReturn, 100 * m.annualVolatility, m.sharpe,
                  100 * m.maxDrawdown, 100 * m.averageTurnover * 252);
      if (r.kind == "selector") std::printf(" %8zu  %s", r.switches, r.holding.c_str());
      std::printf("\n");
    }
  }

  // Today's portfolio: the construction the selector would pick at the last close.
  const std::size_t last = T - 1;
  std::size_t pick = 0;
  {
    double bestScore = -1e18;
    for (std::size_t c = 0; c < books.size(); ++c) {
      const auto& r = books[c].returns;
      std::vector<double> past(r.end() - static_cast<std::ptrdiff_t>(o.lookback), r.end());
      const double s = stdev(past), score = s > 0 ? mean(past) / s : 0.0;
      if (score > bestScore) bestScore = score, pick = c;
    }
  }
  struct Trade {
    std::string ticker;
    double weight, price, shares, value, cost;
  };
  struct Orders {
    std::vector<Trade> trades;
    double spent = 0, fees = 0;
  };
  // Whole-share orders for the budget at the last close, largest weight first.
  auto ordersFor = [&](std::size_t c) {
    const auto w = constructions[c].second(last, eligibleAt(last));
    std::vector<std::size_t> held;
    for (std::size_t i = 0; i < N; ++i)
      if (w[i] > 0) held.push_back(i);
    std::sort(held.begin(), held.end(), [&](std::size_t a, std::size_t b) { return w[a] > w[b]; });
    Orders out;
    for (std::size_t i : held) {
      const double price = d.close(last, i) / o.divisor;
      const double unitCost = price * (1 + (o.costBps + o.stampBps) * 1e-4);
      const double shares = std::floor(w[i] * o.budget / unitCost);
      const double value = shares * price, cost = value * (o.costBps + o.stampBps) * 1e-4;
      out.trades.push_back({d.tickers[i], w[i], price, shares, value, cost});
      out.spent += value + cost;
      out.fees += cost;
    }
    return out;
  };
  auto printOrders = [&](const std::string& title, const Orders& x) {
    std::printf("\n%s\n  %-10s %7s %10s %8s %12s %10s %10s\n", title.c_str(), "ticker", "weight", "price", "shares", "consideration", "costs", "total");
    for (const auto& t : x.trades)
      std::printf("  %-10s %6.1f%% %10.2f %8.0f %12.2f %10.2f %10.2f\n", t.ticker.c_str(), 100 * t.weight, t.price, t.shares, t.value, t.cost, t.value + t.cost);
    std::printf("  invested %.2f, costs %.2f, cash left %.2f of %.2f\n", x.spent - x.fees, x.fees, o.budget - x.spent, o.budget);
  };
  std::vector<Orders> orders;
  for (std::size_t c = 0; c < constructions.size(); ++c) orders.push_back(ordersFor(c));
  // Composite strategy orders: its long positions as whole-share buys (short positions of the
  // selector's long-short rules are listed but not bought).
  auto ordersFromWeights = [&](const std::vector<double>& w, std::size_t at) {
    std::vector<std::size_t> held;
    for (std::size_t i = 0; i < N; ++i)
      if (w[i] > 1e-9) held.push_back(i);
    std::sort(held.begin(), held.end(), [&](std::size_t a, std::size_t b) { return w[a] > w[b]; });
    Orders out;
    for (std::size_t i : held) {
      const double price = d.close(at, i) / o.divisor;
      const double unitCost = price * (1 + (o.costBps + o.stampBps) * 1e-4);
      const double shares = std::floor(w[i] * o.budget / unitCost);
      const double value = shares * price, cost = value * (o.costBps + o.stampBps) * 1e-4;
      out.trades.push_back({d.tickers[i], w[i], price, shares, value, cost});
      out.spent += value + cost;
      out.fees += cost;
    }
    return out;
  };
  const std::size_t compLast = comp.start + compDays - 1;  // the last date the strategy traded from
  const Orders compOrders = ordersFromWeights(comp.target, compLast);
  std::vector<std::pair<std::string, double>> compShorts;
  for (std::size_t i = 0; i < N; ++i)
    if (comp.target[i] < -1e-9) compShorts.push_back({d.tickers[i], comp.target[i]});

  printOrders("Portfolio for " + d.dates[last] + " close, chosen construction: " + constructions[pick].first, orders[pick]);
  for (std::size_t c = 0; c < constructions.size(); ++c)
    if (c != pick) printOrders("Orders if holding " + constructions[c].first + " instead", orders[c]);
  printOrders("Composite strategy positions at the " + d.dates[compLast] + " close (long side)", compOrders);
  for (const auto& [t, w] : compShorts) std::printf("  short %-10s %6.1f%% (not bought)\n", t.c_str(), 100 * w);
  const auto& trades = orders[pick].trades;
  const double spent = orders[pick].spent;

  if (!o.report.empty()) {
    std::ofstream f(o.report);
    f << "{\"from\":" << json(d.dates[aStart]) << ",\"to\":" << json(d.dates[last]) << ",\"stocks\":" << N << ",\"costBps\":" << o.costBps
      << ",\"stampBps\":" << o.stampBps << ",\"rebalance\":" << o.rebalance << ",\"lookback\":" << o.lookback << ",\"budget\":" << o.budget
      << ",\"chosen\":" << json(constructions[pick].first) << ",\"models\":[";
    for (std::size_t k = 0; k < p.models.size(); ++k) f << (k ? "," : "") << "{\"name\":" << json(p.models[k].name) << ",\"auc\":" << p.models[k].oos.auc << ",\"accuracy\":" << p.models[k].oos.accuracy << "}";
    f << "],\"dates\":[";
    for (std::size_t t = aStart; t < T; ++t) f << (t > aStart ? "," : "") << json(d.dates[t]);
    f << "],\"portfolios\":[";
    for (std::size_t k = 0; k < rows.size(); ++k) {
      const auto m = evaluatePerformance(rows[k].r, rows[k].turnover);
      f << (k ? "," : "") << "{\"name\":" << json(rows[k].name) << ",\"cagr\":" << m.annualReturn << ",\"vol\":" << m.annualVolatility << ",\"sharpe\":" << m.sharpe
        << ",\"maxDrawdown\":" << m.maxDrawdown << ",\"turnoverPerYear\":" << m.averageTurnover * 252 << ",\"equity\":[";
      const auto eq = equityCurve(rows[k].r);
      for (std::size_t j = 0; j < eq.size(); ++j) f << (j ? "," : "") << eq[j];
      f << "]}";
    }
    f << "],\"selections\":[";
    for (std::size_t k = 0; k < chosen.size(); ++k) f << (k ? "," : "") << json(constructions[static_cast<std::size_t>(chosen[k])].first);
    f << "],\"trades\":[";
    for (std::size_t k = 0; k < trades.size(); ++k)
      f << (k ? "," : "") << "{\"ticker\":" << json(trades[k].ticker) << ",\"weight\":" << trades[k].weight << ",\"price\":" << trades[k].price
        << ",\"shares\":" << trades[k].shares << ",\"value\":" << trades[k].value << ",\"cost\":" << trades[k].cost << "}";
    f << "],\"cash\":" << o.budget - spent << ",\"orders\":{";
    for (std::size_t c = 0; c < constructions.size(); ++c) {
      f << (c ? "," : "") << json(constructions[c].first) << ":{\"cash\":" << o.budget - orders[c].spent << ",\"trades\":[";
      for (std::size_t k = 0; k < orders[c].trades.size(); ++k) {
        const auto& t = orders[c].trades[k];
        f << (k ? "," : "") << "{\"ticker\":" << json(t.ticker) << ",\"weight\":" << t.weight << ",\"price\":" << t.price << ",\"shares\":" << t.shares
          << ",\"value\":" << t.value << ",\"cost\":" << t.cost << "}";
      }
      f << "]}";
    }
    f << "}";
    f << ",\"composite\":{\"from\":" << json(d.dates[comp.start]) << ",\"to\":" << json(d.dates[comp.start + compDays])
      << ",\"asOf\":" << json(d.dates[compLast]) << ",\"costBps\":" << compExp.costBps << ",\"sleeves\":" << comp.sleeves << ",\"dates\":[";
    for (std::size_t t = comp.start; t <= comp.start + compDays; ++t) f << (t > comp.start ? "," : "") << json(d.dates[t]);
    f << "],\"series\":[";
    for (std::size_t k = 0; k < comp.names.size(); ++k) {
      const auto& m = comp.metrics[k];
      f << (k ? "," : "") << "{\"name\":" << json(comp.names[k]) << ",\"cagr\":" << m.annualReturn << ",\"vol\":" << m.annualVolatility
        << ",\"sharpe\":" << m.sharpe << ",\"maxDrawdown\":" << m.maxDrawdown << ",\"equity\":[";
      const auto eq = equityCurve(comp.returns[k]);
      for (std::size_t j = 0; j < eq.size(); ++j) f << (j ? "," : "") << eq[j];
      f << "]";
      if (k < comp.sleeves) {
        f << ",\"weights\":[";
        for (std::size_t j = 0; j < comp.weights[k].size(); ++j) f << (j ? "," : "") << comp.weights[k][j];
        f << "]";
      }
      f << "}";
    }
    f << "],\"cashWeights\":[";
    for (std::size_t j = 0; j < comp.cash.size(); ++j) f << (j ? "," : "") << comp.cash[j];
    f << "],\"trades\":[";
    for (std::size_t k = 0; k < compOrders.trades.size(); ++k) {
      const auto& t = compOrders.trades[k];
      f << (k ? "," : "") << "{\"ticker\":" << json(t.ticker) << ",\"weight\":" << t.weight << ",\"price\":" << t.price << ",\"shares\":" << t.shares
        << ",\"value\":" << t.value << ",\"cost\":" << t.cost << "}";
    }
    f << "],\"shorts\":[";
    for (std::size_t k = 0; k < compShorts.size(); ++k)
      f << (k ? "," : "") << "{\"ticker\":" << json(compShorts[k].first) << ",\"weight\":" << compShorts[k].second << "}";
    f << "],\"cash\":" << o.budget - compOrders.spent << "}";
    if (!robust.empty()) {
      f << ",\"robustness\":{\"from\":" << json(d.dates[rStart]) << ",\"rows\":[";
      for (std::size_t k = 0; k < robust.size(); ++k) {
        const auto m = evaluatePerformance(robust[k].r, robust[k].turnover);
        f << (k ? "," : "") << "{\"name\":" << json(robust[k].name) << ",\"kind\":" << json(robust[k].kind) << ",\"cagr\":" << m.annualReturn
          << ",\"vol\":" << m.annualVolatility << ",\"sharpe\":" << m.sharpe << ",\"maxDrawdown\":" << m.maxDrawdown
          << ",\"turnoverPerYear\":" << m.averageTurnover * 252 << ",\"switches\":" << robust[k].switches
          << ",\"holding\":" << json(robust[k].holding) << "}";
      }
      f << "]}";
    }
    f << "}\n";
    std::printf("\nreport written to %s\n", o.report.c_str());
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
