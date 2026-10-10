// Adaptive parametrisation of the trading level: can the selector do without hand-set numbers?
//
//   adaptive_params [markets per set=6] [days=2520] [--csv data.csv]
//   adaptive_params 0 --csv data.csv     (real data only)
//
// The forecast is the default pipeline's self-adaptive forecast. The selector then trades it with:
//  A. the hand-set default: five fixed rules (top 5 daily, top 10 every 5 days, long-short 3
//     daily, P > 0.52 daily, probability-weighted daily), a 63-day look-back, a choice every
//     21 days;
//  B. adaptive memory: the same rules, no look-back or cadence (each candidate's record is an
//     ADWIN window, also cut back at changes of market state; re-scored daily);
//  C. a rule grid instead of five picked rules: top-k and long-short with k at 5%, 10%, 20%
//     (and 33% for top-k) of the universe, and probability-weighted, each held 1, 5 or 21 days
//     (the selector picks the size and holding period), with the fixed 63/21 memory;
//  D. the rule grid and adaptive memory;
//  E. D with the market itself (equal weight, held) as one more candidate.
// Decision (fixed in advance): among B-E, the best mean Sharpe ratio on the selection markets
// (average of 10 bp and UK costs: 10 bp plus 50 bp stamp duty on purchases) is chosen; it
// replaces the default only if, on the unseen markets, its mean Sharpe ratio is at least the
// default's at both cost levels and it beats the default on at least 4 of 6 markets at 10 bp,
// and on UK data, at UK costs, it is at least as good as the default.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "sat/sat.hpp"

using namespace sat;

namespace {

struct Costs {
  const char* name;
  double cost, stamp;
};
const Costs kCosts[] = {{"10 bp", 10, 0}, {"UK costs (10 bp + 50 bp stamp duty on purchases)", 10, 50}};
const char* kNames[] = {"A. hand-set default (5 rules, 63/21)", "B. adaptive memory, 5 rules", "C. rule grid, 63/21",
                        "D. rule grid + adaptive memory", "E. D + the market as a candidate"};
constexpr std::size_t kVariants = std::size(kNames);

std::vector<StrategySpec> ruleGrid(std::size_t N, bool withMarket) {
  std::vector<StrategySpec> g;
  auto k = [&](double f) { return std::max(1.0, std::round(f * static_cast<double>(N))); };
  for (std::size_t hold : {1, 5, 21}) {
    for (double f : {0.05, 0.10, 0.20, 0.33}) g.push_back({StrategyKind::LongTopK, k(f), hold});
    for (double f : {0.05, 0.10, 0.20}) g.push_back({StrategyKind::LongShort, k(f), hold});
    g.push_back({StrategyKind::ProbabilityWeighted, 0.5, hold});
  }
  // Duplicates (small universes round two fractions to the same k) are dropped.
  std::vector<StrategySpec> out;
  for (const auto& s : g)
    if (std::none_of(out.begin(), out.end(), [&](const StrategySpec& o) { return o.kind == s.kind && o.param == s.param && o.holding == s.holding; }))
      out.push_back(s);
  if (withMarket) out.push_back({StrategyKind::LongTopK, static_cast<double>(N), 21});
  return out;
}

struct Market {
  std::string name;
  std::size_t evalFrom = 0, mid = 0, end = 0;  // offsets into the book's evaluation series
  std::vector<std::vector<AdaptiveResult>> runs;  // [cost][variant]
};

Market prepare(const std::string& name, const MarketData& data) {
  ExperimentSpec e;
  e.stateSpecialists = true;
  e.composite.method = CompositeMethod::Adaptive;
  e.composite.window = ScoringWindow::MarketAdwin;
  e.composite.decision = ScoringDecision::Evidence;
  const PredictionSet p = runPredictions(data, e);
  std::size_t first = 0;
  for (const auto& mp : p.models) first = std::max(first, mp.start);
  first += SelectorSpec{}.lookback;
  const std::vector<ModelPredictions> forecast = {p.models.back()};
  const std::size_t N = p.nextReturns.assets();
  Market m{name, 0, 0, 0, {}};
  for (const auto& c : kCosts) {
    std::vector<AdaptiveResult> row;
    SelectorSpec fixed, adaptive;
    adaptive.window = ScoringWindow::MarketAdwin;
    const CandidateBook five(forecast, ExperimentSpec::defaultStrategies(), p.nextReturns, c.cost, c.stamp);
    const CandidateBook grid(forecast, ruleGrid(N, false), p.nextReturns, c.cost, c.stamp);
    const CandidateBook gridMarket(forecast, ruleGrid(N, true), p.nextReturns, c.cost, c.stamp);
    const std::size_t from = first - five.start();
    row.push_back(runSelector(five, fixed, from));
    row.push_back(runSelector(five, adaptive, from));
    row.push_back(runSelector(grid, fixed, from));
    row.push_back(runSelector(grid, adaptive, from));
    row.push_back(runSelector(gridMarket, adaptive, from));
    m.runs.push_back(std::move(row));
  }
  m.end = m.runs[0][0].net.size();
  m.mid = m.end / 2;
  std::fprintf(stderr, "  %s ready (%zu rules in the grid)\n", name.c_str(), ruleGrid(N, false).size());
  return m;
}

PerformanceMetrics part(const AdaptiveResult& a, std::size_t from, std::size_t to) {
  return evaluatePerformance(std::vector<double>(a.net.begin() + static_cast<std::ptrdiff_t>(from), a.net.begin() + static_cast<std::ptrdiff_t>(to)),
                             std::vector<double>(a.turnover.begin() + static_cast<std::ptrdiff_t>(from), a.turnover.begin() + static_cast<std::ptrdiff_t>(to)));
}

// Mean Sharpe per [cost][variant]; prints the tables. half: 0 = all, 1 = first, 2 = second.
std::vector<std::vector<double>> report(const std::string& title, const std::vector<Market>& set, int half = 0, std::vector<std::vector<int>>* beatsOut = nullptr) {
  const double n = static_cast<double>(set.size());
  std::vector<std::vector<double>> mean(std::size(kCosts), std::vector<double>(kVariants, 0.0));
  if (beatsOut) beatsOut->assign(std::size(kCosts), std::vector<int>(kVariants, 0));
  for (std::size_t c = 0; c < std::size(kCosts); ++c) {
    std::printf("\n%s, %s\n  %-40s %8s %8s %8s %8s %8s %6s %7s\n", title.c_str(), kCosts[c].name, "", "ann.ret", "Sharpe", "worst", "max DD", "turn/yr",
                "beats", "switch");
    for (std::size_t v = 0; v < kVariants; ++v) {
      double ret = 0, worst = 1e9, dd = 0, turn = 0, sw = 0;
      int beats = 0;
      for (const auto& m : set) {
        const std::size_t from = half == 2 ? m.mid : 0, to = half == 1 ? m.mid : m.end;
        const auto pm = part(m.runs[c][v], from, to), base = part(m.runs[c][0], from, to);
        ret += pm.annualReturn / n, mean[c][v] += pm.sharpe / n, dd += pm.maxDrawdown / n, turn += 252 * pm.averageTurnover / n;
        worst = std::min(worst, pm.sharpe);
        sw += 252.0 * static_cast<double>(m.runs[c][v].switches) / static_cast<double>(m.end) / n;
        beats += v > 0 && pm.sharpe > base.sharpe + 1e-9 ? 1 : 0;
      }
      if (beatsOut) (*beatsOut)[c][v] = beats;
      std::printf("  %-40s %7.1f%% %8.2f %8.2f %7.1f%% %7.0fx", kNames[v], 100 * ret, mean[c][v], worst, 100 * dd, turn);
      if (v > 0) std::printf(" %3d/%-2zu", beats, set.size());
      else std::printf("       ");
      std::printf(" %6.1f/yr\n", sw);
    }
  }
  return mean;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::size_t markets = 6, days = 2520;
  std::string csv;
  std::vector<std::string> pos;
  for (int k = 1; k < argc; ++k) {
    const std::string a = argv[k];
    if (a == "--csv" && k + 1 < argc) csv = argv[++k];
    else pos.push_back(a);
  }
  if (pos.size() > 0) markets = std::strtoul(pos[0].c_str(), nullptr, 10);
  if (pos.size() > 1) days = std::strtoul(pos[1].c_str(), nullptr, 10);
  if (markets == 0 && csv.empty()) {
    std::fprintf(stderr, "adaptive_params 0 needs --csv\n");
    return 2;
  }
  std::size_t chosen = 0;
  bool adopt = true;
  if (markets > 0) {
    auto build = [&](std::uint64_t base) {
      std::vector<Market> set;
      for (std::size_t k = 0; k < markets; ++k) {
        SyntheticMarketSpec ms;
        ms.seed = base + k;
        ms.numDates = days;
        set.push_back(prepare("seed " + std::to_string(ms.seed), generateSyntheticMarket(ms)));
      }
      return set;
    };
    std::fprintf(stderr, "selection markets\n");
    const auto sel = report("Selection markets", build(101));
    chosen = 1;
    for (std::size_t v = 2; v < kVariants; ++v)
      if (sel[0][v] + sel[1][v] > sel[0][chosen] + sel[1][chosen]) chosen = v;
    std::printf("\nChosen on the selection markets: %s\n", kNames[chosen]);
    std::fprintf(stderr, "validation markets\n");
    std::vector<std::vector<int>> beats;
    const auto val = report("Validation markets (unseen)", build(201), 0, &beats);
    const bool ok = val[0][chosen] >= val[0][0] && val[1][chosen] >= val[1][0] && beats[0][chosen] >= 4;
    std::printf("\nUnseen markets: %.2f vs %.2f at 10 bp (beats %d/%zu), %.2f vs %.2f at UK costs: %s\n", val[0][chosen], val[0][0], beats[0][chosen],
                markets, val[1][chosen], val[1][0], ok ? "confirmed" : "NOT confirmed");
    adopt = ok;
  }
  if (!csv.empty()) {
    std::ifstream f(csv);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::vector<Market> uk = {prepare(csv, parseCsv(ss.str()))};
    const auto all = report("Real data: " + csv, uk);
    report("Real data, first half: " + csv, uk, 1);
    report("Real data, second half: " + csv, uk, 2);
    if (markets > 0) {
      const bool ok = all[1][chosen] >= all[1][0];
      std::printf("\nUK at UK costs: %.2f vs the default's %.2f: %s\n", all[1][chosen], all[1][0], ok ? "confirmed" : "NOT confirmed");
      adopt = adopt && ok;
    }
  }
  if (markets > 0) std::printf("\nVerdict: %s\n", adopt ? (std::string("adopt ") + kNames[chosen]).c_str() : "keep the hand-set default");
  return 0;
}
