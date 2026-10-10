// Why do the forecasts point the wrong way on UK data, and what fixes it?
//
//   sign_study [markets=6] [days=2520] [--csv data.csv]
//   sign_study 0 --csv data.csv     (real data only)
//
// 1. Diagnosis: each forecast's mean daily cross-sectional rank correlation (IC) with the next
//    1, 5 and 21 days' returns (t-statistics from non-overlapping dates), next to two simple
//    untrained signals: 5-day reversal and 21-day momentum. Real data: per half of the period.
// 2. Fixes, fixed in advance:
//    - labels: the default labels a stock 1 when its own next-day return is positive
//      (Direction), while the rules rank stocks against each other. Variants: relative labels
//      (ExcessDirection: beats the cross-sectional median) over 1 and 5 days; Direction over 5;
//    - sign by selection: the selector's pool holds the forecast and its inverse, so the
//      selector picks the sign from the recent record;
//    - reversal: the 5-day reversal signal added to the selector's pool.
//    Rule: the variant with the best UK Sharpe ratio at UK costs (10 bp plus 50 bp stamp duty
//    on purchases) in the first half of the UK period is chosen; it is adopted only if it also
//    beats the default in the second half at UK costs, and on the unseen synthetic markets at
//    10 bp its mean Sharpe ratio is no more than 0.10 below the default's.
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
const Costs kTen{"10 bp", 10, 0}, kUk{"UK costs", 10, 50};

struct LabelVariant {
  const char* name;
  LabelKind kind;
  std::size_t horizon;
};
const LabelVariant kLabels[] = {{"default: own direction, 1 day", LabelKind::Direction, 1},
                                {"relative to the median, 1 day", LabelKind::ExcessDirection, 1},
                                {"relative to the median, 5 days", LabelKind::ExcessDirection, 5},
                                {"own direction, 5 days", LabelKind::Direction, 5}};
const char* kPoolNames[] = {"sign chosen by the selector (forecast + inverse)", "default forecast + 5-day reversal"};
constexpr std::size_t kVariants = std::size(kLabels) + std::size(kPoolNames);

ExperimentSpec spec(const LabelVariant& l) {
  ExperimentSpec e;
  e.label.kind = l.kind;
  e.label.horizon = l.horizon;
  e.stateSpecialists = true;
  e.composite.method = CompositeMethod::Adaptive;
  e.composite.window = ScoringWindow::MarketAdwin;
  e.composite.decision = ScoringDecision::Evidence;
  return e;
}

// A signal as a pseudo-forecast: its cross-sectional rank scaled to (0, 1), over [start, end).
ModelPredictions signalForecast(const std::string& name, const Panel& signal, std::size_t start, std::size_t end) {
  ModelPredictions f;
  f.name = name;
  f.probability = Panel(signal.dates(), signal.assets());
  f.start = start, f.end = end;
  for (std::size_t t = start; t < end; ++t) {
    std::vector<double> v;
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < signal.assets(); ++i)
      if (std::isfinite(signal(t, i))) v.push_back(signal(t, i)), idx.push_back(i);
    if (v.size() < 2) continue;
    const auto rk = averageRanks(v);
    for (std::size_t j = 0; j < v.size(); ++j) f.probability(t, idx[j]) = rk[j] / static_cast<double>(v.size() + 1);
  }
  return f;
}

// Trailing return over k days up to each date's close (negated for reversal).
Panel trailing(const MarketData& d, std::size_t k, double sign) {
  Panel p(d.numDates(), d.numAssets());
  for (std::size_t t = k; t < d.numDates(); ++t)
    for (std::size_t i = 0; i < d.numAssets(); ++i) {
      const double a = d.close(t - k, i), b = d.close(t, i);
      if (std::isfinite(a) && std::isfinite(b) && a > 0) p(t, i) = sign * (b / a - 1.0);
    }
  return p;
}

struct Ic {
  double mean = 0, t = 0;
};

// Mean daily rank IC of a forecast with forward returns over [from, to), t-statistic from every
// h-th date (non-overlapping).
Ic rankIc(const Panel& f, const Panel& fwd, std::size_t from, std::size_t to, std::size_t h) {
  std::vector<double> all, sparse;
  for (std::size_t t = from; t < to; ++t) {
    std::vector<double> x, y;
    for (std::size_t i = 0; i < f.assets(); ++i)
      if (std::isfinite(f(t, i)) && std::isfinite(fwd(t, i))) x.push_back(f(t, i)), y.push_back(fwd(t, i));
    if (x.size() < 5) continue;
    const double r = correlation(averageRanks(x), averageRanks(y));
    if (!std::isfinite(r)) continue;
    all.push_back(r);
    if ((t - from) % h == 0) sparse.push_back(r);
  }
  Ic out;
  if (all.empty()) return out;
  for (double r : all) out.mean += r / static_cast<double>(all.size());
  if (sparse.size() > 2) {
    double m = 0;
    for (double r : sparse) m += r / static_cast<double>(sparse.size());
    out.t = m / (stdev(sparse) / std::sqrt(static_cast<double>(sparse.size())) + 1e-12);
  }
  return out;
}

struct Market {
  std::string name;
  std::size_t evalFrom = 0, mid = 0, end = 0;  // dates
  // [variant][cost]: selector net returns and turnover over the evaluation dates
  std::vector<std::vector<std::vector<double>>> net, turnover;
  std::vector<std::string> forecastNames;
  // diagnosis per [half][forecast][horizon]
  std::vector<std::vector<std::vector<Ic>>> ic;
};

const std::size_t kHorizons[] = {1, 5, 21};

Market prepare(const std::string& name, const MarketData& data, bool halves) {
  Market m;
  m.name = name;
  m.net.assign(kVariants, {}), m.turnover.assign(kVariants, {});
  const Panel rev5 = trailing(data, 5, -1.0), mom21 = trailing(data, 21, 1.0);
  std::vector<Panel> fwd;
  for (std::size_t h : kHorizons) fwd.push_back(data.forwardReturns(h));
  auto run = [&](std::size_t v, const std::vector<ModelPredictions>& pool, const Panel& next) {
    for (const auto& c : {kTen, kUk}) {
      const CandidateBook book(pool, ExperimentSpec::defaultStrategies(), next, c.cost, c.stamp);
      const auto a = runSelector(book, SelectorSpec{}, m.evalFrom - book.start());
      m.net[v].push_back(a.net), m.turnover[v].push_back(a.turnover);
    }
  };
  for (std::size_t l = 0; l < std::size(kLabels); ++l) {
    const PredictionSet p = runPredictions(data, spec(kLabels[l]));
    const auto [start, end] = commonRange(p.models);
    if (l == 0) {
      // Every variant is evaluated over the default's dates (longer labels end earlier; the
      // last dates are dropped for all).
      m.evalFrom = start + SelectorSpec{}.lookback;
      m.end = std::min(end, p.nextReturns.dates()) - 4;
      m.mid = halves ? (m.evalFrom + m.end) / 2 : m.end;
    }
    std::vector<ModelPredictions> pool = {p.models.back()};
    for (auto& f : pool) f.end = std::min(f.end, m.end);
    run(l, pool, p.nextReturns);
    if (l != 0) continue;
    // Pools on the default forecast.
    ModelPredictions inv = pool[0];
    inv.name = "inverse";
    for (double& x : inv.probability.data())
      if (std::isfinite(x)) x = 1.0 - x;
    run(std::size(kLabels), {pool[0], inv}, p.nextReturns);
    run(std::size(kLabels) + 1, {pool[0], signalForecast("5-day reversal", rev5, pool[0].start, pool[0].end)}, p.nextReturns);
    // Diagnosis.
    std::vector<const Panel*> fs;
    for (const auto& f : p.models) m.forecastNames.push_back(f.name), fs.push_back(&f.probability);
    m.forecastNames.push_back("5-day reversal (untrained)"), fs.push_back(&rev5);
    m.forecastNames.push_back("21-day momentum (untrained)"), fs.push_back(&mom21);
    const std::size_t bounds[3] = {m.evalFrom, m.mid, m.end};
    for (int half = 0; half < (halves ? 2 : 1); ++half) {
      std::vector<std::vector<Ic>> byF;
      for (const Panel* f : fs) {
        std::vector<Ic> byH;
        for (std::size_t h = 0; h < std::size(kHorizons); ++h)
          byH.push_back(rankIc(*f, fwd[h], bounds[half], halves ? bounds[half + 1] : m.end, kHorizons[h]));
        byF.push_back(byH);
      }
      m.ic.push_back(byF);
    }
  }
  std::fprintf(stderr, "  %s ready\n", name.c_str());
  return m;
}

PerformanceMetrics slice(const Market& m, std::size_t v, std::size_t c, std::size_t from, std::size_t to) {
  const auto& n = m.net[v][c];
  const auto& t = m.turnover[v][c];
  const std::size_t a = from - m.evalFrom, b = std::min(to - m.evalFrom, n.size());
  return evaluatePerformance(std::vector<double>(n.begin() + a, n.begin() + b), std::vector<double>(t.begin() + a, t.begin() + b));
}

const char* variantName(std::size_t v) { return v < std::size(kLabels) ? kLabels[v].name : kPoolNames[v - std::size(kLabels)]; }

void diagnosis(const char* title, const std::vector<Market>& set) {
  const std::size_t halves = set[0].ic.size();
  for (std::size_t half = 0; half < halves; ++half) {
    std::printf("\n%s: rank IC with the next 1 / 5 / 21 days' returns%s (t-statistic)\n", title,
                halves == 1 ? " (mean over markets)" : half == 0 ? ", first half" : ", second half");
    for (std::size_t f = 0; f < set[0].forecastNames.size(); ++f) {
      std::printf("  %-34s", set[0].forecastNames[f].c_str());
      for (std::size_t h = 0; h < std::size(kHorizons); ++h) {
        double mean = 0, t = 0;
        for (const auto& m : set) mean += m.ic[half][f][h].mean / static_cast<double>(set.size()), t += m.ic[half][f][h].t / static_cast<double>(set.size());
        std::printf("  %+.4f (%+5.1f)", mean, t);
      }
      std::printf("\n");
    }
  }
}

// Mean Sharpe per variant over [from, to) of each market; prints a table.
std::vector<double> results(const char* title, const std::vector<Market>& set, std::size_t c, bool firstHalf, bool secondHalf) {
  std::vector<double> mean(kVariants, 0.0);
  const double n = static_cast<double>(set.size());
  std::printf("\n%s, %s\n  %-50s %8s %8s %8s %8s %6s\n", title, c == 0 ? kTen.name : kUk.name, "variant", "ann.ret", "Sharpe", "max DD", "turn/yr",
              "beats");
  for (std::size_t v = 0; v < kVariants; ++v) {
    double ret = 0, dd = 0, turn = 0;
    int beats = 0;
    for (const auto& m : set) {
      const std::size_t from = secondHalf ? m.mid : m.evalFrom, to = firstHalf ? m.mid : m.end;
      const auto pm = slice(m, v, c, from, to), base = slice(m, 0, c, from, to);
      ret += pm.annualReturn / n, mean[v] += pm.sharpe / n, dd += pm.maxDrawdown / n, turn += 252 * pm.averageTurnover / n;
      beats += v > 0 && pm.sharpe > base.sharpe + 1e-9 ? 1 : 0;
    }
    std::printf("  %-50s %7.1f%% %8.2f %7.1f%% %7.0fx", variantName(v), 100 * ret, mean[v], 100 * dd, turn);
    if (v > 0) std::printf(" %3d/%-2zu", beats, set.size());
    std::printf("\n");
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
    std::fprintf(stderr, "sign_study 0 needs --csv\n");
    return 2;
  }
  std::vector<double> synth;
  if (markets > 0) {
    std::vector<Market> set;
    for (std::size_t k = 0; k < markets; ++k) {
      SyntheticMarketSpec ms;
      ms.seed = 201 + k;  // the unseen markets
      ms.numDates = days;
      set.push_back(prepare("seed " + std::to_string(ms.seed), generateSyntheticMarket(ms), false));
    }
    diagnosis("Unseen synthetic markets", set);
    synth = results("Unseen synthetic markets", set, 0, false, false);
    results("Unseen synthetic markets", set, 1, false, false);
  }
  if (!csv.empty()) {
    std::ifstream f(csv);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::vector<Market> uk = {prepare(csv, parseCsv(ss.str()), true)};
    diagnosis(("Real data: " + csv).c_str(), uk);
    const auto first = results(("Real data, first half: " + csv).c_str(), uk, 1, true, false);
    const auto second = results(("Real data, second half: " + csv).c_str(), uk, 1, false, true);
    results(("Real data, first half: " + csv).c_str(), uk, 0, true, false);
    results(("Real data, second half: " + csv).c_str(), uk, 0, false, true);
    std::size_t chosen = 0;
    for (std::size_t v = 1; v < kVariants; ++v)
      if (first[v] > first[chosen]) chosen = v;
    std::printf("\nChosen on the first half at UK costs: %s\n", variantName(chosen));
    if (chosen == 0) {
      std::printf("No variant beats the default in the first half: keep the default.\n");
    } else {
      const bool confirm = second[chosen] > second[0];
      std::printf("Second half at UK costs: %.2f against the default's %.2f: %s\n", second[chosen], second[0], confirm ? "confirmed" : "NOT confirmed");
      if (!synth.empty()) {
        const bool harmless = synth[chosen] >= synth[0] - 0.10;
        std::printf("Unseen synthetic markets at 10 bp: %.2f against the default's %.2f: %s\n", synth[chosen], synth[0],
                    harmless ? "no harm" : "HARMS");
        std::printf("Verdict: %s\n", confirm && harmless ? "adopt" : "keep the default");
      } else {
        std::printf("(the synthetic do-no-harm check is not run here: sign_study [markets] --csv)\n");
      }
    }
  }
  return 0;
}
