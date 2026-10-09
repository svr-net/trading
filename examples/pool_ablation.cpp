// Which models and trading rules earn their place in the self-adaptive strategy's pool?
//
//   pool_ablation [markets per set=6] [days=2520] [--csv data.csv]
//
// Every model family (8) is trained once per market, walk-forward. The candidate pool is
// every model x rule. Starting from the full pool, backward elimination removes, one at a
// time, the model or rule whose absence costs the self-adaptive strategy the least Sharpe
// ratio on average over a set of synthetic markets (selection), as long as the loss stays
// within a tolerance: an item that adds nothing goes, which also shortens training and lowers
// the number of trials the deflated Sharpe ratio has to pay for. The pruned pool is then
// compared with the full one on unseen markets (validation) and, with --csv, on real data.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sat/sat.hpp"

using namespace sat;

namespace {

constexpr double kTolerance = 0.02;  // Sharpe a removal may cost and still count as "adds nothing"
constexpr double kCostBps = 10.0;

std::vector<ModelSpec> allModels() {
  auto m = ExperimentSpec::defaultModels();  // logistic, forest, xgboost, lightgbm, mlp, lstm
  ModelSpec svm;
  svm.type = "svm";
  svm.epochs = 3;
  ModelSpec tree;
  tree.type = "tree";
  tree.maxDepth = 5;
  tree.minLeaf = 50;
  m.push_back(svm);
  m.push_back(tree);
  return m;
}

struct Market {
  std::string name;
  PredictionSet p;
  std::vector<double> trainMs;  // per model
  std::size_t first = 0;        // first traded date, the same for every pool
};

Market prepare(const std::string& name, const MarketData& data) {
  ExperimentSpec exp;
  exp.models = allModels();
  Market m{name, runPredictions(data, exp), {}, 0};
  std::size_t start = 0;
  for (const auto& mp : m.p.models) {
    m.trainMs.push_back(mp.elapsedMs);
    start = std::max(start, mp.start);
  }
  m.first = start + SelectorSpec{}.lookback;
  std::fprintf(stderr, "  %s ready\n", name.c_str());
  return m;
}

struct Pool {
  std::vector<bool> model, rule;
  std::size_t models() const { return static_cast<std::size_t>(std::count(model.begin(), model.end(), true)); }
  std::size_t rules() const { return static_cast<std::size_t>(std::count(rule.begin(), rule.end(), true)); }
};

struct Outcome {
  PerformanceMetrics metrics;
  std::vector<double> net;
  double trialVariance = 0;  // variance of the candidates' per-day Sharpe ratios (the trials)
};

// The default self-adaptive strategy on the pool, traded from the market's common first date.
Outcome selector(const Market& m, const Pool& pool, const std::vector<StrategySpec>& rules) {
  std::vector<ModelPredictions> models;
  std::vector<StrategySpec> strategies;
  for (std::size_t k = 0; k < pool.model.size(); ++k)
    if (pool.model[k]) models.push_back(m.p.models[k]);
  for (std::size_t k = 0; k < pool.rule.size(); ++k)
    if (pool.rule[k]) strategies.push_back(rules[k]);
  const CandidateBook book(models, strategies, m.p.nextReturns, kCostBps);
  const std::size_t from = m.first - book.start();
  const auto a = runSelector(book, SelectorSpec{}, from);
  std::vector<double> sharpes;
  for (std::size_t c = 0; c < book.size(); ++c) sharpes.push_back(afml::periodSharpe(book.netSeries(c, from)));
  const double sd = sharpes.size() > 1 ? stdev(sharpes) : 0.0;
  return {a.metrics, a.net, sd * sd};
}

double meanSharpe(const std::vector<Market>& set, const Pool& pool, const std::vector<StrategySpec>& rules) {
  double s = 0;
  for (const auto& m : set) s += selector(m, pool, rules).metrics.sharpe / static_cast<double>(set.size());
  return s;
}

std::string describe(const Pool& pool, const std::vector<std::string>& modelNames, const std::vector<StrategySpec>& rules) {
  std::string out = "models:";
  for (std::size_t k = 0; k < pool.model.size(); ++k)
    if (pool.model[k]) out += " " + modelNames[k] + ",";
  out.back() = ';';
  out += " rules:";
  for (std::size_t k = 0; k < pool.rule.size(); ++k)
    if (pool.rule[k]) out += " " + rules[k].label() + ",";
  out.pop_back();
  return out;
}

void validate(const char* title, const std::vector<Market>& set, const Pool& full, const Pool& pruned,
              const std::vector<StrategySpec>& rules) {
  std::printf("\n%s\n  %-14s %8s %8s %8s %8s %10s %12s\n", title, "pool", "ann.ret", "Sharpe", "worst", "max DD", "DSR", "train ms");
  for (const auto& [name, pool] : {std::pair<const char*, const Pool*>{"full", &full}, {"pruned", &pruned}}) {
    double ret = 0, sharpe = 0, worst = 1e9, dd = 0, dsr = 0, train = 0;
    const double n = static_cast<double>(set.size());
    for (const auto& m : set) {
      const auto o = selector(m, *pool, rules);
      ret += o.metrics.annualReturn / n;
      sharpe += o.metrics.sharpe / n;
      worst = std::min(worst, o.metrics.sharpe);
      dd += o.metrics.maxDrawdown / n;
      // Deflated Sharpe ratio: the candidates in the pool are the trials.
      const double trials = static_cast<double>(pool->models() * pool->rules());
      const double sr = afml::periodSharpe(o.net);
      dsr += afml::deflatedSharpe(sr, static_cast<double>(o.net.size()), skewness(o.net), kurtosis(o.net), trials, std::max(1e-12, o.trialVariance)) / n;
      for (std::size_t k = 0; k < pool->model.size(); ++k)
        if (pool->model[k]) train += m.trainMs[k] / n;
    }
    std::printf("  %-14s %7.1f%% %8.2f %8.2f %7.1f%% %9.0f%% %12.0f\n", name, 100 * ret, sharpe, worst, 100 * dd, 100 * dsr, train);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // progress lines as they come
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

  const auto rules = ExperimentSpec::defaultStrategies();
  std::vector<std::string> modelNames;
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
  const auto selection = build(101);
  for (const auto& mp : selection[0].p.models) modelNames.push_back(mp.name);

  // Each model on its own merits: out-of-sample AUC and training time.
  std::printf("Models (selection markets, average)\n  %-22s %8s %10s\n", "model", "AUC", "train ms");
  for (std::size_t k = 0; k < modelNames.size(); ++k) {
    double auc = 0, ms = 0;
    for (const auto& m : selection) {
      auc += m.p.models[k].oos.auc / static_cast<double>(selection.size());
      ms += m.trainMs[k] / static_cast<double>(selection.size());
    }
    std::printf("  %-22s %8.4f %10.0f\n", modelNames[k].c_str(), auc, ms);
  }

  Pool full{std::vector<bool>(modelNames.size(), true), std::vector<bool>(rules.size(), true)};
  const double fullSharpe = meanSharpe(selection, full, rules);
  std::printf("\nLeave one out from the full pool (selection markets, mean selector Sharpe %.3f)\n", fullSharpe);
  for (std::size_t k = 0; k < modelNames.size() + rules.size(); ++k) {
    Pool p = full;
    const bool isModel = k < modelNames.size();
    (isModel ? p.model[k] : p.rule[k - modelNames.size()]) = false;
    const double s = meanSharpe(selection, p, rules);
    std::printf("  without %-24s %+.3f\n", isModel ? modelNames[k].c_str() : rules[k - modelNames.size()].label().c_str(), s - fullSharpe);
  }

  // Backward elimination.
  Pool pool = full;
  double current = fullSharpe;
  std::vector<std::pair<std::string, Pool>> path = {{"full pool", full}};
  std::printf("\nBackward elimination (remove while the loss is within %.2f Sharpe)\n", kTolerance);
  for (;;) {
    double bestScore = -1e9;
    std::size_t best = SIZE_MAX;
    for (std::size_t k = 0; k < modelNames.size() + rules.size(); ++k) {
      const bool isModel = k < modelNames.size();
      if (isModel ? !pool.model[k] : !pool.rule[k - modelNames.size()]) continue;
      if (isModel ? pool.models() == 1 : pool.rules() == 1) continue;
      Pool p = pool;
      (isModel ? p.model[k] : p.rule[k - modelNames.size()]) = false;
      const double s = meanSharpe(selection, p, rules);
      if (s > bestScore) bestScore = s, best = k;
    }
    if (best == SIZE_MAX || bestScore < current - kTolerance) {
      if (best != SIZE_MAX) {
        const bool isModel = best < modelNames.size();
        std::printf("  stop: removing %s would cost %.3f\n", isModel ? modelNames[best].c_str() : rules[best - modelNames.size()].label().c_str(), current - bestScore);
      }
      break;
    }
    const bool isModel = best < modelNames.size();
    (isModel ? pool.model[best] : pool.rule[best - modelNames.size()]) = false;
    std::printf("  drop %-24s mean Sharpe %.3f (%+.3f)\n", isModel ? modelNames[best].c_str() : rules[best - modelNames.size()].label().c_str(), bestScore, bestScore - current);
    path.push_back({"- " + (isModel ? modelNames[best] : rules[best - modelNames.size()].label()), pool});
    current = bestScore;
  }
  std::printf("\nPruned pool: %s (%zu candidates instead of %zu)\n", describe(pool, modelNames, rules).c_str(), pool.models() * pool.rules(),
              full.models() * full.rules());

  validate("Selection markets", selection, full, pool, rules);
  std::fprintf(stderr, "validation markets\n");
  const auto validation = build(201);
  validate("Validation markets (unseen)", validation, full, pool, rules);
  // Every step of the elimination on the unseen markets: how small can the pool get before
  // the validation result stops improving?
  std::printf("\nElimination path on the validation markets\n  %-28s %6s %8s %8s %8s\n", "after", "cands", "sel SR", "val SR", "val DD");
  for (const auto& [label, p] : path) {
    double v = 0, dd = 0;
    for (const auto& m : validation) {
      const auto o = selector(m, p, rules);
      v += o.metrics.sharpe / static_cast<double>(validation.size());
      dd += o.metrics.maxDrawdown / static_cast<double>(validation.size());
    }
    std::printf("  %-28s %6zu %8.3f %8.3f %7.1f%%\n", label.c_str(), p.models() * p.rules(), meanSharpe(selection, p, rules), v, 100 * dd);
  }
  if (!csv.empty()) {
    std::ifstream f(csv);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::vector<Market> real = {prepare(csv, parseCsv(ss.str()))};
    validate(("Real data: " + csv).c_str(), real, full, pool, rules);
  }
  return 0;
}
