// Embind wrapper exposing the library to JavaScript.
//
// Every entry point takes a plain JS "spec" object (market, factors, labels, models,
// walk-forward schedule, strategies, costs, selector) and returns a plain JS object of
// numbers, strings and typed arrays. Errors are returned as { error: "message" } rather
// than thrown across the boundary.
//
// Training the models is the expensive step, so the walk-forward predictions are cached:
// the analyses of a page (and the GPU jobs followed by their read-back) reuse them as long
// as the market, factor, label, model and walk-forward settings are unchanged.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sat/sat.hpp"

using emscripten::val;
using namespace sat;

namespace {

// ------------------------------------------------------------------ JS conversion helpers

bool has(const val& o, const char* key) {
  if (o.isUndefined() || o.isNull()) return false;
  const val v = o[key];
  return !(v.isUndefined() || v.isNull());
}

double num(const val& o, const char* key, double fallback) { return has(o, key) ? o[key].as<double>() : fallback; }

std::size_t count(const val& o, const char* key, std::size_t fallback) {
  const double v = num(o, key, static_cast<double>(fallback));
  if (!(v >= 0.0) || !std::isfinite(v)) throw std::invalid_argument(std::string(key) + " must be a non-negative number");
  return static_cast<std::size_t>(std::llround(v));
}

bool flag(const val& o, const char* key, bool fallback) { return has(o, key) ? o[key].as<bool>() : fallback; }

std::string str(const val& o, const char* key, const std::string& fallback) {
  return has(o, key) ? o[key].as<std::string>() : fallback;
}

std::vector<double> vec(const val& v) { return emscripten::vecFromJSArray<double>(v); }

val arr(const std::vector<double>& v) {
  return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

template <class T>
val typedArray(const std::vector<T>& v) {
  return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

val strings(const std::vector<std::string>& v) {
  val out = val::array();
  for (const auto& s : v) out.call<void>("push", s);
  return out;
}

val arrOfArr(const std::vector<std::vector<double>>& rows) {
  val out = val::array();
  for (const auto& r : rows) out.call<void>("push", arr(r));
  return out;
}

double nowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class F>
val guarded(F&& body) {
  try {
    return body();
  } catch (const std::exception& e) {
    val o = val::object();
    o.set("error", std::string(e.what()));
    return o;
  } catch (...) {
    val o = val::object();
    o.set("error", std::string("unknown C++ exception"));
    return o;
  }
}

std::string stringify(const val& v) {
  if (v.isUndefined() || v.isNull()) return "null";
  return val::global("JSON").call<std::string>("stringify", v);
}

// ------------------------------------------------------------------ spec parsing

SyntheticMarketSpec parseMarket(const val& m) {
  SyntheticMarketSpec s;
  s.numAssets = count(m, "numAssets", s.numAssets);
  s.numDates = count(m, "numDates", s.numDates);
  s.seed = static_cast<std::uint64_t>(num(m, "seed", static_cast<double>(s.seed)));
  s.persistence = num(m, "persistence", s.persistence);
  s.idiosyncraticVol = num(m, "idiosyncraticVol", s.idiosyncraticVol);
  s.betaDispersion = num(m, "betaDispersion", s.betaDispersion);
  if (s.numAssets < 4 || s.numAssets > 500) throw std::invalid_argument("the synthetic market needs 4 to 500 stocks");
  if (s.numDates < 200 || s.numDates > 10000) throw std::invalid_argument("the synthetic market needs 200 to 10,000 days");
  if (!(s.persistence >= 0.0 && s.persistence <= 1.0)) throw std::invalid_argument("regime persistence must be in [0, 1]");
  if (has(m, "regimes")) {
    s.regimes.clear();
    const val r = m["regimes"];
    for (std::size_t k = 0; k < r["length"].as<std::size_t>(); ++k) {
      MarketRegime g;
      g.name = str(r[k], "name", "regime " + std::to_string(k + 1));
      g.drift = num(r[k], "drift", 0.0);
      g.volatility = num(r[k], "volatility", 0.01);
      g.momentum = num(r[k], "momentum", 0.0);
      g.volumeReversal = num(r[k], "volumeReversal", 0.0);
      if (!(g.volatility > 0.0)) throw std::invalid_argument("regime volatility must be positive");
      s.regimes.push_back(g);
    }
    if (s.regimes.empty()) throw std::invalid_argument("the synthetic market needs at least one regime");
  }
  return s;
}

ModelSpec parseModel(const val& m) {
  ModelSpec s;
  s.type = str(m, "type", s.type);
  s.name = str(m, "name", "");
  s.l2 = num(m, "l2", s.l2);
  s.trees = count(m, "trees", s.trees);
  s.maxDepth = count(m, "maxDepth", s.maxDepth);
  s.maxLeaves = count(m, "maxLeaves", s.maxLeaves);
  s.minLeaf = count(m, "minLeaf", s.minLeaf);
  s.learningRate = num(m, "learningRate", s.learningRate);
  s.lambda = num(m, "lambda", s.lambda);
  s.gamma = num(m, "gamma", s.gamma);
  s.subsample = num(m, "subsample", s.subsample);
  s.colsample = num(m, "colsample", s.colsample);
  s.bins = count(m, "bins", s.bins);
  s.hidden = count(m, "hidden", s.hidden);
  s.epochs = count(m, "epochs", s.epochs);
  s.batch = count(m, "batch", s.batch);
  s.seqLen = count(m, "seqLen", s.seqLen);
  s.maxSamples = count(m, "maxSamples", s.maxSamples);
  s.weightDecay = num(m, "weightDecay", s.weightDecay);
  s.seed = static_cast<std::uint64_t>(num(m, "seed", 1.0));
  if (std::find(modelTypes().begin(), modelTypes().end(), s.type) == modelTypes().end())
    throw std::invalid_argument("unknown model type '" + s.type + "'");
  if (s.trees > 2000 || s.epochs > 500 || s.hidden > 256) throw std::invalid_argument(s.displayName() + ": settings too large for the browser");
  return s;
}

StrategySpec parseStrategy(const val& s) {
  StrategySpec out;
  out.kind = parseStrategyKind(str(s, "kind", "topk"));
  out.param = num(s, "param", out.param);
  out.holding = count(s, "holding", 1);
  if (out.holding < 1) throw std::invalid_argument("holding period must be at least 1 day");
  return out;
}

SelectorSpec parseSelector(const val& s) {
  SelectorSpec out;
  out.lookback = count(s, "lookback", out.lookback);
  out.adaptEvery = count(s, "adaptEvery", out.adaptEvery);
  out.metric = parseScoreMetric(str(s, "metric", "sharpe"));
  out.topM = count(s, "topM", 1);
  out.allowCash = flag(s, "allowCash", true);
  out.minScore = num(s, "minScore", 0.0);
  if (out.lookback < 2 || out.adaptEvery < 1 || out.topM < 1) throw std::invalid_argument("selector: look-back >= 2, step >= 1, top M >= 1");
  return out;
}

struct Robustness {
  std::vector<std::size_t> lookbacks = {21, 42, 63, 126, 252};
  std::vector<std::size_t> steps = {5, 10, 21, 42, 63};
  std::vector<ScoreMetric> metrics = {ScoreMetric::Return, ScoreMetric::Sharpe, ScoreMetric::Sortino};
  std::vector<double> costs = {0, 5, 10, 20, 40};
};

struct Env {
  std::string marketKey;
  ExperimentSpec exp;
  Robustness rob;
  std::string predictionKey;
};

Env parseEnv(const val& spec) {
  Env env;
  const std::string csv = str(spec, "csv", "");
  env.marketKey = csv.empty() ? "synthetic:" + stringify(spec["market"]) : "csv:" + csv;
  ExperimentSpec& e = env.exp;
  if (has(spec, "alphas")) {
    for (double id : vec(spec["alphas"])) e.alphaIds.push_back(static_cast<int>(id));
    if (e.alphaIds.empty()) throw std::invalid_argument("select at least one alpha factor");
  }
  e.normalisation = parseNormalisation(str(spec, "normalisation", "rank"));
  if (has(spec, "label")) {
    const val l = spec["label"];
    e.label.kind = parseLabelKind(str(l, "kind", "direction"));
    e.label.horizon = std::max<std::size_t>(1, count(l, "horizon", 1));
    e.label.window = std::max<std::size_t>(2, count(l, "window", 10));
    e.label.barrierWidth = num(l, "barrierWidth", 1.0);
    e.label.volSpan = num(l, "volSpan", 50.0);
    if (!(e.label.barrierWidth > 0) || !(e.label.volSpan >= 2)) throw std::invalid_argument("triple barrier: width > 0 and volatility span >= 2");
  }
  if (has(spec, "extraFeatures")) {
    const val f = spec["extraFeatures"];
    for (std::size_t k = 0; k < f["length"].as<std::size_t>(); ++k) e.extraFeatures.push_back(f[k].as<std::string>());
  }
  e.ffdOrder = num(spec, "ffdOrder", e.ffdOrder);
  if (!(e.ffdOrder >= 0 && e.ffdOrder <= 1)) throw std::invalid_argument("the order of fractional differentiation must be in [0, 1]");
  e.cusumMultiple = num(spec, "cusumMultiple", 0.0);
  if (!(e.cusumMultiple >= 0)) throw std::invalid_argument("CUSUM multiple must be non-negative");
  if (has(spec, "models")) {
    const val m = spec["models"];
    for (std::size_t k = 0; k < m["length"].as<std::size_t>(); ++k) e.models.push_back(parseModel(m[k]));
    if (e.models.empty()) throw std::invalid_argument("select at least one model");
  } else {
    e.models = ExperimentSpec::defaultModels();
  }
  if (has(spec, "composite")) {
    const val c = spec["composite"];
    e.composite.method = parseCompositeMethod(str(c, "method", "none"));
    e.composite.window = std::max<std::size_t>(5, count(c, "window", e.composite.window));
    e.composite.refitEvery = std::max<std::size_t>(1, count(c, "refitEvery", e.composite.refitEvery));
    e.composite.ridge = num(c, "ridge", e.composite.ridge);
    e.composite.keepMembers = flag(c, "keepMembers", e.composite.keepMembers);
    if (!(e.composite.ridge >= 0)) throw std::invalid_argument("composite model: the ridge penalty must be non-negative");
  }
  if (has(spec, "walkForward")) {
    const val w = spec["walkForward"];
    e.walkForward.trainWindow = count(w, "trainWindow", e.walkForward.trainWindow);
    e.walkForward.retrainEvery = std::max<std::size_t>(1, count(w, "retrainEvery", e.walkForward.retrainEvery));
    e.walkForward.maxTrainRows = count(w, "maxTrainRows", e.walkForward.maxTrainRows);
    e.walkForward.seed = static_cast<std::uint64_t>(num(w, "seed", 11));
    e.walkForward.weighting = parseSampleWeighting(str(w, "weighting", "none"));
    e.walkForward.decayOldest = num(w, "decayOldest", 0.5);
  }
  if (has(spec, "strategies")) {
    const val s = spec["strategies"];
    for (std::size_t k = 0; k < s["length"].as<std::size_t>(); ++k) e.strategies.push_back(parseStrategy(s[k]));
  }
  if (e.strategies.empty()) e.strategies = ExperimentSpec::defaultStrategies();
  e.costBps = num(spec, "costBps", e.costBps);
  if (!(e.costBps >= 0.0)) throw std::invalid_argument("transaction cost must be non-negative");
  if (has(spec, "selector")) e.selector = parseSelector(spec["selector"]);
  if (has(spec, "robustness")) {
    const val r = spec["robustness"];
    auto sizes = [&](const char* key, std::vector<std::size_t>& out) {
      if (!has(r, key)) return;
      out.clear();
      for (double v : vec(r[key])) out.push_back(static_cast<std::size_t>(std::max(1.0, std::round(v))));
    };
    sizes("lookbacks", env.rob.lookbacks);
    sizes("steps", env.rob.steps);
    if (has(r, "metrics")) {
      env.rob.metrics.clear();
      const val m = r["metrics"];
      for (std::size_t k = 0; k < m["length"].as<std::size_t>(); ++k) env.rob.metrics.push_back(parseScoreMetric(m[k].as<std::string>()));
    }
    if (has(r, "costs")) env.rob.costs = vec(r["costs"]);
    for (auto lb : env.rob.lookbacks)
      if (lb < 2) throw std::invalid_argument("look-backs must be at least 2 days");
  }
  env.predictionKey = env.marketKey + "|" + stringify(spec["alphas"]) + "|" + str(spec, "normalisation", "rank") + "|" +
                      stringify(spec["label"]) + "|" + stringify(spec["models"]) + "|" + stringify(spec["walkForward"]) + "|" +
                      stringify(spec["extraFeatures"]) + "|" + std::to_string(e.ffdOrder) + "|" + std::to_string(e.cusumMultiple);
  return env;
}

// ------------------------------------------------------------------ caches

struct MarketCache {
  std::string key;
  std::shared_ptr<const MarketData> data;
};

struct PredictionCache {
  std::string key;
  std::shared_ptr<const PredictionSet> set;
  double elapsedMs = 0.0;
};

MarketCache& marketCache() {
  static MarketCache c;
  return c;
}

PredictionCache& predictionCache() {
  static PredictionCache c;
  return c;
}

std::shared_ptr<const MarketData> market(const val& spec, const Env& env) {
  auto& c = marketCache();
  if (c.data && c.key == env.marketKey) return c.data;
  const std::string csv = str(spec, "csv", "");
  auto data = std::make_shared<MarketData>(csv.empty() ? generateSyntheticMarket(parseMarket(spec["market"])) : parseCsv(csv));
  if (data->numAssets() < 4) throw std::invalid_argument("the universe needs at least 4 stocks");
  c.key = env.marketKey;
  c.data = data;
  return data;
}

struct Predictions {
  std::shared_ptr<const MarketData> data;
  std::shared_ptr<const PredictionSet> set;
  double trainMs = 0.0;
  bool cached = false;
};

Predictions predictions(const val& spec, const Env& env) {
  Predictions p;
  p.data = market(spec, env);
  auto& c = predictionCache();
  if (c.set && c.key == env.predictionKey) {
    p.set = c.set;
    p.trainMs = c.elapsedMs;
    p.cached = true;
  } else {
    // The members are trained without the composite, so changing only the composite model
    // never re-trains them.
    const double t0 = nowMs();
    ExperimentSpec members = env.exp;
    members.composite.method = CompositeMethod::None;
    auto set = std::make_shared<PredictionSet>(runPredictions(*p.data, members));
    c.key = env.predictionKey;
    c.set = set;
    c.elapsedMs = nowMs() - t0;
    p.set = set;
    p.trainMs = c.elapsedMs;
  }
  const CompositeSpec& cs = env.exp.composite;
  if (cs.method == CompositeMethod::None) return p;
  static PredictionCache composite;
  const std::string key = env.predictionKey + "|composite:" + compositeMethodName(cs.method) + "|" + std::to_string(cs.window) + "|" +
                          std::to_string(cs.refitEvery) + "|" + std::to_string(cs.ridge) + "|" + std::to_string(cs.keepMembers);
  if (!(composite.set && composite.key == key)) {
    auto set = std::make_shared<PredictionSet>(*p.set);
    auto cp = compositePredictions(set->models, set->labels, set->labelEnds, cs);
    if (!cs.keepMembers) set->models.clear();
    set->models.push_back(std::move(cp.model));
    set->compositeMembers = std::move(cp.members);
    set->compositeWeights = std::move(cp.weights);
    composite.key = key;
    composite.set = set;
  }
  p.set = composite.set;
  return p;
}

// ------------------------------------------------------------------ result conversion

val metricsToJs(const PerformanceMetrics& m) {
  val o = val::object();
  o.set("days", static_cast<double>(m.days));
  o.set("totalReturn", m.totalReturn);
  o.set("annualReturn", m.annualReturn);
  o.set("annualVolatility", m.annualVolatility);
  o.set("sharpe", m.sharpe);
  o.set("sortino", m.sortino);
  o.set("maxDrawdown", m.maxDrawdown);
  o.set("calmar", m.calmar);
  o.set("winRate", m.winRate);
  o.set("averageTurnover", m.averageTurnover);
  return o;
}

val classificationToJs(const ClassificationMetrics& m) {
  val o = val::object();
  o.set("count", static_cast<double>(m.count));
  o.set("accuracy", m.accuracy);
  o.set("precision", m.precision);
  o.set("recall", m.recall);
  o.set("f1", m.f1);
  o.set("auc", m.auc);
  o.set("logLoss", m.logLoss);
  o.set("baseRate", m.baseRate);
  return o;
}

std::string dateLabel(const MarketData& d, std::size_t t) { return d.dates.empty() ? std::to_string(t) : d.dates[t]; }

val dateRange(const MarketData& d, std::size_t from, std::size_t to) {
  std::vector<std::string> out;
  for (std::size_t t = from; t < to; ++t) out.push_back(dateLabel(d, t));
  return strings(out);
}

std::vector<double> regimeSlice(const MarketData& d, std::size_t from, std::size_t to) {
  std::vector<double> out;
  for (std::size_t t = from; t < to; ++t) out.push_back(d.regime.empty() ? -1.0 : d.regime[t]);
  return out;
}

val regimeNames(const MarketData& d) { return strings(d.regimeNames); }

// ------------------------------------------------------------------ entry points

val version() {
  val o = val::object();
  o.set("library", std::string("trading sat ") + SAT_VERSION);  // from the CMake project VERSION
  o.set("wasm", true);
  val ids = val::array();
  for (int id : paperAlphaIds()) ids.call<void>("push", id);
  o.set("alphas", ids);
  o.set("models", strings(modelTypes()));
  return o;
}

// Core numerics: the operator algebra on real series, the generator, and the look-ahead check.
val coreDemo(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const auto data = market(spec, env);
    const MarketData& d = *data;
    const std::size_t T = d.numDates(), N = d.numAssets(), from = T > 250 ? T - 250 : 0;
    using namespace ops;
    const Panel& close = d.close;
    const Panel mean20 = tsMean(close, 20), sd20 = tsStddev(close, 20), rank10 = tsRank(close, 10), corr10 = tsCorr(close, d.volume, 10);
    auto seriesOf = [&](const Panel& p) {
      std::vector<double> v;
      for (std::size_t t = from; t < T; ++t) v.push_back(p(t, 0));
      return v;
    };
    val out = val::object();
    out.set("ticker", d.tickers[0]);
    out.set("dates", dateRange(d, from, T));
    out.set("close", arr(seriesOf(close)));
    out.set("mean20", arr(seriesOf(mean20)));
    std::vector<double> up, lo;
    for (std::size_t t = from; t < T; ++t) {
      up.push_back(mean20(t, 0) + 2 * sd20(t, 0));
      lo.push_back(mean20(t, 0) - 2 * sd20(t, 0));
    }
    out.set("upper", arr(up));
    out.set("lower", arr(lo));
    out.set("tsRank10", arr(seriesOf(rank10)));
    out.set("corr10", arr(seriesOf(corr10)));
    // Cross-section of the last date.
    std::vector<double> lastClose(close.row(T - 1), close.row(T - 1) + N);
    const Panel r = rank(close), sc = scale(close);
    out.set("tickers", strings(d.tickers));
    out.set("lastClose", arr(lastClose));
    out.set("lastRank", arr(std::vector<double>(r.row(T - 1), r.row(T - 1) + N)));
    out.set("lastScale", arr(std::vector<double>(sc.row(T - 1), sc.row(T - 1) + N)));
    // Generator: moments of standard normals.
    Rng rng(static_cast<std::uint64_t>(num(spec["market"], "seed", 7)));
    const std::size_t n = 20000;
    std::vector<double> z(n);
    double m1 = 0, m2 = 0, m3 = 0, m4 = 0;
    for (auto& v : z) v = rng.normal();
    for (double v : z) m1 += v / n;
    for (double v : z) {
      m2 += (v - m1) * (v - m1) / n;
      m3 += std::pow(v - m1, 3) / n;
      m4 += std::pow(v - m1, 4) / n;
    }
    val g = val::object();
    g.set("samples", arr(z));
    g.set("mean", m1);
    g.set("sd", std::sqrt(m2));
    g.set("skew", m3 / std::pow(m2, 1.5));
    g.set("kurtosis", m4 / (m2 * m2));
    out.set("normals", g);
    // No look-ahead: every alpha on date t is unchanged when the last 50 dates are removed.
    const std::size_t cut = T > 300 ? T - 50 : T / 2;
    const AlphaInputs full(d), part(d.slice(0, cut));
    val look = val::array();
    for (int id : paperAlphaIds()) {
      const Panel a = computeAlpha(id, full), b = computeAlpha(id, part);
      double diff = 0.0;
      std::size_t mismatched = 0;
      for (std::size_t t = 0; t < cut; ++t)
        for (std::size_t i = 0; i < N; ++i) {
          const bool na = std::isnan(a(t, i)), nb = std::isnan(b(t, i));
          if (na != nb) ++mismatched;
          else if (!na) diff = std::max(diff, std::fabs(a(t, i) - b(t, i)));
        }
      val o = val::object();
      o.set("id", id);
      o.set("maxDiff", diff);
      o.set("nanMismatches", static_cast<double>(mismatched));
      look.call<void>("push", o);
    }
    out.set("lookahead", look);
    out.set("lookaheadCut", static_cast<double>(cut));
    return out;
  });
}

// Market data: prices, regimes and summary statistics.
val marketData(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const auto data = market(spec, env);
    const MarketData& d = *data;
    const std::size_t T = d.numDates(), N = d.numAssets();
    const Panel ret = d.returns();
    std::vector<double> index(T, 1.0), market(T, 0.0);
    for (std::size_t t = 1; t < T; ++t) {
      double s = 0;
      for (std::size_t i = 0; i < N; ++i) s += ret(t, i) / N;
      market[t] = s;
      index[t] = index[t - 1] * (1 + s);
    }
    val out = val::object();
    out.set("source", str(spec, "csv", "").empty() ? std::string("synthetic") : std::string("csv"));
    out.set("numDates", static_cast<double>(T));
    out.set("numAssets", static_cast<double>(N));
    out.set("dates", strings(d.dates));
    out.set("tickers", strings(d.tickers));
    out.set("regimeNames", regimeNames(d));
    out.set("regime", arr(regimeSlice(d, 0, T)));
    out.set("index", arr(index));
    std::vector<std::vector<double>> paths;
    for (std::size_t i = 0; i < std::min<std::size_t>(N, 12); ++i) {
      std::vector<double> p(T);
      for (std::size_t t = 0; t < T; ++t) p[t] = d.close(t, i) / d.close(0, i);
      paths.push_back(p);
    }
    out.set("paths", arrOfArr(paths));
    val stocks = val::array();
    const double mVar = [&] {
      double s = 0, s2 = 0;
      for (std::size_t t = 1; t < T; ++t) {
        s += market[t];
        s2 += market[t] * market[t];
      }
      return s2 / (T - 1) - (s / (T - 1)) * (s / (T - 1));
    }();
    for (std::size_t i = 0; i < N; ++i) {
      std::vector<double> r(T - 1), m(T - 1);
      double vol = 0;
      for (std::size_t t = 1; t < T; ++t) {
        r[t - 1] = ret(t, i);
        m[t - 1] = market[t];
        vol += d.volume(t, i) / (T - 1);
      }
      const PerformanceMetrics pm = evaluatePerformance(r);
      double cov = 0, mr = mean(r), mm = mean(m);
      for (std::size_t k = 0; k < r.size(); ++k) cov += (r[k] - mr) * (m[k] - mm) / r.size();
      val o = val::object();
      o.set("ticker", d.tickers[i]);
      o.set("annualReturn", pm.annualReturn);
      o.set("annualVolatility", pm.annualVolatility);
      o.set("sharpe", pm.sharpe);
      o.set("maxDrawdown", pm.maxDrawdown);
      o.set("beta", mVar > 0 ? cov / mVar : 0.0);
      o.set("averageVolume", vol);
      o.set("lastClose", d.close(T - 1, i));
      stocks.call<void>("push", o);
    }
    out.set("stocks", stocks);
    // Statistics of the equal-weight market in each regime.
    val regimes = val::array();
    for (std::size_t g = 0; g < d.regimeNames.size(); ++g) {
      std::vector<double> r;
      for (std::size_t t = 1; t < T; ++t)
        if (!d.regime.empty() && d.regime[t] == static_cast<int>(g)) r.push_back(market[t]);
      val o = val::object();
      o.set("name", d.regimeNames[g]);
      o.set("days", static_cast<double>(r.size()));
      const PerformanceMetrics pm = evaluatePerformance(r);
      o.set("annualReturn", r.empty() ? 0.0 : std::exp(std::log1p(mean(r)) * 252) - 1);
      o.set("annualVolatility", pm.annualVolatility);
      regimes.call<void>("push", o);
    }
    out.set("regimes", regimes);
    // Daily bars of the first stock over the last 120 days.
    const std::size_t from = T > 120 ? T - 120 : 0;
    val bars = val::object();
    std::vector<double> o_, h_, l_, c_, v_, w_;
    for (std::size_t t = from; t < T; ++t) {
      o_.push_back(d.open(t, 0));
      h_.push_back(d.high(t, 0));
      l_.push_back(d.low(t, 0));
      c_.push_back(d.close(t, 0));
      v_.push_back(d.volume(t, 0));
      w_.push_back(d.vwap(t, 0));
    }
    bars.set("dates", dateRange(d, from, T));
    bars.set("open", arr(o_));
    bars.set("high", arr(h_));
    bars.set("low", arr(l_));
    bars.set("close", arr(c_));
    bars.set("volume", arr(v_));
    bars.set("vwap", arr(w_));
    out.set("bars", bars);
    return out;
  });
}

// Alpha factors: information coefficients, coverage and correlations.
val factors(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const auto data = market(spec, env);
    const MarketData& d = *data;
    const auto ids = env.exp.alphaIds.empty() ? paperAlphaIds() : env.exp.alphaIds;
    const double t0 = nowMs();
    const FeatureSet fs = buildFeatures(d, ids, env.exp.normalisation);
    const Panel next = d.forwardReturns(1);
    const AlphaInputs in(d);
    const std::size_t T = d.numDates(), N = d.numAssets(), F = fs.size();
    val list = val::array();
    std::vector<double> icMean(F);
    for (std::size_t f = 0; f < F; ++f) {
      std::vector<double> ic, cum;
      double c = 0;
      std::vector<std::vector<double>> byRegime(d.regimeNames.size());
      for (std::size_t t = fs.warmup; t + 1 < T; ++t) {
        const double v = rankCorrelation(std::vector<double>(fs.panels[f].row(t), fs.panels[f].row(t) + N),
                                         std::vector<double>(next.row(t), next.row(t) + N));
        const double x = std::isfinite(v) ? v : 0.0;
        ic.push_back(x);
        c += x;
        cum.push_back(c);
        if (!d.regime.empty() && d.regime[t] >= 0 && static_cast<std::size_t>(d.regime[t]) < byRegime.size())
          byRegime[static_cast<std::size_t>(d.regime[t])].push_back(x);
      }
      const Panel raw = computeAlpha(ids[f], in);
      std::size_t finiteCount = 0, total = 0;
      for (std::size_t t = fs.warmup; t < T; ++t)
        for (std::size_t i = 0; i < N; ++i) {
          ++total;
          finiteCount += std::isfinite(raw(t, i)) ? 1 : 0;
        }
      double pos = 0;
      for (double v : ic) pos += v > 0 ? 1 : 0;
      val o = val::object();
      o.set("id", ids[f]);
      o.set("name", fs.names[f]);
      o.set("formula", alphaFormula(ids[f]));
      o.set("lookback", static_cast<double>(alphaLookback(ids[f])));
      icMean[f] = mean(ic);
      o.set("icMean", icMean[f]);
      o.set("icStd", stdev(ic));
      o.set("icir", stdev(ic) > 0 ? icMean[f] / stdev(ic) * std::sqrt(252.0) : 0.0);
      o.set("icPositive", ic.empty() ? 0.0 : pos / ic.size());
      o.set("coverage", total ? static_cast<double>(finiteCount) / total : 0.0);
      o.set("cumulativeIc", arr(cum));
      std::vector<double> reg;
      for (const auto& r : byRegime) reg.push_back(r.empty() ? 0.0 : mean(r));
      o.set("icByRegime", arr(reg));
      list.call<void>("push", o);
    }
    // Average cross-sectional correlation between the normalised factors.
    std::vector<std::vector<double>> corr(F, std::vector<double>(F, 0.0));
    std::size_t days = 0;
    for (std::size_t t = fs.warmup; t < T; t += 5, ++days)
      for (std::size_t a = 0; a < F; ++a) {
        corr[a][a] += 1.0;
        for (std::size_t b = a + 1; b < F; ++b) {
          const double c = correlation(std::vector<double>(fs.panels[a].row(t), fs.panels[a].row(t) + N),
                                       std::vector<double>(fs.panels[b].row(t), fs.panels[b].row(t) + N));
          const double x = std::isfinite(c) ? c : 0.0;
          corr[a][b] += x;
          corr[b][a] += x;
        }
      }
    for (auto& row : corr)
      for (double& v : row) v /= std::max<std::size_t>(1, days);
    val out = val::object();
    out.set("factors", list);
    out.set("correlation", arrOfArr(corr));
    out.set("dates", dateRange(d, fs.warmup, T - 1));
    out.set("regimeNames", regimeNames(d));
    out.set("warmup", static_cast<double>(fs.warmup));
    out.set("elapsedMs", nowMs() - t0);
    return out;
  });
}

// Labelling schemes and an example of N-period min-max labels.
val labels(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const auto data = market(spec, env);
    const MarketData& d = *data;
    const std::size_t T = d.numDates(), N = d.numAssets();
    const LabelSpec chosen = env.exp.label;
    std::vector<std::pair<std::string, LabelSpec>> kinds = {
        {"Direction, next day", {LabelKind::Direction, 1, chosen.window}},
        {"Direction, 5 days", {LabelKind::Direction, 5, chosen.window}},
        {"Excess over median, next day", {LabelKind::ExcessDirection, 1, chosen.window}},
        {"Min-max, " + std::to_string(chosen.window) + "-day window", {LabelKind::MinMax, 1, chosen.window}},
    };
    const Panel base = makeLabels(d, kinds[0].second);
    val list = val::array();
    for (const auto& [name, ls] : kinds) {
      const Panel y = makeLabels(d, ls);
      double labelled = 0, pos = 0, agree = 0, both = 0;
      for (std::size_t t = 0; t < T; ++t)
        for (std::size_t i = 0; i < N; ++i)
          if (std::isfinite(y(t, i))) {
            labelled += 1;
            pos += y(t, i);
            if (std::isfinite(base(t, i))) {
              both += 1;
              agree += y(t, i) == base(t, i);
            }
          }
      val o = val::object();
      o.set("name", name);
      o.set("labelled", labelled / (T * N));
      o.set("positive", labelled > 0 ? pos / labelled : 0.0);
      o.set("agreement", both > 0 ? agree / both : 0.0);
      o.set("lookahead", static_cast<double>(ls.lookahead()));
      list.call<void>("push", o);
    }
    const std::size_t from = T > 300 ? T - 300 : 0;
    const Panel mm = makeLabels(d, {LabelKind::MinMax, 1, chosen.window});
    const Panel cl = makeLabels(d, chosen);
    std::vector<double> close, buy, sell, label;
    for (std::size_t t = from; t < T; ++t) {
      close.push_back(d.close(t, 0));
      buy.push_back(mm(t, 0) == 1.0 ? d.close(t, 0) : NAN);
      sell.push_back(mm(t, 0) == 0.0 ? d.close(t, 0) : NAN);
      label.push_back(cl(t, 0));
    }
    val out = val::object();
    out.set("kinds", list);
    out.set("ticker", d.tickers[0]);
    out.set("dates", dateRange(d, from, T));
    out.set("close", arr(close));
    out.set("buy", arr(buy));
    out.set("sell", arr(sell));
    out.set("chosen", arr(label));
    out.set("chosenLookahead", static_cast<double>(chosen.lookahead()));
    return out;
  });
}

// Walk-forward training of every model: out-of-sample quality and what the models use.
val models(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const Predictions p = predictions(spec, env);
    const MarketData& d = *p.data;
    const PredictionSet& set = *p.set;
    val list = val::array();
    for (const auto& m : set.models) {
      val o = val::object();
      o.set("name", m.name);
      o.set("type", m.spec.type);
      o.set("oos", classificationToJs(m.oos));
      o.set("elapsedMs", m.elapsedMs);
      o.set("start", dateLabel(d, m.start));
      o.set("end", dateLabel(d, m.end - 1));
      o.set("importance", arr(m.importance));
      val fits = val::array();
      for (const auto& r : m.retrains) {
        val f = val::object();
        f.set("date", dateLabel(d, r.date));
        f.set("trainRows", static_cast<double>(r.trainRows));
        f.set("trainFrom", dateLabel(d, r.trainFrom));
        f.set("trainTo", dateLabel(d, r.trainTo - 1));
        f.set("fitMs", r.fitMs);
        f.set("accuracy", r.test.accuracy);
        f.set("auc", r.test.auc);
        fits.call<void>("push", f);
      }
      o.set("retrains", fits);
      // ROC, calibration and accuracy by regime over the out-of-sample samples.
      std::vector<double> y, q;
      std::vector<std::vector<double>> ry(d.regimeNames.size()), rq(d.regimeNames.size());
      for (std::size_t t = m.start; t < m.end; ++t)
        for (std::size_t i = 0; i < d.numAssets(); ++i) {
          if (!std::isfinite(set.labels(t, i))) continue;
          y.push_back(set.labels(t, i));
          q.push_back(m.probability(t, i));
          if (!d.regime.empty() && d.regime[t] >= 0) {
            ry[static_cast<std::size_t>(d.regime[t])].push_back(y.back());
            rq[static_cast<std::size_t>(d.regime[t])].push_back(q.back());
          }
        }
      const auto roc = rocCurve(y, q, 40);
      std::vector<double> fpr, tpr;
      for (const auto& pt : roc) {
        fpr.push_back(pt[0]);
        tpr.push_back(pt[1]);
      }
      o.set("rocFpr", arr(fpr));
      o.set("rocTpr", arr(tpr));
      std::vector<double> binP(10, 0.0), binY(10, 0.0), binN(10, 0.0);
      const double lo = quantile(q, 0.01), hi = quantile(q, 0.99);
      for (std::size_t k = 0; k < q.size(); ++k) {
        const auto b = static_cast<std::size_t>(std::clamp((q[k] - lo) / std::max(hi - lo, 1e-9) * 10.0, 0.0, 9.0));
        binP[b] += q[k];
        binY[b] += y[k];
        binN[b] += 1;
      }
      std::vector<double> calP, calY;
      for (std::size_t b = 0; b < 10; ++b)
        if (binN[b] > 0) {
          calP.push_back(binP[b] / binN[b]);
          calY.push_back(binY[b] / binN[b]);
        }
      o.set("calibrationPredicted", arr(calP));
      o.set("calibrationObserved", arr(calY));
      std::vector<double> regAcc;
      for (std::size_t g = 0; g < ry.size(); ++g) regAcc.push_back(classificationMetrics(ry[g], rq[g]).accuracy);
      o.set("accuracyByRegime", arr(regAcc));
      list.call<void>("push", o);
    }
    val out = val::object();
    out.set("models", list);
    out.set("features", strings(set.features.names));
    out.set("regimeNames", regimeNames(d));
    out.set("trainMs", p.trainMs);
    out.set("cached", p.cached);
    out.set("trainWindow", static_cast<double>(env.exp.walkForward.trainWindow));
    out.set("retrainEvery", static_cast<double>(env.exp.walkForward.retrainEvery));
    out.set("lookahead", static_cast<double>(env.exp.label.lookahead()));
    return out;
  });
}

// ------------------------------------------------------------------ strategy analyses
//
// Each analysis is a list of jobs over the same predictions: a transaction cost and the
// selectors to run over the candidate pool. A job runs on the CPU (CandidateBook +
// evaluateGrid) or on the GPU kernels (gpu::compile -> kernels -> gpu::summarise); both give
// a GridResult, and combine() builds the page's result from them.

struct Job {
  std::string label;
  double costBps = 10.0;
  std::vector<SelectorSpec> selectors;
};

struct Analysis {
  std::vector<Job> jobs;
  std::size_t evalFrom = 0;
  std::size_t gridSize = 0;  // robustness: selectors of the grid in job 0 (the spec's selector follows)
};

std::string analysisOf(const val& spec) { return str(spec, "analysis", "adaptive"); }

Analysis analysisJobs(const Env& env, const std::string& analysis) {
  Analysis a;
  const SelectorSpec& sel = env.exp.selector;
  if (analysis == "strategies" || analysis == "adaptive") {
    a.jobs.push_back({"base", env.exp.costBps, {sel}});
    a.evalFrom = sel.lookback;
  } else if (analysis == "robustness" || analysis == "validation") {
    Job grid{"selector grid", env.exp.costBps, {}};
    for (ScoreMetric m : env.rob.metrics)
      for (std::size_t lb : env.rob.lookbacks)
        for (std::size_t step : env.rob.steps) grid.selectors.push_back({lb, step, m, 1, sel.allowCash, sel.minScore});
    a.gridSize = grid.selectors.size();
    grid.selectors.push_back(sel);
    a.jobs.push_back(grid);
    a.evalFrom = sel.lookback;
    for (std::size_t lb : env.rob.lookbacks) a.evalFrom = std::max(a.evalFrom, lb);
    if (analysis == "robustness")
      for (double c : env.rob.costs) a.jobs.push_back({"cost " + std::to_string(c), c, {sel}});
  } else if (analysis == "book") {
    // Only the fixed candidates: their backtests, read back for an analysis that runs its
    // own selectors and statistics on the CPU (tournament, hedging, overfitting).
    a.jobs.push_back({"candidates", env.exp.costBps, {}});
    a.evalFrom = 0;
  } else {
    throw std::invalid_argument("unknown analysis '" + analysis + "'");
  }
  return a;
}

struct JobResult {
  GridResult grid;
  std::shared_ptr<CandidateBook> book;  // CPU runs only
};

JobResult runJobOnCpu(const PredictionSet& set, const ExperimentSpec& exp, const Job& job, std::size_t evalFrom) {
  JobResult r;
  const double t0 = nowMs();
  r.book = std::make_shared<CandidateBook>(set.models, exp.strategies, set.nextReturns, job.costBps);
  if (evalFrom >= r.book->days()) throw std::invalid_argument("the selector look-back is longer than the out-of-sample period");
  r.grid = evaluateGrid(*r.book, job.selectors, evalFrom);
  r.grid.elapsedMs = nowMs() - t0;
  return r;
}

std::vector<double> netSlice(const CandidateBook& book, std::size_t c, std::size_t from) { return book.netSeries(c, from); }

// The candidate with the highest Sharpe ratio, and its daily net returns (one CPU backtest,
// cheap enough on either engine).
std::size_t bestCandidate(const GridResult& g) {
  std::size_t best = 0;
  for (std::size_t c = 1; c < g.candidates.size(); ++c)
    if (g.candidates[c].sharpe > g.candidates[best].sharpe) best = c;
  return best;
}

std::vector<double> candidateNet(const PredictionSet& set, const ExperimentSpec& exp, double costBps, std::size_t c,
                                 std::size_t start, std::size_t end, std::size_t evalFrom) {
  const std::size_t s = exp.strategies.size();
  const auto& m = set.models.at(c / s);
  const auto r = backtest(exp.strategies[c % s], m.probability, rankTable(m.probability), set.nextReturns, start, end, costBps);
  return std::vector<double>(r.series.net.begin() + static_cast<std::ptrdiff_t>(evalFrom), r.series.net.end());
}

val candidatesToJs(const PredictionSet& set, const ExperimentSpec& exp, const GridResult& g) {
  val list = val::array();
  const std::size_t S = exp.strategies.size();
  for (std::size_t c = 0; c < g.candidates.size(); ++c) {
    val o = val::object();
    o.set("model", set.models[c / S].name);
    o.set("modelIndex", static_cast<double>(c / S));
    o.set("strategy", exp.strategies[c % S].label());
    o.set("kind", strategyKindName(exp.strategies[c % S].kind));
    o.set("strategyIndex", static_cast<double>(c % S));
    o.set("metrics", metricsToJs(g.candidates[c]));
    list.call<void>("push", o);
  }
  return list;
}

val equity(const std::vector<double>& net) { return arr(equityCurve(net)); }

val adaptiveToJs(const GridResult& g, std::size_t s, const SelectorSpec& spec, std::size_t numCandidates) {
  val o = val::object();
  o.set("label", spec.label());
  o.set("metrics", metricsToJs(g.selectors[s]));
  o.set("equity", equity(g.adaptiveNet[s]));
  o.set("drawdown", arr(drawdownCurve(g.adaptiveNet[s])));
  std::vector<double> sel(g.selection[s].begin(), g.selection[s].end());
  o.set("selection", arr(sel));
  o.set("switches", static_cast<double>(g.switches(s)));
  std::vector<double> share(numCandidates + 1, 0.0);
  for (int c : g.selection[s]) share[c < 0 ? numCandidates : static_cast<std::size_t>(c)] += 1.0 / g.selection[s].size();
  o.set("share", arr(share));
  return o;
}

val combine(const val& spec, const Env& env, const Predictions& p, const std::string& analysis, const Analysis& a,
            const std::vector<JobResult>& results, bool onGpu) {
  const MarketData& d = *p.data;
  const PredictionSet& set = *p.set;
  const ExperimentSpec& exp = env.exp;
  const GridResult& g0 = results.at(0).grid;
  const std::size_t start = g0.start, evalFrom = a.evalFrom, end = start + g0.days;
  const std::size_t C = g0.candidates.size();
  val out = val::object();
  out.set("dates", dateRange(d, start + evalFrom - 1, end));  // equity curves start the day before
  out.set("regime", arr(regimeSlice(d, start + evalFrom, end)));
  out.set("regimeNames", regimeNames(d));
  out.set("models", strings([&] {
            std::vector<std::string> n;
            for (const auto& m : set.models) n.push_back(m.name);
            return n;
          }()));
  out.set("strategyLabels", strings([&] {
            std::vector<std::string> n;
            for (const auto& s : exp.strategies) n.push_back(s.label());
            return n;
          }()));
  out.set("trainMs", p.trainMs);
  out.set("cachedPredictions", p.cached);
  out.set("costBps", exp.costBps);
  out.set("evalDays", static_cast<double>(end - start - evalFrom));
  out.set("numCandidates", static_cast<double>(C));
  double strategyMs = 0;
  for (const auto& r : results) strategyMs += r.grid.elapsedMs;
  out.set("strategyMs", strategyMs);
  out.set("candidates", candidatesToJs(set, exp, g0));

  const auto bench = equalWeightBenchmark(set.nextReturns, start + evalFrom, end, exp.costBps);
  val benchmark = val::object();
  benchmark.set("label", bench.label);
  benchmark.set("metrics", metricsToJs(bench.metrics));
  benchmark.set("equity", equity(bench.series.net));
  benchmark.set("drawdown", arr(drawdownCurve(bench.series.net)));
  out.set("benchmark", benchmark);

  const std::size_t best = bestCandidate(g0);
  const auto bestNet = candidateNet(set, exp, exp.costBps, best, start, end, evalFrom);
  val bestFixed = val::object();
  bestFixed.set("index", static_cast<double>(best));
  bestFixed.set("label", set.models[best / exp.strategies.size()].name + " · " + exp.strategies[best % exp.strategies.size()].label());
  bestFixed.set("metrics", metricsToJs(g0.candidates[best]));
  bestFixed.set("equity", equity(bestNet));
  bestFixed.set("drawdown", arr(drawdownCurve(bestNet)));
  out.set("bestFixed", bestFixed);

  // Median fixed strategy by Sharpe ratio.
  std::vector<double> sharpes;
  for (const auto& m : g0.candidates) sharpes.push_back(m.sharpe);
  out.set("medianFixedSharpe", quantile(sharpes, 0.5));
  std::vector<double> annual;
  for (const auto& m : g0.candidates) annual.push_back(m.annualReturn);
  out.set("medianFixedAnnualReturn", quantile(annual, 0.5));

  if (analysis == "strategies" || analysis == "adaptive") {
    out.set("adaptive", adaptiveToJs(g0, 0, exp.selector, C));
    if (!onGpu) {
      const CandidateBook& book = *results[0].book;
      // Every candidate's equity curve (path-level data: WebAssembly only).
      std::vector<std::vector<double>> curves;
      for (std::size_t c = 0; c < C; ++c) curves.push_back(equityCurve(netSlice(book, c, evalFrom)));
      out.set("equityCurves", arrOfArr(curves));
      // The best candidate of each 63-day block: no fixed strategy stays on top.
      const std::size_t block = 63;
      val blocks = val::array();
      for (std::size_t b = evalFrom; b + 21 <= book.days(); b += block) {
        const std::size_t e = std::min(book.days(), b + block);
        std::vector<double> ret(C);
        for (std::size_t c = 0; c < C; ++c) {
          double w = 1;
          for (std::size_t k = b; k < e; ++k) w *= 1 + book.net(c, k);
          ret[c] = w - 1;
        }
        const auto order = argsortDescending(ret.data(), C);
        std::size_t bestRank = 0;
        while (order[bestRank] != best) ++bestRank;
        double aw = 1;
        for (std::size_t k = b; k < e; ++k) aw *= 1 + g0.adaptiveNet[0][k - evalFrom];
        val o = val::object();
        o.set("from", dateLabel(d, start + b));
        o.set("to", dateLabel(d, start + e - 1));
        o.set("winner", static_cast<double>(order[0]));
        o.set("winnerReturn", ret[order[0]]);
        o.set("bestFixedRank", static_cast<double>(bestRank + 1));
        o.set("bestFixedReturn", ret[best]);
        o.set("adaptiveReturn", aw - 1);
        if (!d.regime.empty()) {
          std::vector<double> counts(d.regimeNames.size(), 0.0);
          for (std::size_t k = b; k < e; ++k) counts[static_cast<std::size_t>(d.regime[start + k])] += 1;
          o.set("regime", static_cast<double>(std::max_element(counts.begin(), counts.end()) - counts.begin()));
        }
        blocks.call<void>("push", o);
      }
      out.set("blocks", blocks);
      // Average of all fixed strategies (equal-weight mix of their returns).
      std::vector<double> avg(book.days() - evalFrom, 0.0);
      for (std::size_t c = 0; c < C; ++c)
        for (std::size_t k = evalFrom; k < book.days(); ++k) avg[k - evalFrom] += book.net(c, k) / C;
      val mix = val::object();
      mix.set("metrics", metricsToJs(evaluatePerformance(avg)));
      mix.set("equity", equity(avg));
      out.set("averageFixed", mix);
    }
  } else if (analysis == "robustness") {
    const std::size_t L = env.rob.lookbacks.size(), K = env.rob.steps.size();
    val grids = val::array();
    for (std::size_t mi = 0; mi < env.rob.metrics.size(); ++mi) {
      std::vector<std::vector<double>> sh(L, std::vector<double>(K)), ar(L, std::vector<double>(K)), dd(L, std::vector<double>(K));
      for (std::size_t l = 0; l < L; ++l)
        for (std::size_t k = 0; k < K; ++k) {
          const auto& m = g0.selectors[(mi * L + l) * K + k];
          sh[l][k] = m.sharpe;
          ar[l][k] = m.annualReturn;
          dd[l][k] = m.maxDrawdown;
        }
      val o = val::object();
      o.set("metric", scoreMetricName(env.rob.metrics[mi]));
      o.set("sharpe", arrOfArr(sh));
      o.set("annualReturn", arrOfArr(ar));
      o.set("maxDrawdown", arrOfArr(dd));
      grids.call<void>("push", o);
    }
    out.set("grids", grids);
    std::vector<double> lbs(env.rob.lookbacks.begin(), env.rob.lookbacks.end()), steps(env.rob.steps.begin(), env.rob.steps.end());
    out.set("lookbacks", arr(lbs));
    out.set("steps", arr(steps));
    double beatMedian = 0, beatBench = 0;
    for (std::size_t s = 0; s < a.gridSize; ++s) {
      beatMedian += g0.selectors[s].sharpe > quantile(sharpes, 0.5);
      beatBench += g0.selectors[s].sharpe > bench.metrics.sharpe;
    }
    out.set("gridSize", static_cast<double>(a.gridSize));
    out.set("shareBeatingMedianFixed", a.gridSize ? beatMedian / a.gridSize : 0.0);
    out.set("shareBeatingBenchmark", a.gridSize ? beatBench / a.gridSize : 0.0);
    out.set("adaptive", adaptiveToJs(g0, a.gridSize, exp.selector, C));
    // Year by year: the spec's selector against the best fixed strategy and the market.
    val years = val::array();
    const auto& an = g0.adaptiveNet[a.gridSize];
    std::size_t k = 0;
    while (k < an.size()) {
      const std::string y = dateLabel(d, start + evalFrom + k).substr(0, 4);
      std::size_t e = k;
      double wa = 1, wb = 1, wm = 1;
      while (e < an.size() && dateLabel(d, start + evalFrom + e).substr(0, 4) == y) {
        wa *= 1 + an[e];
        wb *= 1 + bestNet[e];
        wm *= 1 + bench.series.net[e];
        ++e;
      }
      val o = val::object();
      o.set("year", y);
      o.set("days", static_cast<double>(e - k));
      o.set("adaptive", wa - 1);
      o.set("bestFixed", wb - 1);
      o.set("benchmark", wm - 1);
      years.call<void>("push", o);
      k = e;
    }
    out.set("years", years);
    // Transaction costs: the selector, the best fixed strategy at that cost and the market.
    val costs = val::array();
    for (std::size_t j = 1; j < results.size(); ++j) {
      const GridResult& g = results[j].grid;
      const std::size_t b = bestCandidate(g);
      val o = val::object();
      o.set("costBps", a.jobs[j].costBps);
      o.set("adaptive", metricsToJs(g.selectors[0]));
      o.set("bestFixed", metricsToJs(g.candidates[b]));
      o.set("bestFixedLabel", set.models[b / exp.strategies.size()].name + " · " + exp.strategies[b % exp.strategies.size()].label());
      o.set("benchmark", metricsToJs(equalWeightBenchmark(set.nextReturns, start + evalFrom, end, a.jobs[j].costBps).metrics));
      double sumSharpe = 0;
      for (const auto& m : g.candidates) sumSharpe += m.sharpe / g.candidates.size();
      o.set("averageFixedSharpe", sumSharpe);
      costs.call<void>("push", o);
    }
    out.set("costs", costs);
  }
  (void)spec;
  return out;
}

val runAnalysisOnCpu(const val& spec, const std::string& analysis) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const Predictions p = predictions(spec, env);
    const Analysis a = analysisJobs(env, analysis);
    std::vector<JobResult> results;
    for (const auto& job : a.jobs) results.push_back(runJobOnCpu(*p.set, env.exp, job, a.evalFrom));
    return combine(spec, env, p, analysis, a, results, false);
  });
}

val strategies(val spec) { return runAnalysisOnCpu(spec, "strategies"); }
val adaptive(val spec) { return runAnalysisOnCpu(spec, "adaptive"); }
val robustness(val spec) { return runAnalysisOnCpu(spec, "robustness"); }

// ------------------------------------------------------------------ WebGPU path
//
// gpuJobs(spec) trains (or reuses) the models and compiles every job into a plan of
// ready-made buffers; the browser runs the kernels; gpuAnalyse(spec + gpuOutputs) turns the
// read-backs into GridResults and runs the same combine step as the CPU analysis.
// gpuEmulate(spec) runs the kernels on the CPU instead.

val gpuKernels(val) {
  val o = val::object();
  o.set("candidateBacktest", gpu::candidateBacktestKernel());
  o.set("adaptiveSelect", gpu::adaptiveSelectKernel());
  o.set("seriesSummary", gpu::seriesSummaryKernel());
  return o;
}

val planToJs(const gpu::FusedPlan& p, const std::string& label) {
  val o = val::object();
  o.set("label", label);
  o.set("numCandidates", static_cast<double>(p.numCandidates));
  o.set("numSelectors", static_cast<double>(p.numSelectors));
  o.set("numDays", static_cast<double>(p.numDays));
  o.set("numAssets", static_cast<double>(p.numAssets));
  o.set("numModels", static_cast<double>(p.numModels));
  o.set("header", typedArray(p.header));
  o.set("tables", typedArray(p.tables));
  val dispatch = val::object();
  dispatch.set("candidates", static_cast<double>(p.candidateDispatch()));
  dispatch.set("selectors", static_cast<double>(p.selectorDispatch()));
  dispatch.set("summary", static_cast<double>(p.summaryDispatch()));
  o.set("dispatch", dispatch);
  val bytes = val::object();
  bytes.set("tables", static_cast<double>(p.tables.size() * 4));
  bytes.set("book", static_cast<double>(std::max<std::size_t>(16, p.bookBytes())));
  bytes.set("prefix", static_cast<double>(std::max<std::size_t>(16, p.prefixBytes())));
  bytes.set("adapt", static_cast<double>(p.adaptBytes()));
  bytes.set("stats", static_cast<double>(p.statsBytes()));
  o.set("bytes", bytes);
  return o;
}

struct GpuJobs {
  Analysis analysis;
  std::vector<gpu::FusedPlan> plans;
  std::string unsupported;
};

GpuJobs compileJobs(const Env& env, const PredictionSet& set, const std::string& analysis) {
  GpuJobs g;
  g.analysis = analysisJobs(env, analysis);
  for (const auto& job : g.analysis.jobs) {
    if (const std::string why = gpu::limitation(set.models, env.exp.strategies, job.selectors); !why.empty()) {
      g.unsupported = why;
      return g;
    }
    g.plans.push_back(gpu::compile(set.models, env.exp.strategies, set.nextReturns, job.costBps, job.selectors, g.analysis.evalFrom));
  }
  return g;
}

val gpuJobs(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const Predictions p = predictions(spec, env);
    const double t0 = nowMs();
    const GpuJobs g = compileJobs(env, *p.set, analysisOf(spec));
    val out = val::object();
    out.set("compileMs", nowMs() - t0);
    out.set("trainMs", p.trainMs);
    out.set("cachedPredictions", p.cached);
    if (!g.unsupported.empty()) {
      out.set("unsupported", g.unsupported);
      return out;
    }
    val plans = val::array();
    for (std::size_t i = 0; i < g.plans.size(); ++i) plans.call<void>("push", planToJs(g.plans[i], g.analysis.jobs[i].label));
    out.set("plans", plans);
    return out;
  });
}

// GPU validation: the kernels against the CPU library on the same grid.
val validation(const Env& env, const Predictions& p, const Analysis& a, const GridResult& g) {
  const JobResult w = runJobOnCpu(*p.set, env.exp, a.jobs.at(0), a.evalFrom);
  double candSharpe = 0, candReturn = 0, selSharpe = 0, agree = 0, total = 0, turnover = 0;
  for (std::size_t c = 0; c < g.candidates.size(); ++c) {
    candSharpe = std::max(candSharpe, std::fabs(g.candidates[c].sharpe - w.grid.candidates[c].sharpe));
    candReturn = std::max(candReturn, std::fabs(g.candidates[c].totalReturn - w.grid.candidates[c].totalReturn) /
                                          (1.0 + std::fabs(w.grid.candidates[c].totalReturn)));
    turnover = std::max(turnover, std::fabs(g.candidates[c].averageTurnover - w.grid.candidates[c].averageTurnover));
  }
  for (std::size_t s = 0; s < g.selectors.size(); ++s) {
    selSharpe = std::max(selSharpe, std::fabs(g.selectors[s].sharpe - w.grid.selectors[s].sharpe));
    for (std::size_t d = 0; d < g.selection[s].size(); ++d) {
      agree += g.selection[s][d] == w.grid.selection[s][d];
      total += 1;
    }
  }
  val checks = val::object();
  checks.set("candidateSharpeDiff", candSharpe);
  checks.set("candidateReturnDiff", candReturn);
  checks.set("candidateTurnoverDiff", turnover);
  checks.set("selectorSharpeDiff", selSharpe);
  checks.set("selectionAgreement", total > 0 ? agree / total : 1.0);
  val out = val::object();
  out.set("checks", checks);
  out.set("cpuMs", w.grid.elapsedMs);
  out.set("numCandidates", static_cast<double>(g.candidates.size()));
  out.set("numSelectors", static_cast<double>(g.selectors.size()));
  out.set("evalDays", static_cast<double>(g.days - g.evalFrom));
  out.set("numAssets", static_cast<double>(p.data->numAssets()));
  out.set("numModels", static_cast<double>(p.set->models.size()));
  std::vector<double> gs, ws, gc, wc;
  for (std::size_t s = 0; s < a.gridSize; ++s) {
    gs.push_back(g.selectors[s].sharpe);
    ws.push_back(w.grid.selectors[s].sharpe);
  }
  for (std::size_t c = 0; c < g.candidates.size(); ++c) {
    gc.push_back(g.candidates[c].sharpe);
    wc.push_back(w.grid.candidates[c].sharpe);
  }
  out.set("gpuSelectorSharpe", arr(gs));
  out.set("cpuSelectorSharpe", arr(ws));
  out.set("gpuCandidateSharpe", arr(gc));
  out.set("cpuCandidateSharpe", arr(wc));
  const std::size_t last = a.gridSize;
  out.set("gpuEquity", equity(g.adaptiveNet[last]));
  out.set("cpuEquity", equity(w.grid.adaptiveNet[last]));
  out.set("dates", dateRange(*p.data, g.start + g.evalFrom - 1, g.start + g.days));
  return out;
}

val analyseFused(const val& spec, const std::function<gpu::FusedOutput(std::size_t, const gpu::FusedPlan&)>& output) {
  const Env env = parseEnv(spec);
  const Predictions p = predictions(spec, env);
  const std::string analysis = analysisOf(spec);
  const GpuJobs g = compileJobs(env, *p.set, analysis);
  if (!g.unsupported.empty()) throw std::invalid_argument(g.unsupported);
  std::vector<JobResult> results;
  for (std::size_t i = 0; i < g.plans.size(); ++i) {
    JobResult r;
    r.grid = gpu::summarise(g.plans[i], output(i, g.plans[i]));
    results.push_back(std::move(r));
  }
  if (analysis == "validation") return validation(env, p, g.analysis, results.at(0).grid);
  return combine(spec, env, p, analysis, g.analysis, results, true);
}

val gpuAnalyse(val spec) {
  return guarded([&] {
    const val outputs = spec["gpuOutputs"];
    return analyseFused(spec, [&](std::size_t i, const gpu::FusedPlan&) {
      if (outputs.isUndefined() || outputs.isNull() || i >= outputs["length"].as<std::size_t>() || outputs[i].isNull() ||
          outputs[i].isUndefined())
        throw std::invalid_argument("missing GPU output for job " + std::to_string(i));
      gpu::FusedOutput o;
      o.stats = emscripten::convertJSArrayToNumberVector<float>(outputs[i]["stats"]);
      o.adapt = emscripten::convertJSArrayToNumberVector<float>(outputs[i]["adapt"]);
      return o;
    });
  });
}

val gpuEmulate(val spec) {
  return guarded([&] {
    return analyseFused(spec, [](std::size_t, const gpu::FusedPlan& plan) { return gpu::runFusedReference(plan); });
  });
}

// The emulated GPU device: runs one plan from gpuJobs on the CPU reference of the kernels and
// returns the same read-back buffers as the WebGPU engine ({ stats, adapt }), so the browser's
// pipeline (gpuJobs -> device.run per plan -> gpuAnalyse) is unchanged when WebGPU is missing.
val gpuRunPlan(val plan) {
  return guarded([&] {
    gpu::FusedPlan p;
    p.header = emscripten::convertJSArrayToNumberVector<std::uint32_t>(plan["header"]);
    p.tables = emscripten::convertJSArrayToNumberVector<float>(plan["tables"]);
    const double t0 = nowMs();
    const gpu::FusedOutput out = gpu::runFusedReference(p);
    val o = val::object();
    o.set("stats", typedArray(out.stats));
    o.set("adapt", typedArray(out.adapt));
    if (flag(plan, "readBook", false)) o.set("book", typedArray(out.book));
    o.set("ms", nowMs() - t0);
    return o;
  });
}

// ------------------------------------------------------------------ Advances in Financial Machine Learning
//
// One entry point per topic page. They reuse the cached market and predictions where the
// topic builds on them (labels, validation, overfitting) and are otherwise self-contained.

val metricsOfBars(const afml::BarStatistics& s) {
  val o = val::object();
  o.set("count", static_cast<double>(s.count));
  o.set("barsPerDayMean", s.barsPerDayMean);
  o.set("barsPerDaySd", s.barsPerDaySd);
  o.set("returnSd", s.returnSd);
  o.set("skewness", s.skewness);
  o.set("kurtosis", s.kurtosis);
  o.set("jarqueBera", s.jarqueBera);
  o.set("serialCorrelation", s.serialCorrelation);
  o.set("varianceOfVariance", s.varianceOfVariance);
  return o;
}

// Information-driven bars on a synthetic trade stream.
val afmlBars(val spec) {
  return guarded([&] {
    const val b = spec["bars"];
    afml::TradeStreamSpec ts;
    ts.days = count(b, "days", ts.days);
    ts.tradesPerDay = num(b, "tradesPerDay", ts.tradesPerDay);
    ts.activityDispersion = num(b, "activityDispersion", ts.activityDispersion);
    ts.persistence = num(b, "persistence", ts.persistence);
    ts.seed = static_cast<std::uint64_t>(num(b, "seed", 5));
    const double perDay = std::max(1.0, num(b, "barsPerDay", 20));
    if (ts.days > 400 || ts.tradesPerDay > 20000) throw std::invalid_argument("bars: at most 400 days and 20,000 trades a day in the browser");
    const double t0 = nowMs();
    const auto trades = afml::generateTrades(ts);
    double volume = 0, dollars = 0;
    for (const auto& t : trades) {
      volume += t.volume;
      dollars += t.volume * t.price;
    }
    const double target = static_cast<double>(ts.days) * perDay;
    std::vector<std::pair<std::string, std::vector<afml::Bar>>> kinds = {
        {"time", afml::timeBars(trades, 1.0 / perDay)},
        {"tick", afml::tickBars(trades, std::max<std::size_t>(1, static_cast<std::size_t>(trades.size() / target)))},
        {"volume", afml::volumeBars(trades, volume / target)},
        {"dollar", afml::dollarBars(trades, dollars / target)},
        {"tick imbalance", afml::tickImbalanceBars(trades, std::max<std::size_t>(10, static_cast<std::size_t>(trades.size() / target)))},
    };
    val list = val::array();
    for (const auto& [name, bars] : kinds) {
      val o = val::object();
      o.set("name", name);
      o.set("stats", metricsOfBars(afml::barStatistics(bars, ts.days)));
      // Standardised returns for the distribution chart.
      auto r = afml::barReturns(bars);
      const double m = mean(r), sd = stdev(r);
      for (auto& x : r) x = sd > 0 ? (x - m) / sd : 0.0;
      if (r.size() > 6000) r.resize(6000);
      o.set("standardised", arr(r));
      std::vector<double> perDayCount(ts.days, 0.0);
      for (const auto& bar : bars) perDayCount[std::min<std::size_t>(ts.days - 1, static_cast<std::size_t>(bar.timeClose))] += 1;
      o.set("barsPerDay", arr(perDayCount));
      list.call<void>("push", o);
    }
    std::vector<double> activity(ts.days, 0.0), dayClose(ts.days, 0.0);
    for (const auto& t : trades) {
      const auto d = std::min<std::size_t>(ts.days - 1, static_cast<std::size_t>(t.time));
      activity[d] += 1;
      dayClose[d] = t.price;
    }
    val out = val::object();
    out.set("kinds", list);
    out.set("tradesPerDay", arr(activity));
    out.set("dailyClose", arr(dayClose));
    out.set("trades", static_cast<double>(trades.size()));
    out.set("days", static_cast<double>(ts.days));
    out.set("elapsedMs", nowMs() - t0);
    return out;
  });
}

// Fractional differentiation of a log-price series and the stationarity / memory trade-off.
val afmlFracDiff(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const auto data = market(spec, env);
    const MarketData& d = *data;
    const std::size_t T = d.numDates(), N = d.numAssets();
    std::vector<double> series(T, 0.0);
    const std::string which = str(spec, "fracdiffSeries", "index");
    if (which == "index") {
      double lw = 0;  // log of the equal-weight index
      for (std::size_t t = 1; t < T; ++t) {
        double s = 0;
        for (std::size_t i = 0; i < N; ++i) s += (d.close(t, i) / d.close(t - 1, i) - 1.0) / N;
        series[t] = lw += std::log1p(s);
      }
    } else {
      for (std::size_t t = 0; t < T; ++t) series[t] = std::log(d.close(t, 0));
    }
    const double thr = num(spec, "fracdiffThreshold", 1e-3);
    const auto scan = afml::scanFracDiff(series, 0.05, thr);
    const double chosen = std::clamp(num(spec, "ffdOrder", scan.minimumD), 0.0, 1.0);
    val out = val::object();
    out.set("series", which == "index" ? std::string("equal-weight index") : d.tickers[0]);
    out.set("dates", strings(d.dates));
    out.set("logPrice", arr(series));
    out.set("d", arr(scan.d));
    out.set("adf", arr(scan.adf));
    out.set("correlation", arr(scan.correlation));
    std::vector<double> width(scan.width.begin(), scan.width.end());
    out.set("width", arr(width));
    out.set("minimumD", scan.minimumD);
    out.set("critical5", afml::AdfResult::critical5);
    out.set("critical1", afml::AdfResult::critical1);
    out.set("chosenD", chosen);
    out.set("ffdMinimum", arr(afml::fracDiff(series, scan.minimumD, thr)));
    out.set("ffdChosen", arr(afml::fracDiff(series, chosen, thr)));
    out.set("firstDifference", arr(afml::fracDiff(series, 1.0, thr)));
    val weights = val::array();
    for (double dd : {0.2, 0.4, 0.6, 0.8, 1.0}) {
      auto w = afml::fracDiffWeights(dd, 1e-4, 60);
      w.resize(std::min<std::size_t>(w.size(), 30));
      val o = val::object();
      o.set("d", dd);
      o.set("weights", arr(w));
      weights.call<void>("push", o);
    }
    out.set("weights", weights);
    out.set("threshold", thr);
    return out;
  });
}

// Event sampling, triple-barrier labels, uniqueness and meta-labeling.
val afmlLabeling(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const auto data = market(spec, env);
    const MarketData& d = *data;
    const std::size_t T = d.numDates(), N = d.numAssets();
    const val L = spec["labeling"];
    const double cusumMult = num(L, "cusumMultiple", 2.0);
    afml::BarrierSpec bs;
    bs.profitTaking = num(L, "profitTaking", 1.0);
    bs.stopLoss = num(L, "stopLoss", 1.0);
    bs.maxHolding = std::max<std::size_t>(1, count(L, "maxHolding", 10));
    const double volSpan = num(L, "volSpan", 50.0);
    const std::size_t momentum = std::max<std::size_t>(2, count(L, "momentum", 20));
    // Stock 0 in detail.
    const auto close0 = d.close.series(0);
    std::vector<double> lp(T);
    for (std::size_t t = 0; t < T; ++t) lp[t] = std::log(close0[t]);
    const auto vol0 = afml::ewmVolatility(close0, volSpan);
    std::vector<double> absr;
    for (std::size_t t = 1; t < T; ++t) absr.push_back(std::fabs(close0[t] / close0[t - 1] - 1.0));
    const double h0 = cusumMult * quantile(absr, 0.5) * 1.4826;
    const auto ev0 = afml::cusumFilter(lp, h0);
    // Barrier targets in returns over the holding horizon: daily volatility x sqrt(days).
    std::vector<double> trg0(T);
    for (std::size_t t = 0; t < T; ++t) trg0[t] = vol0[t] * std::sqrt(static_cast<double>(bs.maxHolding));
    const auto lab0 = afml::tripleBarrier(close0, ev0, trg0, bs);
    val out = val::object();
    out.set("ticker", d.tickers[0]);
    out.set("dates", strings(d.dates));
    out.set("close", arr(close0));
    out.set("cusumThreshold", h0);
    val events = val::array();
    double up = 0, down = 0, vert = 0, pos = 0;
    std::vector<afml::Span> spans0;
    for (const auto& e : lab0) {
      val o = val::object();
      o.set("t0", static_cast<double>(e.t0));
      o.set("t1", static_cast<double>(e.t1));
      o.set("barrier", e.barrier);
      o.set("label", e.label);
      o.set("ret", e.ret);
      o.set("upper", close0[e.t0] * (1 + bs.profitTaking * e.target));
      o.set("lower", close0[e.t0] * (1 - bs.stopLoss * e.target));
      events.call<void>("push", o);
      (e.barrier > 0 ? up : e.barrier < 0 ? down : vert) += 1;
      pos += e.label > 0;
      spans0.push_back({e.t0, e.t1});
    }
    out.set("events", events);
    val counts = val::object();
    counts.set("upper", up);
    counts.set("lower", down);
    counts.set("vertical", vert);
    counts.set("positive", pos);
    out.set("counts", counts);
    const auto conc = afml::concurrency(spans0, T);
    out.set("concurrency", arr(conc));
    out.set("uniqueness", arr(afml::averageUniqueness(spans0, conc)));
    // Sequential vs standard bootstrap on these spans.
    {
      Rng a(17), b(18);
      std::vector<double> seq, plain;
      const std::size_t draws = spans0.size();
      for (int rep = 0; rep < 30 && draws > 1; ++rep) {
        seq.push_back(afml::sampleUniqueness(spans0, afml::sequentialBootstrap(spans0, T, draws, a), T));
        std::vector<std::size_t> s(draws);
        for (auto& k : s) k = b.below(draws);
        plain.push_back(afml::sampleUniqueness(spans0, s, T));
      }
      out.set("bootstrapSequential", arr(seq));
      out.set("bootstrapStandard", arr(plain));
    }
    // Meta-labeling across all stocks: the primary model is a momentum rule (side = sign of
    // the last `momentum` days' return) at CUSUM events, held to a short triple barrier; the
    // secondary model learns from the alpha factors, signed by the side, whether to act on it.
    // It is trained on the events of the first 60% of dates whose outcome is known by then.
    const FeatureSet fs = buildFeatures(d, env.exp.alphaIds.empty() ? paperAlphaIds() : env.exp.alphaIds, env.exp.normalisation);
    const Panel mask = afml::cusumEventMask(d, num(L, "metaCusumMultiple", 1.0));
    afml::BarrierSpec ms = bs;
    ms.maxHolding = std::max<std::size_t>(1, count(L, "metaHolding", 2));
    struct Row {
      std::size_t t0, t1, asset;
      int side, label;
      double ret, vol;
    };
    std::vector<Row> rows;
    for (std::size_t i = 0; i < N; ++i) {
      const auto c = d.close.series(i);
      const auto v = afml::ewmVolatility(c, volSpan);
      std::vector<double> trg(T);
      for (std::size_t t = 0; t < T; ++t) trg[t] = v[t] * std::sqrt(static_cast<double>(ms.maxHolding));
      std::vector<std::size_t> evs;
      std::vector<int> sides;
      for (std::size_t t = std::max(fs.warmup, momentum); t < T; ++t)
        if (mask(t, i) > 0) {
          evs.push_back(t);
          sides.push_back(c[t] >= c[t - momentum] ? 1 : -1);
        }
      for (const auto& e : afml::tripleBarrier(c, evs, trg, ms, &sides)) rows.push_back({e.t0, e.t1, i, e.side, e.label, e.ret, v[e.t0]});
    }
    if (rows.size() < 100) throw std::invalid_argument("meta-labeling: fewer than 100 events; lower the CUSUM multiple");
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.t0 < b.t0 || (a.t0 == b.t0 && a.asset < b.asset); });
    const std::size_t split = fs.warmup + (T - fs.warmup) * 6 / 10;
    std::vector<std::size_t> trainIdx, testIdx;
    for (std::size_t k = 0; k < rows.size(); ++k) {
      if (rows[k].t1 < split) trainIdx.push_back(k);  // purged: the label is known before the test period
      else if (rows[k].t0 >= split) testIdx.push_back(k);
    }
    if (trainIdx.size() < 30 || testIdx.size() < 30) throw std::invalid_argument("meta-labeling: too few events on one side of the split");
    const std::size_t F = fs.size();
    // Features in the direction of the bet, the side itself and the volatility.
    auto rowOf = [&](const Row& r, double* x) {
      for (std::size_t f = 0; f < F; ++f) x[f] = fs.panels[f](r.t0, r.asset) * r.side;
      x[F] = r.side;
      x[F + 1] = std::isfinite(r.vol) ? r.vol : 0.0;
    };
    Matrix Xtr(trainIdx.size(), F + 2), Xte(testIdx.size(), F + 2);
    std::vector<double> ytr, yte;
    for (std::size_t k = 0; k < trainIdx.size(); ++k) {
      rowOf(rows[trainIdx[k]], Xtr.row(k));
      ytr.push_back(rows[trainIdx[k]].label);
    }
    for (std::size_t k = 0; k < testIdx.size(); ++k) {
      rowOf(rows[testIdx[k]], Xte.row(k));
      yte.push_back(rows[testIdx[k]].label);
    }
    ModelSpec mm;
    mm.type = str(L, "metaModel", "logistic");
    mm.trees = 40;
    auto model = makeClassifier(mm);
    model->fit(Xtr, ytr);
    const auto p = model->predictProba(Xte);
    // Precision, recall and F1 of "the primary bet pays", before and after the filter.
    auto scoreOf = [&](const std::vector<int>& act) {
      double tp = 0, fp = 0, fn = 0, ret = 0, n = 0, sizedRet = 0;
      for (std::size_t k = 0; k < yte.size(); ++k) {
        const bool good = yte[k] > 0.5;
        if (act[k]) {
          tp += good;
          fp += !good;
          ret += rows[testIdx[k]].ret;
          n += 1;
        } else {
          fn += good;
        }
        sizedRet += act[k] ? std::max(0.0, afml::betSize(p[k])) * rows[testIdx[k]].ret : 0.0;
      }
      val o = val::object();
      const double prec = tp + fp > 0 ? tp / (tp + fp) : 0, rec = tp + fn > 0 ? tp / (tp + fn) : 0;
      o.set("bets", n);
      o.set("precision", prec);
      o.set("recall", rec);
      o.set("f1", prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0.0);
      o.set("meanReturn", n > 0 ? ret / n : 0.0);
      o.set("sizedMeanReturn", sizedRet / static_cast<double>(yte.size()));
      return o;
    };
    std::vector<int> all(yte.size(), 1), filtered(yte.size());
    for (std::size_t k = 0; k < yte.size(); ++k) filtered[k] = p[k] > 0.5;
    val meta = val::object();
    meta.set("events", static_cast<double>(rows.size()));
    meta.set("trainEvents", static_cast<double>(trainIdx.size()));
    meta.set("testEvents", static_cast<double>(testIdx.size()));
    meta.set("splitDate", dateLabel(d, split));
    meta.set("primary", scoreOf(all));
    meta.set("filtered", scoreOf(filtered));
    meta.set("auc", classificationMetrics(yte, p).auc);
    meta.set("probabilities", arr(p));
    meta.set("labels", arr(yte));
    meta.set("model", mm.displayName());
    meta.set("holding", static_cast<double>(ms.maxHolding));
    meta.set("momentum", static_cast<double>(momentum));
    out.set("meta", meta);
    return out;
  });
}

// Cross-validation with overlapping labels, combinatorial purged CV and feature importance.
val afmlValidation(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const auto data = market(spec, env);
    const MarketData& d = *data;
    const val V = spec["validation"];
    const std::size_t folds = std::max<std::size_t>(2, count(V, "folds", 5));
    const std::size_t embargo = count(V, "embargo", 5);
    const std::size_t maxRows = std::max<std::size_t>(500, count(V, "maxRows", 6000));
    LabelSpec label = env.exp.label;
    if (has(V, "horizon")) label.horizon = std::max<std::size_t>(1, count(V, "horizon", 5));
    ModelSpec model;
    model.type = str(V, "model", "xgboost");
    model.trees = count(V, "trees", 40);
    const double t0 = nowMs();
    FeatureSet fs = buildFeatures(d, env.exp.alphaIds.empty() ? paperAlphaIds() : env.exp.alphaIds, env.exp.normalisation);
    if (!env.exp.extraFeatures.empty()) afml::appendExtraFeatures(fs, d, env.exp.extraFeatures, env.exp.normalisation, env.exp.ffdOrder);
    Panel ends;
    const Panel y = makeLabels(d, label, &ends);
    // Every k-th date so that the sample stays below maxRows; the dates stay contiguous blocks.
    const std::size_t T = d.numDates(), N = d.numAssets();
    const std::size_t usable = T - fs.warmup;
    const std::size_t stride = std::max<std::size_t>(1, (usable * N + maxRows - 1) / maxRows);
    Dataset all = assemble(fs, y, fs.warmup, T, 1, true);
    std::vector<std::size_t> keep;
    for (std::size_t r = 0; r < all.X.rows(); ++r)
      if ((all.date[r] - fs.warmup) % stride == 0) keep.push_back(r);
    Matrix X = all.X.selectRows(keep);
    std::vector<double> yy;
    std::vector<afml::Span> spans;
    for (std::size_t r : keep) {
      yy.push_back(all.y[r]);
      const double e = ends(all.date[r], all.asset[r]);
      spans.push_back({all.date[r], std::isfinite(e) ? static_cast<std::size_t>(e) : all.date[r]});
    }
    // Shuffled k-fold: rows assigned to folds at random, the textbook leak.
    std::vector<afml::Split> shuffled(folds);
    {
      std::vector<std::size_t> idx(yy.size());
      for (std::size_t k = 0; k < idx.size(); ++k) idx[k] = k;
      Rng rng(23);
      shuffle(idx, rng);
      for (std::size_t k = 0; k < idx.size(); ++k) shuffled[k % folds].test.push_back(idx[k]);
      for (auto& f : shuffled) {
        std::sort(f.test.begin(), f.test.end());
        std::vector<char> in(yy.size(), 0);
        for (std::size_t k : f.test) in[k] = 1;
        for (std::size_t k = 0; k < yy.size(); ++k)
          if (!in[k]) f.train.push_back(k);
      }
    }
    const auto blocked = afml::purgedKFold(spans, T, folds, 0, false);
    const auto purged = afml::purgedKFold(spans, T, folds, embargo, true);
    auto cvToJs = [&](const char* name, const std::vector<afml::Split>& splits) {
      const auto s = afml::crossValidate(model, X, yy, splits);
      double trainRows = 0;
      for (const auto& f : splits) trainRows += static_cast<double>(f.train.size()) / splits.size();
      val o = val::object();
      o.set("name", std::string(name));
      o.set("accuracy", s.accuracy);
      o.set("auc", s.auc);
      o.set("logLoss", s.logLoss);
      o.set("foldAccuracy", arr(s.foldAccuracy));
      o.set("trainRows", trainRows);
      return o;
    };
    val cv = val::array();
    cv.call<void>("push", cvToJs("shuffled k-fold", shuffled));
    cv.call<void>("push", cvToJs("blocked k-fold", blocked));
    cv.call<void>("push", cvToJs("purged k-fold + embargo", purged));
    // Feature importance on the purged folds.
    const auto mdi = afml::meanDecreaseImpurity(model, X, yy);
    const auto mda = afml::meanDecreaseAccuracy(model, X, yy, purged);
    ModelSpec single;
    single.type = "logistic";
    const auto sfi = afml::singleFeatureImportance(single, X, yy, purged);
    // Combinatorial purged CV: per path, the out-of-sample accuracy and the Sharpe ratio of a
    // long-short trade on the predictions (top fifth long, bottom fifth short, held one day).
    const std::size_t groups = std::clamp<std::size_t>(count(V, "groups", 6), 3, 10);
    const std::size_t testGroups = std::clamp<std::size_t>(count(V, "testGroups", 2), 1, groups - 1);
    const auto cp = afml::combinatorialPurgedSplits(spans, T, groups, testGroups, embargo);
    std::vector<std::vector<double>> predBySplit(cp.splits.size());
    for (std::size_t s = 0; s < cp.splits.size(); ++s) {
      const auto& sp = cp.splits[s];
      std::vector<double> ytr;
      for (std::size_t k : sp.train) ytr.push_back(yy[k]);
      auto m = makeClassifier(model);
      m->fit(X.selectRows(sp.train), ytr);
      predBySplit[s] = m->predictProba(X.selectRows(sp.test));
    }
    const Panel next = d.forwardReturns(1);
    val paths = val::array();
    for (std::size_t p = 0; p < cp.paths; ++p) {
      // Predictions of path p: each group's from the split assigned to it.
      std::vector<double> prob(yy.size(), NAN);
      for (std::size_t g = 0; g < groups; ++g) {
        const std::size_t s = cp.pathSplit[p][g];
        const auto& sp = cp.splits[s];
        for (std::size_t k = 0; k < sp.test.size(); ++k)
          if (cp.groupOfDate[spans[sp.test[k]].first] == g) prob[sp.test[k]] = predBySplit[s][k];
      }
      // Daily long-short returns on the sampled dates.
      std::map<std::size_t, std::vector<std::pair<double, double>>> byDate;
      for (std::size_t k = 0; k < yy.size(); ++k)
        if (std::isfinite(prob[k]) && std::isfinite(next(spans[k].first, all.asset[keep[k]])))
          byDate[spans[k].first].push_back({prob[k], next(spans[k].first, all.asset[keep[k]])});
      std::vector<double> ret;
      for (auto& [date, v] : byDate) {
        if (v.size() < 5) continue;
        std::sort(v.begin(), v.end());
        const std::size_t q = v.size() / 5;
        double r = 0;
        for (std::size_t k = 0; k < q; ++k) r += (v[v.size() - 1 - k].second - v[k].second) / q;
        ret.push_back(r);
      }
      const auto pm = evaluatePerformance(ret);
      std::vector<double> yt, pt;
      for (std::size_t k = 0; k < yy.size(); ++k)
        if (std::isfinite(prob[k])) {
          yt.push_back(yy[k]);
          pt.push_back(prob[k]);
        }
      val o = val::object();
      o.set("accuracy", classificationMetrics(yt, pt).accuracy);
      o.set("sharpe", pm.sharpe / std::sqrt(static_cast<double>(stride)));  // dates are `stride` days apart
      o.set("equity", arr(equityCurve(ret)));
      paths.call<void>("push", o);
    }
    val cpcv = val::object();
    cpcv.set("groups", static_cast<double>(groups));
    cpcv.set("testGroups", static_cast<double>(testGroups));
    cpcv.set("splits", static_cast<double>(cp.splits.size()));
    cpcv.set("paths", paths);
    val out = val::object();
    out.set("rows", static_cast<double>(yy.size()));
    out.set("stride", static_cast<double>(stride));
    out.set("horizon", static_cast<double>(label.lookahead()));
    out.set("folds", static_cast<double>(folds));
    out.set("embargo", static_cast<double>(embargo));
    out.set("model", model.displayName());
    out.set("cv", cv);
    out.set("features", strings(fs.names));
    out.set("mdi", arr(mdi.mean));
    out.set("mda", arr(mda.mean));
    out.set("mdaSe", arr(mda.sd));
    out.set("sfi", arr(sfi.mean));
    out.set("cpcv", cpcv);
    out.set("elapsedMs", nowMs() - t0);
    return out;
  });
}

// Hierarchical risk parity on the stock universe and in a Monte Carlo study.
val afmlPortfolio(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const auto data = market(spec, env);
    const MarketData& d = *data;
    const val P = spec["portfolio"];
    const std::size_t window = std::max<std::size_t>(30, count(P, "window", 252));
    const std::size_t rebalance = std::max<std::size_t>(1, count(P, "rebalance", 21));
    const std::size_t T = d.numDates(), N = d.numAssets();
    if (T <= window + rebalance + 1) throw std::invalid_argument("portfolio: the estimation window is longer than the data");
    const Panel ret = d.returns();
    auto windowMatrix = [&](std::size_t from, std::size_t to) {
      Matrix R(to - from, N);
      for (std::size_t t = from; t < to; ++t)
        for (std::size_t i = 0; i < N; ++i) R(t - from, i) = std::isfinite(ret(t, i)) ? ret(t, i) : 0.0;
      return R;
    };
    // The last window in detail.
    const Matrix cov = afml::covarianceMatrix(windowMatrix(T - window, T)), corr = afml::correlationFromCovariance(cov);
    const auto L = afml::clusterAssets(corr);
    const auto hrp = afml::hierarchicalRiskParity(cov, L.order), ivp = afml::inverseVarianceWeights(cov),
               mv = afml::minimumVarianceWeights(cov);
    std::vector<std::vector<double>> c0(N, std::vector<double>(N)), c1(N, std::vector<double>(N));
    std::vector<std::string> orderedTickers;
    for (std::size_t i = 0; i < N; ++i) {
      orderedTickers.push_back(d.tickers[L.order[i]]);
      for (std::size_t j = 0; j < N; ++j) {
        c0[i][j] = corr(i, j);
        c1[i][j] = corr(L.order[i], L.order[j]);
      }
    }
    // Walk-forward backtest: weights re-estimated every `rebalance` days on the past window.
    std::vector<std::vector<double>> daily(4);
    for (std::size_t t = window + 1; t + 1 < T; t += rebalance) {
      const Matrix c = afml::covarianceMatrix(windowMatrix(t - window, t));
      const auto lk = afml::clusterAssets(afml::correlationFromCovariance(c));
      const std::vector<std::vector<double>> ws = {afml::hierarchicalRiskParity(c, lk.order), afml::inverseVarianceWeights(c),
                                                   afml::minimumVarianceWeights(c), std::vector<double>(N, 1.0 / N)};
      for (std::size_t s = t; s < std::min(T, t + rebalance); ++s)
        for (std::size_t m = 0; m < 4; ++m) {
          double r = 0;
          for (std::size_t i = 0; i < N; ++i) r += ws[m][i] * (std::isfinite(ret(s, i)) ? ret(s, i) : 0.0);
          daily[m].push_back(r);
        }
    }
    const char* names[] = {"HRP", "inverse variance", "minimum variance", "equal weight"};
    val backtest = val::array();
    for (std::size_t m = 0; m < 4; ++m) {
      val o = val::object();
      o.set("name", std::string(names[m]));
      o.set("metrics", metricsToJs(evaluatePerformance(daily[m])));
      o.set("equity", arr(equityCurve(daily[m])));
      backtest.call<void>("push", o);
    }
    afml::AllocationTrial trial;
    trial.trials = std::min<std::size_t>(500, count(P, "trials", 100));
    trial.assets = std::clamp<std::size_t>(count(P, "simAssets", 10), 3, 50);
    const auto cmp = afml::compareAllocations(trial);
    val mc = val::object();
    mc.set("hrp", arr(cmp.hrpVariance));
    mc.set("ivp", arr(cmp.ivpVariance));
    mc.set("minVar", arr(cmp.minVarVariance));
    mc.set("longOnly", arr(cmp.longOnlyMinVarVariance));
    mc.set("hrpMaxWeight", cmp.hrpMaxWeight);
    mc.set("ivpMaxWeight", cmp.ivpMaxWeight);
    mc.set("minVarMaxWeight", cmp.minVarMaxWeight);
    mc.set("longOnlyMaxWeight", cmp.longOnlyMaxWeight);
    mc.set("minVarGross", cmp.minVarGross);
    mc.set("trials", static_cast<double>(trial.trials));
    mc.set("assets", static_cast<double>(trial.assets));
    val out = val::object();
    out.set("tickers", strings(d.tickers));
    out.set("orderedTickers", strings(orderedTickers));
    out.set("correlation", arrOfArr(c0));
    out.set("orderedCorrelation", arrOfArr(c1));
    out.set("hrp", arr(hrp));
    out.set("ivp", arr(ivp));
    out.set("minVar", arr(mv));
    std::vector<double> heights(L.height.begin(), L.height.end());
    out.set("linkageHeights", arr(heights));
    out.set("variances", arr({afml::portfolioVariance(cov, hrp) * 252, afml::portfolioVariance(cov, ivp) * 252,
                              afml::portfolioVariance(cov, mv) * 252}));
    out.set("backtest", backtest);
    out.set("backtestDates", dateRange(d, window, window + daily[0].size() + 1));
    out.set("monteCarlo", mc);
    out.set("window", static_cast<double>(window));
    out.set("rebalance", static_cast<double>(rebalance));
    return out;
  });
}

// The candidate book of an analysis. When the page ran the candidate-backtest kernel (on
// WebGPU or the emulated GPU) for this spec it passes the read-back as spec.kernelBook, and
// the book is built from it; otherwise the candidates are backtested here.
std::shared_ptr<const CandidateBook> candidateBook(const val& spec, const Env& env, const Predictions& p, bool& fromKernels) {
  fromKernels = has(spec, "kernelBook");
  if (fromKernels)
    return std::make_shared<CandidateBook>(p.set->models, env.exp.strategies, p.set->nextReturns, env.exp.costBps,
                                           emscripten::convertJSArrayToNumberVector<float>(spec["kernelBook"]));
  return std::make_shared<CandidateBook>(p.set->models, env.exp.strategies, p.set->nextReturns, env.exp.costBps);
}

// Backtest overfitting of the candidate pool and the self-adaptive strategy.
val afmlOverfitting(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const Predictions p = predictions(spec, env);
    bool kernels = false;
    const auto bookPtr = candidateBook(spec, env, p, kernels);
    const CandidateBook& book = *bookPtr;
    const std::size_t evalFrom = std::min(env.exp.selector.lookback, book.days() - 1);
    const AdaptiveResult a = runSelector(book, env.exp.selector, evalFrom);
    const std::size_t blocks = std::clamp<std::size_t>(count(spec["overfitting"], "blocks", 16), 2, 20) / 2 * 2;
    const auto rep = afml::assessOverfitting(book, a.net, evalFrom, blocks);
    auto assessment = [](const afml::StrategyAssessment& s) {
      val o = val::object();
      o.set("sharpe", s.sharpe);
      o.set("annualSharpe", s.annualSharpe);
      o.set("skew", s.skew);
      o.set("kurtosis", s.kurt);
      o.set("psr", s.psr);
      o.set("dsr", s.dsr);
      o.set("maxDrawdown", s.drawdown.maxDrawdown);
      o.set("drawdown95", s.drawdown.drawdown95);
      o.set("longestUnderWater", static_cast<double>(s.drawdown.longestUnderWater));
      o.set("concentration", s.concentration);
      return o;
    };
    val out = val::object();
    out.set("trials", static_cast<double>(rep.trials));
    out.set("days", static_cast<double>(rep.days));
    out.set("trialVariance", rep.trialVariance);
    out.set("expectedMaxSharpe", rep.expectedMaxSharpe);
    out.set("candidateSharpe", arr(rep.candidateSharpe));
    const std::size_t S = env.exp.strategies.size();
    out.set("bestFixedLabel", p.set->models[rep.bestFixed / S].name + " · " + env.exp.strategies[rep.bestFixed % S].label());
    out.set("best", assessment(rep.best));
    out.set("adaptive", assessment(rep.adaptive));
    out.set("adaptiveLabel", env.exp.selector.label());
    val pbo = val::object();
    pbo.set("pbo", rep.pbo.pbo);
    pbo.set("combinations", static_cast<double>(rep.pbo.combinations));
    pbo.set("logits", arr(rep.pbo.logits));
    pbo.set("inSample", arr(rep.pbo.inSampleSharpe));
    pbo.set("outOfSample", arr(rep.pbo.outOfSampleSharpe));
    pbo.set("probabilityOfLoss", rep.pbo.probabilityOfLoss);
    pbo.set("degradationSlope", rep.pbo.degradationSlope);
    pbo.set("blocks", static_cast<double>(blocks));
    out.set("pbo", pbo);
    // Expected maximum Sharpe ratio of unskilled trials against their number.
    std::vector<double> n, e;
    for (double k : {1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0}) {
      n.push_back(k);
      e.push_back(afml::expectedMaxSharpe(k, rep.trialVariance) * std::sqrt(kTradingDaysPerYear));
    }
    out.set("trialCounts", arr(n));
    out.set("expectedMaxAnnual", arr(e));
    // Bet size as a function of the predicted probability.
    std::vector<double> prob, size, step;
    for (int k = 1; k < 100; ++k) {
      prob.push_back(k / 100.0);
      size.push_back(afml::betSize(k / 100.0));
      step.push_back(afml::discretizeBet(size.back(), 0.1));
    }
    out.set("betProbability", arr(prob));
    out.set("betSize", arr(size));
    out.set("betDiscrete", arr(step));
    out.set("trainMs", p.trainMs);
    out.set("cachedPredictions", p.cached);
    out.set("candidateBacktests", std::string(kernels ? "kernels" : "cpu"));
    return out;
  });
}

// ------------------------------------------------- hedging and algorithmic trading

val seriesToJs(const std::string& name, const std::vector<double>& daily) {
  val o = val::object();
  o.set("name", name);
  o.set("metrics", metricsToJs(evaluatePerformance(daily)));
  o.set("equity", arr(equityCurve(daily)));
  o.set("drawdown", arr(drawdownCurve(daily)));
  return o;
}

// Equal-weight average of the finite entries of row t.
double rowMean(const Panel& p, std::size_t t) {
  double s = 0;
  std::size_t n = 0;
  for (std::size_t i = 0; i < p.assets(); ++i)
    if (std::isfinite(p(t, i))) {
      s += p(t, i);
      ++n;
    }
  return n ? s / static_cast<double>(n) : 0.0;
}

// Equal-weight market: value t is the return from date t to t + 1.
std::vector<double> equalWeightReturns(const MarketData& d) { return algo::equalWeightMarket(d); }

double correlationOrZero(const std::vector<double>& a, const std::vector<double>& b) {
  const double c = correlation(a, b);
  return std::isfinite(c) ? c : 0.0;
}

// Beta hedging, volatility targeting and Kelly sizing of the self-adaptive strategy.
val hedgeOverlays(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const Predictions p = predictions(spec, env);
    const MarketData& d = *market(spec, env);
    bool kernels = false;
    const auto bookPtr = candidateBook(spec, env, p, kernels);
    const CandidateBook& book = *bookPtr;
    const std::size_t evalFrom = std::min(env.exp.selector.lookback, book.days() - 1);
    const AdaptiveResult a = runSelector(book, env.exp.selector, evalFrom);
    const val H = spec["hedging"];
    const std::size_t betaWindow = std::max<std::size_t>(10, count(H, "betaWindow", 63));
    const double delta = num(H, "delta", 1e-4), hedgeCost = num(H, "hedgeCostBps", 2.0);
    const double targetVol = num(H, "targetVol", 0.10), volSpan = num(H, "volSpan", 36.0), maxLev = num(H, "maxLeverage", 3.0);
    const double kellyFraction = num(H, "kellyFraction", 0.5);
    const std::size_t kellyWindow = std::max<std::size_t>(20, count(H, "kellyWindow", 126));
    if (!(targetVol > 0) || !(maxLev > 0) || !(volSpan >= 2) || !(kellyFraction > 0) || !(delta > 0 && delta < 1))
      throw std::invalid_argument("hedging: positive target volatility, leverage cap and Kelly fraction, span >= 2, 0 < delta < 1");
    const std::size_t first = book.start() + evalFrom;
    const std::vector<double>& r = a.net;
    std::vector<double> m(r.size());
    for (std::size_t k = 0; k < r.size(); ++k) m[k] = rowMean(p.set->nextReturns, first + k);
    // Observation noise of the Kalman regression: the strategy's residual variance, roughly.
    const double obs = std::max(1e-10, std::pow(stdev(r), 2));
    const auto kal = hedge::kalmanRegression(r, m, delta, obs);
    const auto rolling = hedge::rollingBeta(r, m, betaWindow);
    const auto rollHedged = hedge::applyHedge(r, m, rolling, hedgeCost);
    const auto kalHedged = hedge::applyHedge(r, m, kal.beta, hedgeCost);
    std::vector<double> volLev, kellyLev, comboLev;
    const auto volTargeted = hedge::volatilityTarget(r, targetVol, volSpan, maxLev, &volLev);
    const auto kelly = hedge::kellyScale(r, kellyWindow, kellyFraction, maxLev, &kellyLev);
    const auto combo = hedge::volatilityTarget(kalHedged, targetVol, volSpan, maxLev, &comboLev);
    const double inHindsight = hedge::minimumVarianceHedgeRatio(r, m);
    // Every overlay is compared over the same days: after the longest warm-up (beta window,
    // Kelly window), so none of them is credited or blamed for days it sat out.
    const std::size_t live = std::max(betaWindow, kellyWindow);
    if (live + 20 >= r.size()) throw std::invalid_argument("hedging: the beta and Kelly windows leave too few evaluation days");
    auto tail = [&](const std::vector<double>& v) { return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(live), v.end()); };
    const std::vector<double> mt = tail(m);
    val series = val::array();
    const std::vector<std::pair<std::string, const std::vector<double>*>> all = {
        {"self-adaptive (unhedged)", &r},       {"rolling-beta hedge", &rollHedged}, {"Kalman-beta hedge", &kalHedged},
        {"volatility target", &volTargeted},     {"fractional Kelly", &kelly},        {"Kalman hedge + vol target", &combo},
        {"equal-weight market", &m}};
    val corr = val::array();
    for (const auto& [name, s] : all) {
      const auto t = tail(*s);
      series.call<void>("push", seriesToJs(name, t));
      corr.call<void>("push", correlationOrZero(t, mt));
    }
    val out = val::object();
    out.set("series", series);
    out.set("marketCorrelation", corr);
    out.set("rollingBeta", arr(tail(rolling)));
    out.set("kalmanBeta", arr(tail(kal.beta)));
    out.set("hindsightBeta", inHindsight);
    out.set("volLeverage", arr(tail(volLev)));
    out.set("kellyLeverage", arr(tail(kellyLev)));
    out.set("dates", dateRange(d, first + live, first + r.size() + 1));
    out.set("warmup", static_cast<double>(live));
    out.set("selector", env.exp.selector.label());
    out.set("trainMs", p.trainMs);
    out.set("cachedPredictions", p.cached);
    out.set("candidateBacktests", std::string(kernels ? "kernels" : "cpu"));
    return out;
  });
}

// Delta hedging by simulation, and protective put / collar overlays on the market index.
val hedgeOptions(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const MarketData& d = *market(spec, env);
    const val O = spec["options"];
    hedge::DeltaHedgeSpec h;
    h.vol = num(O, "vol", 0.2);
    h.impliedVol = num(O, "impliedVol", h.vol);
    h.years = num(O, "years", 0.25);
    h.strike = h.spot * num(O, "strike", 1.0);
    h.costBps = num(O, "costBps", 0.0);
    h.paths = std::clamp<std::size_t>(count(O, "paths", 2000), 100, 20000);
    h.seed = static_cast<std::uint64_t>(num(O, "seed", 13));
    if (!(h.vol > 0) || !(h.impliedVol > 0) || !(h.years > 0 && h.years <= 2) || !(h.strike > 0) || !(h.costBps >= 0))
      throw std::invalid_argument("options: positive volatilities and strike, 0 < years <= 2, non-negative cost");
    const std::size_t perDay = h.stepsPerYear / 252;
    val freq = val::array();
    val hist = val::object();
    for (std::size_t every : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16}, std::size_t{40}, std::size_t{80}}) {
      const auto res = hedge::simulateDeltaHedge(h, every);
      val o = val::object();
      o.set("perDay", static_cast<double>(perDay) / static_cast<double>(every));
      o.set("rebalances", static_cast<double>(res.rebalances));
      o.set("mean", res.mean);
      o.set("sd", res.sd);
      o.set("q05", quantile(res.pnl, 0.05));
      freq.call<void>("push", o);
      if (every == perDay) hist.set("daily", arr(res.pnl));
      if (every == perDay * 5) hist.set("weekly", arr(res.pnl));
      hist.set("premium", res.premium);
    }
    hedge::OptionOverlaySpec ov;
    ov.tenorDays = std::max<std::size_t>(5, count(O, "tenorDays", 21));
    ov.putMoneyness = num(O, "putMoneyness", 0.95);
    ov.callMoneyness = num(O, "callMoneyness", 1.05);
    ov.volPremium = num(O, "volPremium", 0.02);
    if (!(ov.putMoneyness > 0 && ov.putMoneyness <= 1.2) || !(ov.callMoneyness >= 0.8)) throw std::invalid_argument("options: put moneyness in (0, 1.2], call moneyness >= 0.8");
    const auto mr = equalWeightReturns(d);
    std::vector<double> price{100.0};
    for (double x : mr) price.push_back(price.back() * (1 + x));
    const auto o = hedge::optionOverlay(price, ov);
    val overlays = val::array();
    overlays.call<void>("push", seriesToJs("equal-weight index", o.unhedged));
    overlays.call<void>("push", seriesToJs("protective put", o.protectivePut));
    overlays.call<void>("push", seriesToJs("collar", o.collar));
    val out = val::object();
    out.set("frequencies", freq);
    out.set("histograms", hist);
    out.set("premium", hedge::blackScholes(h.spot, h.strike, h.years, h.rate, h.impliedVol, true));
    out.set("overlays", overlays);
    out.set("averagePutCost", o.averagePutCost);
    out.set("averageCallIncome", o.averageCallIncome);
    out.set("rolls", static_cast<double>(o.rolls));
    out.set("dates", dateRange(d, 0, d.numDates()));
    // Payoff of the overlays at expiry against the index move (premia at the average cost).
    std::vector<double> move, put, collar;
    for (int k = -20; k <= 20; ++k) {
      const double x = k / 100.0, s = 1 + x;
      move.push_back(x);
      put.push_back(x + std::max(0.0, ov.putMoneyness - s) - o.averagePutCost);
      collar.push_back(x + std::max(0.0, ov.putMoneyness - s) - std::max(0.0, s - ov.callMoneyness) - o.averagePutCost + o.averageCallIncome);
    }
    out.set("payoffMove", arr(move));
    out.set("payoffPut", arr(put));
    out.set("payoffCollar", arr(collar));
    return out;
  });
}

val cointegrationToJs(const algo::Cointegration& c) {
  val o = val::object();
  o.set("hedgeRatio", c.hedgeRatio);
  o.set("intercept", c.intercept);
  o.set("adf", c.adf);
  o.set("critical5", algo::Cointegration::critical5);
  o.set("halfLife", c.halfLife);
  o.set("cointegrated", c.cointegrated());
  return o;
}

algo::PairsSpec parsePairs(const val& P) {
  algo::PairsSpec s;
  s.delta = num(P, "delta", s.delta);
  s.observationVariance = num(P, "observationVariance", s.observationVariance);
  s.entryZ = num(P, "entryZ", s.entryZ);
  s.exitZ = num(P, "exitZ", s.exitZ);
  s.costBps = num(P, "costBps", s.costBps);
  if (!(s.delta > 0 && s.delta < 1) || !(s.observationVariance > 0) || !(s.entryZ > s.exitZ) || !(s.costBps >= 0))
    throw std::invalid_argument("pairs: 0 < delta < 1, positive observation variance, entry z above exit z");
  return s;
}

val pairsToJs(const algo::PairsResult& r) {
  val o = val::object();
  o.set("metrics", metricsToJs(evaluatePerformance(r.returns)));
  o.set("equity", arr(equityCurve(r.returns)));
  o.set("beta", arr(r.beta));
  o.set("zscore", arr(r.zscore));
  std::vector<double> pos(r.position.begin(), r.position.end());
  o.set("position", arr(pos));
  o.set("trades", static_cast<double>(r.trades));
  return o;
}

// Cointegration and Kalman-filter pairs trading.
val algoPairs(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const MarketData& d = *market(spec, env);
    const val P = spec["pairs"];
    const auto ps = parsePairs(P);
    algo::PairSpec g;
    g.days = std::clamp<std::size_t>(count(P, "days", 1000), 100, 20000);
    g.beta = num(P, "beta", 1.5);
    g.halfLifeDays = num(P, "halfLife", 10.0);
    g.spreadVol = num(P, "spreadVol", 0.01);
    g.seed = static_cast<std::uint64_t>(num(P, "seed", 21));
    if (!(g.halfLifeDays > 0) || !(g.spreadVol > 0)) throw std::invalid_argument("pairs: positive half-life and spread volatility");
    std::vector<double> y, x;
    algo::generatePair(g, y, x);
    val synth = val::object();
    synth.set("y", arr(y));
    synth.set("x", arr(x));
    synth.set("cointegration", cointegrationToJs(algo::engleGranger(y, x)));
    synth.set("strategy", pairsToJs(algo::kalmanPairs(y, x, ps)));
    // Universe scan on the first half; the best pairs are traded on the second half.
    const std::size_t T = d.numDates(), N = d.numAssets(), half = T / 2;
    if (half < 60) throw std::invalid_argument("pairs: the universe scan needs at least 120 days");
    struct Scan {
      std::size_t a, b;
      algo::Cointegration c;
    };
    std::vector<Scan> scans;
    std::vector<std::vector<double>> px(N);
    for (std::size_t i = 0; i < N; ++i) px[i] = d.close.series(i);
    for (std::size_t a = 0; a < N; ++a)
      for (std::size_t b = a + 1; b < N; ++b) {
        const std::vector<double> ya(px[a].begin(), px[a].begin() + static_cast<std::ptrdiff_t>(half)),
            xb(px[b].begin(), px[b].begin() + static_cast<std::ptrdiff_t>(half));
        scans.push_back({a, b, algo::engleGranger(ya, xb)});
      }
    std::sort(scans.begin(), scans.end(), [](const Scan& l, const Scan& r) { return l.c.adf < r.c.adf; });
    std::size_t found = 0;
    for (const auto& s : scans) found += s.c.cointegrated();
    const std::size_t top = std::min<std::size_t>(5, scans.size());
    val table = val::array();
    std::vector<double> oosAll;
    for (std::size_t k = 0; k < top; ++k) {
      const auto& s = scans[k];
      const std::vector<double> yb(px[s.a].begin() + static_cast<std::ptrdiff_t>(half), px[s.a].end()),
          xb(px[s.b].begin() + static_cast<std::ptrdiff_t>(half), px[s.b].end());
      const auto res = algo::kalmanPairs(yb, xb, ps);
      const auto oos = algo::engleGranger(yb, xb);
      val o = val::object();
      o.set("pair", d.tickers[s.a] + " / " + d.tickers[s.b]);
      o.set("inSample", cointegrationToJs(s.c));
      o.set("outOfSample", cointegrationToJs(oos));
      o.set("metrics", metricsToJs(evaluatePerformance(res.returns)));
      o.set("equity", arr(equityCurve(res.returns)));
      o.set("trades", static_cast<double>(res.trades));
      table.call<void>("push", o);
    }
    std::vector<double> adf;
    for (const auto& s : scans) adf.push_back(s.c.adf);
    val out = val::object();
    out.set("synthetic", synth);
    out.set("scan", table);
    out.set("scanAdf", arr(adf));
    out.set("pairsTested", static_cast<double>(scans.size()));
    out.set("pairsCointegrated", static_cast<double>(found));
    out.set("scanDays", static_cast<double>(half));
    out.set("tradeDates", dateRange(d, half, T));
    return out;
  });
}

// Carver-style trend following on the universe, with static and adaptive rule weights.
val algoTrend(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const MarketData& d = *market(spec, env);
    const val R = spec["trend"];
    algo::TrendSpec s;
    s.targetVol = num(R, "targetVol", s.targetVol);
    s.volSpan = num(R, "volSpan", s.volSpan);
    s.buffer = num(R, "buffer", s.buffer);
    s.costBps = num(R, "costBps", s.costBps);
    s.reweightEvery = std::max<std::size_t>(1, count(R, "reweightEvery", s.reweightEvery));
    s.reweightWindow = std::max<std::size_t>(20, count(R, "reweightWindow", s.reweightWindow));
    s.longOnly = flag(R, "longOnly", false);
    if (!(s.targetVol > 0) || !(s.volSpan >= 2) || !(s.buffer >= 0) || !(s.costBps >= 0))
      throw std::invalid_argument("trend: positive volatility target, span >= 2, non-negative buffer and cost");
    if (d.numDates() <= s.warmup + 30) throw std::invalid_argument("trend: needs more than " + std::to_string(s.warmup + 30) + " days");
    algo::TrendSpec fixed = s;
    fixed.adaptiveWeights = false;
    const auto adaptive = algo::trendFollowing(d.close, s), stat = algo::trendFollowing(d.close, fixed);
    const auto mr = equalWeightReturns(d);
    const std::vector<double> bench(mr.begin() + static_cast<std::ptrdiff_t>(s.warmup), mr.end());
    val series = val::array();
    series.call<void>("push", seriesToJs("adaptive rule weights", adaptive.returns));
    series.call<void>("push", seriesToJs("equal rule weights", stat.returns));
    series.call<void>("push", seriesToJs("equal-weight market", bench));
    val rules = val::array();
    for (std::size_t r = 0; r < s.rules.size(); ++r) {
      val o = val::object();
      o.set("name", "EWMAC " + std::to_string(s.rules[r].first) + "/" + std::to_string(s.rules[r].second));
      o.set("metrics", metricsToJs(evaluatePerformance(adaptive.ruleReturns[r])));
      o.set("weights", arr(adaptive.weights[r]));
      o.set("scalar", adaptive.forecastScalars[r]);
      rules.call<void>("push", o);
    }
    val out = val::object();
    out.set("series", series);
    out.set("rules", rules);
    out.set("grossLeverage", arr(adaptive.grossLeverage));
    out.set("turnover", arr(adaptive.turnover));
    out.set("idm", arr(adaptive.idm));
    out.set("forecast", arr(adaptive.combinedForecastMean));
    out.set("dates", dateRange(d, s.warmup, d.numDates()));
    out.set("regime", arr(regimeSlice(d, s.warmup + 1, d.numDates())));
    out.set("regimeNames", regimeNames(d));
    return out;
  });
}

// Hidden Markov regimes of the market and regime-switched exposure.
val algoRegimes(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const MarketData& d = *market(spec, env);
    const val G = spec["regimes"];
    algo::RegimeSwitchSpec rs;
    rs.states = std::clamp<std::size_t>(count(G, "states", 2), 2, 4);
    rs.window = std::max<std::size_t>(60, count(G, "window", 504));
    rs.refitEvery = std::max<std::size_t>(1, count(G, "refitEvery", 63));
    rs.threshold = num(G, "threshold", 0.5);
    rs.riskOffExposure = num(G, "riskOffExposure", 0.0);
    rs.costBps = num(G, "costBps", 5.0);
    const auto r = equalWeightReturns(d);
    if (r.size() <= rs.window + 20) throw std::invalid_argument("regimes: the estimation window is longer than the data");
    const auto model = algo::fitHmm(r, rs.states);
    const auto filtered = algo::filterHmm(model, r);
    std::vector<std::vector<double>> probs(rs.states);
    for (const auto& row : filtered)
      for (std::size_t k = 0; k < rs.states; ++k) probs[k].push_back(row[k]);
    // Accuracy against the generator's regimes: the most volatile HMM state against the
    // most volatile true regime (only for the synthetic market).
    double accuracy = -1;
    if (str(spec, "csv", "").empty() && !d.regime.empty()) {
      const auto regs = parseMarket(spec["market"]).regimes;
      int volatileRegime = 0;
      for (std::size_t k = 1; k < regs.size(); ++k)
        if (regs[k].volatility > regs[static_cast<std::size_t>(volatileRegime)].volatility) volatileRegime = static_cast<int>(k);
      std::size_t hit = 0;
      for (std::size_t t = 0; t < r.size(); ++t) hit += (filtered[t].back() > 0.5) == (d.regime[t + 1] == volatileRegime);
      accuracy = static_cast<double>(hit) / static_cast<double>(r.size());
    }
    const auto sw = algo::regimeSwitch(r, rs);
    const std::vector<double> bench(r.begin() + static_cast<std::ptrdiff_t>(sw.start), r.end());
    // Volatility targeting as the continuous alternative to switching.
    const auto vt = hedge::volatilityTarget(bench, num(G, "targetVol", 0.12), 36.0, 2.0);
    val series = val::array();
    series.call<void>("push", seriesToJs("equal-weight market", bench));
    series.call<void>("push", seriesToJs("HMM regime switch", sw.returns));
    series.call<void>("push", seriesToJs("volatility target", vt));
    val m = val::object();
    m.set("mean", arr(model.mean));
    m.set("sd", arr(model.sd));
    m.set("transition", arrOfArr(model.transition));
    m.set("logLikelihood", model.logLikelihood);
    m.set("iterations", static_cast<double>(model.iterations));
    val out = val::object();
    out.set("model", m);
    out.set("probabilities", arrOfArr(probs));
    out.set("returns", arr(r));
    out.set("regime", arr(regimeSlice(d, 1, d.numDates())));
    out.set("regimeNames", regimeNames(d));
    out.set("accuracy", accuracy);
    out.set("series", series);
    out.set("exposure", arr(sw.exposure));
    out.set("switchProbability", arr(sw.highVolProbability));
    out.set("dates", dateRange(d, 1, d.numDates()));
    out.set("switchStart", static_cast<double>(sw.start));
    return out;
  });
}

// Almgren-Chriss optimal execution.
val algoExecution(val spec) {
  return guarded([&] {
    const val X = spec["execution"];
    algo::ExecutionSpec s;
    s.shares = num(X, "shares", s.shares);
    s.price = num(X, "price", s.price);
    s.horizonDays = num(X, "horizonDays", s.horizonDays);
    s.periods = std::clamp<std::size_t>(count(X, "periods", 25), 1, 500);
    s.sigma = num(X, "sigma", s.sigma);
    s.epsilon = num(X, "epsilon", s.epsilon);
    s.eta = num(X, "eta", s.eta);
    s.gamma = num(X, "gamma", s.gamma);
    s.riskAversion = num(X, "riskAversion", s.riskAversion);
    const std::size_t paths = std::clamp<std::size_t>(count(X, "paths", 5000), 100, 50000);
    if (!(s.sigma >= 0) || !(s.epsilon >= 0) || !(s.gamma >= 0) || !(s.price > 0)) throw std::invalid_argument("execution: non-negative volatility and impact, positive price");
    algo::ExecutionSpec twapSpec = s, urgent = s;
    twapSpec.riskAversion = 0.0;
    urgent.riskAversion = s.riskAversion * 10;
    const auto twap = algo::almgrenChriss(twapSpec), opt = algo::almgrenChriss(s), fast = algo::almgrenChriss(urgent);
    std::vector<double> immediate(s.periods + 1, 0.0);
    immediate[0] = s.shares;
    double ie, iv;
    algo::scheduleCostVariance(s, immediate, ie, iv);
    auto planJs = [&](const std::string& name, const std::vector<double>& holdings, double e, double v, double kappa, std::uint64_t seed) {
      const auto sim = algo::simulateShortfall(s, holdings, paths, seed);
      val o = val::object();
      o.set("name", name);
      o.set("holdings", arr(holdings));
      o.set("expectedCost", e);
      o.set("sd", std::sqrt(v));
      o.set("kappa", kappa);
      o.set("simMean", mean(sim));
      o.set("simSd", stdev(sim));
      o.set("simQ95", quantile(sim, 0.95));
      o.set("shortfall", arr(sim));
      return o;
    };
    val plans = val::array();
    plans.call<void>("push", planJs("TWAP (λ = 0)", twap.holdings, twap.expectedCost, twap.variance, 0.0, 1));
    plans.call<void>("push", planJs("Almgren-Chriss (λ)", opt.holdings, opt.expectedCost, opt.variance, opt.kappa, 1));
    plans.call<void>("push", planJs("Almgren-Chriss (10 λ)", fast.holdings, fast.expectedCost, fast.variance, fast.kappa, 1));
    plans.call<void>("push", planJs("immediate", immediate, ie, iv, 0.0, 1));
    const double lam = s.riskAversion > 0 ? s.riskAversion : 1e-6;
    const auto fr = algo::efficientFrontier(s, lam / 1000, lam * 1000, 40);
    std::vector<double> fl, fc, fs;
    for (const auto& f : fr) {
      fl.push_back(f.riskAversion);
      fc.push_back(f.expectedCost);
      fs.push_back(f.sd);
    }
    std::vector<double> times;
    for (std::size_t j = 0; j <= s.periods; ++j) times.push_back(s.horizonDays * static_cast<double>(j) / static_cast<double>(s.periods));
    val out = val::object();
    out.set("plans", plans);
    out.set("times", arr(times));
    out.set("frontierLambda", arr(fl));
    out.set("frontierCost", arr(fc));
    out.set("frontierSd", arr(fs));
    out.set("kappa", opt.kappa);
    out.set("halfLifeDays", opt.halfLifeDays);
    out.set("notional", s.shares * s.price);
    out.set("paths", static_cast<double>(paths));
    return out;
  });
}

// Every approach of the three books on the same days, and self-adaptive allocation across them.
val strategyTournament(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const Predictions p = predictions(spec, env);
    const MarketData& d = *market(spec, env);
    const val Q = spec["tournament"];
    algo::TournamentSpec ts;
    algo::AllocationSpec& as = ts.allocation;
    as.lookback = std::max<std::size_t>(5, count(Q, "lookback", as.lookback));
    as.rebalanceEvery = std::max<std::size_t>(1, count(Q, "rebalanceEvery", as.rebalanceEvery));
    as.topN = std::max<std::size_t>(1, count(Q, "topN", 1));
    as.costBps = num(Q, "costBps", as.costBps);
    as.eta = num(Q, "eta", as.eta);
    if (has(Q, "method")) as.method = algo::parseAllocationMethod(str(Q, "method", ""));
    as.allowCash = flag(Q, "allowCash", true);
    if (!(as.costBps >= 0) || !(as.eta > 0)) throw std::invalid_argument("tournament: non-negative cost, positive eta");
    ts.regimes.window = std::min<std::size_t>(ts.regimes.window, d.numDates() / 3);
    bool kernels = false;
    const auto book = candidateBook(spec, env, p, kernels);
    const auto res = algo::runTournament(d, *p.set, env.exp, ts, book.get());
    val sl = val::array();
    std::vector<std::string> names;
    for (const auto& s : res.sleeves) {
      val o = seriesToJs(s.name, s.returns);
      o.set("family", s.family);
      sl.call<void>("push", o);
      names.push_back(s.name);
    }
    val al = val::array();
    for (const auto& a : res.allocators) {
      val o = seriesToJs(algo::allocationName(a.method), a.result.returns);
      o.set("weights", arrOfArr(a.result.weights));
      o.set("cash", arr(a.result.cash));
      o.set("averageTurnover", a.result.averageTurnover);
      o.set("dsr", a.deflatedSharpe);
      al.call<void>("push", o);
    }
    std::vector<std::vector<double>> corr(res.sleeves.size(), std::vector<double>(res.sleeves.size()));
    for (std::size_t i = 0; i < corr.size(); ++i)
      for (std::size_t j = 0; j < corr.size(); ++j) corr[i][j] = i == j ? 1.0 : correlationOrZero(res.sleeves[i].returns, res.sleeves[j].returns);
    const std::size_t n = res.sleeves[0].returns.size();
    val out = val::object();
    out.set("sleeves", sl);
    out.set("allocators", al);
    out.set("names", strings(names));
    out.set("correlation", arrOfArr(corr));
    out.set("dates", dateRange(d, res.start, res.start + n + 1));
    out.set("regime", arr(regimeSlice(d, res.start + 1, res.start + n + 1)));
    out.set("regimeNames", regimeNames(d));
    out.set("chosen", algo::allocationName(as.method));
    out.set("lookback", static_cast<double>(as.lookback));
    out.set("rebalanceEvery", static_cast<double>(as.rebalanceEvery));
    out.set("trainMs", p.trainMs);
    out.set("cachedPredictions", p.cached);
    out.set("candidateBacktests", std::string(kernels ? "kernels" : "cpu"));
    return out;
  });
}
// Composite model (members' forecasts combined three ways) and the composite strategy.
val compositeStrategy(val spec) {
  return guarded([&] {
    const Env env = parseEnv(spec);
    const Predictions p = predictions(spec, env);
    const MarketData& d = *market(spec, env);
    std::vector<ModelPredictions> members;
    for (const auto& m : p.set->models)
      if (m.spec.type != "composite") members.push_back(m);
    CompositeSpec cs = env.exp.composite;
    CompositeMethod chosen = cs.method == CompositeMethod::None ? CompositeMethod::Average : cs.method;
    val mem = val::array();
    auto quality = [](const std::string& name, const ClassificationMetrics& m) {
      val o = val::object();
      o.set("name", name);
      o.set("auc", m.auc);
      o.set("logLoss", m.logLoss);
      o.set("accuracy", m.accuracy);
      return o;
    };
    for (const auto& m : members) mem.call<void>("push", quality(m.name, m.oos));
    val comps = val::array();
    ModelPredictions forecast;
    std::vector<std::string> memberNames;
    if (members.size() > 1) {
      for (auto method : {CompositeMethod::Average, CompositeMethod::Stacked, CompositeMethod::Online}) {
        cs.method = method;
        auto c = compositePredictions(members, p.set->labels, p.set->labelEnds, cs);
        val o = quality(c.model.name, c.model.oos);
        o.set("method", compositeMethodName(method));
        std::vector<std::vector<double>> w(members.size());
        for (std::size_t t = 0; t < c.weights.size(); t += 5)
          for (std::size_t k = 0; k < members.size(); ++k) w[k].push_back(c.weights[t][k]);
        o.set("weights", arrOfArr(w));
        comps.call<void>("push", o);
        memberNames = c.members;
        if (method == chosen) forecast = std::move(c.model);
      }
    } else {
      if (members.empty()) throw std::invalid_argument("composite: the composite model needs member models");
      forecast = members[0];
      memberNames = {members[0].name};
    }
    const val M = spec["multiSignal"];
    algo::CompositeStrategySpec ss;
    algo::MultiSignalSpec ms;
    ms.useMl = flag(M, "useMl", ms.useMl);
    ms.useMomentum = flag(M, "useMomentum", ms.useMomentum);
    ms.useTrailingSharpe = flag(M, "useTrailingSharpe", ms.useTrailingSharpe);
    ms.momentumLookback = count(M, "momentumLookback", ms.momentumLookback);
    ms.momentumSkip = count(M, "momentumSkip", ms.momentumSkip);
    ms.sharpeLookback = count(M, "sharpeLookback", ms.sharpeLookback);
    if (has(M, "weighting")) ms.weighting = algo::parseSignalWeighting(str(M, "weighting", "adaptive"));
    ms.icLookback = count(M, "icLookback", ms.icLookback);
    ms.eta = num(M, "eta", ms.eta);
    ms.holdings = count(M, "holdings", ms.holdings);
    ms.rebalanceEvery = count(M, "rebalanceEvery", ms.rebalanceEvery);
    ms.targetVol = num(M, "targetVol", ms.targetVol);
    ms.maxLeverage = num(M, "maxLeverage", ms.maxLeverage);
    ms.regimeGate = flag(M, "regimeGate", ms.regimeGate);
    ms.costBps = env.exp.costBps;
    ms.regimes.window = std::min<std::size_t>(ms.regimes.window, d.numDates() / 3);
    if (ms.holdings < 1 || ms.rebalanceEvery < 1 || !(ms.targetVol >= 0) || !(ms.maxLeverage > 0))
      throw std::invalid_argument("stock-selection books: holdings and rebalance >= 1, target volatility >= 0, leverage > 0");
    ss.books.clear();
    if (flag(M, "blendBook", true)) ss.books.push_back(ms);
    if (flag(M, "separateBooks", false))
      for (int k = 0; k < 3; ++k) {
        algo::MultiSignalSpec one = ms;
        one.useMl = k == 0, one.useMomentum = k == 1, one.useTrailingSharpe = k == 2;
        ss.books.push_back(one);
      }
    ss.marketSleeve = flag(M, "marketSleeve", false);
    const val Q = spec["tournament"];
    algo::AllocationSpec& as = ss.allocation;
    as.lookback = std::max<std::size_t>(5, count(Q, "lookback", as.lookback));
    as.rebalanceEvery = std::max<std::size_t>(1, count(Q, "rebalanceEvery", as.rebalanceEvery));
    as.topN = std::max<std::size_t>(1, count(Q, "topN", 1));
    as.costBps = num(Q, "costBps", as.costBps);
    as.eta = num(Q, "eta", as.eta);
    if (has(Q, "method")) as.method = algo::parseAllocationMethod(str(Q, "method", ""));
    as.allowCash = flag(Q, "allowCash", true);
    if (!(as.costBps >= 0) || !(as.eta > 0)) throw std::invalid_argument("allocation: non-negative cost, positive eta");
    const auto r = algo::runCompositeStrategy(d, forecast, p.set->nextReturns, env.exp, ss);
    const std::size_t n = r.returns[0].size();
    val series = val::array();
    for (std::size_t k = 0; k < r.names.size(); ++k) series.call<void>("push", seriesToJs(r.names[k], r.returns[k]));
    // The library's default: the selector on the separate members, over the same days.
    if (members.size() > 1) {
      const CandidateBook book(members, env.exp.strategies, p.set->nextReturns, env.exp.costBps);
      const auto a = runSelector(book, env.exp.selector);
      const std::size_t first = book.start() + a.evalFrom;  // a.net[0] is earned from this date
      if (r.start >= first && r.start - first + n <= a.net.size()) {
        const auto from = a.net.begin() + static_cast<std::ptrdiff_t>(r.start - first);
        series.call<void>("push", seriesToJs("Self-adaptive selector (separate members)", std::vector<double>(from, from + static_cast<std::ptrdiff_t>(n))));
      }
    }
    val books = val::array();
    for (std::size_t q = 0; q < r.books.size(); ++q) {
      const auto& msr = r.books[q];
      std::vector<std::vector<double>> sw(msr.signals.size());
      for (const auto& row : msr.signalWeights)
        for (std::size_t k = 0; k < row.size(); ++k) sw[k].push_back(row[k]);
      std::vector<std::size_t> order(d.numAssets());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return msr.book[a] > msr.book[b]; });
      val holdings = val::array();
      for (auto i : order) {
        if (!(msr.book[i] > 0)) break;
        val o = val::object();
        o.set("ticker", d.tickers[i]);
        o.set("weight", msr.book[i]);
        o.set("held", msr.holdings[i]);
        o.set("score", i < msr.lastScore.size() ? msr.lastScore[i] : Panel::kMissing);
        holdings.call<void>("push", o);
      }
      val o = val::object();
      o.set("name", r.names[1 + q]);
      o.set("signals", strings(msr.signals));
      o.set("signalWeights", arrOfArr(sw));
      o.set("exposure", arr(msr.exposure));
      o.set("holdings", holdings);
      books.call<void>("push", o);
    }
    const std::size_t lb = as.lookback;
    val out = val::object();
    out.set("members", mem);
    out.set("memberNames", strings(memberNames));
    out.set("composites", comps);
    out.set("chosen", compositeMethodName(chosen));
    out.set("forecast", forecast.name);
    out.set("series", series);
    out.set("weights", arrOfArr(r.weights));
    out.set("cash", arr(r.cash));
    out.set("sleeves", static_cast<double>(r.sleeves));
    out.set("books", books);
    out.set("selectorCandidates", strings(r.selectorCandidates));
    out.set("selectorShare", arr(r.selector.share));
    out.set("selectorSwitches", static_cast<double>(r.selector.switches));
    out.set("start", static_cast<double>(r.start));
    out.set("allocationLookback", static_cast<double>(lb));
    out.set("dates", dateRange(d, r.start, r.start + n + 1));
    out.set("bookDates", dateRange(d, r.start - lb, r.start + n + 1));
    out.set("regime", arr(regimeSlice(d, r.start + 1, r.start + n + 1)));
    out.set("regimeNames", regimeNames(d));
    out.set("trainMs", p.trainMs);
    out.set("cachedPredictions", p.cached);
    return out;
  });
}
}  // namespace

EMSCRIPTEN_BINDINGS(sat) {
  emscripten::function("version", &version);
  emscripten::function("coreDemo", &coreDemo);
  emscripten::function("marketData", &marketData);
  emscripten::function("factors", &factors);
  emscripten::function("labels", &labels);
  emscripten::function("models", &models);
  emscripten::function("strategies", &strategies);
  emscripten::function("adaptive", &adaptive);
  emscripten::function("robustness", &robustness);
  emscripten::function("gpuKernels", &gpuKernels);
  emscripten::function("gpuJobs", &gpuJobs);
  emscripten::function("gpuAnalyse", &gpuAnalyse);
  emscripten::function("gpuEmulate", &gpuEmulate);
  emscripten::function("gpuRunPlan", &gpuRunPlan);
  emscripten::function("afmlBars", &afmlBars);
  emscripten::function("afmlFracDiff", &afmlFracDiff);
  emscripten::function("afmlLabeling", &afmlLabeling);
  emscripten::function("afmlValidation", &afmlValidation);
  emscripten::function("afmlPortfolio", &afmlPortfolio);
  emscripten::function("afmlOverfitting", &afmlOverfitting);
  emscripten::function("hedgeOverlays", &hedgeOverlays);
  emscripten::function("hedgeOptions", &hedgeOptions);
  emscripten::function("algoPairs", &algoPairs);
  emscripten::function("algoTrend", &algoTrend);
  emscripten::function("algoRegimes", &algoRegimes);
  emscripten::function("algoExecution", &algoExecution);
  emscripten::function("strategyTournament", &strategyTournament);
  emscripten::function("compositeStrategy", &compositeStrategy);
}
