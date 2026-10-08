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
  }
  if (has(spec, "models")) {
    const val m = spec["models"];
    for (std::size_t k = 0; k < m["length"].as<std::size_t>(); ++k) e.models.push_back(parseModel(m[k]));
    if (e.models.empty()) throw std::invalid_argument("select at least one model");
  } else {
    e.models = ExperimentSpec::defaultModels();
  }
  if (has(spec, "walkForward")) {
    const val w = spec["walkForward"];
    e.walkForward.trainWindow = count(w, "trainWindow", e.walkForward.trainWindow);
    e.walkForward.retrainEvery = std::max<std::size_t>(1, count(w, "retrainEvery", e.walkForward.retrainEvery));
    e.walkForward.maxTrainRows = count(w, "maxTrainRows", e.walkForward.maxTrainRows);
    e.walkForward.seed = static_cast<std::uint64_t>(num(w, "seed", 11));
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
                      stringify(spec["label"]) + "|" + stringify(spec["models"]) + "|" + stringify(spec["walkForward"]);
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
    return p;
  }
  const double t0 = nowMs();
  auto set = std::make_shared<PredictionSet>(runPredictions(*p.data, env.exp));
  c.key = env.predictionKey;
  c.set = set;
  c.elapsedMs = nowMs() - t0;
  p.set = set;
  p.trainMs = c.elapsedMs;
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
}
