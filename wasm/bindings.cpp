// WebAssembly entry points (embind). The page's JavaScript only moves data: it hands the C++
// library the CSV text, and on WebGPU it uploads the plan's tables, dispatches the kernels whose
// WGSL the library supplies, and returns the read-backs. Every number is computed here.
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ofm/kernels.hpp"
#include "ofm/report.hpp"

using emscripten::val;

namespace {

template <class T>
val copyOut(const std::vector<T>& v) {
  return val(emscripten::typed_memory_view(v.size(), v.data())).call<val>("slice");
}

class Session {
 public:
  std::string loadSample() {
    auto s = ofm::syntheticMarket();
    market_ = std::move(s.market), series_ = std::move(s.series);
    return reset();
  }

  std::string loadCsv(const std::string& stocks, const std::string& series) {
    market_ = ofm::parseMarketCsv(stocks);
    series_ = series.empty() ? std::map<std::string, std::vector<double>>{} : ofm::parseSeriesCsv(series, market_.dates);
    return reset();
  }

  /// What the WebGPU host needs: sizes, chunks and the kernel sources.
  val plan() const {
    val o = val::object();
    o.set("T", plan_.T), o.set("N", plan_.N), o.set("K", plan_.K), o.set("H", plan_.H);
    o.set("stride", plan_.stride), o.set("chunks", plan_.numChunks()), o.set("chunkDays", plan_.chunkDays);
    o.set("maxAssets", ofm::kMaxAssets);
    val src = val::object();
    src.set("zscore", ofm::kernels::zscoreSource()), src.set("gram", ofm::kernels::gramSource()), src.set("expect", ofm::kernels::expectSource());
    o.set("sources", src);
    return o;
  }
  val tables() const { return copyOut(plan_.tables); }
  val header(int c) const { return copyOut(plan_.header(static_cast<std::size_t>(c))); }
  int chunkLength(int c) const { return static_cast<int>(plan_.chunkLength(static_cast<std::size_t>(c))); }

  /// The Gram read-back of chunk c, in order: returns v for the expected-return kernel.
  val absorb(int c, val gram) {
    const auto g32 = emscripten::convertJSArrayToNumberVector<float>(gram);
    const std::size_t cd = plan_.chunkLength(static_cast<std::size_t>(c)), c0 = plan_.chunkStart(static_cast<std::size_t>(c));
    if (g32.size() != cd * plan_.stride) throw std::invalid_argument("gram read-back has the wrong size");
    const std::vector<double> g(g32.begin(), g32.end());
    std::vector<float> v(cd * plan_.K, 0.0f);
    valid_.assign(cd, 0);
    for (std::size_t td = 0; td < cd; ++td) {
      const auto vt = forecaster_->absorb(c0 + td, &g[td * plan_.stride]);
      if (vt.empty()) continue;
      valid_[td] = 1;
      for (std::size_t k = 0; k < plan_.K; ++k) v[td * plan_.K + k] = static_cast<float>(vt[k]);
    }
    return copyOut(v);
  }

  /// The expected-return read-back of chunk c.
  void takeExpected(int c, val e) {
    const auto e32 = emscripten::convertJSArrayToNumberVector<float>(e);
    const std::size_t cd = plan_.chunkLength(static_cast<std::size_t>(c)), c0 = plan_.chunkStart(static_cast<std::size_t>(c)), N = plan_.N;
    if (e32.size() != cd * N) throw std::invalid_argument("expected-return read-back has the wrong size");
    const std::size_t TN = plan_.T * N;
    for (std::size_t td = 0; td < cd; ++td) {
      if (!valid_[td]) continue;
      for (std::size_t a = 0; a < N; ++a)
        if (plan_.tables[4 * TN + (c0 + td) * N + a] > 0.5f) forecast_.E(c0 + td, a) = e32[td * N + a];
    }
  }

  /// After the last chunk on WebGPU: the backtest and the report.
  std::string finishGpu(double buyBps, double sellBps, double futBps, double gpuMs, const std::string& adapter) {
    forecast_.mean = forecaster_->mean, forecast_.tstat = forecaster_->tstat, forecast_.premium = forecaster_->premium;
    forecast_.records = forecaster_->records();
    forecast_.kernelMs = gpuMs;
    forecast_.engine = "WebGPU (" + adapter + ")";
    return report(buyBps, sellBps, futBps);
  }

  /// The whole run on the CPU: "emulated" (the kernels in single precision) or "reference".
  std::string runCpu(const std::string& engine, double buyBps, double sellBps, double futBps) {
    forecast_ = ofm::runModel(market_, engine);
    forecast_.engine = engine == "emulated" ? "Emulated GPU (CPU)" : "Reference (double precision)";
    return report(buyBps, sellBps, futBps);
  }

 private:
  std::string reset() {
    plan_ = ofm::compilePlan(market_);
    forecaster_ = std::make_unique<ofm::Forecaster>(plan_);
    forecast_ = ofm::Forecast{};
    forecast_.E = ofm::Panel(plan_.T, plan_.N);
    std::string j = "{\"stocks\":" + std::to_string(market_.N()) + ",\"days\":" + std::to_string(market_.T()) + ",\"from\":\"" +
                    market_.dates.front() + "\",\"to\":\"" + market_.dates.back() + "\",\"series\":[";
    bool first = true;
    for (const auto& [name, v] : series_) j += (first ? "\"" : ",\"") + name + "\"", first = false;
    return j + "]}";
  }

  std::string report(double buyBps, double sellBps, double futBps) {
    ofm::Costs c;
    c.buyBps = buyBps, c.sellBps = sellBps, c.futuresBps = futBps;
    const ofm::Backtest bt = ofm::backtest(market_, forecast_, c, series_);
    return ofm::reportJson(market_, forecast_, bt, c);
  }

  ofm::Market market_;
  std::map<std::string, std::vector<double>> series_;
  ofm::Plan plan_;
  std::unique_ptr<ofm::Forecaster> forecaster_;
  ofm::Forecast forecast_;
  std::vector<char> valid_;
};

std::string version() { return OFM_VERSION; }

}  // namespace

EMSCRIPTEN_BINDINGS(ofm) {
  emscripten::function("version", &version);
  emscripten::class_<Session>("Session")
      .constructor<>()
      .function("loadSample", &Session::loadSample)
      .function("loadCsv", &Session::loadCsv)
      .function("plan", &Session::plan)
      .function("tables", &Session::tables)
      .function("header", &Session::header)
      .function("chunkLength", &Session::chunkLength)
      .function("absorb", &Session::absorb)
      .function("takeExpected", &Session::takeExpected)
      .function("finishGpu", &Session::finishGpu)
      .function("runCpu", &Session::runCpu);
}
