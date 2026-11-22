// The Lab's GPU back end (--backend webgpu, the WebAssembly build docker/wasm/Dockerfile makes with OFM_WEBGPU), all of it
// here in C++ on WebGPU's C API (webgpu.h): the engine's own loops run as the serial back end's, and the nested Monte
// Carlo's trade kernel (include/strategy/mc_gpu.hpp: every reset's paths and every stop search) runs as one batch on
// the GPU: the adapter and device opened once, the kernel (include/strategy/mc_wgsl.hpp) compiled once, each batch
// uploaded, dispatched, read back. WebGPU answers through callbacks; the engine waits for them (emscripten_sleep under
// Asyncify) and goes on. No GPU, or a batch the GPU cannot run (a buffer over its limits, a validation error, the
// device lost): that batch runs on the CPU, so a run always ends. The page's JavaScript only starts the engine.
//
// The same device trains the adaptive nested Monte Carlo's differential network (include/strategy/diffnet_gpu.hpp):
// diffnetOffload() runs a fit's fused kernel (diffnetFused, include/strategy/diffnet_wgsl.hpp) in one command buffer per
// 64 steps, each one dispatch of one workgroup, then reads the weights back once. Set when the build loads, so the
// adaptive nested Monte Carlo's WebAssembly build (ofm_anmc_gpu.js, which links this file) trains on the GPU; a fit the
// GPU cannot run returns to the kernel's f32 reference on the CPU.
// Compiled only into those builds: the native engine and the plain WebAssembly builds have no such back end.
#if defined(__EMSCRIPTEN__) && defined(OFM_WEBGPU)
#include <emscripten.h>
#include <webgpu/webgpu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "strategy/backend.hpp"
#include "strategy/diffnet_gpu.hpp"
#include "strategy/diffnet_wgsl.hpp"
#include "strategy/mc_gpu.hpp"
#include "strategy/mc_wgsl.hpp"

namespace {

struct Gpu {
  bool tried = false, lost = false;
  WGPUInstance instance = nullptr;
  WGPUAdapter adapter = nullptr;
  WGPUDevice device = nullptr;
  WGPUQueue queue = nullptr;
  WGPUComputePipeline pipeline = nullptr;
  WGPUBindGroupLayout layout = nullptr;
  std::string name, why;
  std::uint64_t maxBinding = 128ull << 20;
  std::size_t batches = 0, fallbacks = 0;
  // the differential network's fused kernel: compiled on its first fit
  bool deviceTried = false, diffTried = false;
  WGPUComputePipeline diffPipeline = nullptr;
  WGPUBindGroupLayout diffLayout = nullptr;
  std::string diffWhy;
  std::size_t fits = 0, fitFallbacks = 0;
  ~Gpu() {
    if (batches || fallbacks || (!why.empty() && !fits && !fitFallbacks))
      std::printf("strategy: webgpu: %zu Monte Carlo batches on %s%s%s\n", batches, name.empty() ? "no GPU" : name.c_str(),
                  fallbacks ? (", " + std::to_string(fallbacks) + " on the CPU").c_str() : "", why.empty() ? "" : (" (" + why + ")").c_str());
    if (fits || fitFallbacks) {
      const std::string w = !diffWhy.empty() ? diffWhy : why;
      std::printf("anmc: webgpu: %zu network fits on %s%s%s\n", fits, name.empty() ? "no GPU" : name.c_str(),
                  fitFallbacks ? (", " + std::to_string(fitFallbacks) + " on the CPU").c_str() : "", w.empty() ? "" : (" (" + w + ")").c_str());
    }
  }
};
Gpu& gpu() {
  static Gpu g;
  return g;
}

// Waits for a callback to set `done` (the browser runs it while the engine sleeps); false after `ms`.
bool waitFor(const volatile bool& done, int ms = 60000) {
  for (int t = 0; !done && t < ms; ++t) emscripten_sleep(1);
  return done;
}

// The adapter and the device, once (both kernels share them).
bool openDevice() {
  Gpu& g = gpu();
  if (g.deviceTried) return g.device && !g.lost;
  g.deviceTried = true;
  g.instance = wgpuCreateInstance(nullptr);
  if (!g.instance) return g.why = "no WebGPU", false;
  struct { volatile bool done = false; WGPUAdapter a = nullptr; } ra;
  WGPURequestAdapterOptions ao = {};
  ao.powerPreference = WGPUPowerPreference_HighPerformance;
  wgpuInstanceRequestAdapter(g.instance, &ao, [](WGPURequestAdapterStatus s, WGPUAdapter a, const char*, void* u) {
    auto* r = static_cast<decltype(ra)*>(u);
    r->a = s == WGPURequestAdapterStatus_Success ? a : nullptr, r->done = true;
  }, &ra);
  if (!waitFor(ra.done) || !ra.a) return g.why = "no GPU adapter", false;
  g.adapter = ra.a;
  WGPUAdapterInfo info = {};
  wgpuAdapterGetInfo(g.adapter, &info);
  for (const char* s : {info.vendor, info.architecture, info.description})
    if (s && *s) g.name += (g.name.empty() ? "" : " ") + std::string(s);
  if (g.name.empty()) g.name = "the GPU";
  WGPUSupportedLimits lim = {};
  if (wgpuAdapterGetLimits(g.adapter, &lim)) g.maxBinding = lim.limits.maxStorageBufferBindingSize;
  WGPURequiredLimits req = {};
  std::memset(&req.limits, 0xFF, sizeof req.limits);  // every limit undefined (the defaults), but the buffers': the adapter's own
  req.limits.maxStorageBufferBindingSize = lim.limits.maxStorageBufferBindingSize;
  req.limits.maxBufferSize = lim.limits.maxBufferSize;
  WGPUDeviceDescriptor dd = {};
  dd.requiredLimits = &req;
  dd.deviceLostCallback = [](WGPUDeviceLostReason, const char*, void*) { gpu().lost = true; };
  struct { volatile bool done = false; WGPUDevice d = nullptr; std::string message; } rd;
  wgpuAdapterRequestDevice(g.adapter, &dd, [](WGPURequestDeviceStatus s, WGPUDevice d, const char* m, void* u) {
    auto* r = static_cast<decltype(rd)*>(u);
    r->d = s == WGPURequestDeviceStatus_Success ? d : nullptr, r->message = m ? m : "", r->done = true;
  }, &rd);
  if (!waitFor(rd.done) || !rd.d) return g.why = "the GPU device could not be opened: " + rd.message.substr(0, 200), false;
  g.device = rd.d;
  g.queue = wgpuDeviceGetQueue(g.device);
  return true;
}

// A kernel's pipeline (WGSL `code`, entry point `entry`) and its bind group layout; false, with `why`: it did not compile.
bool compile(const char* code, const char* entry, WGPUComputePipeline& pipeline, WGPUBindGroupLayout& layout, std::string& why) {
  Gpu& g = gpu();
  WGPUShaderModuleWGSLDescriptor wgsl = {};
  wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
  wgsl.code = code;
  WGPUShaderModuleDescriptor sd = {};
  sd.nextInChain = &wgsl.chain;
  wgpuDevicePushErrorScope(g.device, WGPUErrorFilter_Validation);
  WGPUShaderModule module = wgpuDeviceCreateShaderModule(g.device, &sd);
  WGPUComputePipelineDescriptor pd = {};
  pd.compute.module = module;
  pd.compute.entryPoint = entry;
  pipeline = wgpuDeviceCreateComputePipeline(g.device, &pd);
  struct { volatile bool done = false; bool error = false; std::string message; } es;
  wgpuDevicePopErrorScope(g.device, [](WGPUErrorType t, const char* m, void* u) {
    auto* r = static_cast<decltype(es)*>(u);
    r->error = t != WGPUErrorType_NoError, r->message = m ? m : "", r->done = true;
  }, &es);
  waitFor(es.done);
  wgpuShaderModuleRelease(module);
  if (es.error || !pipeline) return why = "the kernel did not compile: " + es.message.substr(0, 200), pipeline = nullptr, false;
  layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
  return true;
}

// The Monte Carlo's trade kernel, once.
bool open() {
  Gpu& g = gpu();
  if (g.tried) return g.pipeline && !g.lost;
  g.tried = true;
  if (!openDevice()) return false;
  return compile(strat::kMcWgsl, "main", g.pipeline, g.layout, g.why);
}

// The differential network's fused kernel, once.
bool openDiffnet() {
  Gpu& g = gpu();
  if (g.diffTried) return g.diffPipeline && !g.lost;
  g.diffTried = true;
  if (!openDevice()) return false;
  return compile(anmc::kDiffnetFusedWgsl, "diffnetFused", g.diffPipeline, g.diffLayout, g.diffWhy);
}

WGPUBuffer upload(const void* data, std::size_t bytes, WGPUBufferUsageFlags usage) {
  Gpu& g = gpu();
  WGPUBufferDescriptor bd = {};
  bd.size = std::max<std::size_t>(16, (bytes + 3) / 4 * 4);
  bd.usage = usage | WGPUBufferUsage_CopyDst;
  WGPUBuffer b = wgpuDeviceCreateBuffer(g.device, &bd);
  if (bytes) wgpuQueueWriteBuffer(g.queue, b, 0, data, (bytes + 3) / 4 * 4);
  return b;
}

// One batch on the GPU: fills ret and stopped (G * S * P); false: not run (the engine runs it itself).
bool runBatch(const strat::McBatch& b, std::vector<float>& ret, std::vector<std::uint32_t>& stopped) {
  static_assert(sizeof(strat::McGpuShare) == 21 * 4 && sizeof(strat::McGpuSpec) == 8 * 4, "the kernel's layout");
  Gpu& g = gpu();
  if (!open()) return ++g.fallbacks, false;
  const std::size_t n = std::size_t(b.G) * b.S * b.P;
  if (!n || b.table.size() * 4 > g.maxBinding || n * 4 > g.maxBinding) return ++g.fallbacks, false;
  const std::uint32_t prm[8] = {b.P, b.H, b.S, b.G, b.D, 0, 0, 0};
  wgpuDevicePushErrorScope(g.device, WGPUErrorFilter_OutOfMemory);
  wgpuDevicePushErrorScope(g.device, WGPUErrorFilter_Validation);
  WGPUBuffer in[5] = {
      upload(prm, sizeof prm, WGPUBufferUsage_Uniform),
      upload(b.days.data(), b.days.size() * 4, WGPUBufferUsage_Storage),
      upload(b.table.data(), b.table.size() * 4, WGPUBufferUsage_Storage),
      upload(b.shares.data(), b.shares.size() * sizeof(strat::McGpuShare), WGPUBufferUsage_Storage),
      upload(b.specs.data(), b.specs.size() * sizeof(strat::McGpuSpec), WGPUBufferUsage_Storage),
  };
  WGPUBufferDescriptor od = {};
  od.size = n * 4;
  od.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
  WGPUBuffer outRet = wgpuDeviceCreateBuffer(g.device, &od), outStop = wgpuDeviceCreateBuffer(g.device, &od);
  WGPUBufferDescriptor rdd = {};
  rdd.size = n * 8;
  rdd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  WGPUBuffer read = wgpuDeviceCreateBuffer(g.device, &rdd);
  WGPUBindGroupEntry e[7] = {};
  WGPUBuffer all[7] = {in[0], in[1], in[2], in[3], in[4], outRet, outStop};
  for (std::uint32_t k = 0; k < 7; ++k) e[k].binding = k, e[k].buffer = all[k], e[k].size = WGPU_WHOLE_SIZE;
  WGPUBindGroupDescriptor bgd = {};
  bgd.layout = g.layout, bgd.entryCount = 7, bgd.entries = e;
  WGPUBindGroup bind = wgpuDeviceCreateBindGroup(g.device, &bgd);
  const std::uint32_t groups = std::uint32_t((n + 63) / 64), gx = std::min<std::uint32_t>(groups, 65535), gy = (groups + gx - 1) / gx;
  WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g.device, nullptr);
  WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
  wgpuComputePassEncoderSetPipeline(pass, g.pipeline);
  wgpuComputePassEncoderSetBindGroup(pass, 0, bind, 0, nullptr);
  wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
  wgpuComputePassEncoderEnd(pass);
  wgpuCommandEncoderCopyBufferToBuffer(enc, outRet, 0, read, 0, n * 4);
  wgpuCommandEncoderCopyBufferToBuffer(enc, outStop, 0, read, n * 4, n * 4);
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
  wgpuQueueSubmit(g.queue, 1, &cmd);
  // The two error scopes and the readback all asked at once, then one wait for the three answers (each wait yields to
  // the browser through a timer: one wait a batch, not three).
  struct Scope { volatile bool done = false; bool error = false; } s1, s2;
  const auto pop = [](WGPUErrorType t, const char*, void* u) {
    auto* r = static_cast<Scope*>(u);
    r->error = t != WGPUErrorType_NoError, r->done = true;
  };
  wgpuDevicePopErrorScope(g.device, pop, &s1);
  wgpuDevicePopErrorScope(g.device, pop, &s2);
  struct { volatile bool done = false; bool good = false; } mp;
  wgpuBufferMapAsync(read, WGPUMapMode_Read, 0, n * 8, [](WGPUBufferMapAsyncStatus st, void* u) {
    auto* r = static_cast<decltype(mp)*>(u);
    r->good = st == WGPUBufferMapAsyncStatus_Success, r->done = true;
  }, &mp);
  for (int t = 0; !(s1.done && s2.done && mp.done) && t < 60000; ++t) emscripten_sleep(1);
  bool ok = s1.done && s2.done && mp.done && !s1.error && !s2.error && mp.good && !g.lost;
  if (ok) {
    const auto* p = static_cast<const std::uint8_t*>(wgpuBufferGetConstMappedRange(read, 0, n * 8));
    ok = p != nullptr;
    if (ok) {
      ret.resize(n), stopped.resize(n);
      std::memcpy(ret.data(), p, n * 4), std::memcpy(stopped.data(), p + n * 4, n * 4);
    }
  }
  if (mp.done && mp.good) wgpuBufferUnmap(read);
  wgpuCommandBufferRelease(cmd);
  wgpuComputePassEncoderRelease(pass);
  wgpuCommandEncoderRelease(enc);
  wgpuBindGroupRelease(bind);
  for (WGPUBuffer x : {in[0], in[1], in[2], in[3], in[4], outRet, outStop, read}) wgpuBufferDestroy(x), wgpuBufferRelease(x);
  if (ok) ++g.batches;
  else ++g.fallbacks;
  return ok;
}

// One differential network fit (and its predictions) on the GPU: diffnetFused over the plan's steps, 64 steps a command
// buffer (each a dispatch of one workgroup, its step range at its own 256-byte offset of one uniform buffer), so no one
// submission runs long enough for the browser's GPU watchdog; the weights, predictions and step losses read back once.
// False: not run (the kernel's f32 reference runs the fit).
bool runDiffnet(const anmc::DiffnetFusedPlan& p, anmc::DiffnetFusedResult& r) {
  Gpu& g = gpu();
  if (!openDiffnet()) return ++g.fitFallbacks, false;
  const std::size_t P = p.parameters, outF = p.outFloats();
  for (std::size_t bytes : {p.xg.size() * 4, p.datau.size() * 4, p.rowData.size() * 4, p.workFloats() * 4, outF * 4})
    if (bytes > g.maxBinding) return ++g.fitFallbacks, false;
  constexpr std::size_t kStepsPerSubmit = 64;
  const std::size_t chunks = std::max<std::size_t>(1, (p.steps + kStepsPerSubmit - 1) / kStepsPerSubmit);
  std::vector<std::uint32_t> ranges(chunks * 64, 0u);  // (first, last, predict) at every 256 bytes
  for (std::size_t c = 0; c < chunks; ++c) {
    ranges[c * 64] = std::uint32_t(c * kStepsPerSubmit);
    ranges[c * 64 + 1] = std::uint32_t(std::min(p.steps, (c + 1) * kStepsPerSubmit));
    ranges[c * 64 + 2] = c + 1 == chunks ? 1u : 0u;
  }
  wgpuDevicePushErrorScope(g.device, WGPUErrorFilter_OutOfMemory);
  wgpuDevicePushErrorScope(g.device, WGPUErrorFilter_Validation);
  WGPUBuffer header = upload(p.header.data(), p.header.size() * 4, WGPUBufferUsage_Uniform);
  WGPUBuffer range = upload(ranges.data(), ranges.size() * 4, WGPUBufferUsage_Uniform);
  WGPUBuffer theta = upload(p.theta.data(), P * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
  WGPUBuffer xg = upload(p.xg.data(), p.xg.size() * 4, WGPUBufferUsage_Storage);
  WGPUBuffer rows = upload(p.rowData.data(), p.rowData.size() * 4, WGPUBufferUsage_Storage);
  WGPUBuffer datau = upload(p.datau.data(), p.datau.size() * 4, WGPUBufferUsage_Storage);
  WGPUBufferDescriptor wd = {};
  wd.size = std::max<std::size_t>(16, p.workFloats() * 4);
  wd.usage = WGPUBufferUsage_Storage;
  WGPUBuffer work = wgpuDeviceCreateBuffer(g.device, &wd);  // zeroed by WebGPU: Adam's moments start at 0
  WGPUBufferDescriptor od = {};
  od.size = std::max<std::size_t>(16, outF * 4);
  od.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
  WGPUBuffer out = wgpuDeviceCreateBuffer(g.device, &od);
  const std::size_t readBytes = (P + outF) * 4;
  WGPUBufferDescriptor rdd = {};
  rdd.size = std::max<std::size_t>(16, readBytes);
  rdd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
  WGPUBuffer read = wgpuDeviceCreateBuffer(g.device, &rdd);
  std::vector<WGPUBindGroup> binds(chunks);
  for (std::size_t c = 0; c < chunks; ++c) {
    WGPUBindGroupEntry e[8] = {};
    WGPUBuffer all[8] = {header, range, theta, xg, rows, datau, work, out};
    for (std::uint32_t k = 0; k < 8; ++k) e[k].binding = k, e[k].buffer = all[k], e[k].size = WGPU_WHOLE_SIZE;
    e[1].offset = c * 256, e[1].size = 16;
    WGPUBindGroupDescriptor bgd = {};
    bgd.layout = g.diffLayout, bgd.entryCount = 8, bgd.entries = e;
    binds[c] = wgpuDeviceCreateBindGroup(g.device, &bgd);
  }
  for (std::size_t c = 0; c < chunks; ++c) {
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g.device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, g.diffPipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, binds[c], 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    if (c + 1 == chunks) {
      wgpuCommandEncoderCopyBufferToBuffer(enc, theta, 0, read, 0, P * 4);
      if (outF) wgpuCommandEncoderCopyBufferToBuffer(enc, out, 0, read, P * 4, outF * 4);
    }
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(g.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(enc);
  }
  struct Scope { volatile bool done = false; bool error = false; } s1, s2;
  const auto pop = [](WGPUErrorType t, const char*, void* u) {
    auto* q = static_cast<Scope*>(u);
    q->error = t != WGPUErrorType_NoError, q->done = true;
  };
  wgpuDevicePopErrorScope(g.device, pop, &s1);
  wgpuDevicePopErrorScope(g.device, pop, &s2);
  struct { volatile bool done = false; bool good = false; } mp;
  wgpuBufferMapAsync(read, WGPUMapMode_Read, 0, readBytes, [](WGPUBufferMapAsyncStatus st, void* u) {
    auto* q = static_cast<decltype(mp)*>(u);
    q->good = st == WGPUBufferMapAsyncStatus_Success, q->done = true;
  }, &mp);
  // a fit can take minutes on a software GPU: up to ten of them (each turn at least a millisecond)
  for (int t = 0; !(s1.done && s2.done && mp.done) && t < 600000; ++t) emscripten_sleep(1);
  bool ok = s1.done && s2.done && mp.done && !s1.error && !s2.error && mp.good && !g.lost;
  if (ok) {
    const auto* f = static_cast<const float*>(wgpuBufferGetConstMappedRange(read, 0, readBytes));
    ok = f != nullptr;
    if (ok) {
      const std::size_t split = p.predictRows * (p.inputs + 1);
      r.theta.assign(f, f + P);
      r.predictions.assign(f + P, f + P + split);
      r.stepLoss.assign(f + P + split, f + P + split + p.steps);
      for (float x : r.theta)
        if (!std::isfinite(x)) ok = false;
    }
  }
  if (mp.done && mp.good) wgpuBufferUnmap(read);
  for (WGPUBindGroup b : binds) wgpuBindGroupRelease(b);
  for (WGPUBuffer x : {header, range, theta, xg, rows, datau, work, out, read}) wgpuBufferDestroy(x), wgpuBufferRelease(x);
  if (ok) ++g.fits;
  else ++g.fitFallbacks;
  return ok;
}

// Set when the build loads: any differential network fit in this build is offered to the GPU first.
[[maybe_unused]] const bool kDiffnetOffload = (anmc::diffnetOffload() = runDiffnet, true);

class WebGpu : public strat::Backend {
 public:
  WebGpu() { strat::mcOffload() = runBatch; }
  std::string about() const override { return "the nested Monte Carlo on the GPU (WebGPU), the rest on one thread"; }
  void parallelFor(std::size_t n, const std::function<void(std::size_t, std::size_t)>& f) const override {
    if (n) f(0, n);
  }
};

}  // namespace

REGISTER_BACKEND("webgpu", WebGpu);
#endif
