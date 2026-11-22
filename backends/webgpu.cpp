// The Lab's GPU back end (--backend webgpu, the WebAssembly build docker/wasm/Dockerfile makes with OFM_WEBGPU), all of it
// here in C++ on WebGPU's C API (webgpu.h): the engine's own loops run as the serial back end's, and the nested Monte
// Carlo's trade kernel (include/strategy/mc_gpu.hpp: every reset's paths and every stop search) runs as one batch on
// the GPU: the adapter and device opened once, the kernel (include/strategy/mc_wgsl.hpp) compiled once, each batch
// uploaded, dispatched, read back. WebGPU answers through callbacks; the engine waits for them (emscripten_sleep under
// Asyncify) and goes on. No GPU, or a batch the GPU cannot run (a buffer over its limits, a validation error, the
// device lost): that batch runs on the CPU, so a run always ends. The page's JavaScript only starts the engine.
// Compiled only into that build: the native engine and the plain WebAssembly build have no such back end.
#if defined(__EMSCRIPTEN__) && defined(OFM_WEBGPU)
#include <emscripten.h>
#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "strategy/backend.hpp"
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
  ~Gpu() {
    if (batches || fallbacks || !why.empty())
      std::printf("strategy: webgpu: %zu Monte Carlo batches on %s%s%s\n", batches, name.empty() ? "no GPU" : name.c_str(),
                  fallbacks ? (", " + std::to_string(fallbacks) + " on the CPU").c_str() : "", why.empty() ? "" : (" (" + why + ")").c_str());
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

bool open() {
  Gpu& g = gpu();
  if (g.tried) return g.pipeline && !g.lost;
  g.tried = true;
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
  WGPUShaderModuleWGSLDescriptor wgsl = {};
  wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
  wgsl.code = strat::kMcWgsl;
  WGPUShaderModuleDescriptor sd = {};
  sd.nextInChain = &wgsl.chain;
  wgpuDevicePushErrorScope(g.device, WGPUErrorFilter_Validation);
  WGPUShaderModule module = wgpuDeviceCreateShaderModule(g.device, &sd);
  WGPUComputePipelineDescriptor pd = {};
  pd.compute.module = module;
  pd.compute.entryPoint = "main";
  g.pipeline = wgpuDeviceCreateComputePipeline(g.device, &pd);
  struct { volatile bool done = false; bool error = false; std::string message; } es;
  wgpuDevicePopErrorScope(g.device, [](WGPUErrorType t, const char* m, void* u) {
    auto* r = static_cast<decltype(es)*>(u);
    r->error = t != WGPUErrorType_NoError, r->message = m ? m : "", r->done = true;
  }, &es);
  waitFor(es.done);
  wgpuShaderModuleRelease(module);
  if (es.error || !g.pipeline) return g.why = "the kernel did not compile: " + es.message.substr(0, 200), g.pipeline = nullptr, false;
  g.layout = wgpuComputePipelineGetBindGroupLayout(g.pipeline, 0);
  return true;
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
