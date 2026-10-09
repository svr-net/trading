// WebGPU host for the strategy-search kernels of the C++ library (sat::gpu).
//
// No numerics live here. The library supplies the WGSL sources (gpuKernels) and, for every
// job, a plan with one packed table, dispatch sizes and buffer sizes (gpuJobs). This class
// uploads the table, dispatches candidate-backtest -> adaptive-select -> series-summary and
// reads back two arrays, which go back to the library (gpuAnalyse) for all post-processing.

// Adapter requests, in order. Some mobile drivers (seen on Android) return no adapter for a
// high-performance request but do for a plain one, so a null answer is not final. The last
// request asks for a WebGPU *compatibility* adapter: where Chrome cannot offer core WebGPU
// (seen on Android with an OpenGL compositor and a blocklisted Vulkan/GL interop) it may still
// offer WebGPU on OpenGL ES in compatibility mode, which is enough for compute kernels.
const ADAPTER_OPTIONS = [
  { powerPreference: 'high-performance' },
  {},
  { powerPreference: 'low-power' },
  { featureLevel: 'compatibility', compatibilityMode: true },
];

// What the kernels need: 4 storage buffers in adaptive-select, 64-invocation workgroups and
// the 8.5 KiB day of the universe that candidate-backtest stages in workgroup memory. All
// within the compatibility-mode defaults, so even an OpenGL ES adapter qualifies.
const KERNEL_LIMITS = {
  maxStorageBuffersPerShaderStage: 4,
  maxComputeInvocationsPerWorkgroup: 64,
  maxComputeWorkgroupSizeX: 64,
  maxComputeWorkgroupStorageSize: 8704,
};

/** The kernel limits this adapter cannot meet, e.g. ["maxStorageBuffersPerShaderStage 4 < 7"]. */
export function missingLimits(adapter) {
  return Object.entries(KERNEL_LIMITS)
    .filter(([k, need]) => !(adapter.limits[k] >= need))
    .map(([k, need]) => `${k} ${adapter.limits[k]} < ${need}`);
}

/** Why WebGPU cannot start here, with what the user can check. */
export function noAdapterReason() {
  const ua = navigator.userAgent || '';
  const android = /Android/.test(ua);
  return android
    ? 'no WebGPU adapter: Chrome has WebGPU blocked or unsupported on this device (it needs Android 12+ and a supported GPU; see chrome://gpu)'
    : 'no WebGPU adapter: WebGPU is blocked or unsupported for this GPU or driver (see chrome://gpu)';
}

async function requestAdapter() {
  if (!('gpu' in navigator)) throw new Error('WebGPU is not available in this browser');
  const tooLimited = [];
  for (const options of ADAPTER_OPTIONS) {
    const adapter = await navigator.gpu.requestAdapter(options).catch(() => null);
    if (!adapter) continue;
    const missing = missingLimits(adapter);
    if (!missing.length) return { adapter, compatibility: 'featureLevel' in options };
    tooLimited.push(missing.join(', '));
  }
  if (tooLimited.length) throw new Error(`WebGPU adapter too limited for the kernels (${tooLimited[0]})`);
  throw new Error(noAdapterReason());
}

// Ask for the adapter's own limits where the kernels need more than the defaults (required in
// compatibility mode) and for its largest buffers; without the large buffers if refused.
async function requestDevice(adapter) {
  const lim = adapter.limits;
  const needed = Object.fromEntries(Object.keys(KERNEL_LIMITS).map((k) => [k, lim[k]]));
  try {
    return await adapter.requestDevice({
      requiredLimits: { ...needed, maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize },
    });
  } catch (_) {
    return adapter.requestDevice({ requiredLimits: needed });
  }
}

/** Every adapter request with its answer, for the diagnostics panel. */
export async function probeAdapters() {
  const report = { secureContext: globalThis.isSecureContext, api: 'gpu' in navigator, userAgent: navigator.userAgent, adapters: [] };
  if (!report.api) return report;
  for (const options of ADAPTER_OPTIONS) {
    const entry = { request: JSON.stringify(options) };
    try {
      const a = await navigator.gpu.requestAdapter(options);
      if (a) {
        const info = a.info || {};
        entry.adapter = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' · ') || 'adapter (no info)';
        entry.maxStorageBufferBindingSize = a.limits.maxStorageBufferBindingSize;
        entry.maxComputeWorkgroupStorageSize = a.limits.maxComputeWorkgroupStorageSize;
        entry.missing = missingLimits(a);
      } else entry.adapter = null;
    } catch (e) {
      entry.error = e.message;
    }
    report.adapters.push(entry);
  }
  return report;
}

export class GpuEngine {
  static async create(kernels) {
    const { adapter, compatibility } = await requestAdapter();
    const device = await requestDevice(adapter);
    const engine = new GpuEngine(adapter, device, compatibility);
    await engine.compile(kernels);
    return engine;
  }

  constructor(adapter, device, compatibility = false) {
    this.adapter = adapter;
    this.device = device;
    this.info = adapter.info || {};
    this.name = ([this.info.vendor, this.info.architecture].filter(Boolean).join(' ') || 'GPU') + (compatibility ? ', compatibility mode' : '');
    device.lost.then((info) => { this.lost = info; });
  }

  async compile(kernels) {
    const make = async (code, label) => {
      const module = this.device.createShaderModule({ code, label });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      if (errors.length) throw new Error(`${label}: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
      return this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
    };
    [this.backtest, this.select, this.summary] = await Promise.all([
      make(kernels.candidateBacktest, 'candidate-backtest'),
      make(kernels.adaptiveSelect, 'adaptive-select'),
      make(kernels.seriesSummary, 'series-summary'),
    ]);
  }

  upload(data, usage, label) {
    const buf = this.device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: usage | GPUBufferUsage.COPY_DST, label, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buf.unmap();
    return buf;
  }

  /** Runs one plan from gpuJobs. Resolves to { stats, adapt, gpuMs }, plus book with plan.readBook. */
  async run(plan) {
    const d = this.device;
    const { bytes, dispatch } = plan;
    const largest = Math.max(bytes.tables, bytes.book, bytes.prefix, bytes.adapt);
    if (largest > d.limits.maxStorageBufferBindingSize)
      throw new Error(`candidates × days too large for one storage buffer (${(largest / 2 ** 20).toFixed(0)} MiB > ${(d.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB)`);
    if (Math.max(dispatch.candidates, dispatch.summary) > d.limits.maxComputeWorkgroupsPerDimension) throw new Error('too many candidates for one dispatch');

    const S = GPUBufferUsage.STORAGE;
    const scratch = (size, usage, label) => d.createBuffer({ size: Math.max(16, size), usage, label });
    const bufs = {
      header: this.upload(plan.header, GPUBufferUsage.UNIFORM, 'header'),
      tables: this.upload(plan.tables, S, 'tables'),
      book: scratch(bytes.book, S | GPUBufferUsage.COPY_SRC, 'book'),
      prefix: scratch(bytes.prefix, S, 'prefix'),
      adapt: scratch(bytes.adapt, S | GPUBufferUsage.COPY_SRC, 'adapt'),
      stats: scratch(bytes.stats, S | GPUBufferUsage.COPY_SRC, 'stats'),
      readAdapt: scratch(bytes.adapt, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-adapt'),
      readStats: scratch(bytes.stats, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-stats'),
    };
    // The candidate book (gross return, turnover per candidate and day), for analyses that
    // run their own statistics on the CPU.
    const bookBytes = plan.numCandidates * plan.numDays * 8;
    if (plan.readBook) bufs.readBook = scratch(bookBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'read-book');
    const group = (pipeline, entries) => d.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });

    d.pushErrorScope('validation');
    d.pushErrorScope('out-of-memory');
    const t0 = performance.now();
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.backtest);
    pass.setBindGroup(0, group(this.backtest, [bufs.header, bufs.tables, bufs.book, bufs.prefix]));
    pass.dispatchWorkgroups(dispatch.candidates);
    if (dispatch.selectors > 0) {
      pass.setPipeline(this.select);
      pass.setBindGroup(0, group(this.select, [bufs.header, bufs.tables, bufs.book, bufs.prefix, bufs.adapt]));
      pass.dispatchWorkgroups(dispatch.selectors);
    }
    pass.setPipeline(this.summary);
    pass.setBindGroup(0, group(this.summary, [bufs.header, bufs.book, bufs.adapt, bufs.stats]));
    pass.dispatchWorkgroups(dispatch.summary);
    pass.end();
    enc.copyBufferToBuffer(bufs.adapt, 0, bufs.readAdapt, 0, Math.max(16, bytes.adapt));
    enc.copyBufferToBuffer(bufs.stats, 0, bufs.readStats, 0, Math.max(16, bytes.stats));
    if (plan.readBook) enc.copyBufferToBuffer(bufs.book, 0, bufs.readBook, 0, Math.max(16, bookBytes));
    d.queue.submit([enc.finish()]);
    globalThis.__satGpuRuns = (globalThis.__satGpuRuns || 0) + 1; // kernel pipelines submitted (checked by the e2e test)
    await Promise.all([bufs.readAdapt.mapAsync(GPUMapMode.READ), bufs.readStats.mapAsync(GPUMapMode.READ),
      ...(plan.readBook ? [bufs.readBook.mapAsync(GPUMapMode.READ)] : [])]);
    const gpuMs = performance.now() - t0;
    const oom = await d.popErrorScope();
    const validation = await d.popErrorScope();
    const out = validation || oom ? null : {
      stats: new Float32Array(bufs.readStats.getMappedRange().slice(0, bytes.stats)),
      adapt: new Float32Array(bufs.readAdapt.getMappedRange().slice(0, bytes.adapt)),
      ...(plan.readBook ? { book: new Float32Array(bufs.readBook.getMappedRange().slice(0, bookBytes)) } : {}),
      gpuMs,
    };
    for (const b of Object.values(bufs)) b.destroy();
    if (!out) throw new Error((validation || oom).message);
    if (this.lost) throw new Error('GPU device lost: ' + this.lost.message);
    return out;
  }
}
