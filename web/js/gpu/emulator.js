// The emulated GPU: a drop-in for GpuEngine that runs the same plans on the library's CPU
// reference of the WGSL kernels (gpu::runFusedReference) in the WebAssembly worker. Each
// kernel invocation runs in turn, in 32-bit floats, over the same header and table buffers,
// and the read-back has the same layout, so gpuAnalyse treats it exactly like WebGPU's.
// It is the fallback when the browser has no usable WebGPU adapter.
import { run } from '../sat-client.js';

export class CpuGpuEngine {
  constructor() {
    this.name = 'CPU emulation of the kernels';
    this.emulated = true;
  }

  /** Runs one plan from gpuJobs. Resolves to { stats, adapt, gpuMs } like GpuEngine.run. */
  async run(plan) {
    const out = await run('gpuRunPlan', { header: plan.header, tables: plan.tables });
    return { stats: out.stats, adapt: out.adapt, gpuMs: out.ms };
  }
}
