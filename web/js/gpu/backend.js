// Runs a strategy analysis of the library on the WebGPU kernels, on an emulated GPU, or on
// the WebAssembly build. Auto picks WebGPU whenever the browser supports it and the library
// can compile the analysis for the kernels; without a usable WebGPU adapter it runs the same
// kernels on the emulated GPU (the library's CPU reference of them, in the worker). Model
// training always runs in WebAssembly; the (emulated) GPU runs the backtests of every
// candidate and the self-adaptive selectors.
//
// Kernel path: gpuJobs (C++ compiles the plans) -> device.run per plan (WebGPU: upload,
// dispatch, read back; emulator: gpuRunPlan) -> gpuAnalyse (C++ turns the read-backs into
// the same result as the WASM analysis). This file only routes calls; all numerics are in
// the C++ library.
import { run } from '../sat-client.js';
import { el } from '../ui.js';
import { CpuGpuEngine } from './emulator.js';
import { GpuEngine } from './engine.js';

const STORAGE_KEY = 'sat-engine';
const MODES = { auto: 'Auto', gpu: 'WebGPU', emulator: 'Emulated GPU (CPU)', wasm: 'WebAssembly' };

export function getEngineMode() {
  try { return MODES[localStorage.getItem(STORAGE_KEY)] ? localStorage.getItem(STORAGE_KEY) : 'auto'; } catch (_) { return 'auto'; }
}

export function setEngineMode(mode) {
  try { localStorage.setItem(STORAGE_KEY, mode); } catch (_) { /* storage unavailable */ }
}

let engine = null;
export function gpuEngine() {
  if (!engine) engine = run('gpuKernels', {}).then((k) => GpuEngine.create(k)).catch((e) => { engine = null; throw e; });
  return engine;
}

let emulator = null;
export function emulatedGpu() {
  if (!emulator) emulator = new CpuGpuEngine();
  return emulator;
}

/** Runs every plan of gpuJobs on the GPU and hands the read-backs to gpuAnalyse. */
export function runOnGpu(analysis, spec, options) {
  return runOnDevice(gpuEngine, analysis, spec, options);
}

/** The same, on the emulated GPU. */
export function runOnEmulator(analysis, spec, options) {
  return runOnDevice(emulatedGpu, analysis, spec, options);
}

// `device` creates (or returns) the engine; it is only called once the plans compiled.
async function runOnDevice(device, analysis, spec, { warmUp = false } = {}) {
  const jobs = await run('gpuJobs', { ...spec, analysis });
  if (jobs.unsupported) return { unsupported: jobs.unsupported };
  const gpu = await device();
  if (warmUp && jobs.plans[0]) await gpu.run(jobs.plans[0]); // pipeline and driver caches
  const gpuOutputs = [];
  let gpuMs = 0;
  for (const plan of jobs.plans) {
    const out = plan ? await gpu.run(plan) : null;
    gpuOutputs.push(out && { stats: out.stats, adapt: out.adapt });
    gpuMs += out ? out.gpuMs : 0;
  }
  const result = await run('gpuAnalyse', { ...spec, analysis, gpuOutputs });
  const first = jobs.plans.find(Boolean);
  return { result: { ...result, gpuMs, adapter: gpu.name, emulated: !!gpu.emulated, plans: jobs.plans.length, compileMs: jobs.compileMs, numCandidates: first?.numCandidates, numSelectors: first?.numSelectors, numDays: first?.numDays } };
}

/**
 * Runs `analysis` (a library entry point: strategies, adaptive, robustness) on the engine
 * chosen by the selector. Resolves to { result, engine, reason } with engine 'gpu',
 * 'emulator' or 'wasm'. Without a usable WebGPU adapter, or when the GPU run fails, the
 * kernels run on the emulated GPU; an analysis the kernels cannot express (a top-M mix, too
 * many stocks) runs on WebAssembly, as does anything the emulator cannot complete.
 */
export async function runAnalysis(analysis, spec) {
  const mode = getEngineMode();
  const wasm = async (reason) => ({ result: await run(analysis, spec), engine: 'wasm', reason });
  const emulate = async (reason) => {
    try {
      const g = await runOnEmulator(analysis, spec);
      if (g.unsupported) return wasm(g.unsupported);
      return { ...g, engine: 'emulator', reason };
    } catch (e) {
      console.warn('Emulated GPU failed, falling back to WebAssembly:', e);
      return wasm(`emulated GPU failed (${e.message}); used WebAssembly`);
    }
  };
  if (mode === 'wasm') return wasm('WebAssembly selected');
  if (mode === 'emulator') return emulate('emulated GPU selected');
  if (!('gpu' in navigator)) return emulate('WebGPU is not available in this browser; kernels emulated on the CPU');
  try {
    const g = await runOnGpu(analysis, spec);
    if (g.unsupported) return wasm(g.unsupported);
    return { ...g, engine: 'gpu', reason: mode === 'auto' ? 'Auto: WebGPU available' : 'WebGPU selected' };
  } catch (e) {
    console.warn('WebGPU run failed, falling back to the emulated GPU:', e);
    // No adapter is a property of the device, not a failure of the run: say so plainly.
    if (/no WebGPU adapter|too limited/.test(e.message)) return emulate(`WebGPU unavailable: ${e.message.replace('no WebGPU adapter: ', '')}; kernels emulated on the CPU`);
    return emulate(`WebGPU failed (${e.message}); kernels emulated on the CPU`);
  }
}

/** Toolbar control: Engine [Auto | WebGPU | Emulated GPU | WebAssembly]. Changing it re-runs via `onChange`. */
export function engineSelector(page, onChange) {
  const select = el('select', { 'aria-label': 'Compute engine' },
    Object.entries(MODES).map(([value, label]) => el('option', {
      value, selected: value === getEngineMode(),
      text: value === 'auto' ? 'Auto (WebGPU, else emulated GPU)' : label,
    })));
  select.addEventListener('change', () => { setEngineMode(select.value); onChange(); });
  page.toolbar.insertBefore(el('label', { class: 'engine-select' }, 'Engine ', select), page.status);
  return select;
}

/** One-line note for the status bar: which engine ran and why. */
export function engineNote(r, ms) {
  const name = r.engine === 'gpu' ? `WebGPU (${r.result.adapter})` : r.engine === 'emulator' ? 'Emulated GPU' : 'WebAssembly';
  return `Done in ${Math.round(ms).toLocaleString('en-US')} ms · ${name} · ${r.reason}`;
}

/** Card shown in GPU mode listing what needs per-candidate daily data from the WASM engine. */
export function gpuScopeNote(parent, items, engine = 'gpu') {
  parent.append(el('p', { class: 'note engine-note' },
    el('b', { text: engine === 'emulator' ? 'Computed by the GPU kernels, emulated on the CPU. ' : 'Computed on WebGPU. ' }),
    `The kernels return each candidate's statistics and the self-adaptive series, not every candidate's daily returns, so ${items} are only available with the WebAssembly engine.`));
}
