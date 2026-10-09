import { run } from '../sat-client.js';
import { barChart, dateAxis, fmt, lineChart, scatterChart } from '../charts.js';
import { probeAdapters } from '../gpu/engine.js';
import { runOnEmulator, runOnGpu } from '../gpu/backend.js';
import { card, el, grid, initPage, passFail, runButton, specEditor, table, tiles } from '../ui.js';

const page = initPage({
  id: 'gpu',
  title: 'WebGPU strategy search',
  context: 'sat/gpu: compile · summarise · runFusedReference · WGSL kernels',
  description:
    'The library packs the candidate pool and the selector grid into one table: every model\'s out-of-sample predictions and ranks, the next-day returns, the rules and the selector settings. ' +
    'Three WebGPU compute kernels, also part of the library, then do the search. <b>candidate-backtest</b> runs one invocation per candidate; the 64 candidates of a workgroup share each day of the universe through workgroup memory. ' +
    '<b>adaptive-select</b> runs one invocation per selector setting and scores every candidate in O(1) from prefix sums. <b>series-summary</b> computes the performance statistics. ' +
    'The browser only uploads the table, dispatches the kernels and hands the read-back to the library, which checks it below against its own CPU run. ' +
    'The GPU works in 32-bit floats, so near-ties between candidates can occasionally resolve differently. ' +
    'Every run also executes the same plans on the <b>emulated GPU</b>, the library\'s CPU reference of the three kernels, invocation by invocation in 32-bit floats, in the WebAssembly worker. It is what the strategy pages use when the device has no usable WebGPU adapter, and here it is checked against both WebGPU and the WebAssembly library.',
});

// The same plans on WebGPU (when the device has it) and on the emulated GPU, which is
// always run so the page shows both devices side by side.
async function runKernels(analysis, spec, options) {
  let gpu = null, why = 'WebGPU is not available in this browser';
  if ('gpu' in navigator) {
    try {
      gpu = await runOnGpu(analysis, spec, options);
    } catch (e) {
      console.warn('WebGPU run failed; showing the emulated GPU only:', e);
      why = e.message;
    }
  }
  const emu = await runOnEmulator(analysis, spec, options);
  const unsupported = (gpu && gpu.unsupported) || emu.unsupported;
  if (unsupported) return { unsupported };
  return { gpu: gpu && gpu.result, emu: emu.result, why: gpu ? null : why };
}

const maxDiff = (a, b) => Array.from(a).reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
const checkRows = (name, c) => [
  [`${name}: candidates max |ΔSharpe|`, c.candidateSharpeDiff.toExponential(1), '1e-3', passFail(c.candidateSharpeDiff < 1e-3)],
  [`${name}: candidates max |Δ total return| / (1 + |R|)`, c.candidateReturnDiff.toExponential(1), '1e-3', passFail(c.candidateReturnDiff < 1e-3)],
  [`${name}: candidates max |Δ turnover/day|`, c.candidateTurnoverDiff.toExponential(1), '1e-4', passFail(c.candidateTurnoverDiff < 1e-4)],
  [`${name}: selectors holding the same candidate`, fmt.pct(c.selectionAgreement, 2), '≥ 95%', passFail(c.selectionAgreement >= 0.95)],
  [`${name}: selectors max |ΔSharpe|`, c.selectorSharpeDiff.toFixed(3), '0.1', passFail(c.selectorSharpeDiff < 0.1)],
];

runButton(page, 'Run the kernels and validate against WASM', async (spec) => {
  const k = await runKernels('validation', spec, { warmUp: true });
  if (k.unsupported) throw new Error('not on GPU: ' + k.unsupported);
  const g = k.gpu, e = k.emu, r = g || e, c = r.checks;
  if (k.why) page.content.append(el('p', { class: 'note engine-note' }, el('b', { text: 'Emulated GPU only. ' }),
    `WebGPU could not run here (${k.why}), so the plans ran on the CPU reference of the kernels. The checks below compare that emulation with the WebAssembly library; timings are CPU timings.`));
  const gpuVsEmu = g ? { candidates: maxDiff(g.gpuCandidateSharpe, e.gpuCandidateSharpe), selectors: maxDiff(g.gpuSelectorSharpe, e.gpuSelectorSharpe) } : null;
  tiles(page.content, [
    ...(g ? [{ label: 'WebGPU kernels', value: fmt.num(g.gpuMs, 1) + ' ms', hint: `${fmt.num(r.numCandidates)} candidates × ${fmt.num(r.numSelectors)} selectors × ${fmt.num(r.evalDays)} days` }] : []),
    { label: 'Emulated GPU (CPU)', value: fmt.num(e.gpuMs, 1) + ' ms', hint: 'same kernels and plans, on the CPU in the worker' },
    { label: 'WASM library (CPU, 1 thread)', value: fmt.num(r.cpuMs, 0) + ' ms', hint: 'backtests + selectors' },
    ...(g ? [{ label: 'Speed-up: WebGPU vs WASM', value: (r.cpuMs / g.gpuMs).toFixed(1) + '×', hint: 'includes upload & read-back' }] : []),
    { label: 'Plan compile (C++)', value: fmt.num(r.compileMs, 1) + ' ms', hint: `${r.numModels} models × ${r.numAssets} stocks` },
    { label: 'Same choice', value: fmt.pct(c.selectionAgreement, 2), hint: `selector-days holding the same candidate${g ? ' (WebGPU vs WASM)' : ''}` },
  ]);
  const devices = [...(g ? [['WebGPU', g, 0]] : []), ['Emulated GPU', e, 2]];
  const gr = grid(page.content);
  scatterChart(card(gr, 'Candidates: Sharpe ratio, kernels vs WASM', 'Every fixed candidate, on each device. The points should sit on the diagonal; WebGPU\'s and the emulated GPU\'s coincide.'), {
    xLabel: 'WASM', yLabel: 'kernels', diagonal: true, square: true,
    series: devices.map(([name, x, colorIndex]) => ({ name, x: Array.from(x.cpuCandidateSharpe), y: Array.from(x.gpuCandidateSharpe), colorIndex })),
  });
  scatterChart(card(gr, 'Selector grid: Sharpe ratio, kernels vs WASM', 'Every look-back × step × score setting.'), {
    xLabel: 'WASM', yLabel: 'kernels', diagonal: true, square: true,
    series: devices.map(([name, x, colorIndex]) => ({ name, x: Array.from(x.cpuSelectorSharpe), y: Array.from(x.gpuSelectorSharpe), colorIndex: colorIndex + 1 })),
  });
  lineChart(card(gr, 'Your selector: wealth on every engine', 'Solid: kernels; dashed: WASM library.'), {
    x: r.dates.map((_, i) => i), xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [...devices.map(([name, x, colorIndex]) => ({ name, y: Array.from(x.gpuEquity), colorIndex })), { name: 'WASM', y: Array.from(r.cpuEquity), dash: true, colorIndex: 1 }],
  });
  const checks = card(page.content, 'Agreement checks', 'Computed by the library from each device\'s read-back and its own CPU run. The two kernel devices run the same f32 arithmetic.');
  table(checks, ['check', 'value', 'tolerance', 'result'], [
    ...(g ? checkRows('WebGPU vs WASM', g.checks) : []),
    ...checkRows('Emulated GPU vs WASM', e.checks),
    ...(gpuVsEmu ? [
      ['WebGPU vs emulated GPU: candidates max |ΔSharpe|', gpuVsEmu.candidates.toExponential(1), '1e-4', passFail(gpuVsEmu.candidates < 1e-4)],
      ['WebGPU vs emulated GPU: selectors max |ΔSharpe|', gpuVsEmu.selectors.toExponential(1), '0.1', passFail(gpuVsEmu.selectors < 0.1)],
    ] : []),
  ]);
  window.__satLastRun = { checks: c, device: g ? 'GPU' : 'Emulated GPU', emulatorChecks: e.checks, gpuVsEmu };
  return g ? `GPU ${fmt.num(g.gpuMs, 1)} ms · emulated GPU ${fmt.num(e.gpuMs, 1)} ms · WASM ${fmt.num(r.cpuMs)} ms · adapter: ${g.adapter}`
    : `Emulated GPU ${fmt.num(e.gpuMs, 1)} ms · WASM ${fmt.num(r.cpuMs)} ms · adapter: ${e.adapter}`;
});

// More candidates: extra rules with nearby parameters, every model, against the same selector grid.
function widen(strategies, n) {
  const out = [];
  for (let k = 0; out.length < n; k++)
    for (const s of strategies) {
      if (out.length >= n) break;
      const step = s.kind === 'topk' || s.kind === 'longshort' ? 1 : 0.005;
      out.push({ ...s, param: s.param + step * k, holding: 1 + (k % 3) });
    }
  return out;
}

const bench = el('button', { text: 'Benchmark candidate scaling' });
page.toolbar.insertBefore(bench, page.status);
bench.addEventListener('click', async () => {
  bench.disabled = true;
  page.status.textContent = 'Benchmarking…';
  try {
    const rows = [];
    for (const n of [8, 32, 128, 512]) {
      const spec = { ...page.spec, strategies: widen(page.spec.strategies, n) };
      const k = await runKernels('validation', spec);
      if (k.unsupported) throw new Error('not on GPU: ' + k.unsupported);
      const r = k.gpu || k.emu;
      rows.push({ n: r.numCandidates, gpuMs: k.gpu ? k.gpu.gpuMs : null, emuMs: k.emu.gpuMs, wasmMs: r.cpuMs });
    }
    const box = card(page.content, 'Throughput (candidate-days per second)', 'Every engine runs the same candidates and selector grid. The models are trained once, in WebAssembly, and not timed.');
    barChart(box, {
      labels: rows.map((x) => fmt.num(x.n) + ' candidates'),
      series: [
        ...(rows[0].gpuMs != null ? [{ name: 'WebGPU kernels', values: rows.map((x) => (1000 * x.n) / x.gpuMs) }] : []),
        { name: 'Emulated GPU (CPU)', values: rows.map((x) => (1000 * x.n) / x.emuMs), colorIndex: 2 },
        { name: 'WASM library', values: rows.map((x) => (1000 * x.n) / x.wasmMs), colorIndex: 1 },
      ],
    });
    const hasGpu = rows[0].gpuMs != null;
    table(box, ['candidates', ...(hasGpu ? ['WebGPU ms'] : []), 'emulated GPU ms', 'WASM ms', ...(hasGpu ? ['WebGPU speed-up'] : [])],
      rows.map((x) => [fmt.num(x.n), ...(hasGpu ? [fmt.num(x.gpuMs, 1)] : []), fmt.num(x.emuMs, 1), fmt.num(x.wasmMs), ...(hasGpu ? [(x.wasmMs / x.gpuMs).toFixed(1) + '×'] : [])]));
    page.status.textContent = 'Benchmark done';
  } catch (e) {
    page.status.className = 'status error';
    page.status.textContent = 'Error: ' + e.message;
  } finally {
    bench.disabled = false;
  }
});

run('gpuKernels', {}).then((k) => page.main.append(el('details', { class: 'spec' }, el('summary', { text: 'WGSL source of the kernels (from the C++ library)' }),
  el('h3', { text: 'candidate-backtest' }), el('pre', { class: 'code', text: k.candidateBacktest }),
  el('h3', { text: 'adaptive-select' }), el('pre', { class: 'code', text: k.adaptiveSelect }),
  el('h3', { text: 'series-summary' }), el('pre', { class: 'code', text: k.seriesSummary }))));

// What this browser and device offer: useful when Auto falls back to the emulated GPU.
const diag = el('details', { class: 'spec', id: 'webgpu-diagnostics' }, el('summary', { text: 'WebGPU diagnostics for this device' }));
page.main.append(diag);
probeAdapters().then((r) => {
  const rows = [
    ['secure context (https or file)', r.secureContext ? 'yes' : 'no: WebGPU needs https'],
    ['navigator.gpu (WebGPU API)', r.api ? 'present' : 'missing: this browser has no WebGPU'],
    ...r.adapters.map((a) => [`requestAdapter(${a.request})`, a.error ? `error: ${a.error}` : a.adapter
      ? `${a.adapter} (storage binding ${fmt.num(a.maxStorageBufferBindingSize / 2 ** 20)} MiB, workgroup storage ${fmt.num(a.maxComputeWorkgroupStorageSize)} B)` +
        (a.missing.length ? ` — too limited for the kernels: ${a.missing.join(', ')}` : ' — usable')
      : 'no adapter']),
    ['browser', r.userAgent],
  ];
  diag.append(el('dl', { class: 'diagnostics' }, rows.flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })])));
  if (r.api && r.adapters.every((a) => !a.adapter || a.missing.length))
    diag.append(el('p', { class: 'note', text: 'The browser has the WebGPU API but offers no adapter, so Chrome has WebGPU blocked or unsupported for this GPU or driver. chrome://gpu shows the reason. The pages keep working on WebAssembly.' }));
  window.__satWebGpuProbe = r;
});
