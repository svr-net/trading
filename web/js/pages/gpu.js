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
    'Without a usable WebGPU adapter the page runs the same plans on the <b>emulated GPU</b>: the library\'s CPU reference of the three kernels, invocation by invocation in 32-bit floats, in the WebAssembly worker.',
});

// The kernels on WebGPU when the device has it, otherwise on the emulated GPU.
async function runKernels(analysis, spec, options) {
  let why = 'WebGPU is not available in this browser';
  if ('gpu' in navigator) {
    try {
      return { ...(await runOnGpu(analysis, spec, options)), device: 'GPU' };
    } catch (e) {
      console.warn('WebGPU run failed, using the emulated GPU:', e);
      why = e.message;
    }
  }
  return { ...(await runOnEmulator(analysis, spec, options)), device: 'Emulated GPU', why };
}
specEditor(page, ['market', 'csv', 'models', 'strategies', 'costs', 'selector', 'robustness']);

runButton(page, 'Run on GPU and validate against WASM', async (spec) => {
  const g = await runKernels('validation', spec, { warmUp: true });
  if (g.unsupported) throw new Error('not on GPU: ' + g.unsupported);
  const r = g.result, c = r.checks, dev = g.device;
  if (g.why) page.content.append(el('p', { class: 'note engine-note' }, el('b', { text: 'Emulated GPU. ' }),
    `WebGPU could not run here (${g.why}), so the same plans ran on the CPU reference of the kernels. The checks below compare that emulation with the WebAssembly library; timings are CPU timings.`));
  tiles(page.content, [
    { label: `${dev} kernels`, value: fmt.num(r.gpuMs, 1) + ' ms', hint: `${fmt.num(r.numCandidates)} candidates × ${fmt.num(r.numSelectors)} selectors × ${fmt.num(r.evalDays)} days` },
    { label: 'WASM (CPU, 1 thread)', value: fmt.num(r.cpuMs, 0) + ' ms', hint: 'backtests + selectors' },
    { label: 'Speed-up', value: (r.cpuMs / r.gpuMs).toFixed(1) + '×', hint: 'includes upload & read-back' },
    { label: 'Plan compile (C++)', value: fmt.num(r.compileMs, 1) + ' ms', hint: `${r.numModels} models × ${r.numAssets} stocks` },
    { label: 'Same choice', value: fmt.pct(c.selectionAgreement, 2), hint: 'selector-days holding the same candidate' },
  ]);
  const gr = grid(page.content);
  scatterChart(card(gr, `Candidates: Sharpe ratio, ${dev} vs WASM`, 'Every fixed candidate. The points should sit on the diagonal.'), {
    xLabel: 'WASM', yLabel: dev, diagonal: true, square: true, series: [{ name: 'candidate', x: Array.from(r.cpuCandidateSharpe), y: Array.from(r.gpuCandidateSharpe) }],
  });
  scatterChart(card(gr, `Selector grid: Sharpe ratio, ${dev} vs WASM`, 'Every look-back × step × score setting.'), {
    xLabel: 'WASM', yLabel: dev, diagonal: true, square: true, series: [{ name: 'selector', x: Array.from(r.cpuSelectorSharpe), y: Array.from(r.gpuSelectorSharpe), colorIndex: 1 }],
  });
  lineChart(card(gr, 'Your selector: wealth on both engines', `Solid: ${dev} kernels; dashed: WASM library.`), {
    x: r.dates.map((_, i) => i), xFormat: dateAxis(r.dates), xLabel: 'date', yFormat: (v) => v.toFixed(2),
    series: [{ name: dev, y: Array.from(r.gpuEquity) }, { name: 'WASM', y: Array.from(r.cpuEquity), dash: true, colorIndex: 1 }],
  });
  const checks = card(gr, 'Agreement checks', `Computed by the library from the ${dev === 'GPU' ? 'GPU' : 'emulated GPU'} read-back and its own CPU run.`);
  table(checks, ['check', 'value', 'tolerance', 'result'], [
    ['candidates: max |ΔSharpe|', c.candidateSharpeDiff.toExponential(1), '1e-3', passFail(c.candidateSharpeDiff < 1e-3)],
    ['candidates: max |Δ total return| / (1 + |R|)', c.candidateReturnDiff.toExponential(1), '1e-3', passFail(c.candidateReturnDiff < 1e-3)],
    ['candidates: max |Δ turnover/day|', c.candidateTurnoverDiff.toExponential(1), '1e-4', passFail(c.candidateTurnoverDiff < 1e-4)],
    ['selectors: same candidate held', fmt.pct(c.selectionAgreement, 2), '≥ 95%', passFail(c.selectionAgreement >= 0.95)],
    ['selectors: max |ΔSharpe|', c.selectorSharpeDiff.toFixed(3), '0.1', passFail(c.selectorSharpeDiff < 0.1)],
  ]);
  window.__satLastRun = { checks: c, device: dev };
  return `${dev} ${fmt.num(r.gpuMs, 1)} ms · WASM ${fmt.num(r.cpuMs)} ms · adapter: ${r.adapter}`;
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
      const g = await runKernels('validation', spec);
      if (g.unsupported) throw new Error('not on GPU: ' + g.unsupported);
      rows.push({ n: g.result.numCandidates, gpuMs: g.result.gpuMs, wasmMs: g.result.cpuMs, dev: g.device });
    }
    const box = card(page.content, 'Throughput (candidate-days per second)', 'Both engines run the same candidates and selector grid. The models are trained once, in WebAssembly, and not timed.');
    barChart(box, {
      labels: rows.map((x) => fmt.num(x.n) + ' candidates'),
      series: [{ name: `${rows[0].dev} kernels`, values: rows.map((x) => (1000 * x.n) / x.gpuMs) }, { name: 'WASM library', values: rows.map((x) => (1000 * x.n) / x.wasmMs) }],
    });
    table(box, ['candidates', `${rows[0].dev} ms`, 'WASM ms', 'speed-up'], rows.map((x) => [fmt.num(x.n), fmt.num(x.gpuMs, 1), fmt.num(x.wasmMs), (x.wasmMs / x.gpuMs).toFixed(1) + '×']));
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
