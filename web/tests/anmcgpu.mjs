// The differential network's fused kernels (include/strategy/diffnet_wgsl.hpp) on WebGPU, in headless Chromium (its
// software GPU, SwiftShader), against their C++ reference in single precision (include/strategy/diffnet_gpu.hpp):
//   two kernels   a fit as one command buffer of (diffnetGrad, diffnetAdam) pairs, a step each by a dynamic uniform
//                 offset, then diffnetPredict
//   one kernel    diffnetFused, the whole fit and the predictions in one workgroup: in one dispatch, and split into
//                 dispatches of 16 steps (as a host would under a browser's GPU watchdog)
// The plan (the WGSL, the buffers and the reference's weights, predictions and step losses) comes from the C++:
//   build/ofm_anmc --kernels --dump anmc-gpu.json && node web/tests/anmcgpu.mjs anmc-gpu.json
// Everything is computed in C++; this file only uploads the buffers, dispatches and reads back.
// Environment: PLAYWRIGHT_MODULE (path to playwright's index.mjs), CHROMIUM_PATH.
import fs from 'node:fs';
import http from 'node:http';

const planPath = process.argv[2];
if (!planPath) { console.error('usage: node web/tests/anmcgpu.mjs PLAN_JSON'); process.exit(2); }
const plan = JSON.parse(fs.readFileSync(planPath, 'utf8'));

// ---------------------------------------------------------------------------------------------- the hosts (in the page)
const STEP_BYTES = 256;  // kDiffStepWords * 4: the dynamic uniform offset alignment

async function compile(device, wgsl) {
  const module = device.createShaderModule({ code: wgsl });
  const errors = (await module.getCompilationInfo()).messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error('WGSL: ' + errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
  return module;
}
function buffers(device) {
  const U = GPUBufferUsage, made = [];
  const upload = (data, usage) => {
    const b = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage: usage | U.COPY_DST });
    device.queue.writeBuffer(b, 0, data);
    made.push(b);
    return b;
  };
  const storage = (bytes, extra = 0) => { const b = device.createBuffer({ size: Math.max(16, bytes), usage: U.STORAGE | extra }); made.push(b); return b; };
  const reader = (bytes) => { const b = device.createBuffer({ size: Math.max(16, bytes), usage: U.COPY_DST | U.MAP_READ }); made.push(b); return b; };
  return { upload, storage, reader, destroy: () => made.forEach((b) => b.destroy()) };
}
const entry = (binding, type, dynamic = false) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type, hasDynamicOffset: dynamic } });

async function runTwoKernels(device, wgsl, fit, predict) {
  const U = GPUBufferUsage, b = buffers(device), module = await compile(device, wgsl);
  const header = b.upload(new Uint32Array(fit.header), U.UNIFORM), steps = b.upload(new Uint32Array(fit.stepTable), U.UNIFORM);
  const theta = b.upload(new Float32Array(fit.theta), U.STORAGE | U.COPY_SRC), x = b.upload(new Float32Array(fit.x), U.STORAGE);
  const g = b.upload(new Float32Array(fit.g), U.STORAGE), rows = b.upload(new Float32Array(fit.rowData), U.STORAGE);
  const order = b.upload(new Uint32Array(fit.order), U.STORAGE), grads = b.storage(fit.batch * (fit.parameters + 1) * 4);
  const adam = b.storage(fit.parameters * 8);  // (m, v) per weight, zero-initialised by WebGPU
  const pHeader = b.upload(new Uint32Array(predict.header), U.UNIFORM), pX = b.upload(new Float32Array(predict.x), U.STORAGE);
  const outBytes = predict.rows * (predict.inputs + 1) * 4, out = b.storage(outBytes, U.COPY_SRC);
  const layout = (entries) => device.createBindGroupLayout({ entries });
  const gradL = layout([entry(0, 'uniform'), entry(1, 'uniform', true), entry(2, 'storage'), entry(3, 'read-only-storage'),
    entry(4, 'read-only-storage'), entry(5, 'read-only-storage'), entry(6, 'read-only-storage'), entry(7, 'storage')]);
  const adamL = layout([entry(0, 'uniform'), entry(1, 'uniform', true), entry(2, 'storage'), entry(7, 'storage'), entry(8, 'storage')]);
  const predL = layout([entry(0, 'uniform'), entry(2, 'storage'), entry(3, 'read-only-storage'), entry(9, 'storage')]);
  const pipe = (l, entryPoint) => device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [l] }), compute: { module, entryPoint } });
  const bind = (l, list) => device.createBindGroup({ layout: l, entries: list.map(([binding, resource]) => ({ binding, resource })) });
  const step = { buffer: steps, offset: 0, size: 32 };
  const gradBG = bind(gradL, [[0, { buffer: header }], [1, step], [2, { buffer: theta }], [3, { buffer: x }], [4, { buffer: g }], [5, { buffer: rows }],
    [6, { buffer: order }], [7, { buffer: grads }]]);
  const adamBG = bind(adamL, [[0, { buffer: header }], [1, step], [2, { buffer: theta }], [7, { buffer: grads }], [8, { buffer: adam }]]);
  const predBG = bind(predL, [[0, { buffer: pHeader }], [2, { buffer: theta }], [3, { buffer: pX }], [9, { buffer: out }]]);
  const gradP = pipe(gradL, 'diffnetGrad'), adamP = pipe(adamL, 'diffnetAdam'), predP = pipe(predL, 'diffnetPredict');
  const t0 = performance.now();
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  for (let s = 0; s < fit.steps; ++s) {
    pass.setPipeline(gradP);
    pass.setBindGroup(0, gradBG, [s * STEP_BYTES]);
    pass.dispatchWorkgroups(Math.ceil(fit.stepTable[s * 64 + 1] / 64));
    pass.setPipeline(adamP);
    pass.setBindGroup(0, adamBG, [s * STEP_BYTES]);
    pass.dispatchWorkgroups(Math.ceil(fit.parameters / 64));
  }
  pass.setPipeline(predP);
  pass.setBindGroup(0, predBG);
  pass.dispatchWorkgroups(Math.ceil(predict.rows / 64));
  pass.end();
  const tR = b.reader(fit.parameters * 4), oR = b.reader(outBytes);
  enc.copyBufferToBuffer(theta, 0, tR, 0, fit.parameters * 4);
  enc.copyBufferToBuffer(out, 0, oR, 0, outBytes);
  device.queue.submit([enc.finish()]);
  await Promise.all([tR.mapAsync(GPUMapMode.READ), oR.mapAsync(GPUMapMode.READ)]);
  const res = { theta: Array.from(new Float32Array(tR.getMappedRange())), out: Array.from(new Float32Array(oR.getMappedRange())), ms: performance.now() - t0 };
  b.destroy();
  return res;
}

async function runFused(device, wgsl, fused, stepsPerDispatch) {
  const U = GPUBufferUsage, b = buffers(device), module = await compile(device, wgsl);
  const chunks = [];
  for (let first = 0; first < fused.steps; first += stepsPerDispatch) {
    const last = Math.min(fused.steps, first + stepsPerDispatch);
    chunks.push([first, last, last === fused.steps ? 1 : 0]);
  }
  const ranges = new Uint32Array(chunks.length * 64);
  chunks.forEach(([f, l, p], i) => ranges.set([f, l, p, 0], i * 64));
  const header = b.upload(new Uint32Array(fused.header), U.UNIFORM), range = b.upload(ranges, U.UNIFORM);
  const theta = b.upload(new Float32Array(fused.theta), U.STORAGE | U.COPY_SRC), xg = b.upload(new Float32Array(fused.xg), U.STORAGE);
  const rows = b.upload(new Float32Array(fused.rowData), U.STORAGE), datau = b.upload(new Uint32Array(fused.datau), U.STORAGE);
  const work = b.storage(fused.workFloats * 4), out = b.storage(fused.outFloats * 4, U.COPY_SRC);  // WORK zeroed: Adam starts at 0
  const l = device.createBindGroupLayout({ entries: [entry(0, 'uniform'), entry(1, 'uniform', true), entry(2, 'storage'), entry(3, 'read-only-storage'),
    entry(4, 'read-only-storage'), entry(5, 'read-only-storage'), entry(6, 'storage'), entry(7, 'storage')] });
  const pipe = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [l] }), compute: { module, entryPoint: 'diffnetFused' } });
  const bg = device.createBindGroup({ layout: l, entries: [[0, { buffer: header }], [1, { buffer: range, offset: 0, size: 16 }], [2, { buffer: theta }],
    [3, { buffer: xg }], [4, { buffer: rows }], [5, { buffer: datau }], [6, { buffer: work }], [7, { buffer: out }]].map(([binding, resource]) => ({ binding, resource })) });
  const t0 = performance.now();
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  chunks.forEach((_, i) => { pass.setBindGroup(0, bg, [i * STEP_BYTES]); pass.dispatchWorkgroups(1); });
  pass.end();
  const tR = b.reader(fused.parameters * 4), oR = b.reader(fused.outFloats * 4);
  enc.copyBufferToBuffer(theta, 0, tR, 0, fused.parameters * 4);
  enc.copyBufferToBuffer(out, 0, oR, 0, fused.outFloats * 4);
  device.queue.submit([enc.finish()]);
  await Promise.all([tR.mapAsync(GPUMapMode.READ), oR.mapAsync(GPUMapMode.READ)]);
  const all = Array.from(new Float32Array(oR.getMappedRange())), split = fused.predictRows * (fused.inputs + 1);
  const res = { theta: Array.from(new Float32Array(tR.getMappedRange())), predictions: all.slice(0, split), stepLoss: all.slice(split, split + fused.steps),
    dispatches: chunks.length, ms: performance.now() - t0 };
  b.destroy();
  return res;
}

// ---------------------------------------------------------------------------------------------- the run
const server = http.createServer((req, res) => { res.writeHead(200, { 'content-type': 'text/html' }); res.end('<!doctype html><title>anmcgpu</title>'); });
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader'],
});
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/`);
await page.addScriptTag({ content: [STEP_BYTES, compile, buffers, entry, runTwoKernels, runFused].map((f, i) =>
  i === 0 ? `const STEP_BYTES = ${f};` : i === 3 ? `const entry = ${f};` : String(f)).join('\n') });

let ok = true;
const fail = (m) => { console.log(`FAIL: ${m}`); ok = false; };
const maxAbs = (a) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
const rel = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0) / Math.max(maxAbs(b), 1e-30);
const finite = (...arrays) => arrays.every((a) => a.every(Number.isFinite));
const d = plan.predict.inputs;
const values = (a) => a.filter((_, i) => i % (d + 1) === 0), grads = (a) => a.filter((_, i) => i % (d + 1) !== 0);
const bound = 1e-3;  // the GPU's own exp, log and order of rounding against the C++ reference's
try {
  const adapter = await page.evaluate(async () => {
    const a = await navigator.gpu?.requestAdapter();
    return a ? [a.info?.vendor, a.info?.architecture, a.info?.description].filter(Boolean).join(' ') || 'an adapter' : null;
  });
  if (!adapter) throw new Error('no WebGPU adapter');
  const two = await page.evaluate(async ({ wgsl, fit, predict }) =>
    runTwoKernels(await (await navigator.gpu.requestAdapter()).requestDevice(), wgsl, fit, predict), { wgsl: plan.wgsl, fit: plan.fit, predict: plan.predict });
  const e = plan.expected;
  console.log(`WebGPU (${adapter}): ${plan.fit.steps} Adam steps of ${plan.fit.batch} samples, ${plan.fit.parameters} weights, ${plan.predict.rows} predictions`);
  console.log(`two kernels, ${2 * plan.fit.steps + 1} dispatches: ${two.ms.toFixed(0)} ms; weights ${rel(two.theta, e.theta).toExponential(2)}, values ${rel(values(two.out), values(e.out)).toExponential(2)}, gradients ${rel(grads(two.out), grads(e.out)).toExponential(2)} from the C++ reference`);
  if (!finite(two.theta, two.out)) fail('two kernels: a value is not finite');
  for (const [what, a, b] of [['weights', two.theta, e.theta], ['values', values(two.out), values(e.out)], ['gradients', grads(two.out), grads(e.out)]])
    if (!(rel(a, b) < bound)) fail(`two kernels: ${what} ${rel(a, b).toExponential(2)} from the C++ reference`);
  const f = plan.fused, ef = plan.expectedFused;
  for (const per of [f.steps, 16]) {
    const one = await page.evaluate(async ({ wgsl, fused, per }) =>
      runFused(await (await navigator.gpu.requestAdapter()).requestDevice(), wgsl, fused, per), { wgsl: plan.fusedWgsl, fused: f, per });
    console.log(`one kernel, ${one.dispatches} dispatch(es) of one workgroup: ${one.ms.toFixed(0)} ms; weights ${rel(one.theta, ef.theta).toExponential(2)}, values ${rel(values(one.predictions), values(ef.predictions)).toExponential(2)}, gradients ${rel(grads(one.predictions), grads(ef.predictions)).toExponential(2)}, step losses ${rel(one.stepLoss, ef.stepLoss).toExponential(2)} from the C++ reference; weights ${rel(one.theta, two.theta).toExponential(2)} from the two kernels on the GPU`);
    if (!finite(one.theta, one.predictions, one.stepLoss)) fail(`one kernel (${one.dispatches} dispatches): a value is not finite`);
    for (const [what, a, b] of [['weights', one.theta, ef.theta], ['values', values(one.predictions), values(ef.predictions)],
      ['gradients', grads(one.predictions), grads(ef.predictions)], ['step losses', one.stepLoss, ef.stepLoss]])
      if (!(rel(a, b) < bound)) fail(`one kernel (${one.dispatches} dispatches): ${what} ${rel(a, b).toExponential(2)} from the C++ reference`);
  }
} catch (err) {
  fail(err.message);
}
await browser.close();
server.close();
console.log(ok ? 'anmcgpu: the differential network\'s fused kernels on WebGPU give the C++ reference\'s results; OK' : 'anmcgpu: FAILED');
process.exit(ok ? 0 : 1);
