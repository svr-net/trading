// WebGPU host for the model's fused kernels. No numerics here: the C++ library (WebAssembly)
// supplies the WGSL sources, the packed tables and each chunk's header; this file uploads them,
// dispatches zscore -> gram, hands the Gram read-back to the library (absorb), uploads the vector
// it returns, dispatches expect, and hands that read-back back (takeExpected).

// Adapters in order of preference: the graphics hardware, then the browser's software adapter (a
// GPU simulated on the CPU, e.g. SwiftShader, running the same WGSL) where there is no graphics
// support. Without WebGPU at all, the page falls back to the kernels' C++ translation (WebAssembly).
const HARDWARE = [{ powerPreference: 'high-performance' }, {}, { featureLevel: 'compatibility' }];
const SOFTWARE = [{ forceFallbackAdapter: true }, { forceFallbackAdapter: true, featureLevel: 'compatibility' }];
const NEED = { maxComputeInvocationsPerWorkgroup: 64, maxComputeWorkgroupSizeX: 64, maxStorageBuffersPerShaderStage: 8 };

const isSoftware = (adapter) => Boolean(adapter.isFallbackAdapter || (adapter.info && adapter.info.isFallbackAdapter));

/** Opens a device. mode: 'auto' (hardware, else the software adapter), 'hardware' or 'software'. */
export async function openDevice(mode = 'auto') {
  if (!('gpu' in navigator)) throw new Error('this browser has no WebGPU');
  const tries = mode === 'software' ? SOFTWARE : mode === 'hardware' ? HARDWARE : [...HARDWARE, ...SOFTWARE];
  let soft = null;
  for (const options of tries) {
    const adapter = await navigator.gpu.requestAdapter(options).catch(() => null);
    if (!adapter || Object.entries(NEED).some(([k, v]) => !(adapter.limits[k] >= v))) continue;
    const software = isSoftware(adapter) || Boolean(options.forceFallbackAdapter);
    if (mode === 'hardware' && software) continue;
    // A software adapter offered for a hardware request: keep it in reserve, prefer real hardware.
    if (mode === 'auto' && software && !options.forceFallbackAdapter) { soft = soft || adapter; continue; }
    return open(adapter, software);
  }
  if (soft) return open(soft, true);
  throw new Error(mode === 'software' ? 'no software WebGPU adapter' : 'no WebGPU adapter');
}

async function open(adapter, software) {
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: { ...NEED, maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize },
  });
  const info = adapter.info || {};
  const name = [info.vendor, info.architecture].filter(Boolean).join(' ') || 'GPU';
  return { device, name: software ? `CPU-simulated GPU: ${name}` : name, software };
}

async function pipeline(device, code, label) {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(`${label}: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
  return device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
}

async function readBack(device, src, bytes, Type = Float32Array) {
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, dst, 0, bytes);
  device.queue.submit([enc.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const out = new Type(dst.getMappedRange(0, bytes)).slice();
  dst.unmap();
  dst.destroy();
  return out;
}

/** Runs the model's kernels for every chunk of days. Resolves to { ms, adapter }. */
export async function runKernels(session, onProgress = () => {}, mode = 'auto') {
  const plan = session.plan();
  const { device, name } = await openDevice(mode);
  const [zscore, gram, expect] = await Promise.all([
    pipeline(device, plan.sources.zscore, 'zscore'), pipeline(device, plan.sources.gram, 'gram'), pipeline(device, plan.sources.expect, 'expect'),
  ]);
  const t0 = performance.now();
  const S = GPUBufferUsage.STORAGE;
  const buffer = (bytes, usage) => device.createBuffer({ size: Math.max(16, bytes), usage });
  const tablesData = session.tables();
  const tables = buffer(tablesData.byteLength, S | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(tables, 0, tablesData);
  const uniform = buffer(128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const cdMax = plan.chunkDays;
  const sig = buffer(cdMax * plan.N * plan.K * 4, S);
  const z = buffer(cdMax * plan.N * plan.K * 4, S);
  const g = buffer(cdMax * plan.stride * 4, S | GPUBufferUsage.COPY_SRC);
  const v = buffer(cdMax * plan.K * 4, S | GPUBufferUsage.COPY_DST);
  const e = buffer(cdMax * plan.N * 4, S | GPUBufferUsage.COPY_SRC);
  const group = (p, entries) => device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: entries.map(([binding, buf]) => ({ binding, resource: { buffer: buf } })),
  });
  const gZ = group(zscore, [[0, uniform], [1, tables], [2, sig], [3, z]]);
  const gG = group(gram, [[0, uniform], [1, tables], [2, z], [3, g]]);
  const gE = group(expect, [[0, uniform], [2, z], [3, v], [4, e]]);
  for (let c = 0; c < plan.chunks; c++) {
    const cd = session.chunkLength(c);
    device.queue.writeBuffer(uniform, 0, session.header(c));
    let enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setPipeline(zscore); pass.setBindGroup(0, gZ); pass.dispatchWorkgroups(plan.K, cd);
    pass.setPipeline(gram); pass.setBindGroup(0, gG); pass.dispatchWorkgroups(plan.gramTiles, cd);
    pass.end();
    device.queue.submit([enc.finish()]);
    const gramOut = await readBack(device, g, cd * plan.stride * 4);
    device.queue.writeBuffer(v, 0, session.absorb(c, gramOut));
    enc = device.createCommandEncoder();
    pass = enc.beginComputePass();
    pass.setPipeline(expect); pass.setBindGroup(0, gE); pass.dispatchWorkgroups(Math.ceil(plan.N / 64), cd);
    pass.end();
    device.queue.submit([enc.finish()]);
    session.takeExpected(c, await readBack(device, e, cd * plan.N * 4));
    onProgress((c + 1) / plan.chunks);
  }
  const ms = performance.now() - t0;
  for (const b of [tables, uniform, sig, z, g, v, e]) b.destroy();
  device.destroy();
  return { ms, adapter: name };
}

/** Runs the day-trade kernels (levels with the trades fused, per chunk of days, then book) on the model's
 * expected returns, and hands the read-backs to the library. Resolves to the report (JSON text). */
export async function runDayTrades(session, onProgress = () => {}, mode = 'auto') {
  const plan = session.dayTradePlan();
  if (!plan.ok) return null;
  const { device } = await openDevice(mode);
  const [levels, book] = await Promise.all([pipeline(device, plan.sources.levels, 'levels'), pipeline(device, plan.sources.book, 'book')]);
  const { T, N, K, s } = plan;
  const S = GPUBufferUsage.STORAGE, DST = GPUBufferUsage.COPY_DST, SRC = GPUBufferUsage.COPY_SRC;
  const buffer = (bytes, usage) => device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage });
  const upload = (data) => { const b = buffer(data.byteLength, S | DST); device.queue.writeBuffer(b, 0, data); return b; };
  const uniform = buffer(112, GPUBufferUsage.UNIFORM | DST);
  const bars = upload(plan.bars), life = upload(plan.life), state = upload(plan.initialState);
  const expected = upload(plan.expected), elig = upload(plan.elig);
  const obs = buffer(N * 4 * T * 4, S), idx = buffer(N * 2 * T * 4, S);
  const lev = buffer(T * N * plan.level * 4, S | SRC), tra = buffer(T * N * plan.trade * 4, S | SRC);
  const days = buffer(T * plan.day * 4, S | SRC), booked = buffer((T + 1) * K * 4, S | SRC);
  const group = (p, entries) => device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: entries.map(([binding, buf]) => ({ binding, resource: { buffer: buf } })),
  });
  const gL = group(levels, [[0, uniform], [1, bars], [2, life], [3, state], [4, obs], [5, idx], [6, lev], [7, tra]]);
  const gB = group(book, [[0, uniform], [1, bars], [6, lev], [7, tra], [8, expected], [9, elig], [10, days], [11, booked]]);
  const step = (p, g, x, y = 1) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(x, y);
    pass.end();
    device.queue.submit([enc.finish()]);
  };
  const t0 = performance.now();
  for (let a = 0; a < T; a += plan.chunkDays) {
    device.queue.writeBuffer(uniform, 0, session.dayTradeHeader(a, Math.min(T, a + plan.chunkDays)));
    step(levels, gL, N);
    await device.queue.onSubmittedWorkDone();
    onProgress(`Day trades on WebGPU: day ${Math.min(T, a + plan.chunkDays)} of ${T}`);
  }
  device.queue.writeBuffer(uniform, 0, session.dayTradeHeader(0, T));
  step(book, gB, Math.ceil((T - s) / 64));
  const [L, R, D, B] = await Promise.all([
    readBack(device, lev, T * N * plan.level * 4), readBack(device, tra, T * N * plan.trade * 4),
    readBack(device, days, T * plan.day * 4), readBack(device, booked, (T + 1) * K * 4, Uint32Array),
  ]);
  const ms = performance.now() - t0;
  device.destroy();
  return { json: session.finishDayTrades(L, R, D, B), ms };
}
