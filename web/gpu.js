// WebGPU host for the model's fused kernels. No numerics here: the C++ library (WebAssembly)
// supplies the WGSL sources, the packed tables and each chunk's header; this file uploads them,
// dispatches zscore -> gram, hands the Gram read-back to the library (absorb), uploads the vector
// it returns, dispatches expect, and hands that read-back back (takeExpected).

const ADAPTERS = [{ powerPreference: 'high-performance' }, {}, { featureLevel: 'compatibility' }];
const NEED = { maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxStorageBuffersPerShaderStage: 3 };

export async function openDevice() {
  if (!('gpu' in navigator)) throw new Error('this browser has no WebGPU');
  for (const options of ADAPTERS) {
    const adapter = await navigator.gpu.requestAdapter(options).catch(() => null);
    if (!adapter) continue;
    if (Object.entries(NEED).some(([k, v]) => !(adapter.limits[k] >= v))) continue;
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: { ...NEED, maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize },
    });
    const info = adapter.info || {};
    return { device, name: [info.vendor, info.architecture].filter(Boolean).join(' ') || 'GPU' };
  }
  throw new Error('no WebGPU adapter with 256-invocation workgroups');
}

async function pipeline(device, code, label) {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(`${label}: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')}`);
  return device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
}

async function readBack(device, src, bytes) {
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, dst, 0, bytes);
  device.queue.submit([enc.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(dst.getMappedRange(0, bytes)).slice();
  dst.unmap();
  dst.destroy();
  return out;
}

/** Runs the model's kernels for every chunk of days. Resolves to { ms, adapter }. */
export async function runKernels(session, onProgress = () => {}) {
  const plan = session.plan();
  if (plan.N > plan.maxAssets) throw new Error(`the kernels rank at most ${plan.maxAssets} stocks a day`);
  const { device, name } = await openDevice();
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
  const z = buffer(cdMax * plan.N * plan.K * 4, S);
  const g = buffer(cdMax * plan.stride * 4, S | GPUBufferUsage.COPY_SRC);
  const v = buffer(cdMax * plan.K * 4, S | GPUBufferUsage.COPY_DST);
  const e = buffer(cdMax * plan.N * 4, S | GPUBufferUsage.COPY_SRC);
  const group = (p, entries) => device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: entries.map(([binding, buf]) => ({ binding, resource: { buffer: buf } })),
  });
  const gZ = group(zscore, [[0, uniform], [1, tables], [2, z]]);
  const gG = group(gram, [[0, uniform], [1, tables], [2, z], [3, g]]);
  const gE = group(expect, [[0, uniform], [2, z], [3, v], [4, e]]);
  for (let c = 0; c < plan.chunks; c++) {
    const cd = session.chunkLength(c);
    device.queue.writeBuffer(uniform, 0, session.header(c));
    let enc = device.createCommandEncoder();
    let pass = enc.beginComputePass();
    pass.setPipeline(zscore); pass.setBindGroup(0, gZ); pass.dispatchWorkgroups(plan.K, cd);
    pass.setPipeline(gram); pass.setBindGroup(0, gG); pass.dispatchWorkgroups(Math.ceil(plan.stride / 64), cd);
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
  for (const b of [tables, uniform, z, g, v, e]) b.destroy();
  device.destroy();
  return { ms, adapter: name };
}
