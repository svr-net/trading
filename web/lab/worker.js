// The page's Lab: the strategy engine (apps/ofm_strategy, built to WebAssembly by docker/wasm/Dockerfile) in a background thread,
// run on the market's daily bars from the bridge (prices: never published). One fresh instance per run. gpu: the GPU build
// where the browser has WebGPU (--backend webgpu: the engine itself runs its Monte Carlo on the GPU, backends/webgpu.cpp).
self.onmessage = async ({ data }) => {
  const { csv, env, files } = data, gpu = !!data.gpu && 'gpu' in navigator, lines = [];
  const args = gpu ? [...data.args.filter((a, k, all) => a !== '--backend' && all[k - 1] !== '--backend'), '--backend', 'webgpu'] : data.args;
  try {
    const t0 = performance.now();
    const create = (await import(gpu ? './ofm_strategy_gpu.js' : './ofm_strategy.js')).default;
    let ended;
    const end = new Promise((r) => { ended = r; });
    const m = await create({
      print: (s) => { lines.push(s); if (/^strategy: /.test(s)) self.postMessage({ progress: s }); },
      printErr: (s) => lines.push(s),
      preRun: [(mod) => Object.assign(mod.ENV, env)],
      onExit: ended,
    });
    m.FS.writeFile('/bars.csv', csv);
    for (const [name, text] of Object.entries(files || {})) m.FS.writeFile(`/${name}`, text);
    try { m.callMain([...args, '--json', '/report.json', '--orders', '/orders.json']); } catch (e) { if (!(e && e.name === 'ExitStatus')) throw e; }
    const code = await end;  // the GPU build waits for the GPU: its run ends after callMain returns
    if (code) throw new Error(`the engine stopped (exit ${code})`);
    const read = (f) => { try { return JSON.parse(m.FS.readFile(f, { encoding: 'utf8' })); } catch { return null; } };
    const ran = (lines.find((l) => /^strategy: webgpu: /.test(l)) || '').replace(/^strategy: webgpu: /, '') || 'on the CPU';
    self.postMessage({ done: true, ms: performance.now() - t0, report: read('/report.json'), orders: read('/orders.json'), log: lines, ran });
  } catch (e) {
    self.postMessage({ error: String(e && e.message || e), log: lines });
  }
};
