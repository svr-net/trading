// The page's Lab: the strategy engine (apps/ofm_strategy, built to WebAssembly by docker/wasm/Dockerfile) in a background thread,
// run on the market's daily bars from the bridge (prices: never published). One fresh instance per run.
import createOfmStrategy from './ofm_strategy.js';

self.onmessage = async ({ data }) => {
  const { csv, args, env, files } = data;
  const lines = [];
  try {
    const t0 = performance.now();
    const m = await createOfmStrategy({
      print: (s) => { lines.push(s); if (/^strategy: /.test(s)) self.postMessage({ progress: s }); },
      printErr: (s) => lines.push(s),
      preRun: [(mod) => Object.assign(mod.ENV, env)],
    });
    m.FS.writeFile('/bars.csv', csv);
    for (const [name, text] of Object.entries(files || {})) m.FS.writeFile(`/${name}`, text);
    try { m.callMain([...args, '--json', '/report.json', '--orders', '/orders.json']); } catch (e) { if (!(e && e.name === 'ExitStatus')) throw e; }
    const read = (f) => { try { return JSON.parse(m.FS.readFile(f, { encoding: 'utf8' })); } catch { return null; } };
    self.postMessage({ done: true, ms: performance.now() - t0, report: read('/report.json'), orders: read('/orders.json'), log: lines });
  } catch (e) {
    self.postMessage({ error: String(e && e.message || e), log: lines });
  }
};
