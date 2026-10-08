// Promise-based client for the WASM worker: await sat.call("adaptive", spec).

let worker = null;
let nextId = 1;
const pending = new Map();

function createWorker() {
  // Standalone build: the whole worker (glue + embedded .wasm) ships as a string in a classic
  // script, so it runs from a blob URL without any fetch, from file:// or any static host.
  if (globalThis.__SAT_WORKER_SOURCE__) {
    const blob = new Blob([globalThis.__SAT_WORKER_SOURCE__], { type: 'text/javascript' });
    return new Worker(URL.createObjectURL(blob));
  }
  return new Worker(new URL('./sat-worker.js', import.meta.url), { type: 'module' });
}

function ensureWorker() {
  if (worker) return worker;
  worker = createWorker();
  worker.onmessage = (e) => {
    const p = pending.get(e.data.id);
    if (!p) return;
    pending.delete(e.data.id);
    if (e.data.error) p.reject(new Error(e.data.error));
    else p.resolve({ result: e.data.result, ms: e.data.ms });
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message || 'WASM worker failed to load'));
    pending.clear();
  };
  return worker;
}

/** Calls a library entry point in the worker. Resolves to { result, ms }. */
export function call(fn, spec) {
  const w = ensureWorker();
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    w.postMessage({ id, fn, spec }); // structured clone: keeps the typed arrays of GPU read-backs
  });
}

/** Convenience: resolves to the result only. */
export async function run(fn, spec) {
  return (await call(fn, spec)).result;
}
