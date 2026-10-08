// Builds the copy-deployable standalone site from web/.
//
//   node tools/standalone/build.mjs [--out dist] [--web web]
//
// Output (dist/standalone/):
//   *.html                  one page per library context
//   assets/style.<hash>.css
//   assets/app.<hash>.js    every page bundled into one classic script
//   assets/sat-runtime.<hash>.js
//                           the WASM worker (Emscripten glue + library) as a string, with the
//                           .wasm bytes embedded; started from a blob URL
//   manifest.json, README.txt
// and dist/trading-sat-standalone.zip.
//
// The result needs no server, no MIME configuration and no build step where it is deployed:
// there are no ES modules, no .wasm file and no fetch, so it works when opened from file://,
// copied to a share or USB stick, or uploaded to any static host (S3, GitHub Pages, nginx...).
import { build } from 'esbuild';
import { createHash } from 'node:crypto';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import zlib from 'node:zlib';

const here = path.dirname(fileURLToPath(import.meta.url));
const arg = (name, fallback) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const repo = path.resolve(here, '..', '..');
const web = path.resolve(arg('--web', path.join(repo, 'web')));
const outRoot = path.resolve(arg('--out', path.join(repo, 'dist')));
const out = path.join(outRoot, 'standalone');
const assets = path.join(out, 'assets');

// The value only feeds code paths that never run here (module-worker fallback, Emscripten's
// default .wasm lookup); defining it keeps `new URL(..., import.meta.url)` valid in a classic script.
const define = { 'import.meta.url': '"file:///trading-sat-standalone/"' };
const common = { bundle: true, format: 'iife', platform: 'browser', target: 'es2022', minify: true, write: false, logLevel: 'warning', define };

const hash = (data) => createHash('sha256').update(data).digest('hex').slice(0, 10);
const kb = (n) => `${(n / 1024).toFixed(0)} KiB`;

const wasmPath = path.join(web, 'wasm', 'sat.wasm');
if (!fs.existsSync(wasmPath)) throw new Error(`missing ${wasmPath}: build the WASM module first`);
const wasm = fs.readFileSync(wasmPath);

// 1. Worker: Emscripten glue + worker loop, with the .wasm bytes injected as module options.
const worker = await build({
  ...common,
  entryPoints: [path.join(web, 'js', 'sat-worker.js')],
  external: ['module'], // Node-only branch of the Emscripten glue, never taken in a browser
  banner: {
    js: `self.__SAT_MODULE_OPTIONS__={wasmBinary:Uint8Array.from(atob(${JSON.stringify(wasm.toString('base64'))}),c=>c.charCodeAt(0))};`,
  },
});
const runtime = `/* trading SAT: WebAssembly worker with embedded library */\nglobalThis.__SAT_WORKER_SOURCE__=${JSON.stringify(worker.outputFiles[0].text)};\n`;

// 2. Application: all pages in one classic script, lazily initialised per <body data-page>.
const app = await build({ ...common, entryPoints: [path.join(web, 'js', 'app.js')] });
const appJs = app.outputFiles[0].text;

const css = fs.readFileSync(path.join(web, 'css', 'style.css'), 'utf8');

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(assets, { recursive: true });
const files = {};
const emit = (rel, data) => {
  fs.writeFileSync(path.join(out, rel), data);
  files[rel] = { bytes: Buffer.byteLength(data), sha256: createHash('sha256').update(data).digest('hex') };
  return rel;
};
const cssFile = emit(`assets/style.${hash(css)}.css`, css);
const runtimeFile = emit(`assets/sat-runtime.${hash(runtime)}.js`, runtime);
const appFile = emit(`assets/app.${hash(appJs)}.js`, appJs);

// 3. Pages: one HTML file per page found in web/ (same names, so links keep working).
const pages = fs.readdirSync(web).filter((f) => f.endsWith('.html')).map((f) => f.replace(/\.html$/, '')).sort();
for (const page of pages) {
  emit(`${page}.html`, `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>trading SAT</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="${cssFile}">
</head>
<body data-page="${page}">
  <div id="app"></div>
  <noscript>This application needs JavaScript (and WebAssembly).</noscript>
  <script src="${runtimeFile}"></script>
  <script src="${appFile}"></script>
</body>
</html>
`);
}

let commit = process.env.GIT_COMMIT || 'unknown';
if (commit === 'unknown') try { commit = execSync('git rev-parse --short HEAD', { cwd: repo, stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); } catch (_) { /* not a git checkout */ }
emit('README.txt', `trading SAT - standalone build (${commit})

Self-adaptive machine-learning trading library (C++ compiled to WebAssembly) with WebGPU kernels.

Deploy by copying this folder anywhere:
  * open index.html directly in a browser (file:// works, no server needed), or
  * upload the folder as-is to any static host (S3, GitHub Pages, nginx, IIS...).
No server-side code, MIME types or build step are required.

Requirements: a current Chrome, Edge, Firefox or Safari. The strategy pages use
WebGPU when the browser offers it and WebAssembly otherwise; the WebGPU page needs WebGPU.
`);
const manifest = { name: 'trading-sat-standalone', commit, builtAt: new Date().toISOString(), pages, wasmBytes: wasm.length, files };
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');

// 4. Zip archive of the folder (minimal writer: deflate + CRC-32 from node:zlib).
function zip(dir, prefix) {
  const entries = [];
  const walk = (d) => {
    for (const name of fs.readdirSync(d).sort()) {
      const full = path.join(d, name);
      if (fs.statSync(full).isDirectory()) walk(full);
      else entries.push(full);
    }
  };
  walk(dir);
  const chunks = [], central = [];
  let offset = 0;
  for (const full of entries) {
    const name = Buffer.from(`${prefix}/${path.relative(dir, full).split(path.sep).join('/')}`);
    const data = fs.readFileSync(full);
    const packed = zlib.deflateRawSync(data, { level: 9 });
    const crc = zlib.crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); local.writeUInt16LE(20, 4); local.writeUInt16LE(0x0800, 6); local.writeUInt16LE(8, 8);
    local.writeUInt16LE(0, 10); local.writeUInt16LE(0x21, 12); // 1980-01-01: reproducible archives
    local.writeUInt32LE(crc, 14); local.writeUInt32LE(packed.length, 18); local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(name.length, 26); local.writeUInt16LE(0, 28);
    const head = Buffer.alloc(46);
    head.writeUInt32LE(0x02014b50, 0); head.writeUInt16LE(20, 4); head.writeUInt16LE(20, 6); head.writeUInt16LE(0x0800, 8);
    head.writeUInt16LE(8, 10); head.writeUInt16LE(0, 12); head.writeUInt16LE(0x21, 14); head.writeUInt32LE(crc, 16);
    head.writeUInt32LE(packed.length, 20); head.writeUInt32LE(data.length, 24); head.writeUInt16LE(name.length, 28);
    head.writeUInt32LE(offset, 42);
    chunks.push(local, name, packed);
    central.push(head, name);
    offset += local.length + name.length + packed.length;
  }
  const cd = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(cd.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...chunks, cd, end]);
}
const archive = zip(out, 'trading-sat-standalone');
fs.writeFileSync(path.join(outRoot, 'trading-sat-standalone.zip'), archive);

const total = Object.values(files).reduce((s, f) => s + f.bytes, 0);
console.log(`standalone site: ${out}`);
for (const [f, info] of Object.entries(files)) if (f.startsWith('assets/')) console.log(`  ${f.padEnd(36)} ${kb(info.bytes)}`);
console.log(`  ${pages.length} pages, ${kb(total)} total; zip ${kb(archive.length)} -> ${path.join(outRoot, 'trading-sat-standalone.zip')}`);
