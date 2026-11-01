// The London Stock Exchange's trading segment of every share, from its monthly public instrument
// list (docs.londonstockexchange.com, "Instrument list.xlsx", sheet 1.0 All Equity), as
// ticker,segment,market: ofm_research --segments uses it to tell the shares that trade through the
// SETS order book's opening and closing auctions (SET*, STMM, AMSM) from those quoted by market
// makers (SETSqx and others). Keeps the committed list when the exchange does not answer or the
// file changed shape. Needs unzip.
//   node tools/eoddata/lse_segments.mjs tools/eoddata/lse_segments.csv
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const out = process.argv[2] || 'tools/eoddata/lse_segments.csv';
const keep = (why) => { console.log(`LSE segments: ${why}; kept the committed list`); process.exit(0); };
try {
  const idx = await (await fetch('https://docs.londonstockexchange.com/reports/instruments/latest.json')).json();
  const link = idx?.Data?.[0]?.downloadLink;
  if (!link) keep('no instrument list in the index');
  const r = await fetch(link);
  if (!r.ok) keep(`HTTP ${r.status}`);
  const dir = mkdtempSync(join(tmpdir(), 'lse-'));
  writeFileSync(join(dir, 'list.xlsx'), Buffer.from(await r.arrayBuffer()));
  execFileSync('unzip', ['-oq', join(dir, 'list.xlsx'), 'xl/sharedStrings.xml', 'xl/worksheets/sheet1.xml', '-d', dir]);
  const ss = [...readFileSync(join(dir, 'xl/sharedStrings.xml'), 'utf8').matchAll(/<si>([\s\S]*?)<\/si>/g)]
    .map((m) => [...m[1].matchAll(/<t[^>]*>([^<]*)<\/t>/g)].map((x) => x[1]).join(''));
  const rows = [];
  for (const row of readFileSync(join(dir, 'xl/worksheets/sheet1.xml'), 'utf8').matchAll(/<row[^>]*>([\s\S]*?)<\/row>/g)) {
    const cells = {};
    for (const c of row[1].matchAll(/<c r="([A-Z]+)\d+"([^>]*)>(?:<v>([^<]*)<\/v>|<is><t>([^<]*)<\/t><\/is>)?<\/c>/g))
      cells[c[1]] = c[3] !== undefined ? (/t="s"/.test(c[2]) ? ss[+c[3]] : c[3]) : c[4];
    rows.push(cells);
  }
  const hi = rows.findIndex((x) => Object.values(x).includes('TIDM'));
  if (hi < 0) keep('no TIDM column');
  const col = (n) => Object.keys(rows[hi]).find((k) => rows[hi][k] === n);
  const T = col('TIDM'), S = col('Market Segment Code'), M = col('LSE Market');
  if (!T || !S) keep('no segment column');
  const lines = rows.slice(hi + 1).filter((x) => x[T] && x[S]).map((x) => `${x[T]},${x[S]},${(x[M] || '').replace(/,/g, ' ')}`);
  const old = existsSync(out) ? readFileSync(out, 'utf8').split('\n').length - 2 : 0;
  if (lines.length < Math.max(1000, 0.8 * old)) keep(`only ${lines.length} shares`);
  writeFileSync(out, 'ticker,segment,market\n' + lines.join('\n') + '\n');
  console.log(`LSE segments: ${lines.length} shares (was ${old}), list of ${idx.Data[0].publishedDate}`);
} catch (e) {
  keep(e.message);
}
