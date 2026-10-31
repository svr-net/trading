// The AIM shares' tickers (exempt from the UK's 0.5% stamp duty on buying since April 2014), from
// stockanalysis.com's list of the London Stock Exchange's AIM market, one a line, for
// ofm_research --stamp-free. Keeps the committed list when the site does not answer or changed.
//   node tools/eoddata/aim.mjs tools/eoddata/aim.txt
import { readFileSync, writeFileSync } from 'node:fs';

const out = process.argv[2] || 'tools/eoddata/aim.txt';
const tickers = new Set();
for (let page = 1; page <= 5; page++) {
  const r = await fetch(`https://stockanalysis.com/list/london-stock-exchange-aim/${page > 1 ? `?page=${page}` : ''}`).catch(() => null);
  if (!r || !r.ok) break;
  const found = [...(await r.text()).matchAll(/\/quote\/aim\/([A-Za-z0-9.]+)\//g)].map((m) => m[1]);
  const before = tickers.size;
  found.forEach((t) => tickers.add(t));
  if (tickers.size === before) break;
}
let old = [];
try { old = readFileSync(out, 'utf8').split(/\s+/).filter(Boolean); } catch { /* none yet */ }
// A list far shorter than the last one means the page changed: keep the last.
if (tickers.size < Math.max(300, 0.8 * old.length)) {
  console.log(`AIM list: ${tickers.size} tickers found, kept the last (${old.length})`);
} else {
  writeFileSync(out, [...tickers].sort().join('\n') + '\n');
  console.log(`AIM list: ${tickers.size} tickers (was ${old.length})`);
}
