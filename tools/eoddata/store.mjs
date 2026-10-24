// A persistent store of daily bars, one JSON file per symbol under data/eoddata/store/ (kept
// between workflow runs by actions/cache; git-ignored: EODData's licence does not allow
// redistributing its data). Each run only asks the API for the days a symbol is missing, so a
// symbol's history is downloaded once.
import fs from 'node:fs';
import path from 'node:path';

export const STORE = path.join('data', 'eoddata', 'store');

const safe = (s) => s.replace(/[^A-Za-z0-9.-]/g, '_');
const fileOf = (group, code) => path.join(STORE, safe(group), `${safe(code)}.json`);

/** The stored bars of a symbol: Map date -> {o, h, l, c, v} (empty if none). */
export function load(group, code) {
  const f = fileOf(group, code);
  if (!fs.existsSync(f)) return new Map();
  return new Map(Object.entries(JSON.parse(fs.readFileSync(f, 'utf8'))));
}

/** Whether the store holds the symbol (fetched at least once, possibly with no bars). */
export const has = (group, code) => fs.existsSync(fileOf(group, code));

export function save(group, code, bars) {
  const f = fileOf(group, code);
  fs.mkdirSync(path.dirname(f), { recursive: true });
  const o = {};
  for (const d of [...bars.keys()].sort()) o[d] = bars.get(d);
  fs.writeFileSync(f, JSON.stringify(o));
}

export const lastDate = (bars) => (bars.size ? [...bars.keys()].sort().at(-1) : null);

/** The weekday before a date (YYYY-MM-DD). */
export function previousWeekday(d) {
  const t = new Date(`${d}T12:00:00Z`);
  do t.setUTCDate(t.getUTCDate() - 1); while (t.getUTCDay() === 0 || t.getUTCDay() === 6);
  return t.toISOString().slice(0, 10);
}

export const addDays = (d, n) => {
  const t = new Date(`${d}T12:00:00Z`);
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
};

// Field names are matched case-insensitively and by alias, so small API changes don't break the parse.
export const field = (o, ...names) => {
  for (const n of names) for (const k of Object.keys(o)) if (k.toLowerCase() === n.toLowerCase()) return o[k];
  return undefined;
};
export const day = (v) => String(v).slice(0, 10);

/** A bar from an API record, or null. */
export function bar(b) {
  const d = day(field(b, 'dateStamp', 'date') ?? '');
  const c = +field(b, 'close');
  if (!/^\d{4}-\d{2}-\d{2}$/.test(d) || !(c > 0)) return null;
  const v = +field(b, 'volume');
  return [d, { o: +field(b, 'open') || c, h: +field(b, 'high') || c, l: +field(b, 'low') || c, c, v: Number.isFinite(v) ? v : 0 }];
}

/** Adds API records to stored bars; returns how many days were new. */
export function merge(bars, list) {
  let n = 0;
  for (const rec of list) {
    const b = bar(rec);
    if (!b) continue;
    if (!bars.has(b[0])) ++n;
    bars.set(b[0], b[1]);
  }
  return n;
}

/** Deletes all but the newest `keep` files of a directory whose names start with `prefix`. */
export function prune(dir, prefix, keep = 1) {
  if (!fs.existsSync(dir)) return;
  const files = fs.readdirSync(dir).filter((f) => f.startsWith(prefix)).sort();
  for (const f of files.slice(0, Math.max(0, files.length - keep))) fs.rmSync(path.join(dir, f));
}
