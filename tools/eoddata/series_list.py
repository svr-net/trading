"""A market made of named series (tools/eoddata/index.txt, forex.txt): each code fetched from the first of its
EODData exchanges that has bars (tools/eoddata/fetch_series.mjs, through the store), written as one CSV of
EXCHANGE:CODE rows; with --inverse, each series' inverse beside it (CODE-: 1/price, its high from the low)
so a long-only model can take either side.

  python3 tools/eoddata/series_list.py tools/eoddata/index.txt data/stocks.csv [--years 9] [--inverse]
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile


def read_list(path):
    out = []
    for line in open(path):
        line = line.split('#', 1)[0].strip()
        if line:
            code, *ex = line.split()
            out.append((code, (ex[0] if ex else 'AMEX').split(',')))
    return out


def fetch(symbols, years, fetch_cmd):
    """{EXCHANGE:CODE: bars} for the symbols (0 when EODData has none), and the CSV's rows."""
    if not symbols:
        return {}, []
    out = tempfile.mktemp(suffix='.csv')
    p = subprocess.run([*fetch_cmd, '--symbols', ','.join(symbols), '--years', str(years), '--out', out], capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    counts = {s: 0 for s in symbols}
    for line in p.stdout.splitlines():
        m = re.match(r'^(\S+)\s+(\d+) bars', line)
        if m and m.group(1) in counts:
            counts[m.group(1)] = int(m.group(2))
    rows = list(csv.reader(open(out)))[1:] if os.path.exists(out) else []
    return counts, rows


def main():
    a = argparse.ArgumentParser()
    a.add_argument('list')
    a.add_argument('out')
    a.add_argument('--years', type=int, default=9)
    a.add_argument('--inverse', action='store_true')
    args = a.parse_args()
    fetch_cmd = os.environ.get('FETCH_SERIES', 'node tools/eoddata/fetch_series.mjs').split()
    todo = read_list(args.list)
    rows, found, missing = [], {}, []
    for k in range(max(len(ex) for _, ex in todo)):  # each pass: the codes not found yet, on their next exchange
        ask = {f'{ex[k]}:{code}': code for code, ex in todo if code not in found and k < len(ex)}
        counts, got = fetch(list(ask), args.years, fetch_cmd)
        for s, n in counts.items():
            if n > 0:
                found[ask[s]] = s
        rows += [r for r in got if r[1] in counts and counts[r[1]] > 0]
    missing = [code for code, _ in todo if code not in found]
    if args.inverse:
        inv = []
        for d, t, o, h, l, c, v in rows:
            o, h, l, c = float(o), float(h), float(l), float(c)
            if min(o, h, l, c) > 0:
                inv.append([d, t + '-', f'{1 / o:.8g}', f'{1 / l:.8g}', f'{1 / h:.8g}', f'{1 / c:.8g}', v])
        rows += inv
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
        w.writerows(rows)
    bad = sum(1 for r in rows if not all(float(x or 0) > 0 for x in r[2:6]))
    print(f'{os.path.basename(args.list)}: {len(found)} of {len(todo)} series' + (f' (none at EODData: {", ".join(missing)})' if missing else '')
          + (', each with its inverse' if args.inverse else '') + f'; {len(rows)} rows in {args.out}'
          + (f'; {bad} of them without a full bar (an open, high, low or close of 0: left out by the models)' if bad else ''))


if __name__ == '__main__':
    main()
