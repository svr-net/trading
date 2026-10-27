"""Audits the share bars the model reads (data/stocks.csv from tools/eoddata/fetch.mjs) for defects
that distort measures across days (true range, Bollinger bands, close-to-close moves) more than
moves within the day. Prints counts, tickers, dates and ratios only, never prices (EODData's
licence).

    python3 tools/audit/data_audit.py data/stocks.csv
"""
import csv, math, sys
from collections import defaultdict

rows = defaultdict(list)
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        try:
            o, h, l, c, v = (float(r[k]) for k in ("open", "high", "low", "close", "volume"))
        except ValueError:
            continue
        rows[r["ticker"]].append((r["date"], o, h, l, c, v))
dates = sorted({d for b in rows.values() for d, *_ in b})
pos = {d: k for k, d in enumerate(dates)}
n = sum(len(b) for b in rows.values())
print(f"bars {n}, shares {len(rows)}, days {len(dates)} ({dates[0]} .. {dates[-1]})")

count = defaultdict(int)
worst = defaultdict(list)  # kind -> (size, ticker, date)
per = defaultdict(lambda: defaultdict(int))
tr_ratio = []
for t, b in rows.items():
    b.sort()
    trs, rngs = [], []
    for k, (d, o, h, l, c, v) in enumerate(b):
        if not (l <= min(o, c) and h >= max(o, c)):
            count["inconsistent bar (open or close outside high-low)"] += 1; per[t]["inconsistent"] += 1
            worst["inconsistent"].append((max(math.log(max(o, c) / h) if h > 0 else 9, math.log(l / min(o, c)) if min(o, c) > 0 else 9), t, d))
        if h == l:
            count["flat bar (high = low)"] += 1; per[t]["flat"] += 1
        if v <= 0:
            count["no volume"] += 1; per[t]["novol"] += 1
        if h > 0 and l > 0 and math.log(h / l) > math.log(1.3):
            count["range over 30% in a day"] += 1; worst["range"].append((math.log(h / l), t, d))
        if k == 0:
            continue
        pd_, po, ph, pl, pc, pv = b[k - 1]
        hole = pos[d] - pos[pd_] - 1
        if hole >= 5:
            count["holes of 5+ market days"] += 1; per[t]["holes"] += 1; worst["hole"].append((hole, t, d))
        gap = math.log(o / pc)
        if abs(gap) > math.log(1.2):
            count["opening gap over 20% from the previous close"] += 1; per[t]["gap"] += 1; worst["gap"].append((abs(gap), t, d))
        cc = math.log(c / pc)
        if abs(gap) > math.log(1.2) and abs(cc) < math.log(1.05):
            count["  of them: the close back within 5% (a bad open?)"] += 1; worst["badopen"].append((abs(gap), t, d))
        tr = max(h, pc) - min(l, pc)
        if c > 0 and h > l:
            trs.append(tr / c); rngs.append((h - l) / c)
    if len(trs) > 50 and sum(rngs) > 0:
        tr_ratio.append((sum(trs) / sum(rngs), t, len(b)))

for k in sorted(count):
    print(f"{k}: {count[k]} ({100 * count[k] / n:.3f}% of bars)")
for kind, label in (("gap", "largest opening gaps (log)"), ("badopen", "gaps whose close came back (bad opens?)"), ("inconsistent", "most inconsistent bars"),
                    ("range", "largest day ranges (log)"), ("hole", "longest holes (market days)")):
    w = sorted(worst[kind], reverse=True)[:12]
    if w:
        print(label + ": " + ", ".join(f"{t} {d} {s:.2f}" for s, t, d in w))
tr_ratio.sort(reverse=True)
med = sorted(r for r, *_ in tr_ratio)[len(tr_ratio) // 2] if tr_ratio else 0
print(f"true range / day's range, per share: median {med:.2f}; highest: " + ", ".join(f"{t} {r:.2f}" for r, t, _ in tr_ratio[:12]))
bad = sorted(((sum(c.values()), t) for t, c in per.items()), reverse=True)[:12]
print("shares with the most defects: " + ", ".join(f"{t} {s} {dict(per[t])}" for s, t in bad))
