#!/bin/sh
# The UK model's checks on a generated market (no data licence needed), for CI: it must find a planted
# effect (each share's day, open to close, reverses part of its last day's) and trade it after costs,
# its orders must be sensible (a limit near the close, a stop below it, an amount within the share's
# liquidity), and every order it placed must be the same with or without the later days' bars (no look-ahead).
#   sh tools/check/uk_leak.sh build/ofm_uk
set -eu
BIN=${1:-build/ofm_uk}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
python3 - "$W/full.csv" <<'PY'  # seeded: the same market on every run (mawk's srand is not)
import math, random, sys
r = random.Random(7)
p = [200 + r.random() * 600 for _ in range(60)]
q = [0.0] * 60
vol = [3000 if s < 10 else 1000000 for s in range(60)]  # S000-S009 trade too little to be ordered
with open(sys.argv[1], 'w') as f:
    f.write('date,ticker,open,high,low,close,volume\n')
    for d in range(1100):
        for s in range(60):
            o = p[s] * math.exp((r.random() - 0.5) * 0.01)
            c = o * math.exp(-0.3 * q[s] + (r.random() - 0.5) * 0.03)
            hi, lo = max(o, c) * (1 + r.random() * 0.01), min(o, c) * (1 - r.random() * 0.01)
            f.write(f'D{d:04d},S{s:03d},{o:.4f},{hi:.4f},{lo:.4f},{c:.4f},{int(vol[s] * (0.5 + r.random()))}\n')
            q[s], p[s] = math.log(c / o), c
PY
awk -F, 'NR==1 || $1 < "D0900"' "$W/full.csv" > "$W/cut.csv"
seq 0 29 | awk '{printf "S%03d\n", $1}' > "$W/aim.txt"   # half the shares pay no stamp duty
X_TRADELOG="$W/full.log" "$BIN" --csv "$W/full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --trace "$W/full.tr" --orders "$W/orders.json" > "$W/full.out"
"$BIN" --csv "$W/cut.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --trace "$W/cut.tr" > /dev/null
sh=$(sed -n 's/^uk: the account.* a year, Sharpe \([-0-9.]*\),.*/\1/p' "$W/full.out")
awk -v s="$sh" 'BEGIN { exit !(s > 2) }' || { cat "$W/full.out"; echo "FAIL: the planted effect was not found (the account's Sharpe $sh)"; exit 1; }
n=$(wc -l < "$W/cut.tr")
[ "$n" -gt 200 ] || { cat "$W/full.out"; echo "FAIL: only $n orders placed"; exit 1; }
head -n "$n" "$W/full.tr" | cmp -s - "$W/cut.tr" || { echo 'FAIL: an order changed with later data (look-ahead)'; exit 1; }
# Every order of the backtest and the next session: in a share with enough volume (S000-S009 trade too
# little), a limit within 2% above the close, a stop under the close, an amount of GBP 500 to the capital.
python3 - "$W/orders.json" "$W/full.log" <<'PY'
import csv, json, sys
book = json.load(open(sys.argv[1]))
for o in book['orders']:
    assert o['close'] < o['limit'] <= o['close'] * 1.02 and o['stop'] < o['close'], o
    assert 500 <= o['capital'] <= min(5000, o['maxCapital']) + 1e-6, o
    assert o['ifFilled'] >= 0.0005 and o['take'] is None and o['exit'] == 'close', o
rows = list(csv.DictReader(open(sys.argv[2])))
assert len(rows) > 200, f'only {len(rows)} orders in the backtest'
for o in rows:
    c, lim, stop, v = (float(o[k]) for k in ('close', 'limit', 'stop', 'value'))
    assert o['ticker'] >= 'S010', f'an order in a share that trades too little: {o}'
    assert c < lim <= c * 1.02 and stop < c and 500 <= v <= 5000 and float(o['expected']) - float(o['cost']) >= 0.0005, o
filled = sum(1 for o in rows if o['filled'] != '0')
assert filled > 0.6 * len(rows), f'only {filled} of {len(rows)} orders filled'
PY
echo "uk: the planted effect found (the account's Sharpe $sh); $n orders unchanged by later data; every order sensible"
