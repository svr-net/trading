#!/bin/sh
# The UK model's checks on a generated market (no data licence needed), for CI: it must find a planted
# effect (each share's day, open to close, reverses part of its last day's) and trade it after costs;
# where the day's highs spike and fade (planted), it must learn to take profit, and where they do not,
# not; on 5-minute bars, a moving stop where the day's runs fade and a fixed one on a random walk; its orders must be sensible (bought at the open only, a stop below and a take-profit above the
# entry, an amount within the share's liquidity); and every order it placed must be the same with or
# without the later days' bars (no look-ahead).
#   sh tools/check/uk_leak.sh build/ofm_uk
set -eu
BIN=${1:-build/ofm_uk}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
[ -n "${KEEP:-}" ] && trap - EXIT && echo "kept in $W"
gen() {  # gen <file> <how far a day's high may spike above its path and fade (0: none)> <5-minute bars file> <mean run that fades (0: none)>
python3 - "$1" "$2" "$3" "$4" <<'PY'  # seeded: the same market on every run (mawk's srand is not)
import math, random, sys
r = random.Random(7)
spike, run = float(sys.argv[2]), float(sys.argv[4])
p = [200 + r.random() * 600 for _ in range(60)]
q = [0.0] * 60
vol = [3000 if s < 10 else 1000000 for s in range(60)]  # S000-S009 trade too little to be ordered
with open(sys.argv[1], 'w') as f, open(sys.argv[3], 'w') as fb:
    f.write('date,ticker,open,high,low,close,volume\n')
    fb.write('date,ticker,time,high,low,close,volume\n')
    for d in range(1100):
        for s in range(60):
            o = p[s] * math.exp(r.gauss(0, 0.003))
            # the day as 30 five-minute steps of a random walk from the open, with the planted drift (the
            # last day's open to close reversed in part) and, with `run`, a rise that fades by the close
            # (its size drawn each day, mean `run`: a trailing stop keeps part of it, the close none)
            a = r.expovariate(1 / run) if run > 0 else 0.0
            x, prev, hi, lo, w = 0.0, 0.0, 0.0, 0.0, 0.0
            for k in range(30):
                w += -0.3 * q[s] / 30 + r.gauss(0, 0.009 / math.sqrt(30))
                x = w + a * math.sin(math.pi * (k + 1) / 30)
                bh, bl = max(prev, x) + abs(r.gauss(0, 0.0005)), min(prev, x) - abs(r.gauss(0, 0.0005))
                fb.write(f'D{d:04d},S{s:03d},{8 * 60 + 5 * k:04d},{o * math.exp(bh):.4f},{o * math.exp(bl):.4f},{o * math.exp(x):.4f},1000\n')
                hi, lo, prev = max(hi, bh), min(lo, bl), x
            c = o * math.exp(x)
            h = o * math.exp(hi) * (1 + spike * r.random())
            f.write(f'D{d:04d},S{s:03d},{o:.4f},{h:.4f},{o * math.exp(lo):.4f},{c:.4f},{int(vol[s] * (0.5 + r.random()))}\n')
            q[s], p[s] = w, c
PY
}
gen "$W/full.csv" 0 "$W/full.bars" 0
gen "$W/spiky.csv" 0.03 /dev/null 0  # daily bars only: the levels on the worse of the two orders
gen "$W/runs.csv" 0 "$W/runs.bars" 0.03
awk -F, 'NR==1 || $1 < "D0900"' "$W/full.csv" > "$W/cut.csv"
seq 0 29 | awk '{printf "S%03d\n", $1}' > "$W/aim.txt"   # half the shares pay no stamp duty
X_TRADELOG="$W/full.log" "$BIN" --csv "$W/full.csv" --bars "$W/full.bars" --costs UK --top 60 --stamp-free "$W/aim.txt" --trace "$W/full.tr" --orders "$W/orders.json" > "$W/full.out"
"$BIN" --csv "$W/cut.csv" --bars "$W/full.bars" --costs UK --top 60 --stamp-free "$W/aim.txt" --trace "$W/cut.tr" > /dev/null
sh=$(sed -n 's/^uk: the account.* a year, Sharpe \([-0-9.]*\),.*/\1/p' "$W/full.out")
awk -v s="$sh" 'BEGIN { exit !(s > 2) }' || { cat "$W/full.out"; echo "FAIL: the planted effect was not found (the account's Sharpe $sh)"; exit 1; }
# The levels: with highs that spike and fade, a take-profit on most sessions; on a random walk, none on most.
"$BIN" --csv "$W/spiky.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" > "$W/spiky.out"
takes() { sed -n 's/^uk:   levels used[^:]*://p' "$1" | tr ';' '\n' | awk 'NF >= 4 { split($2, a, "/"); n = $4 + 0; if (a[2] == "none") no += n; else yes += n } END { printf "%d %d", yes, no }'; }
kinds() { sed -n 's/^uk:   levels used[^:]*://p' "$1" | tr ';' '\n' | awk 'NF >= 4 { n = $4 + 0; if ($1 == "fixed") fx += n; else act += n } END { printf "%d %d", act, fx }'; }
set -- $(takes "$W/spiky.out"); [ "$1" -gt "$2" ] || { cat "$W/spiky.out"; echo "FAIL: spiking highs, but a take-profit on only $1 sessions (none on $2)"; exit 1; }
set -- $(takes "$W/full.out"); [ "$2" -gt "$1" ] || { cat "$W/full.out"; echo "FAIL: a random walk, but a take-profit on $1 sessions (none on $2)"; exit 1; }
# Moving stops, on 5-minute bars: on a random walk a fixed stop on most sessions; where the day's runs
# fade by the close, a trailing or break-even stop on most (and on the daily bars alone, which cannot
# tell the order of the high and the low, not chosen there).
set -- $(kinds "$W/full.out"); [ "$2" -gt "$1" ] || { cat "$W/full.out"; echo "FAIL: a random walk on 5-minute bars, but a moving stop on $1 sessions (fixed on $2)"; exit 1; }
"$BIN" --csv "$W/runs.csv" --bars "$W/runs.bars" --costs UK --top 60 --stamp-free "$W/aim.txt" > "$W/runs.out"
set -- $(kinds "$W/runs.out"); [ "$1" -gt "$2" ] || { cat "$W/runs.out"; echo "FAIL: runs that fade, on 5-minute bars, but a moving stop on only $1 sessions (fixed on $2)"; exit 1; }
grep -q '5-minute bars: [1-9]' "$W/runs.out" || { echo 'FAIL: the 5-minute bars were not read'; exit 1; }
n=$(wc -l < "$W/cut.tr")
[ "$n" -gt 200 ] || { cat "$W/full.out"; echo "FAIL: only $n orders placed"; exit 1; }
head -n "$n" "$W/full.tr" | cmp -s - "$W/cut.tr" || { echo 'FAIL: an order changed with later data (look-ahead)'; exit 1; }
# Every order of the backtest and the next session: in a share with enough volume (S000-S009 trade too
# little), bought at the open only (a limit within 2% above the close), a stop below and any take-profit
# above the entry, an amount of GBP 500 to the capital.
python3 - "$W/orders.json" "$W/full.log" <<'PY'
import csv, json, sys
book = json.load(open(sys.argv[1]))
for o in book['orders']:
    assert o['entry'] == 'open' and o['close'] < o['limit'] <= o['close'] * 1.02, o
    assert -0.2 < o['stopRel'] < 0 and (o['takeRel'] is None or o['takeRel'] > 0) and o['stop'] < o['close'], o
    assert 500 <= o['capital'] <= min(5000, o['maxCapital']) + 1e-6 and o['exit'] == 'close', o
    assert o['cost'] > 0 and o['ifFilled'] > 0, o  # the final orders: the broker's costs applied, each still expected to gain
rows = list(csv.DictReader(open(sys.argv[2])))
assert len(rows) > 200, f'only {len(rows)} orders in the backtest'
for o in rows:
    c, lim, v = float(o['close']), float(o['limit']), float(o['value'])
    assert o['ticker'] >= 'S010', f'an order in a share that trades too little: {o}'
    assert c < lim <= c * 1.02 and float(o['stopRel']) < 0 and float(o['takeRel']) >= 0 and 500 <= v <= 5000, o
    assert float(o['expected']) >= 0.0005, o  # the backtest: before costs
    if o['filled'] != '0':  # bought at the open, under its limit; out at its stop, its take-profit or the close
        e, x = float(o['entry']), float(o['exitPrice'])
        assert e <= lim, o
        if o['exit'] == '1': assert abs(x / e - 1 - float(o['stopRel'])) < 1e-4, o
        if o['exit'] == '2': assert float(o['takeRel']) > 0 and abs(x / e - 1 - float(o['takeRel'])) < 1e-4, o
filled = sum(1 for o in rows if o['filled'] != '0')
assert filled > 0.6 * len(rows), f'only {filled} of {len(rows)} orders bought at the open'
PY
echo "uk: the planted effect found (the account's Sharpe $sh); $n orders unchanged by later data; the levels (fixed, trailing, break-even) learnt where they pay; every order sensible"
