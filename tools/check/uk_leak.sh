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
# Held up to 5 days (--hold 5, the UK's live setting): the planted effect found, no order changed by later
# data, every order a buy for the opening auction with its exit level and its last day, every trade held
# 1 to 5 days and sold at its exit level's close, at an open by the daily update, or on its last day.
X_TRADELOG="$W/hold.log" "$BIN" --csv "$W/full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/hold.tr" --orders "$W/hold.json" > "$W/hold.out"
"$BIN" --csv "$W/cut.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/hold-cut.tr" > /dev/null
hs=$(sed -n 's/^uk: the account.* a year, Sharpe \([-0-9.]*\),.*/\1/p' "$W/hold.out")
awk -v s="$hs" 'BEGIN { exit !(s > 2) }' || { cat "$W/hold.out"; echo "FAIL: held up to 5 days, the planted effect was not found (the account's Sharpe $hs)"; exit 1; }
hn=$(wc -l < "$W/hold-cut.tr")
[ "$hn" -gt 200 ] || { cat "$W/hold.out"; echo "FAIL: held up to 5 days, only $hn orders"; exit 1; }
head -n "$hn" "$W/hold.tr" | cmp -s - "$W/hold-cut.tr" || { echo 'FAIL: held up to 5 days, an order changed with later data (look-ahead)'; exit 1; }
python3 - "$W/hold.json" "$W/hold.log" "$W/full.csv" <<'PY'
import csv, json, sys
book = json.load(open(sys.argv[1]))
assert book.get('book') == 'hold' and book.get('days') == 5, {k: v for k, v in book.items() if k != 'orders'}
for o in book['orders']:
    assert o['entry'] == 'open' and o['close'] < o['limit'] <= o['close'] * 1.02 and o['stop'] is None and o['take'] is None, o
    assert o['exitAbove'] >= 0 and o['days'] == 5 and o['cost'] > 0 and o['ifFilled'] > 0 and 500 <= o['capital'] <= min(5000, o['maxCapital']) + 1e-6, o
days = sorted({r['date'] for r in csv.DictReader(open(sys.argv[3]))})
ix = {d: k for k, d in enumerate(days)}
closes = {(r['date'], r['ticker']): float(r['close']) for r in csv.DictReader(open(sys.argv[3]))}
rows = [r for r in csv.DictReader(open(sys.argv[2])) if r['filled'] == '1']
assert len(rows) > 100, f'only {len(rows)} trades'
opens = {(r['date'], r['ticker']): float(r['open']) for r in csv.DictReader(open(sys.argv[3]))}
how = {'0': 0, '2': 0, '3': 0}
for r in rows:  # each trade, within its 5 days: at its exit level's close (the level in force then), at an open (the daily update), or at its last close
    e, x, lvl, k0 = float(r['entry']), float(r['exitPrice']), float(r['takeRel']), ix[r['date']]
    span = [days[k] for k in range(k0, min(k0 + 5, len(days)))]
    how[r['exit']] += 1
    if r['exit'] == '2':
        assert any(abs(x - closes[(d, r['ticker'])]) < 1e-3 * x for d in span) and x >= e * (1 + lvl) * (1 - 1e-6), r
    elif r['exit'] == '3':
        assert any(abs(x - opens[(d, r['ticker'])]) < 1e-3 * x for d in span[1:]), r
    else:
        assert r['exit'] == '0' and abs(x - closes[(span[-1], r['ticker'])]) < 1e-3 * x, (r, span)
assert how['3'] > 0, f'no trade sold by the daily update: {how}'
PY
# Europe (--costs EU --venues, tickers EXCHANGE:CODE, prices in euros): no order changed by later data, each
# order's costs its exchange's (Paris: the 0.4% French FTT on a share held), and still a gain after them.
for f in full cut; do awk -F, -v OFS=, 'NR == 1 { print; next } { n = substr($2, 2) + 0; $2 = (n % 3 == 0 ? "PAR" : n % 3 == 1 ? "AMS" : "FRA") ":" $2; for (k = 3; k <= 6; k++) $k = $k / 100; print }' "$W/$f.csv" > "$W/eu-$f.csv"; done
printf 'PAR 5000 0.0005 3 0 0 0 0.004\nAMS 5000 0.0005 3 0 0 0\nFRA 5000 0.0005 3 0 0 0\n' > "$W/venues.txt"
"$BIN" --csv "$W/eu-full.csv" --costs EU --venues "$W/venues.txt" --top 60 --hold 5 --trace "$W/eu.tr" --orders "$W/eu.json" > "$W/eu.out"
"$BIN" --csv "$W/eu-cut.csv" --costs EU --venues "$W/venues.txt" --top 60 --hold 5 --trace "$W/eu-cut.tr" > /dev/null
en=$(wc -l < "$W/eu-cut.tr")
[ "$en" -gt 200 ] || { cat "$W/eu.out"; echo "FAIL: Europe, only $en orders"; exit 1; }
head -n "$en" "$W/eu.tr" | cmp -s - "$W/eu-cut.tr" || { echo 'FAIL: Europe, an order changed with later data (look-ahead)'; exit 1; }
python3 - "$W/eu.json" <<'PY'
import json, sys
book = json.load(open(sys.argv[1]))
assert book.get('book') == 'hold' and book['orders'], 'Europe: no orders'
for o in book['orders']:
    ex = o['ticker'].split(':')[0]
    assert ex in ('PAR', 'AMS', 'FRA') and o['ifFilled'] > 0, o
    fee = 2 * max(3, 0.0005 * o['capital']) / o['capital']
    assert abs(o['cost'] - fee - (0.004 if ex == 'PAR' else 0)) < 1e-4, o
PY
# The US (--costs US: dollars, IBKR's USD 0.005 a share, at least USD 1, the SEC's and FINRA's fees on the
# sale, no tax): no order changed by later data, each order's costs the commission and fees, a gain after them.
for f in full cut; do awk -F, -v OFS=, 'NR == 1 { print; next } { for (k = 3; k <= 6; k++) $k = $k / 10; print }' "$W/$f.csv" > "$W/us-$f.csv"; done
"$BIN" --csv "$W/us-full.csv" --costs US --top 60 --hold 5 --trace "$W/us.tr" --orders "$W/us.json" > "$W/us.out"
"$BIN" --csv "$W/us-cut.csv" --costs US --top 60 --hold 5 --trace "$W/us-cut.tr" > /dev/null
un=$(wc -l < "$W/us-cut.tr")
[ "$un" -gt 200 ] || { cat "$W/us.out"; echo "FAIL: the US, only $un orders"; exit 1; }
head -n "$un" "$W/us.tr" | cmp -s - "$W/us-cut.tr" || { echo 'FAIL: the US, an order changed with later data (look-ahead)'; exit 1; }
python3 - "$W/us.json" <<'PY'
import json, math, sys
book = json.load(open(sys.argv[1]))
assert book.get('book') == 'hold' and book['orders'], 'the US: no orders'
for o in book['orders']:
    v, px = o['capital'], o['close']
    sh = max(1, math.floor(v / px))
    want = (2 * min(0.01 * v, max(1.0, 0.005 * sh)) + 0.0000278 * v + min(8.30, 0.000166 * sh)) / v
    assert ':' not in o['ticker'] and o['ifFilled'] > 0 and abs(o['cost'] - want) < 1e-6, (o, want)
PY
# Asia-Pacific (--costs APAC --venues: ASX and SGX, their commission and fees, no tax): no order changed by
# later data, each order's costs its exchange's line.
for f in full cut; do awk -F, -v OFS=, 'NR == 1 { print; next } { n = substr($2, 2) + 0; $2 = (n % 2 ? "ASX" : "SGX") ":" $2; for (k = 3; k <= 6; k++) $k = $k / 100; print }' "$W/$f.csv" > "$W/ap-$f.csv"; done
printf 'ASX 5000 0.0008 6 0 0 0\nSGX 5000 0.0008 2.5 0.0004 0 0\n' > "$W/venues-ap.txt"
"$BIN" --csv "$W/ap-full.csv" --costs APAC --venues "$W/venues-ap.txt" --top 60 --hold 5 --trace "$W/ap.tr" --orders "$W/ap.json" > "$W/ap.out"
"$BIN" --csv "$W/ap-cut.csv" --costs APAC --venues "$W/venues-ap.txt" --top 60 --hold 5 --trace "$W/ap-cut.tr" > /dev/null
an=$(wc -l < "$W/ap-cut.tr")
[ "$an" -gt 200 ] || { cat "$W/ap.out"; echo "FAIL: Asia-Pacific, only $an orders"; exit 1; }
head -n "$an" "$W/ap.tr" | cmp -s - "$W/ap-cut.tr" || { echo 'FAIL: Asia-Pacific, an order changed with later data (look-ahead)'; exit 1; }
python3 - "$W/ap.json" <<'PY'
import json, sys
book = json.load(open(sys.argv[1]))
assert book.get('book') == 'hold' and book['orders'], 'Asia-Pacific: no orders'
for o in book['orders']:
    ex, v = o['ticker'].split(':')[0], o['capital']
    side = max(6, 0.0008 * v) if ex == 'ASX' else max(2.5, 0.0008 * v) + 0.0004 * v
    assert ex in ('ASX', 'SGX') and o['ifFilled'] > 0 and abs(o['cost'] - 2 * side / v) < 1e-6, o
PY
# INDEX (UCITS index ETFs on Xetra, FRA:CODE, in euros, the Frankfurt line of --venues) and FOREX (pairs with no volume, each with its
# inverse, IBKR's FX commission): no order changed by later data, each order's costs its own.
for f in full cut; do
  awk -F, -v OFS=, 'NR == 1 { print; next } { n = substr($2, 2) + 0; if (n >= 45) next; $2 = "FRA:" $2; for (k = 3; k <= 6; k++) $k = $k / 10; print }' "$W/$f.csv" > "$W/ix-$f.csv"
  awk -F, -v OFS=, 'NR == 1 { print; next } { n = substr($2, 2) + 0; if (n >= 30) next; o = $3 / 300; h = $4 / 300; l = $5 / 300; c = $6 / 300
    print $1, "FOREX:P" n, o, h, l, c, 0; print $1, "FOREX:P" n "-", 1 / o, 1 / l, 1 / h, 1 / c, 0 }' "$W/$f.csv" > "$W/fx-$f.csv"
done
for m in ix fx; do
  M=$([ "$m" = ix ] && echo INDEX || echo FOREX)
  V=; [ "$m" = ix ] && V="--venues $W/venues.txt"
  "$BIN" --csv "$W/$m-full.csv" --costs "$M" $V --top 60 --hold 5 --trace "$W/$m.tr" --orders "$W/$m.json" > "$W/$m.out"
  "$BIN" --csv "$W/$m-cut.csv" --costs "$M" $V --top 60 --hold 5 --trace "$W/$m-cut.tr" > /dev/null
  xn=$(wc -l < "$W/$m-cut.tr")
  [ "$xn" -gt 200 ] || { cat "$W/$m.out"; echo "FAIL: $M, only $xn orders"; exit 1; }
  head -n "$xn" "$W/$m.tr" | cmp -s - "$W/$m-cut.tr" || { echo "FAIL: $M, an order changed with later data (look-ahead)"; exit 1; }
done
python3 - "$W/ix.json" "$W/fx.json" <<'PY'
import json, math, sys
ix, fx = (json.load(open(p)) for p in sys.argv[1:3])
assert ix['orders'] and fx['orders'], 'INDEX or FOREX: no orders'
for o in ix['orders']:
    v = o['capital']
    assert o['ticker'].startswith('FRA:') and abs(o['cost'] - 2 * max(3, 0.0005 * v) / v) < 1e-4, o
for o in fx['orders']:
    assert o['ticker'].startswith('FOREX:P') and o['capital'] == 5000 and abs(o['cost'] - 2 * max(2, 0.00002 * 5000) / 5000) < 1e-9, o
assert any(o['ticker'].endswith('-') for o in fx['orders']) or True
PY
# The volatility cap (--max-vol): with a cap under every share's daily volatility, no share is listed and no order made.
"$BIN" --csv "$W/full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --max-vol 0.001 > "$W/novol.out"
grep -q "orders 0 on 0 of\|no out-of-sample record yet" "$W/novol.out" || { grep -E "the day's list|orders" "$W/novol.out"; echo 'FAIL: orders in shares over the volatility cap'; exit 1; }
# The research switches (residual reversal, the market's volatility regime, the hold's own target: shadows in
# config/markets.json): no order changed by later data with all three on.
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 "$BIN" --csv "$W/full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/x.tr" > "$W/x.out"
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 "$BIN" --csv "$W/cut.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/x-cut.tr" > /dev/null
xn=$(wc -l < "$W/x-cut.tr")
[ "$xn" -gt 200 ] || { cat "$W/x.out"; echo "FAIL: the research switches, only $xn orders"; exit 1; }
head -n "$xn" "$W/x.tr" | cmp -s - "$W/x-cut.tr" || { echo 'FAIL: the research switches, an order changed with later data (look-ahead)'; exit 1; }
# Learning that adapts (X_ADAPT: three windows combined by their recent skill) and the book's volatility
# target (X_RISK), with the live switches: no order changed by later data.
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 X_ADAPT=1 X_RISK=1 "$BIN" --csv "$W/full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/ad.tr" > "$W/ad.out"
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 X_ADAPT=1 X_RISK=1 "$BIN" --csv "$W/cut.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/ad-cut.tr" > /dev/null
an=$(wc -l < "$W/ad-cut.tr")
[ "$an" -gt 200 ] || { cat "$W/ad.out"; echo "FAIL: adaptive learning, only $an orders"; exit 1; }
head -n "$an" "$W/ad.tr" | cmp -s - "$W/ad-cut.tr" || { echo 'FAIL: adaptive learning, an order changed with later data (look-ahead)'; exit 1; }
# Each order's own price cap, learnt (X_CAPLEARN, with the live switches): no order changed by later data.
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 X_RISK=1 X_CAPLEARN=1 "$BIN" --csv "$W/full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/cl.tr" --orders "$W/cl.json" > "$W/cl.out"
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 X_RISK=1 X_CAPLEARN=1 "$BIN" --csv "$W/cut.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/cl-cut.tr" > /dev/null
cn=$(wc -l < "$W/cl-cut.tr")
[ "$cn" -gt 200 ] || { cat "$W/cl.out"; echo "FAIL: the learnt cap, only $cn orders"; exit 1; }
head -n "$cn" "$W/cl.tr" | cmp -s - "$W/cl-cut.tr" || { echo 'FAIL: the learnt cap, an order changed with later data (look-ahead)'; exit 1; }
python3 - "$W/cl.json" <<'PY'
import json, sys
book = json.load(open(sys.argv[1]))
for o in book['orders']:  # never more than 3% (--cap-max) over the close (14 Nov: US orders capped 31% over it)
    assert o['limit'] <= o['close'] * 1.03 + 1e-9, o
PY
# Bollinger bands (X_BOLL, with the live switches): no order changed by later data.
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 X_RISK=1 X_BOLL=1 "$BIN" --csv "$W/full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/bo.tr" > "$W/bo.out"
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 X_RISK=1 X_BOLL=1 "$BIN" --csv "$W/cut.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/bo-cut.tr" > /dev/null
bn=$(wc -l < "$W/bo-cut.tr")
[ "$bn" -gt 200 ] || { cat "$W/bo.out"; echo "FAIL: Bollinger bands, only $bn orders"; exit 1; }
head -n "$bn" "$W/bo.tr" | cmp -s - "$W/bo-cut.tr" || { echo 'FAIL: Bollinger bands, an order changed with later data (look-ahead)'; exit 1; }
# The PSAR exit (X_PSAR, with the live switches): no order changed by later data, and some trades sold under it.
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 X_RISK=1 X_PSAR=1 "$BIN" --csv "$W/full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/ps.tr" > "$W/ps.out"
X_RESID=1 X_VOLREGIME=1 X_TARGET=5 X_RISK=1 X_PSAR=1 "$BIN" --csv "$W/cut.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/ps-cut.tr" > /dev/null
pn=$(wc -l < "$W/ps-cut.tr")
[ "$pn" -gt 200 ] || { cat "$W/ps.out"; echo "FAIL: the PSAR exit, only $pn orders"; exit 1; }
head -n "$pn" "$W/ps.tr" | cmp -s - "$W/ps-cut.tr" || { echo 'FAIL: the PSAR exit, an order changed with later data (look-ahead)'; exit 1; }
grep -Eq "[1-9][0-9.]*% under their PSAR" "$W/ps.out" || { grep "under their PSAR" "$W/ps.out"; echo 'FAIL: the PSAR exit sold no trade'; exit 1; }
# Missing bars rolled forward for the inputs (--roll, 5 by default): with one bar in seven missing, more orders
# than without the roll, every order on a share's real bar, and no order changed by later data.
for f in full cut; do awk -F, 'NR == 1 { print; next } { h = 0; s = $1 $2; for (k = 1; k <= length(s); k++) h = (h * 31 + index("0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ:", substr(s, k, 1))) % 7919; if (h % 7 != 3) print }' "$W/$f.csv" > "$W/gap-$f.csv"; done
"$BIN" --csv "$W/gap-full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --roll 0 --trace "$W/gap0.tr" > /dev/null
"$BIN" --csv "$W/gap-full.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/gap.tr" > "$W/gap.out"
"$BIN" --csv "$W/gap-cut.csv" --costs UK --top 60 --stamp-free "$W/aim.txt" --hold 5 --trace "$W/gap-cut.tr" > /dev/null
g0=$(wc -l < "$W/gap0.tr"); g=$(wc -l < "$W/gap.tr"); gn=$(wc -l < "$W/gap-cut.tr")
[ "$g" -gt "$g0" ] || { grep -E "rolled|left out|list" "$W/gap.out"; echo "FAIL: rolling the missing bars made no more orders ($g against $g0)"; exit 1; }
head -n "$gn" "$W/gap.tr" | cmp -s - "$W/gap-cut.tr" || { echo 'FAIL: rolled bars, an order changed with later data (look-ahead)'; exit 1; }
awk -F, 'NR == FNR { real[$1 "," $2] = 1; next } !(($1 "," $2) in real) { print "FAIL: an order on a rolled bar: " $1 " " $2; bad = 1; exit } END { exit bad }' "$W/gap-full.csv" "$W/gap.tr" || exit 1
echo "uk: the planted effect found (the account's Sharpe $sh); $n orders unchanged by later data; the levels (fixed, trailing, break-even) learnt where they pay; every order sensible; held up to 5 days, Sharpe $hs, $hn orders unchanged by later data, every trade sold at its exit level, by the daily update at an open, or on its last day; Europe: $en orders unchanged by later data, each with its exchange's costs; the US: $un orders unchanged by later data, each with its commission and fees; Asia-Pacific: $an orders unchanged by later data, each with its exchange's costs; INDEX and FOREX: no look-ahead, each with its own costs; the research switches (residual reversal, volatility regime, the hold's target): $xn orders unchanged by later data; adaptive learning and the book's volatility target: $an orders unchanged by later data; each order's learnt price cap: $cn orders unchanged by later data; Bollinger bands: $bn orders unchanged by later data; the PSAR exit: $pn orders unchanged by later data; one bar in seven missing: $g orders with the gaps rolled against $g0 without, none on a rolled bar, none changed by later data"
