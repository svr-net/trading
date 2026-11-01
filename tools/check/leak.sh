#!/bin/sh
# The model's no-look-ahead checks, on a generated market (no data licence needed), for CI:
#  1. the future changed: every bar from a date on replaced by noise; every trade before it must be
#     identical (no price, volume or result from later enters an earlier decision);
#  2. a day's bars partly removed (shares suspended): that day's picks and every earlier trade must be
#     identical (whether a share trades tomorrow is not known at the close);
#  3. X_VERIFY: the fused kernel and the trade path that makes the orders agree on every checked trade.
#   sh tools/check/leak.sh build/ofm_research
set -eu
BIN=${1:-build/ofm_research}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
# 1,100 weekdays from 2018 (GNU date), 300 shares: random walks with gaps, ranges and volumes.
i=0; n=0
: > "$W/days"
while [ "$n" -lt 1100 ]; do
  d=$(date -u -d "2018-01-01 +$i day" +%F); u=$(date -u -d "$d" +%u); i=$((i + 1))
  [ "$u" -le 5 ] && { echo "$d" >> "$W/days"; n=$((n + 1)); }
done
awk 'BEGIN { srand(42); print "date,ticker,open,high,low,close,volume" }
  { day[NR] = $1 } END {
    for (s = 1; s <= 300; s++) { p = 20 + 80 * rand(); v = 1e5 * (1 + 20 * rand()); vol = 0.01 + 0.03 * rand()
      for (k = 1; k <= NR; k++) {
        o = p * exp(vol * (rand() - 0.5)); c = o * exp(2 * vol * (rand() - 0.5))
        h = (o > c ? o : c) * (1 + vol * rand()); l = (o < c ? o : c) / (1 + vol * rand())
        printf "%s,S%03d,%.4f,%.4f,%.4f,%.4f,%d\n", day[k], s, o, h, l, c, v * (0.5 + rand()); p = c } } }' "$W/days" > "$W/base.csv"
CUT=$(sed -n 800p "$W/days"); HALT=$(sed -n 700p "$W/days")
awk -F, -v OFS=, -v cut="$CUT" 'BEGIN { srand(7) } NR > 1 && $1 >= cut { f = exp((rand() - 0.5) * 0.2); $3 *= f; $6 *= f; $4 = ($3 > $6 ? $3 : $6) * 1.02; $5 = ($3 < $6 ? $3 : $6) / 1.02; $7 = int($7 * (0.5 + rand())) } { print }' "$W/base.csv" > "$W/noise.csv"
awk -F, -v halt="$HALT" 'BEGIN { srand(11) } NR > 1 && $1 == halt && rand() < 0.1 { next } { print }' "$W/base.csv" > "$W/halt.csv"
for v in base noise halt; do
  X_VERIFY=1 X_TRADELOG="$W/$v.log" "$BIN" --csv "$W/$v.csv" --lag 1 --scan --costs UK --capital 1000 > "$W/$v.out"
  grep -q "verify: [0-9]* kernel trades against the trade path, 0 differ" "$W/$v.out" || { echo "FAIL: kernel and trade path differ ($v)"; grep verify "$W/$v.out"; exit 1; }
done
pre() { awk -F, -v d="$2" 'NR > 1 && $1 < d' "$1"; }
[ "$(pre "$W/base.log" "$CUT" | wc -l)" -gt 100 ] || { echo "FAIL: too few trades to test"; exit 1; }
pre "$W/base.log" "$CUT" > "$W/a"; pre "$W/noise.log" "$CUT" > "$W/b"
cmp -s "$W/a" "$W/b" || { echo "FAIL: trades before $CUT change when later data changes (look-ahead)"; diff "$W/a" "$W/b" | head; exit 1; }
# Before the day: whole trades; on the day: the decisions (share, rank, prediction, sigma, liquidity
# rank, limit), not the outcomes, which change with the missing bars.
pre "$W/base.log" "$HALT" > "$W/a"; pre "$W/halt.log" "$HALT" > "$W/b"
cmp -s "$W/a" "$W/b" || { echo "FAIL: trades before $HALT change when that day's bars are partly missing (look-ahead)"; diff "$W/a" "$W/b" | head; exit 1; }
for v in base halt; do awk -F, -v OFS=, -v d="$HALT" 'NR > 1 && $1 == d { print $1, $2, $3, $4, $5, $6, $17 }' "$W/$v.log" > "$W/$v.day"; done
[ -s "$W/base.day" ] || { echo "FAIL: no picks on $HALT to test"; exit 1; }
cmp -s "$W/base.day" "$W/halt.day" || { echo "FAIL: the picks on $HALT change when that day's bars are partly missing (look-ahead)"; diff "$W/base.day" "$W/halt.day" | head; exit 1; }
echo "no look-ahead: $(pre "$W/base.log" "$CUT" | wc -l) trades before $CUT unchanged by later data; picks on $HALT unchanged by missing bars; kernel and trade path agree"
