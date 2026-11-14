#!/bin/sh
# The checks of apps/ofm_strategy on a generated market (its --synthetic), for every strategy built in:
#   look-ahead   a run on the market without its last 120 days makes the same orders on every day both have
#   back ends    every back end gives the same orders, trades and report
#   costs        the backtest is before costs; the final orders are in order of expected return (before costs,
#                highest first), each with its costs and still expected to gain after them
# usage: sh tools/check/strategy_check.sh build/ofm_strategy
set -eu
BIN=${1:-build/ofm_strategy}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
"$BIN" --synthetic "$W/full.csv"
cut=$(tail -n +2 "$W/full.csv" | cut -d, -f1 | sort -u | tail -n 120 | head -n 1)
awk -F, -v c="$cut" 'NR == 1 || $1 < c' "$W/full.csv" > "$W/cut.csv"
names=$("$BIN" --list | awk '/^strategies:/ { s = 1; next } /^back ends:/ { s = 0 } s { print $1 }')
backs=$("$BIN" --list | awk '/^back ends:/ { s = 1; next } s { print $1 }')
for s in $names; do
  first=""
  for b in $backs; do
    X_TRADELOG="$W/$s-$b.log" "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --backend "$b" --min-value 0 --json "$W/$s-$b.json" --orders "$W/$s-$b.orders" --trace "$W/$s-$b.tr" > "$W/$s-$b.out"
    sed 's/"backend":"[a-z]*"//' "$W/$s-$b.json" > "$W/$s-$b.cmp"
    if [ -z "$first" ]; then first=$b; continue; fi
    for x in cmp tr orders log; do
      cmp -s "$W/$s-$first.$x" "$W/$s-$b.$x" || { echo "FAIL: $s: the back ends $first and $b differ ($x)"; exit 1; }
    done
  done
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 --trace "$W/$s-cut.tr" > /dev/null
  n=$(wc -l < "$W/$s-cut.tr")
  [ "$n" -gt 10 ] || { echo "FAIL: $s: only $n orders on the shorter market"; exit 1; }
  head -n "$n" "$W/$s-$first.tr" | cmp -s - "$W/$s-cut.tr" || { echo "FAIL: $s: an order changed with later data (look-ahead)"; exit 1; }
  jq -e '.costs.net == false and .costs.atEnd == true
         and ([.orders[] | .gross] as $g | $g == ($g | sort | reverse))
         and all(.orders[]; .cost > 0 and .ifFilled > 0 and ((.gross - .cost - .ifFilled) | fabs) < 1e-6)' "$W/$s-$first.json" > /dev/null \
    || { echo "FAIL: $s: the final orders are not in order of expected return, or a cost is missing"; jq -c '.orders[] | {ticker, gross, cost, ifFilled}' "$W/$s-$first.json"; exit 1; }
  echo "strategy $s: $n orders unchanged by later data; the same on every back end ($(echo $backs | tr ' ' ',')); the final orders by expected return, costs applied at the end"
done
