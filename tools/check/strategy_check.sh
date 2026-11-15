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
  # the nested Monte Carlo: the same on every back end (counter-based draws)
  for b in $backs; do
    "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --backend "$b" --min-value 0 --param psar=1 --param rebuy=0 --param mc=1 --param mcpaths=200 --json "$W/$s-mc-$b.json" > "$W/$s-mc-$b.out"
    sed 's/"backend":"[a-z]*"//' "$W/$s-mc-$b.json" > "$W/$s-mc-$b.cmp"
    cmp -s "$W/$s-mc-$first.cmp" "$W/$s-mc-$b.cmp" || { echo "FAIL: $s: the Monte Carlo differs between the back ends $first and $b"; exit 1; }
  done
  jq -e '.backtest.mc.resets > 5 and .backtest.mc.coverage >= 0 and .backtest.mc.next.p5 <= .backtest.mc.next.p95' "$W/$s-mc-$first.json" > /dev/null || { echo "FAIL: $s: no Monte Carlo in the report"; exit 1; }
  # the Monte Carlo with the neural overlay choosing the book: the same on every back end, no look-ahead
  NN="--param mc=1 --param mcselect=1 --param mcnn=1 --param mcobj=1 --param mccandk=2 --param mcpaths=100"
  for b in $backs; do
    "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --backend "$b" --min-value 0 $NN --json "$W/$s-nn-$b.json" --trace "$W/$s-nn-$b.tr" > "$W/$s-nn-$b.out"
    sed 's/"backend":"[a-z]*"//' "$W/$s-nn-$b.json" > "$W/$s-nn-$b.cmp"
    cmp -s "$W/$s-nn-$first.cmp" "$W/$s-nn-$b.cmp" || { echo "FAIL: $s: the neural Monte Carlo differs between the back ends $first and $b"; exit 1; }
  done
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 $NN --trace "$W/$s-nn-cut.tr" > /dev/null
  nn=$(wc -l < "$W/$s-nn-cut.tr")
  [ "$nn" -gt 10 ] || { echo "FAIL: $s: only $nn orders with the neural Monte Carlo"; exit 1; }
  head -n "$nn" "$W/$s-nn-$first.tr" | cmp -s - "$W/$s-nn-cut.tr" || { echo "FAIL: $s with the neural Monte Carlo: an order changed with later data (look-ahead)"; exit 1; }
  grep -q "chosen by the simulation" "$W/$s-nn-$first.out" && grep -q "differential neural network" "$W/$s-nn-$first.out" || { cat "$W/$s-nn-$first.out"; echo "FAIL: $s: the neural Monte Carlo did not run"; exit 1; }
  # the PSAR stop and its buy-back: no look-ahead either
  X_TRADELOG="$W/$s-psar.log" "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --min-value 0 --param psar=1 --param rebuy=0 --trace "$W/$s-psar.tr" > "$W/$s-psar.out"
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 --param psar=1 --param rebuy=0 --trace "$W/$s-psar-cut.tr" > /dev/null
  np=$(wc -l < "$W/$s-psar-cut.tr")
  head -n "$np" "$W/$s-psar.tr" | cmp -s - "$W/$s-psar-cut.tr" || { echo "FAIL: $s with the PSAR stop: an order changed with later data (look-ahead)"; exit 1; }
  grep -q "positions stopped" "$W/$s-psar.out" || { cat "$W/$s-psar.out"; echo "FAIL: $s: the PSAR stop did not run"; exit 1; }
  jq -e '.costs.net == false and .costs.atEnd == true
         and ([.orders[] | .gross] as $g | $g == ($g | sort | reverse))
         and all(.orders[]; .cost > 0 and .ifFilled > 0 and ((.gross - .cost - .ifFilled) | fabs) < 1e-6)' "$W/$s-$first.json" > /dev/null \
    || { echo "FAIL: $s: the final orders are not in order of expected return, or a cost is missing"; jq -c '.orders[] | {ticker, gross, cost, ifFilled}' "$W/$s-$first.json"; exit 1; }
  echo "strategy $s: $n orders unchanged by later data ($np with the PSAR stop and its buy-back, $nn with the neural Monte Carlo choosing); the Monte Carlo the same on every back end; the same on every back end ($(echo $backs | tr ' ' ',')); the final orders by expected return, costs applied at the end"
done
