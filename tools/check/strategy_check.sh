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
HEAVY=${HEAVY:-rsi2}
# Each strategy's checks in its own process, all strategies at once (their files are their own: $W/<strategy>-...): half
# the runs use the one-thread back end, so one after another left most of the CPU idle. Each strategy's report is
# printed in the strategies' order once all have finished; any failure fails the check.
check() {
  s=$1
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
  # intraday (bought in the opening auction, sold the same day, a take-profit working): no look-ahead
  "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --min-value 0 --param intraday=1 --param takesd=1.5 --trace "$W/$s-id.tr" > "$W/$s-id.out"
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 --param intraday=1 --param takesd=1.5 --trace "$W/$s-id-cut.tr" > /dev/null
  ni=$(wc -l < "$W/$s-id-cut.tr")
  head -n "$ni" "$W/$s-id.tr" | cmp -s - "$W/$s-id-cut.tr" || { echo "FAIL: $s intraday: an order changed with later data (look-ahead)"; exit 1; }
  grep -q "intraday: every share bought" "$W/$s-id.out" || { cat "$W/$s-id.out"; echo "FAIL: $s: the intraday book did not run"; exit 1; }
  # The Monte Carlo, the network, the stops and the market's states are the engine's own code, the same whatever the
  # strategy: checked once, with one strategy ($HEAVY); every strategy gets the checks above.
  if [ "$s" != "$HEAVY" ]; then
    echo "strategy $s: $n orders unchanged by later data (and intraday); the same on every back end ($(echo $backs | tr ' ' ',')); the final orders by expected return, costs applied at the end"
    return 0
  fi
  # the nested Monte Carlo: the same on every back end (counter-based draws), with the trailing buy-back learnt for each share
  for b in $backs; do
    "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --backend "$b" --min-value 0 --param psar=1 --param rebuy=0 --param psarvol=3 --param rebuystop=0.01 --param rebuystopadapt=1 --param rebuytrail=1 --param mc=1 --param mcpaths=200 --json "$W/$s-mc-$b.json" > "$W/$s-mc-$b.out"
    sed 's/"backend":"[a-z]*"//' "$W/$s-mc-$b.json" > "$W/$s-mc-$b.cmp"
    cmp -s "$W/$s-mc-$first.cmp" "$W/$s-mc-$b.cmp" || { echo "FAIL: $s: the Monte Carlo differs between the back ends $first and $b"; exit 1; }
  done
  jq -e '.backtest.mc.resets > 5 and .backtest.mc.coverage >= 0 and .backtest.mc.next.p5 <= .backtest.mc.next.p95' "$W/$s-mc-$first.json" > /dev/null || { echo "FAIL: $s: no Monte Carlo in the report"; exit 1; }
  # the Monte Carlo with the neural overlay, every strategy injected, the autonomous gate choosing the book and the
  # SAR stop learnt for each share: the same on every back end, no look-ahead
  NN="--param mc=1 --param mcselect=1 --param mcnn=1 --param mcobj=1 --param mccandk=2 --param mcpool=1 --param mcgate=1 --param mcnnstep=5 --param mcpaths=100 --param psar=1 --param rebuy=0 --param psaradapt=1"
  for b in $backs; do
    "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --backend "$b" --min-value 0 $NN --json "$W/$s-nn-$b.json" --trace "$W/$s-nn-$b.tr" > "$W/$s-nn-$b.out"
    sed 's/"backend":"[a-z]*"//' "$W/$s-nn-$b.json" > "$W/$s-nn-$b.cmp"
    cmp -s "$W/$s-nn-$first.cmp" "$W/$s-nn-$b.cmp" || { echo "FAIL: $s: the neural Monte Carlo differs between the back ends $first and $b"; exit 1; }
  done
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 $NN --trace "$W/$s-nn-cut.tr" > /dev/null
  nn=$(wc -l < "$W/$s-nn-cut.tr")
  [ "$nn" -gt 10 ] || { echo "FAIL: $s: only $nn orders with the neural Monte Carlo"; exit 1; }
  head -n "$nn" "$W/$s-nn-$first.tr" | cmp -s - "$W/$s-nn-cut.tr" || { echo "FAIL: $s with the neural Monte Carlo: an order changed with later data (look-ahead)"; exit 1; }
  grep -q "chosen by the simulation" "$W/$s-nn-$first.out" && grep -q "differential neural network" "$W/$s-nn-$first.out" && grep -q "autonomous:" "$W/$s-nn-$first.out" && grep -q "strategies injected" "$W/$s-nn-$first.out" && grep -q "SAR learnt" "$W/$s-nn-$first.out" || { cat "$W/$s-nn-$first.out"; echo "FAIL: $s: the neural Monte Carlo did not run"; exit 1; }
  # the PSAR stop and its buy-back: no look-ahead either
  X_TRADELOG="$W/$s-psar.log" "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --min-value 0 --param psar=1 --param rebuy=0 --trace "$W/$s-psar.tr" > "$W/$s-psar.out"
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 --param psar=1 --param rebuy=0 --trace "$W/$s-psar-cut.tr" > /dev/null
  np=$(wc -l < "$W/$s-psar-cut.tr")
  head -n "$np" "$W/$s-psar.tr" | cmp -s - "$W/$s-psar-cut.tr" || { echo "FAIL: $s with the PSAR stop: an order changed with later data (look-ahead)"; exit 1; }
  grep -q "positions stopped" "$W/$s-psar.out" || { cat "$W/$s-psar.out"; echo "FAIL: $s: the PSAR stop did not run"; exit 1; }
  # the volatility stop, learnt for each share, with the slots it frees refilled: no look-ahead, and it runs
  VS="--param mc=1 --param mcselect=1 --param mcnn=1 --param mcpool=1 --param mcpaths=100 --param vstop=3 --param vstopadapt=1 --param refill=1 --param rebuy=0"
  "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --min-value 0 $VS --trace "$W/$s-vs.tr" > "$W/$s-vs.out"
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 $VS --trace "$W/$s-vs-cut.tr" > /dev/null
  nv=$(wc -l < "$W/$s-vs-cut.tr")
  head -n "$nv" "$W/$s-vs.tr" | cmp -s - "$W/$s-vs-cut.tr" || { echo "FAIL: $s with the volatility stop: an order changed with later data (look-ahead)"; exit 1; }
  grep -q "a volatility stop" "$W/$s-vs.out" && grep -q "volatility stop learnt" "$W/$s-vs.out" && grep -q "slot freed by a stop" "$W/$s-vs.out" || { cat "$W/$s-vs.out"; echo "FAIL: $s: the volatility stop did not run"; exit 1; }
  # the volatility-adjusted SAR, its band learnt for each share, with buy-backs and refills: no look-ahead, and it runs
  PV="--param mc=1 --param mcselect=1 --param mcnn=1 --param mcpool=1 --param mcpaths=100 --param psar=1 --param psarvol=3 --param psarvoladapt=1 --param refill=1 --param rebuy=0"
  "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --min-value 0 $PV --trace "$W/$s-pv.tr" > "$W/$s-pv.out"
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 $PV --trace "$W/$s-pv-cut.tr" > /dev/null
  np2=$(wc -l < "$W/$s-pv-cut.tr")
  head -n "$np2" "$W/$s-pv.tr" | cmp -s - "$W/$s-pv-cut.tr" || { echo "FAIL: $s with the volatility-adjusted SAR: an order changed with later data (look-ahead)"; exit 1; }
  grep -q "volatility-adjusted Parabolic SAR" "$W/$s-pv.out" && grep -q "volatility band learnt" "$W/$s-pv.out" || { cat "$W/$s-pv.out"; echo "FAIL: $s: the volatility-adjusted SAR did not run"; exit 1; }
  # the market's hidden states refitted at each reset, the futures drawn in them: no look-ahead, the same on every back end
  RG="$PV --param mcregime=3"
  for b in $backs; do
    "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --backend "$b" --min-value 0 $RG --json "$W/$s-rg-$b.json" --trace "$W/$s-rg-$b.tr" > "$W/$s-rg-$b.out"
    sed 's/"backend":"[a-z]*"//' "$W/$s-rg-$b.json" > "$W/$s-rg-$b.cmp"
    cmp -s "$W/$s-rg-$first.cmp" "$W/$s-rg-$b.cmp" || { echo "FAIL: $s: the market's states differ between the back ends $first and $b"; exit 1; }
  done
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 $RG --trace "$W/$s-rg-cut.tr" > /dev/null
  nr=$(wc -l < "$W/$s-rg-cut.tr")
  head -n "$nr" "$W/$s-rg-$first.tr" | cmp -s - "$W/$s-rg-cut.tr" || { echo "FAIL: $s with the market's states: an order changed with later data (look-ahead)"; exit 1; }
  grep -q "hidden states, refitted" "$W/$s-rg-$first.out" && jq -e '.backtest.mc.regime.states == 3 and (.backtest.mc.regime.now | add | . > 0.999 and . < 1.001)' "$W/$s-rg-$first.json" > /dev/null \
    || { cat "$W/$s-rg-$first.out"; echo "FAIL: $s: the market's states did not run"; exit 1; }
  # adaptive: the stops learnt again only where the observed moved, the positions not viable stopped early, the network
  # and the market's states learning adaptively between full fits: no look-ahead, the same on every back end
  AD="$RG --param mcnnstep=5 --param psarvoladapt=1 --param rebuystop=0.01 --param rebuystopadapt=1 --param mcreuse=0.1 --param mcprune=2 --param mcprunepaths=64 --param rebalance=1"
  for b in $backs; do
    "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --backend "$b" --min-value 0 $AD --json "$W/$s-ad-$b.json" --trace "$W/$s-ad-$b.tr" > "$W/$s-ad-$b.out"
    sed 's/"backend":"[a-z]*"//' "$W/$s-ad-$b.json" > "$W/$s-ad-$b.cmp"
    cmp -s "$W/$s-ad-$first.cmp" "$W/$s-ad-$b.cmp" || { echo "FAIL: $s: the adaptive Monte Carlo differs between the back ends $first and $b"; exit 1; }
  done
  "$BIN" --csv "$W/cut.csv" --costs US --strategy "$s" --min-value 0 $AD --trace "$W/$s-ad-cut.tr" > /dev/null
  na=$(wc -l < "$W/$s-ad-cut.tr")
  head -n "$na" "$W/$s-ad-$first.tr" | cmp -s - "$W/$s-ad-cut.tr" || { echo "FAIL: $s with the adaptive Monte Carlo: an order changed with later data (look-ahead)"; exit 1; }
  grep -q "viability pilots" "$W/$s-ad-$first.out" && grep -q "learnt again only where" "$W/$s-ad-$first.out" && grep -q "adaptive refits" "$W/$s-ad-$first.out" \
    || { cat "$W/$s-ad-$first.out"; echo "FAIL: $s: the adaptive Monte Carlo did not run"; exit 1; }
  # a synthetic market drawn from the real one (--resample): the same for a seed and on every back end, another for another seed
  for b in $backs; do
    "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --backend "$b" --min-value 0 --param psar=1 --param rebuy=0 --resample 3 --json "$W/$s-rs-$b.json" > /dev/null
    sed 's/"backend":"[a-z]*"//' "$W/$s-rs-$b.json" > "$W/$s-rs-$b.cmp"
    cmp -s "$W/$s-rs-$first.cmp" "$W/$s-rs-$b.cmp" || { echo "FAIL: $s: the resampled market differs between the back ends $first and $b"; exit 1; }
  done
  "$BIN" --csv "$W/full.csv" --costs US --strategy "$s" --min-value 0 --param psar=1 --param rebuy=0 --resample 4 --json "$W/$s-rs4.json" > /dev/null
  [ "$(jq -c '.backtest | del(.resample)' "$W/$s-rs-$first.json")" != "$(jq -c '.backtest | del(.resample)' "$W/$s-rs4.json")" ] || { echo "FAIL: $s: two seeds drew the same market"; exit 1; }
  echo "strategy $s: $n orders unchanged by later data ($np with the PSAR stop and its buy-back, $nn with the neural Monte Carlo choosing); the Monte Carlo the same on every back end; the same on every back end ($(echo $backs | tr ' ' ',')); the final orders by expected return, costs applied at the end"
}
# At most half the CPUs' worth of strategies at once (a run on the threads back end uses every CPU and spin-waits:
# more only oversubscribed the runner).
jobs=$(( $(nproc 2>/dev/null || echo 2) / 2 )); [ "$jobs" -ge 1 ] || jobs=1
fail=0
running=0
pids=""
for s in $names; do
  ( check "$s" ) > "$W/$s.report" 2>&1 &
  pids="$pids $!"
  running=$((running + 1))
  if [ "$running" -ge "$jobs" ]; then
    for p in $pids; do wait "$p" || fail=1; done
    pids=""
    running=0
  fi
done
for p in $pids; do wait "$p" || fail=1; done
for s in $names; do cat "$W/$s.report"; done
[ "$fail" -eq 0 ] || { echo "FAIL: a strategy's checks failed (above)"; exit 1; }
