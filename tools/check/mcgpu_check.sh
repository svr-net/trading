#!/bin/sh
# The Lab's GPU kernel (include/strategy/mc_gpu.hpp, web/lab/mc.wgsl) against the engine's own double precision, on a
# generated market with the UK's live settings (web/data/strategies.json) and with a volatility stop:
#   agreement   every batch also run by the kernel's f64 and f32 references and compared with the engine's results
#               (X_MCGPU_CHECK): in single precision at most 0.05% of the slots more than 1e-4 apart and at most 0.01%
#               of the stop decisions different (decisions that sit exactly on a boundary)
#   decisions   the whole run in the GPU's precision (X_MCGPU_EMULATE) makes the same orders and the same backtest trades
#   dumps       with DUMP=DIR, the first batches written for the shader's own test (web/tests/mcgpu.mjs)
# usage: sh tools/check/mcgpu_check.sh build/ofm_strategy [DUMP_DIR]
set -eu
BIN=$1
DUMP=${2:-}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
"$BIN" --synthetic "$W/syn.csv" > /dev/null
LIVE=$(jq -r '.markets.uk[] | select(.role == "live") | .params | to_entries[] | "--param \(.key)=\(.value)"' web/data/strategies.json | tr '\n' ' ')
check() {  # NAME DUMPDIR ARGS...
  name=$1 dir=$2
  shift 2
  [ -n "$dir" ] && mkdir -p "$dir"
  line=$(X_MCGPU_CHECK=1 X_MCGPU_DUMP="$dir" "$BIN" --csv "$W/syn.csv" --costs US --strategy rsi2 --min-value 0 "$@" | grep '^mcgpu:')
  echo "$name: $line"
  echo "$line" | awk -v name="$name" '{
    for (i = 1; i <= NF; i++) { if ($i == "slots;") slots = $(i-1); if ($i == "apart" && $(i+1) == "over" && $(i+2) == "1e-4") over = $(i-1) }
    n = split($0, a, "stops differ "); stops = a[n] + 0
    if (slots == 0) { print name ": no batches"; exit 1 }
    if (over / slots > 0.0005) { print name ": " over " of " slots " slots more than 1e-4 apart in single precision"; exit 1 }
    if (stops / slots > 0.0001) { print name ": " stops " stop decisions differ"; exit 1 }
  }'
  "$BIN" --csv "$W/syn.csv" --costs US --strategy rsi2 --min-value 0 "$@" --json "$W/a.json" --orders "$W/a-o.json" > /dev/null
  X_MCGPU_EMULATE=1 "$BIN" --csv "$W/syn.csv" --costs US --strategy rsi2 --min-value 0 "$@" --json "$W/b.json" --orders "$W/b-o.json" > /dev/null
  [ "$(jq -c '[.orders[].ticker]' "$W/a-o.json")" = "$(jq -c '[.orders[].ticker]' "$W/b-o.json")" ] || { echo "FAIL: $name: the GPU's precision changes the orders"; exit 1; }
  [ "$(jq '.backtest.trades' "$W/a.json")" = "$(jq '.backtest.trades' "$W/b.json")" ] || { echo "FAIL: $name: the GPU's precision changes the backtest's trades"; exit 1; }
  echo "$name: the same orders and $(jq '.backtest.trades' "$W/a.json") backtest trades in the GPU's precision (Sharpe $(jq '.backtest.sharpe' "$W/a.json") / $(jq '.backtest.sharpe' "$W/b.json"))"
}
# shellcheck disable=SC2086
check "live (UK)" "${DUMP:+$DUMP/live}" $LIVE
check "volatility stop" "${DUMP:+$DUMP/vstop}" --param mc=1 --param vstop=3 --param vstopadapt=1 --param rebuy=0.02
check "fixed stop" "${DUMP:+$DUMP/fixed}" --param mc=1 --param mcselect=1 --param stop=0.08 --param rebuy=0.03
echo "mcgpu: the GPU kernel gives the engine's results in single precision; OK"
