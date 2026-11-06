#!/bin/sh
# The single-share model's checks on a generated market (no data licence needed), for CI: it must find
# a planted effect (each share's day, open to close, reverses part of its last day's), and every order
# it placed must be the same with or without the later days' bars (no look-ahead).
#   sh tools/check/single_leak.sh build/ofm_single
set -eu
BIN=${1:-build/ofm_single}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
awk 'BEGIN{srand(7); print "date,ticker,open,high,low,close,volume"; for(s=0;s<60;s++){p[s]=20+rand()*80; q[s]=0}
  for(d=0;d<1100;d++){ day=sprintf("D%04d",d); for(s=0;s<60;s++){ o=p[s]*exp((rand()-0.5)*0.01); c=o*exp(-0.3*q[s]+(rand()-0.5)*0.03)
    hi=(o>c?o:c)*(1+rand()*0.01); lo=(o<c?o:c)*(1-rand()*0.01); printf "%s,S%03d,%.4f,%.4f,%.4f,%.4f,%d\n",day,s,o,hi,lo,c,1000000*(0.5+rand())
    q[s]=log(c/o); p[s]=c}}}' > "$W/full.csv"
awk -F, 'NR==1 || $1 < "D0900"' "$W/full.csv" > "$W/cut.csv"
"$BIN" --csv "$W/full.csv" --costs US --top 60 --trace "$W/full.tr" > "$W/full.out"
"$BIN" --csv "$W/cut.csv" --costs US --top 60 --trace "$W/cut.tr" > /dev/null
sh=$(sed -n 's/^single: the account.* a year, Sharpe \([-0-9.]*\) .*/\1/p' "$W/full.out")
awk -v s="$sh" 'BEGIN { exit !(s > 2) }' || { cat "$W/full.out"; echo "FAIL: the planted effect was not found (the account's Sharpe $sh)"; exit 1; }
n=$(wc -l < "$W/cut.tr")
[ "$n" -gt 1000 ] || { echo "FAIL: only $n orders placed"; exit 1; }
head -n "$n" "$W/full.tr" | cmp -s - "$W/cut.tr" || { echo 'FAIL: an order changed with later data (look-ahead)'; exit 1; }
echo "single: the planted effect found (the account's Sharpe $sh); $n orders unchanged by later data"
