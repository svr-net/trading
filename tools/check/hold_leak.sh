#!/bin/sh
# The multi-day single-share model's checks on a generated market (no data licence needed), for CI: it
# must find a planted effect (each share's day, open to close, reverses part of its last 5 days), and
# every order it placed must be the same with or without the later days' bars (no look-ahead).
#   sh tools/check/hold_leak.sh build/ofm_hold
set -eu
BIN=${1:-build/ofm_hold}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
awk 'BEGIN{srand(11); print "date,ticker,open,high,low,close,volume"; for(s=0;s<60;s++){p[s]=20+rand()*80; for(k=0;k<5;k++)h[s,k]=0}
  for(d=0;d<1100;d++){ day=sprintf("D%04d",d); for(s=0;s<60;s++){ r5=h[s,0]+h[s,1]+h[s,2]+h[s,3]+h[s,4]; o=p[s]*exp((rand()-0.5)*0.01); c=o*exp(-0.15*r5+(rand()-0.5)*0.03)
    hi=(o>c?o:c)*(1+rand()*0.01); lo=(o<c?o:c)*(1-rand()*0.01); printf "%s,S%03d,%.4f,%.4f,%.4f,%.4f,%d\n",day,s,o,hi,lo,c,1000000*(0.5+rand())
    for(k=4;k>0;k--)h[s,k]=h[s,k-1]; h[s,0]=log(c/p[s]); p[s]=c}}}' > "$W/full.csv"
awk -F, 'NR==1 || $1 < "D0900"' "$W/full.csv" > "$W/cut.csv"
"$BIN" --csv "$W/full.csv" --costs US --top 60 --trace "$W/full.tr" > "$W/full.out"
"$BIN" --csv "$W/cut.csv" --costs US --top 60 --trace "$W/cut.tr" > /dev/null
sh=$(sed -n 's/^hold: the account.* a year, Sharpe \([-0-9.]*\) .*/\1/p' "$W/full.out")
awk -v s="$sh" 'BEGIN { exit !(s > 2) }' || { cat "$W/full.out"; echo "FAIL: the planted effect was not found (the account's Sharpe $sh)"; exit 1; }
n=$(wc -l < "$W/cut.tr")
[ "$n" -gt 1000 ] || { echo "FAIL: only $n orders placed"; exit 1; }
head -n "$n" "$W/full.tr" | cmp -s - "$W/cut.tr" || { echo 'FAIL: an order changed with later data (look-ahead)'; exit 1; }
echo "hold: the planted effect found (the account's Sharpe $sh); $n orders unchanged by later data"
