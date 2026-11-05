#!/bin/sh
# The swing model's checks on a generated market (no data licence needed), for CI: it must find a
# planted effect (an overnight reversal of the last 5 days) in the right window, and its choice on
# every day must be the same with or without the later days' bars (no look-ahead).
#   sh tools/check/swing_leak.sh build/ofm_swing
set -eu
BIN=${1:-build/ofm_swing}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
awk 'BEGIN{srand(5); print "date,ticker,open,high,low,close,volume"; for(s=0;s<300;s++){p[s]=20+rand()*80; for(k=0;k<5;k++)h[s,k]=0}
  for(d=0;d<1100;d++){ day=sprintf("D%04d",d); for(s=0;s<300;s++){ r5=h[s,0]+h[s,1]+h[s,2]+h[s,3]+h[s,4]; o=p[s]*exp(-0.15*r5+(rand()-0.5)*0.01); c=o*exp((rand()-0.5)*0.03)
    hi=(o>c?o:c)*(1+rand()*0.01); lo=(o<c?o:c)*(1-rand()*0.01); printf "%s,S%03d,%.4f,%.4f,%.4f,%.4f,%d\n",day,s,o,hi,lo,c,1000000*(0.5+rand())
    for(k=4;k>0;k--)h[s,k]=h[s,k-1]; h[s,0]=log(c/p[s]); p[s]=c}}}' > "$W/full.csv"
awk -F, 'NR==1 || $1 < "D0900"' "$W/full.csv" > "$W/cut.csv"
"$BIN" --csv "$W/full.csv" --costs US --top 200 --trace "$W/full.tr" > "$W/full.out"
"$BIN" --csv "$W/cut.csv" --costs US --top 200 --trace "$W/cut.tr" > /dev/null
grep -q 'learnt for the next session: close, sold at the next open' "$W/full.out" || { cat "$W/full.out"; echo 'FAIL: the planted overnight effect was not found'; exit 1; }
head -n -1 "$W/cut.tr" > "$W/a"
n=$(wc -l < "$W/a")
[ "$n" -gt 100 ] || { echo "FAIL: only $n days chosen"; exit 1; }
head -n "$n" "$W/full.tr" | cmp -s - "$W/a" || { echo 'FAIL: a choice changed with later data (look-ahead)'; exit 1; }
echo "swing: the planted effect found; $n days' choices unchanged by later data"
