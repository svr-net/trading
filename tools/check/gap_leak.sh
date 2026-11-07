#!/bin/sh
# The buy-on-gap model's checks on a generated market (no data licence needed), for CI: it must find a
# planted effect (a share that opens far under its previous low gains to the close), and its days must
# be the same with or without the later days' bars (no look-ahead).
#   sh tools/check/gap_leak.sh build/ofm_gap
set -eu
BIN=${1:-build/ofm_gap}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
awk 'BEGIN{srand(13); print "date,ticker,open,high,low,close,volume"; for(s=0;s<80;s++){p[s]=20+rand()*80; lo[s]=p[s]}
  for(d=0;d<1100;d++){ day=sprintf("D%04d",d); for(s=0;s<80;s++){ g=(rand()<0.08)?-0.025:(rand()-0.5)*0.01; o=p[s]*exp(g)
    c=o*exp(0.004+(g<-0.02?0.02:0)+(rand()-0.5)*0.016); hi=(o>c?o:c)*(1+rand()*0.01); l=(o<c?o:c)*(1-rand()*0.01)
    printf "%s,S%03d,%.4f,%.4f,%.4f,%.4f,%d\n",day,s,o,hi,l,c,1000000*(0.5+rand()); p[s]=c}}}' > "$W/full.csv"
awk -F, 'NR==1 || $1 < "D0900"' "$W/full.csv" > "$W/cut.csv"
"$BIN" --csv "$W/full.csv" --costs US --top 80 --trades "$W/full.tr" > "$W/full.out"
"$BIN" --csv "$W/cut.csv" --costs US --top 80 --trades "$W/cut.tr" > /dev/null
sh=$(sed -n 's/^gap: learnt day by day: .* a year, Sharpe \([-0-9.]*\) .*/\1/p' "$W/full.out")
awk -v s="$sh" 'BEGIN { exit !(s > 2) }' || { cat "$W/full.out"; echo "FAIL: the planted effect was not found (Sharpe $sh)"; exit 1; }
n=$(wc -l < "$W/cut.tr.days")
[ "$n" -gt 300 ] || { echo "FAIL: only $n days"; exit 1; }
head -n "$n" "$W/full.tr.days" | cmp -s - "$W/cut.tr.days" || { echo 'FAIL: a day changed with later data (look-ahead)'; exit 1; }
echo "gap: the planted effect found (Sharpe $sh); $n days unchanged by later data"
