"""IBKR's daily bars' coverage of the live pool, by exchange (counts only, no prices): which exchanges'
shares IBKR answers for.
  python3 tools/check/ibkr_coverage.py orders/EU.json ibkr/EU/daily.csv"""
import collections
import csv
import json
import sys

pool = json.load(open(sys.argv[1])).get('pool') or []
got = {r[1] for r in csv.reader(open(sys.argv[2])) if len(r) > 1}
ex = lambda t: t.split(':')[0] if ':' in t else 'all'  # noqa: E731
p = collections.Counter(map(ex, pool))
g = collections.Counter(ex(t) for t in pool if t in got)
print('IBKR bars by exchange (pool shares with bars / in the pool):', ', '.join(f'{k} {g[k]}/{p[k]}' for k in sorted(p)) or 'no pool')
