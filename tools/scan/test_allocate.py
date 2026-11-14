"""The account's split over the markets (tools/scan/strategies.py allocate):

  python3 tools/scan/test_allocate.py

A market with no edge gets nothing; two markets that move together share less than an independent one of
the same edge; the shares add up to the whole limit, none over the cap; on paper, each at least the floor.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0]]
import strategies as s  # noqa: E402

r = random.Random(3)
days = [f'2026-{1 + k // 28:02d}-{1 + k % 28:02d}' for k in range(250)]
common = [r.gauss(0, 0.006) for _ in days]
rec = {
    'UK': {d: 0.002 + c + r.gauss(0, 0.003) for d, c in zip(days, common)},   # edge, moves with EU
    'EU': {d: 0.002 + c + r.gauss(0, 0.003) for d, c in zip(days, common)},   # the same edge, moves with UK
    'US': {d: 0.002 + r.gauss(0, 0.0067) for d in days},                        # the same edge and risk, independent
    'APAC': {d: -0.001 + r.gauss(0, 0.01) for d in days},                       # no edge
}
w = s.allocate(rec, cap=0.5)
print('split:', {m: round(v, 3) for m, v in w.items()})
assert w['APAC'] == 0, 'a market with no edge got money'
assert abs(sum(w.values()) - 1) < 1e-9, 'the shares do not add up to the limit'
assert max(w.values()) <= 0.5 + 1e-9, 'a market over the cap'
assert w['US'] > w['UK'] and w['US'] > w['EU'], 'the independent market did not get more than the two that move together'
wp = s.allocate(rec, cap=0.5, floor=0.05)
print('paper:', {m: round(v, 3) for m, v in wp.items()})
assert min(wp.values()) >= 0.05 - 1e-9 and abs(sum(wp.values()) - 1) < 1e-9, wp
assert s.allocate({'UK': {d: -0.001 for d in days}}) == {'UK': 0.0}, 'no edge anywhere: nothing'
one = s.allocate({'INDEX': rec['US'], 'UK': rec['APAC'], 'EU': rec['APAC']}, cap=0.5, floor=0.05)
assert abs(one['INDEX'] - 0.5) < 1e-9 and abs(one['UK'] - 0.05) < 1e-9, f'the floors came out of the capped market, not the unused limit: {one}'
# The outlook (forward_of): the next session's orders, their chance to fill, what the backtest delivered.
same = {'UK': rec['US'], 'EU': {d: v for d, v in rec['US'].items()}}
w0 = s.allocate(same, cap=1.0)
w1 = s.allocate(same, cap=1.0, forward={'UK': 0.004, 'EU': 0.0})
assert abs(w0['UK'] - w0['EU']) < 1e-9 and w1['UK'] > w1['EU'], (w0, w1)
import json, tempfile
d = tempfile.mkdtemp()
json.dump({'slots': 2, 'orders': [{'expected': 0.006, 'holdDays': 3}, {'expected': 0.002, 'holdDays': 2}]}, open(f'{d}/o.json', 'w'))
open(f'{d}/t.csv', 'w').write('date,ticker,expected,filled,ret\n2026-01-02,A,0.01,1,0.004\n2026-01-03,B,0.01,1,0.001\n2026-01-03,C,0.02,0,0\n2025-01-01,D,0.01,1,0.05\n')
out, exp, dl = s.forward_of(f'{d}/o.json', f'{d}/t.csv', '2026-01-01')
assert abs(exp - (0.002 + 0.001) / 2) < 1e-12 and abs(dl - 0.25) < 1e-12 and abs(out - exp * 0.25) < 1e-12, (out, exp, dl)
print('outlook: the next orders and their chance to fill shift the split, discounted by what the backtest delivered; OK')
print('allocate: the split maximises the risk-adjusted return, no money without an edge, capped, floored on paper; OK')
