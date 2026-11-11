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
print('allocate: the split maximises the risk-adjusted return, no money without an edge, capped, floored on paper; OK')
