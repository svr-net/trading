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
Y = dict(window=250, half_life=60)  # the split's maths, judged on a year of made-up days (the live split: SPLIT_WINDOW)
w = s.allocate(rec, cap=0.5, **Y)
print('split:', {m: round(v, 3) for m, v in w.items()})
assert w['APAC'] == 0, 'a market with no edge got money'
assert abs(sum(w.values()) - 1) < 1e-9, 'the shares do not add up to the limit'
assert max(w.values()) <= 0.5 + 1e-9, 'a market over the cap'
assert w['US'] > w['UK'] and w['US'] > w['EU'], 'the independent market did not get more than the two that move together'
wp = s.allocate(rec, cap=0.5, floor=0.05, **Y)
print('paper:', {m: round(v, 3) for m, v in wp.items()})
assert min(wp.values()) >= 0.05 - 1e-9 and abs(sum(wp.values()) - 1) < 1e-9, wp
assert s.allocate({'UK': {d: -0.001 for d in days}}, **Y) == {'UK': 0.0}, 'no edge anywhere: nothing'
one = s.allocate({'INDEX': rec['US'], 'UK': rec['APAC'], 'EU': rec['APAC']}, cap=0.5, floor=0.05, **Y)
assert abs(one['INDEX'] - 0.5) < 1e-9 and abs(one['UK'] - 0.05) < 1e-9, f'the floors came out of the capped market, not the unused limit: {one}'
# The outlook (forward_of): the next session's orders, their chance to fill, what the backtest delivered.
same = {'UK': rec['US'], 'EU': {d: v for d, v in rec['US'].items()}}
w0 = s.allocate(same, cap=1.0, **Y)
w1 = s.allocate(same, cap=1.0, forward={'UK': 0.004, 'EU': 0.0}, **Y)
assert abs(w0['UK'] - w0['EU']) < 1e-9 and w1['UK'] > w1['EU'], (w0, w1)
import json, tempfile
d = tempfile.mkdtemp()
json.dump({'slots': 2, 'orders': [{'expected': 0.006, 'holdDays': 3}, {'expected': 0.002, 'holdDays': 2}]}, open(f'{d}/o.json', 'w'))
open(f'{d}/t.csv', 'w').write('date,ticker,expected,filled,ret\n2026-01-02,A,0.01,1,0.004\n2026-01-03,B,0.01,1,0.001\n2026-01-03,C,0.02,0,0\n2025-01-01,D,0.01,1,0.05\n')
out, exp, dl = s.forward_of(f'{d}/o.json', f'{d}/t.csv', '2026-01-01')
wa = 0.5 ** (1 / s.HALF_LIFE)  # A traded a session before B: it counts a little less
want = (wa * 0.004 + 0.001) / (wa * 0.01 + 0.01)
assert abs(exp - (0.002 + 0.001) / 2) < 1e-12 and abs(dl - want) < 1e-12 and abs(out - exp * want) < 1e-12, (out, exp, dl)
# The most recent days count most: of two markets with the same year, the one whose gains came lately gets more.
late = {d: (0.004 if k >= 190 else -0.001) + r.gauss(0, 0.004) for k, d in enumerate(days)}
early = {d: late[days[249 - k]] for k, d in enumerate(days)}  # the same days, in reverse order
wl = s.allocate({'UK': late, 'EU': early}, cap=1.0, window=250, half_life=60)
assert wl['UK'] > wl['EU'], wl
assert s.allocate({'UK': late, 'EU': early}, cap=1.0, window=250, half_life=0)['UK'] < wl['UK'] + 1e-12, 'no weighting gave the recent gains more'
# The live split looks at the period relevant to the multi-day hold only (SPLIT_WINDOW sessions): a market that
# lost the year but gains in its last SPLIT_WINDOW sessions gets more than one that gained the year but loses now.
n = s.SPLIT_WINDOW
turn = {d: (-0.003 if k < 250 - n else 0.006) + r.gauss(0, 0.001) for k, d in enumerate(days)}
fade = {d: (0.003 if k < 250 - n else -0.002) + r.gauss(0, 0.001) for k, d in enumerate(days)}
ws = s.allocate({'UK': turn, 'EU': fade}, cap=1.0)
assert s.SPLIT_WINDOW == 20 and ws['UK'] > ws['EU'], ws
print('the split: the last', n, 'sessions only (the multi-day hold), not the year; OK')
print('recent days count most (exponential weighting); the outlook: the next orders and their chance to fill shift the split, discounted by what the backtest delivered; OK')
print('allocate: the split maximises the risk-adjusted return, no money without an edge, capped, floored on paper; OK')
