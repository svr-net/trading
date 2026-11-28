#!/usr/bin/env python3
"""The intraday adjustments, learnt from 5-minute bars and tested walk-forward.

Every order of the engine's backtest (X_TRADELOG: its limit, take-profit, stop, buy-back, last close and daily
volatility) is replayed on its day's 5-minute path. At each check (every `--every` bars) the order is in a state:

  an open position:  the time of day and its move since the buy in daily volatilities
  an order not filled (all-day limit): the time of day and the price's distance above the limit in daily volatilities
  an auction order not filled at the open: the same distance (from its limit)

and each action's rest-of-day return is measured from that check (before costs):

  position:   hold | sell now | stop to the buy price (when above it) | stop trailed k volatilities under the price | no take-profit
  not filled: keep | cancel | buy now (the same take-profit and stop)

The policy (a state -> an action) is learnt on the years before and used, check by check, on the next year only:
an action replaces holding only where it beat holding by more than twice its standard error on 30+ cases in the
training years. The report compares the policy's trades with the plain bracket's on the same days, after costs at
the end (an extra round trip for a buy the bracket did not make). --policy writes the policy learnt on every year
(the bridge's table).

usage: intraday_policy.py --trades t.csv --bars dir --daily stocks.csv [--every 6] [--policy out.json]
"""
import argparse, csv, json, math, os, collections

ap = argparse.ArgumentParser()
ap.add_argument('--trades', required=True)
ap.add_argument('--bars', required=True, help='a JSON per symbol: {date: [[time,h,l,c],...]}')
ap.add_argument('--daily', required=True, help='the daily bars (date,ticker,open,...): the day\'s open')
ap.add_argument('--every', type=int, default=6, help='bars between checks (6: 30 minutes)')
ap.add_argument('--slip', type=float, default=0.1, help='a stop sells this many daily volatilities under its level')
ap.add_argument('--cost', type=float, default=0.0002, help='a round trip\'s cost (only for the extra buys)')
ap.add_argument('--policy')
a = ap.parse_args()

rows = list(csv.DictReader(open(a.trades)))
cache = {}
def bars(t, d):
    if t not in cache:
        p = os.path.join(a.bars, t + '.json')
        cache[t] = json.load(open(p)) if os.path.exists(p) else {}
    return cache[t].get(d)

orders = [r for r in rows if r['take'] not in ('', 'null', 'nan') and bars(r['ticker'], r['date'])]
need = {(r['date'], r['ticker']) for r in orders}
opens = {}
with open(a.daily) as f:
    for x in csv.reader(f):
        if (x[0], x[1]) in need: opens[(x[0], x[1])] = float(x[2])

ZB = [-1.5, -0.75, -0.25, 0.25, 0.75, 1.5]  # the move's buckets (daily volatilities)
GB = [0.25, 0.5, 1.0, 2.0]                   # a missed order's distance above its limit
def bucket(x, b): return sum(x > v for v in b)

def run_position(path, j0, entry, take, stop, rb, sig, mods=None):
    """Bars j0.. with a position bought at `entry`: (return to the close, bar of exit or None). mods: {bar: action}
    applied at that bar's close (the bridge acts on a finished bar)."""
    slip = a.slip * sig
    ret_rb = 0.0
    for j in range(j0, len(path)):
        _, h, l, c = path[j][:4]
        if l <= stop:
            out = stop * (1 - slip)
            r = out / entry - 1
            if rb is not None:  # the buy-back: a buy-stop above (m >= 0) or a limit under (m < 0) the stop
                for k in range(j + 1, len(path)):
                    hk, lk = path[k][1], path[k][2]
                    if (rb >= stop and hk >= rb) or (rb < stop and lk <= rb):
                        hit_take = any(path[q][1] >= take for q in range(k + 1, len(path)))
                        ret_rb = (take if hit_take else path[-1][3]) / rb - 1
                        break
            return r + ret_rb, j
        if h >= take: return take / entry - 1, j
        if mods and j in mods:
            act, arg = mods[j]
            if act == 'sell': return c / entry - 1, j
            if act == 'even' and c > entry: stop = max(stop, entry)
            if act == 'trail': stop = max(stop, c * (1 - arg * sig))
            if act == 'notake': take = math.inf
    return path[-1][3] / entry - 1, None

def replay(o, policy=None, record=None):
    """The order's day: (filled, return). policy: state -> action, applied at each check; record(kind, state, action,
    value) gets every action's rest-of-day value at each check (the learning)."""
    path = bars(o['ticker'], o['date'])
    lim, take, stop, sig = float(o['limit']), float(o['take']), float(o['stop']), float(o['sigma'])
    take = take if take > 0 else math.inf  # 0: no take-profit / no stop
    stop = stop if stop > 0 else 0.0
    m = float(o['rebuym']) if o['rebuym'] not in ('', 'nan') else -99
    rb = stop * (1 + m * sig) if m > -98 and stop > 0 else None
    op = opens.get((o['date'], o['ticker']))
    if not op or not sig or sig != sig: return None
    if abs(path[0][3] / op - 1) > 0.05: return None  # the 5-minute bars on another price scale than the daily (a split, another share)
    day = o['dayentry'] == '1'
    checks = list(range(a.every - 1, len(path) - 1, a.every))
    # the buy
    j0, entry = None, None
    if op <= lim: j0, entry = 0, op
    elif day:
        for j, b in enumerate(path):
            if b[2] <= lim: j0, entry = j, lim; break
    # not filled at a check (before its fill): keep, cancel or buy now
    for c in checks:
        if j0 is not None and c >= j0: break
        if not day and c == 0: continue
        px = path[c][3]
        st = (c // a.every, bucket((px / lim - 1) / sig, GB), int(day))
        if record:
            keep = run_position(path, j0, entry, take, stop, rb, sig)[0] if j0 is not None else 0.0
            buy = run_position(path, c + 1, px, take, stop, rb, sig)[0] - a.cost
            record('wait', st, 'keep', keep); record('wait', st, 'cancel', 0.0); record('wait', st, 'buy', buy)
        if policy:
            act = policy.get(('wait',) + st)
            if act == 'cancel': return True, 0.0, 'cancel'
            if act == 'buy':
                return True, run_position(path, c + 1, px, take, stop, rb, sig)[0] - a.cost, 'buy'
    if j0 is None: return False, 0.0, None
    # a position: hold, sell, stop to the buy price, trail, no take-profit
    if record:
        for c in checks:
            if c < j0: continue
            r0, ex = run_position(path, j0, entry, take, stop, rb, sig)
            if ex is not None and ex <= c: break
            px = path[c][3]
            st = (c // a.every, bucket((px / entry - 1) / sig, ZB))
            for act, arg in (('hold', 0), ('sell', 0), ('even', 0), ('trail', 0.5), ('trail', 1.0), ('notake', 0)):
                name = act if act != 'trail' else f'trail{arg}'
                v = run_position(path, j0, entry, take, stop, rb, sig, None if act == 'hold' else {c: (act, arg)})[0]
                record('pos', st, name, v)
    mods = {}
    if policy:
        for c in checks:
            if c < j0: continue
            px = path[c][3]
            st = (c // a.every, bucket((px / entry - 1) / sig, ZB))
            act = policy.get(('pos',) + st)
            if act and act != 'hold':
                mods[c] = ('trail', float(act[5:])) if act.startswith('trail') else (act, 0)
                r, ex = run_position(path, j0, entry, take, stop, rb, sig, mods)
                if ex is not None and ex <= c: break
    return True, run_position(path, j0, entry, take, stop, rb, sig, mods)[0], None

def learn(train):
    vals = collections.defaultdict(lambda: collections.defaultdict(list))
    def rec(kind, st, act, v): vals[(kind,) + st][act].append(v)
    for o in train: replay(o, record=rec)
    pol = {}
    for st, acts in vals.items():
        base = 'hold' if st[0] == 'pos' else 'keep'
        b = acts[base]
        if len(b) < 30: continue
        best, gain = base, 0.0
        for act, v in acts.items():
            if act == base: continue
            dif = [x - y for x, y in zip(v, b)]
            n = len(dif); mu = sum(dif) / n
            sd = math.sqrt(sum((x - mu) ** 2 for x in dif) / max(1, n - 1))
            if mu > 2 * sd / math.sqrt(n) and mu > gain: best, gain = act, mu
        if best != base: pol[st] = best
    return pol

years = sorted({o['date'][:4] for o in orders})
print(f'{len(orders)} orders with 5-minute bars ({years[0]}-{years[-1]}), checks every {a.every * 5} minutes')
tot = collections.Counter()
for y in years[1:]:
    train = [o for o in orders if o['date'][:4] < y]
    test = [o for o in orders if o['date'][:4] == y]
    if len(train) < 300: continue
    pol = learn(train)
    base, polr, n, acted = [], [], 0, collections.Counter()
    for o in test:
        b = replay(o); p = replay(o, policy=pol)
        if b is None: continue
        if b[0] or p[0]:
            base.append(b[1] if b[0] else 0.0); polr.append(p[1] if p[0] else 0.0)
            if p[2]: acted[p[2]] += 1
    k = len(base)
    if not k: continue
    dif = [x - y for x, y in zip(polr, base)]
    mu = sum(dif) / k; sd = math.sqrt(sum((x - mu) ** 2 for x in dif) / max(1, k - 1))
    print(f'{y}: {k} trades; the bracket {100 * sum(base) / k:+.3f}% a trade, with the adjustments {100 * sum(polr) / k:+.3f}% '
          f'(difference {100 * mu:+.3f}% ± {100 * sd / math.sqrt(k):.3f}%); {len(pol)} states with an action; acted on missed orders: {dict(acted)}')
    tot['n'] += k; tot['b'] += sum(base); tot['p'] += sum(polr)
if tot['n']:
    print(f'all test years: {tot["n"]} trades; the bracket {100 * tot["b"] / tot["n"]:+.3f}% a trade, with the adjustments {100 * tot["p"] / tot["n"]:+.3f}% (before costs except the extra buys)')
if a.policy:
    pol = learn(orders)
    json.dump({'every': a.every, 'zBuckets': ZB, 'gapBuckets': GB,
               'states': [{'kind': k[0], 'check': k[1], 'bucket': k[2], **({'dayEntry': bool(k[3])} if k[0] == 'wait' else {}), 'action': v} for k, v in sorted(pol.items())]},
              open(a.policy, 'w'), indent=1)
    print(f'policy: {len(pol)} states with an action -> {a.policy}')
