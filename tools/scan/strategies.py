"""Runs a market's strategies (config/markets.json) on the day's data: the live one, whose orders the
bridge places, and its shadows, whose orders are kept in the private store and never placed. Every
run is checked as the live one always was (X_VERIFY: the kernel's trades against the trade path).

  python3 tools/scan/strategies.py --market EU [--publish-orders] -- <ofm_research's arguments>

Writes report/ofm.json and report/ofm.txt (the live strategy, as before), report/shadow-<name>.json
and .txt for each shadow, and report/strategies.json (the market's strategies, results only). With
--publish-orders (the main branch): the orders to data/eoddata/orders/<MARKET>.json (live) and
data/eoddata/shadow/<MARKET>-<name>.json (shadows). A strategy with a ref is built at that version
first (in .pins/<ref>); one with "only": "aim" sees only the UK's AIM shares (no stamp duty;
tools/eoddata/aim.txt); one with "app": "single" is the single-share model (apps/ofm_single: every
share learnt and backtested on its own bars only), given the same arguments.

The budget (a market's "budget": {"total": T, "slots": [2, 3, 5, 10]}): the live strategy runs once for
each number of trades k, k trades a day of T/k each (--capital, and each venue's amount in --venues,
scaled to it: a larger trade pays its minimum commission on more money, more trades spread the risk).
Each day the k whose daily results over the year before (the days known then, at least 60) have the
best mean shrunk by its t statistic is the one traded the next (the most trades before that); the
latest choice's orders and report are the live ones. The runs' trade logs and orders stay in
data/budget/ (prices: never published); the choice's record is printed and kept in report/budget.json. A shadow that fails is reported and skipped; the live one failing fails the run.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

NAME = re.compile(r'^[a-z0-9][a-z0-9-]{0,40}$')
REF = re.compile(r'^[A-Za-z0-9._/-]{1,80}$')


def strategies(config, market):
    m = json.load(open(config))['markets'][market]
    out = [dict(m['live'], role='live')] + [dict(s, role='shadow') for s in m.get('shadows', [])]
    names = set()
    for s in out:
        if not NAME.match(s['name']) or s['name'] in names:
            sys.exit(f"strategies: {market}: a bad or repeated name {s['name']!r}")
        names.add(s['name'])
        bad = [k for k in s.get('env', {}) if not re.match(r'^X_[A-Z0-9_]+$', k)]
        if bad:
            sys.exit(f"strategies: {market} {s['name']}: only the model's research switches (X_*) may be set, not {bad}")
        if 'ref' in s and not REF.match(s['ref']):
            sys.exit(f"strategies: {market} {s['name']}: a bad ref {s['ref']!r}")
        if s.get('only') not in (None, 'aim'):
            sys.exit(f"strategies: {market} {s['name']}: an unknown share filter {s['only']!r}")
        if s.get('app') not in (None, 'single'):
            sys.exit(f"strategies: {market} {s['name']}: an unknown model {s['app']!r}")
        if s.get('app') and s['role'] == 'live':
            sys.exit(f"strategies: {market} {s['name']}: the single-share model runs as a shadow only (the bridge places the research model's orders)")
        if s.get('app') and 'ref' in s:
            sys.exit(f"strategies: {market} {s['name']}: a pinned version is for the research model only")
    return out


def binary(ref, app=None):
    """The model's binary: this checkout's, or one built at ref."""
    if app == 'single':
        return './build/ofm_single'
    if not ref:
        return './build/ofm_research'
    root = os.path.join('.pins', ref.replace('/', '_'))
    exe = os.path.join(root, 'build', 'ofm_research')
    if not os.path.exists(exe):
        subprocess.run(['git', 'fetch', '-q', '--depth', '1', 'origin', ref], check=True)
        subprocess.run(['git', 'worktree', 'add', '-f', '--detach', root, 'FETCH_HEAD'], check=True)
        subprocess.run(['cmake', '-S', root, '-B', os.path.join(root, 'build'), '-G', 'Ninja'], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['cmake', '--build', os.path.join(root, 'build'), '--target', 'ofm_research'], check=True, stdout=subprocess.DEVNULL)
    return exe


def budget_of(config, market):
    """The market's total budget and the numbers of trades to choose from, or None."""
    b = json.load(open(config))['markets'][market].get('budget')
    if not b:
        return None
    slots = sorted({int(k) for k in b.get('slots', [2, 3, 5, 10]) if int(k) > 0})
    if float(b['total']) <= 0 or not slots:
        sys.exit(f'strategies: {market}: a bad budget {b!r}')
    return float(b['total']), slots


def with_capital(args_m, amount, tag):
    """The model's arguments with --capital at amount, and --venues' amounts scaled to it."""
    a = list(args_m)
    base = None
    if '--capital' in a:
        k = a.index('--capital') + 1
        base, a[k] = float(a[k]), f'{amount:g}'
    else:
        a += ['--capital', f'{amount:g}']
    if '--venues' in a and base:
        k = a.index('--venues') + 1
        out = f'data/budget/venues-{tag}.txt'
        with open(a[k]) as f, open(out, 'w') as o:
            for line in f:
                p = line.split()
                if line.startswith('#') or len(p) < 4:
                    o.write(line)
                    continue
                p[1] = f'{float(p[1]) * amount / base:g}'
                o.write(' '.join(p) + '\n')
        a[k] = out
    return a


def days_of(path):
    """A run's daily results (X_TRADELOG.days: date,ret)."""
    out = {}
    with open(path) as f:
        next(f, None)
        for line in f:
            d, r = line.strip().split(',')[:2]
            out[d] = float(r)
    return out


def shrunk(x):
    n = len(x)
    if n < 2:
        return 0.0
    m = sum(x) / n
    v = sum((y - m) ** 2 for y in x) / n
    t = m / (v ** 0.5) * n ** 0.5 if v > 0 else 0.0
    return m * max(0.0, 1 - 1 / (t * t)) if t else 0.0


def choose(series, window=250, least=60):
    """Walk-forward: each day the k with the best shrunk mean over the window before it (the most
    trades before least days); returns the days, the chosen k each day, its result, and the next choice."""
    ks = sorted(series)
    days = sorted(set.intersection(*(set(series[k]) for k in ks)))
    picks, meta = [], []

    def best(upto):
        w = days[max(0, upto - window):upto]
        if len(w) < least:
            return ks[-1]
        return max(reversed(ks), key=lambda k: shrunk([series[k][d] for d in w]))
    for i, d in enumerate(days):
        k = best(i)
        picks.append(k)
        meta.append(series[k][d])
    return days, picks, meta, best(len(days))


def stats(x):
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    m = sum(x) / n
    v = sum((y - m) ** 2 for y in x) / n
    return 252 * m, (m / v ** 0.5 * 252 ** 0.5 if v > 0 else 0.0)


def only_aim(csv):
    """The share list's AIM shares only (tools/eoddata/aim.txt), as a file beside it."""
    out = csv + '.aim.csv'
    if not os.path.exists(out):
        aim = {l.split(',')[0].strip() for l in open('tools/eoddata/aim.txt') if l.strip()}
        with open(csv) as f, open(out, 'w') as o:
            o.write(next(f))
            for line in f:
                if line.split(',', 2)[1] in aim:
                    o.write(line)
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--market', required=True)
    a.add_argument('--config', default='config/markets.json')
    a.add_argument('--publish-orders', action='store_true')
    a.add_argument('model', nargs=argparse.REMAINDER)
    args = a.parse_args()
    model = args.model[1:] if args.model[:1] == ['--'] else args.model
    os.makedirs('report', exist_ok=True)
    index = {'market': args.market, 'strategies': []}
    # Every strategy's command first (a pinned version built, the AIM share list written), then all of
    # them at once: each is deterministic and writes only its own files, so running them side by side
    # changes nothing but the time (the model's kernels leave cores idle between their dispatches).
    jobs = []
    budget = budget_of(args.config, args.market)
    variants = []  # (k, the live strategy run with k trades of the budget's k-th)
    for s in strategies(args.config, args.market):
        live = s['role'] == 'live'
        stem = 'ofm' if live else f"shadow-{s['name']}"
        args_m = list(model)
        if s.get('only') == 'aim' and '--csv' in args_m:
            k = args_m.index('--csv') + 1
            args_m[k] = only_aim(args_m[k])
        if live and budget:
            os.makedirs('data/budget', exist_ok=True)
            total, slots = budget
            for k in slots:
                env = {x: v for x, v in os.environ.items() if not x.startswith('X_')}
                env.update(s.get('env', {}), X_VERIFY='1', X_NPICK=str(k), X_TRADELOG=f'data/budget/trades-{k}.csv')
                cmd = [binary(s.get('ref'), s.get('app')), *with_capital(args_m, total / k, k), '--json', f'data/budget/report-{k}.json',
                       '--orders', f'data/budget/orders-{k}.json']
                log = open(f'data/budget/report-{k}.txt', 'w')
                variants.append((k, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
            continue
        cmd = [binary(s.get('ref'), s.get('app')), *args_m, '--json', f'report/{stem}.json']
        if args.publish_orders:
            path = f'data/eoddata/orders/{args.market}.json' if live else f"data/eoddata/shadow/{args.market}-{s['name']}.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cmd += ['--orders', path]
        env = {k: v for k, v in os.environ.items() if not k.startswith('X_')}
        env.update(s.get('env', {}), X_VERIFY='1')
        log = open(f'report/{stem}.txt', 'w')
        jobs.append((s, live, stem, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    if variants:  # the budget's runs: the walk-forward choice of how many trades makes the live strategy
        s = next(x for x in strategies(args.config, args.market) if x['role'] == 'live')
        total, slots = budget
        for k, p, log in variants:
            code = p.wait()
            log.close()
            if code:
                sys.stdout.write(open(f'data/budget/report-{k}.txt').read())
                for _, q, _ in variants:
                    q.kill()
                for _, _, _, q, _ in jobs:
                    q.kill()
                sys.exit(f"strategies: the live strategy {s['name']} with {k} trades failed ({code})")
        series = {k: days_of(f'data/budget/trades-{k}.csv.days') for k, _, _ in variants}
        days, picks, meta, nxt = choose(series)
        lines = [f"budget: {args.market} {total:g} a day over k trades (each {', '.join(f'{k}: {total / k:g}' for k in slots)}); each day the k whose year before was best (walk-forward)"]
        for k in slots:
            a, sh = stats([series[k][d] for d in days])
            lines.append(f"budget:   {k:2d} trades of {total / k:g}: {100 * a:+.1f}% a year, Sharpe {sh:.2f}; chosen on {picks.count(k)} of {len(days)} days")
        a, sh = stats(meta)
        h = len(meta) // 2
        a1, s1 = stats(meta[:h])
        a2, s2 = stats(meta[h:])
        lines.append(f"budget:   chosen day by day: {100 * a:+.1f}% a year, Sharpe {sh:.2f} | first half {100 * a1:+.1f}% {s1:.2f} | second half {100 * a2:+.1f}% {s2:.2f}")
        lines.append(f"budget:   next session: {nxt} trades of {total / nxt:g}")
        shutil.copy(f'data/budget/report-{nxt}.json', 'report/ofm.json')
        with open('report/ofm.txt', 'w') as o:
            o.write('\n'.join(lines) + '\n' + open(f'data/budget/report-{nxt}.txt').read())
        if args.publish_orders:
            os.makedirs(f'data/eoddata/orders', exist_ok=True)
            shutil.copy(f'data/budget/orders-{nxt}.json', f'data/eoddata/orders/{args.market}.json')
        with open('report/budget.json', 'w') as f:
            json.dump({'market': args.market, 'total': total, 'slots': slots, 'next': nxt, 'amount': total / nxt,
                       'chosen': {str(k): picks.count(k) for k in slots}, 'annual': {str(k): stats([series[k][d] for d in days])[0] for k in slots},
                       'auto': {'annual': a, 'sharpe': sh}}, f, indent=1)
        print(f"strategies: {args.market} live {s['name']} (budget {total:g}, {nxt} trades of {total / nxt:g}): {' '.join(f'{x}={v}' for x, v in s.get('env', {}).items()) or 'defaults'}", flush=True)
        sys.stdout.write(open('report/ofm.txt').read())
        index['strategies'].append({'name': s['name'], 'role': 'live', 'env': s.get('env', {}), 'ref': s.get('ref'), 'only': s.get('only'), 'app': s.get('app'),
                                    'report': 'ofm.json', 'budget': {'total': total, 'trades': nxt, 'amount': total / nxt}})
    for s, live, stem, p, log in jobs:
        code = p.wait()
        log.close()
        print(f"strategies: {args.market} {s['role']} {s['name']}" + (f" (at {s['ref']})" if s.get('ref') else '') + (' (AIM shares only)' if s.get('only') == 'aim' else '') + (' (single-share model)' if s.get('app') else '') + f": {' '.join(f'{k}={v}' for k, v in s.get('env', {}).items()) or 'defaults'}", flush=True)
        sys.stdout.write(open(f'report/{stem}.txt').read())
        if code:
            if live:
                for _, _, _, q, _ in jobs:
                    q.kill()
                sys.exit(f"strategies: the live strategy {s['name']} failed ({code})")
            print(f"strategies: the shadow {s['name']} failed ({code}): skipped")
            continue
        index['strategies'].append({'name': s['name'], 'role': s['role'], 'env': s.get('env', {}), 'ref': s.get('ref'), 'only': s.get('only'), 'app': s.get('app'), 'report': f'{stem}.json'})
    with open('report/strategies.json', 'w') as f:
        json.dump(index, f, indent=1)


if __name__ == '__main__':
    main()
