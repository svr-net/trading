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
share learnt and backtested on its own bars only), given the same arguments; one with "app": "hold" the
multi-day single-share model (apps/ofm_hold); one with "app": "uk" the auction model (apps/ofm_uk: orders
sized to each share's liquidity, bought in the opening auction and held up to 5 days with --hold 5,
made only when its expected return clears the order's own costs; every market's live strategy); "args" (a list) are added after the model's arguments
(a later one wins: e.g. ["--holds", "up", "--pricing", "tiered", "--capital", "1000"]).

The budget (a market's "budget": {"total": T, "slots": [2, 3, 5, 10]}): the live strategy runs once for
each number of trades k, k trades a day of T/k each (--capital, and each venue's amount in --venues,
scaled to it: a larger trade pays its minimum commission on more money, more trades spread the risk).
Each day the k whose daily results over the year before (the days known then, at least 60) have the
best mean shrunk by its t statistic is the one traded the next (the most trades before that); the
latest choice's orders and report are the live ones. The runs' trade logs and orders stay in
data/budget/ (prices: never published); the choice's record is printed and kept in report/budget.json.

The account's daily limit ("account": {"daily_max": 50000, "currency": "EUR"}): at most that much is
invested in a day across every market. Each market's share of it follows its live strategy's record
(the chosen day by day results' mean over the last year, shrunk by its t statistic; this market's
from this run, the others' from their latest web/data/budget-<market>.json, at most 10 days old): a
market with no positive record gets none, so it places no orders. The share is converted to the
market's currency with the day's rates (--fx, the scan's FOREX series: EURGBP, GBPUSD, EURAUD). A shadow that fails is reported and skipped; the live one failing fails the run.
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
        if s.get('app') not in (None, 'single', 'hold', 'uk'):
            sys.exit(f"strategies: {market} {s['name']}: an unknown model {s['app']!r}")
        if s.get('app') in ('single', 'hold') and s['role'] == 'live':
            sys.exit(f"strategies: {market} {s['name']}: the single-share models run as shadows only (the bridge places the research model's or the UK model's orders)")
        if s.get('app') == 'uk' and market not in ('UK', 'EU', 'US', 'APAC', 'INDEX', 'FOREX'):
            sys.exit(f"strategies: {market} {s['name']}: the auction model (apps/ofm_uk) is for the UK, Europe, the US and Asia-Pacific only")
        bad_args = [x for x in s.get('args', []) if not isinstance(x, str) or not re.match(r'^[A-Za-z0-9_.,:/=-]{1,64}$', x)]
        if bad_args or not isinstance(s.get('args', []), list):
            sys.exit(f"strategies: {market} {s['name']}: bad args {bad_args or s.get('args')!r}")
        if s.get('app') and 'ref' in s:
            sys.exit(f"strategies: {market} {s['name']}: a pinned version is for the research model only")
    return out


def binary(ref, app=None):
    """The model's binary: this checkout's, or one built at ref."""
    if app in ('single', 'hold', 'uk'):
        return f'./build/ofm_{app}'
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


CURRENCY = {'UK': 'GBP', 'EU': 'EUR', 'US': 'USD', 'APAC': 'AUD', 'INDEX': 'USD', 'FOREX': 'USD'}


def fx_rates(path):
    """Each currency's units for one EUR, from the latest closes of the scan's FOREX series."""
    last = {}
    try:
        with open(path) as f:
            next(f, None)
            for line in f:
                c = line.strip().split(',')
                if len(c) >= 6 and c[1].startswith('FOREX:') and float(c[5] or 0) > 0 and c[0] >= last.get(c[1], ('', 0))[0]:
                    last[c[1]] = (c[0], float(c[5]))
    except OSError:
        pass
    r = {'EUR': 1.0}
    if 'FOREX:EURGBP' in last:
        r['GBP'] = last['FOREX:EURGBP'][1]
        if 'FOREX:GBPUSD' in last:
            r['USD'] = r['GBP'] * last['FOREX:GBPUSD'][1]
    if 'FOREX:EURAUD' in last:
        r['AUD'] = last['FOREX:EURAUD'][1]
    return r


def account_of(config):
    """The account's daily limit left for the markets' live strategies (less "reserve": what the
    multi-day books the bridge places hold, e.g. the UK's exit on strength) and its currency."""
    a = json.load(open(config)).get('account') or {}
    return max(0.0, float(a.get('daily_max', 0)) - float(a.get('reserve', 0))), a.get('currency', 'EUR')


def paper_of(config):
    """A paper account ("account": {"paper": true}): every market with a budget trades an equal share,
    whatever its record, so the broker's real fills are measured in each (a paper account risks nothing)."""
    return bool((json.load(open(config)).get('account') or {}).get('paper'))


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


def published_budgets(config):
    """Every market's latest published choice (web/data/budget-<market>.json), results only."""
    out = {}
    for m in json.load(open(config))['markets']:
        try:
            out[m] = json.load(open(f'web/data/budget-{m.lower()}.json'))
        except (OSError, ValueError):
            pass
    return out


def other_scores(market, config, fresh_days=10):
    """The other markets' records (their latest published, at most fresh_days old)."""
    from datetime import date
    out = {}
    for m, b in published_budgets(config).items():
        if m == market or 'score' not in b:
            continue
        try:
            if (date.today() - date.fromisoformat(b.get('date', '')[:10])).days > fresh_days:
                continue
        except ValueError:
            continue
        out[m] = float(b['score'])
    return out


def other_shares(market, config, fresh_days=10):
    """The other markets' latest shares of the daily limit (at most fresh_days old); a market with a
    budget and no fresh published share holds an equal share until it publishes one, so the first
    runs never sum past the limit either."""
    from datetime import date
    markets = [m for m, v in json.load(open(config))['markets'].items() if v.get('budget')]
    pub = published_budgets(config)
    out = {}
    for m in markets:
        if m == market:
            continue
        out[m] = 1.0 / len(markets)
        b = pub.get(m)
        try:
            if b and (date.today() - date.fromisoformat(b.get('date', '')[:10])).days <= fresh_days:
                out[m] = float(b.get('share', 0))
        except ValueError:
            pass
    return out


def previous_share(market, config):
    """This market's share last time; an equal share of the markets with a budget at first."""
    b = published_budgets(config).get(market)
    if b and float(b.get('share', 0)) > 0:
        return float(b['share'])
    n = sum(1 for m in json.load(open(config))['markets'].values() if m.get('budget'))
    return 1.0 / max(1, n)


def orders_of(path):
    """Each day's orders of a run's trade log (X_TRADELOG): their net results, an unfilled order 0."""
    out = {}
    try:
        with open(path) as f:
            head = next(f).strip().split(',')
            cd, cr, cf = head.index('date'), head.index('ret'), head.index('filled')
            for line in f:
                c = line.strip().split(',')
                if len(c) > max(cd, cr, cf):
                    out.setdefault(c[cd], []).append(float(c[cr]) if float(c[cf] or 0) > 0 else 0.0)
    except (OSError, StopIteration, ValueError):
        pass
    return out


def expected(days, picks, orders):
    """The expected return as the user reads it: each day the mean of that day's recommended orders (an
    unfilled order 0: it earns nothing), averaged over every trading day (a day without orders 0), and
    that daily average compounded over a year of 252 days. Also the orders a day and their fill rate."""
    per, n, filled = [], 0, 0
    for d, k in zip(days, picks):
        o = orders.get(k, {}).get(d, [])
        per.append(sum(o) / len(o) if o else 0.0)
        n += len(o)
        filled += sum(1 for x in o if x != 0)
    if not per:
        return 0.0, 0.0, 0.0, 0.0
    daily = sum(per) / len(per)
    return daily, (1 + daily) ** 252 - 1, n / len(per), (filled / n if n else 0.0)


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
    a.add_argument('--fx', default='data/futures.csv', help="the scan's series (FOREX closes for the account's daily limit)")
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
    pending_variants = []  # (k, command, environment): started in waves the runner's memory holds
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
            daily, cur = account_of(args.config)
            rates = fx_rates(args.fx)
            if daily > 0:  # the runs at the whole limit (the record then shares it out)
                mine = CURRENCY.get(args.market)
                if cur != 'EUR' or mine not in rates:
                    sys.exit(f"strategies: {args.market}: no rate for {mine} (or the limit's currency {cur} is not EUR)")
                # The runs at the whole limit: a market's record is what it earns when given the budget,
                # not what its last share earned (a small share's trades pay the minimum commission on
                # little money, so its record fell and its share with it: 7 Nov, the UK).
                total = daily * rates[mine]
                budget = (total, slots)
            for k in slots:
                env = {x: v for x, v in os.environ.items() if not x.startswith('X_')}
                env.update(s.get('env', {}), X_VERIFY='1', X_NPICK=str(k), X_TRADELOG=f'data/budget/trades-{k}.csv')
                cmd = [binary(s.get('ref'), s.get('app')), *with_capital(args_m + list(s.get('args', [])), total / k, k), '--json', f'data/budget/report-{k}.json',
                       '--orders', f'data/budget/orders-{k}.json']
                pending_variants.append((k, cmd, env))
            continue
        cmd = [binary(s.get('ref'), s.get('app')), *args_m, *s.get('args', []), '--json', f'report/{stem}.json']
        if args.publish_orders:
            path = f'data/eoddata/orders/{args.market}.json' if live else f"data/eoddata/shadow/{args.market}-{s['name']}.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cmd += ['--orders', path]
        env = {k: v for k, v in os.environ.items() if not k.startswith('X_')}
        env.update(s.get('env', {}), X_VERIFY='1')
        log = open(f'report/{stem}.txt', 'w')
        jobs.append((s, live, stem, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    if pending_variants:  # the runs the memory allows at once (a large market's run holds about 1 GB per million share-days)
        csv = model[model.index('--csv') + 1] if '--csv' in model else None
        rows = sum(1 for _ in open(csv)) if csv and os.path.exists(csv) else 0
        avail = 0
        try:
            avail = next(int(l.split()[1]) * 1024 for l in open('/proc/meminfo') if l.startswith('MemAvailable'))
        except (OSError, StopIteration):
            pass
        per = max(1, rows) * 1000  # bytes a run holds per share-day, measured roughly (the features, their interactions, the forecasts)
        width = max(1, min(len(pending_variants), int(0.7 * avail / per) - len(jobs))) if avail else len(pending_variants)
        print(f'strategies: {args.market}: {len(pending_variants)} budget runs, {width} at a time ({rows} share-days, {avail / 2**30:.1f} GB free)', flush=True)
        queue = list(pending_variants)
        running = []
        while queue or running:
            while queue and len(running) < width:
                k, cmd, env = queue.pop(0)
                log = open(f'data/budget/report-{k}.txt', 'w')
                p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                running.append((k, p, log))
                variants.append((k, p, log))
            running[0][1].wait()
            running = [r for r in running if r[1].poll() is None]
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
        score = shrunk(meta[-250:])
        run_total = total  # the amount the runs traded (their costs)
        daily, cur = account_of(args.config)
        share, scale = 1.0, 1.0
        if daily > 0:  # this market's share of the account's daily limit, from every market's record
            others = other_scores(args.market, args.config)
            pos = max(0.0, score) + sum(max(0.0, v) for v in others.values())
            share = max(0.0, score) / pos if pos > 0 else 0.0
            if paper_of(args.config):  # paper: an equal share for every market, to measure its real fills
                share = 1.0 / (1 + len(other_shares(args.market, args.config)))
            # Growing only into its part of what is free: the markets run at the same time (UK and EU at 05:30)
            # each see the others' latest shares, so each may take at most its own last share plus the free
            # share split over the markets; together they never pass the limit, whoever runs when.
            others = other_shares(args.market, args.config)
            mine = previous_share(args.market, args.config)
            # Only the markets that could grow at the same time share the free room: a market with no
            # positive record takes none, so the one most needed takes it all at once.
            growers = other_scores(args.market, args.config)
            n = 1 + sum(1 for m in others if growers.get(m, 1.0) > 0)
            free = max(0.0, 1.0 - mine - sum(others.values()))
            # Paper: the equal share at once. Growing into the free room left a new market (INDEX, 11 Nov) at
            # 0 while the others still held their older, larger equal shares; they take their new one at their
            # next runs, and until then the paper account is only briefly over its limit.
            if not paper_of(args.config):
                share = min(share, mine + free / n, max(0.0, 1.0 - sum(others.values())))
            new_total = daily * share * fx_rates(args.fx)[CURRENCY[args.market]]
            scale = new_total / total if total > 0 else 0.0
            total = new_total
        lines = [f"budget: {args.market} {run_total:,.0f} a day run over k trades (each {', '.join(f'{k}: {run_total / k:,.0f}' for k in slots)}); each day the k whose year before was best (walk-forward)"]
        for k in slots:
            a, sh = stats([series[k][d] for d in days])
            lines.append(f"budget:   {k:2d} trades of {run_total / k:,.0f}: {100 * a:+.1f}% a year, Sharpe {sh:.2f}; chosen on {picks.count(k)} of {len(days)} days")
        a, sh = stats(meta)
        h = len(meta) // 2
        a1, s1 = stats(meta[:h])
        a2, s2 = stats(meta[h:])
        lines.append(f"budget:   chosen day by day: {100 * a:+.1f}% a year, Sharpe {sh:.2f} | first half {100 * a1:+.1f}% {s1:.2f} | second half {100 * a2:+.1f}% {s2:.2f}")
        orders = {k: orders_of(f'data/budget/trades-{k}.csv') for k in slots}
        ed, ey, opd, fr = expected(days, picks, orders)
        ed1, ey1, _, _ = expected(days[-252:], picks[-252:], orders)
        lines.append(f"budget:   expected: a day's recommended orders {100 * ed:+.3f}% on average (an unfilled order 0; {opd:.1f} orders a day, {100 * fr:.0f}% filled), "
                     f"compounded {100 * ey:+.1f}% a year | the last year: {100 * ed1:+.3f}% a day, {100 * ey1:+.1f}% a year")
        if daily > 0:
            lines.append(f"budget:   the account's limit {daily:g} {cur} a day across the markets; this market's share {100 * share:.0f}% ({total:,.0f} {CURRENCY[args.market]}; its record {100 * score:+.3f}% a day shrunk, the others': "
                         + (', '.join(f'{m} {100 * v:+.3f}%' for m, v in sorted(other_scores(args.market, args.config).items())) or 'none yet') + ')')
        lines.append(f"budget:   next session: {nxt} trades of {total / nxt:,.0f}" + (' (no orders: no positive record)' if total <= 0 else '') + (' (paper account: an equal share, to measure the fills)' if paper_of(args.config) else ''))
        shutil.copy(f'data/budget/report-{nxt}.json', 'report/ofm.json')
        with open('report/ofm.txt', 'w') as o:
            o.write('\n'.join(lines) + '\n' + open(f'data/budget/report-{nxt}.txt').read())
        book = json.load(open(f'data/budget/orders-{nxt}.json'))  # each order's amount: the market's share over the trades
        for o in book.get('orders', []):  # never more than the model's own cap (apps/ofm_uk: the share's liquidity)
            o['capital'] = min(total / nxt, float(o.get('maxCapital') or float('inf')))
        if total <= 0:
            book['orders'] = []
        book['budget'] = {'total': total, 'trades': nxt, 'amount': total / nxt, 'share': share}
        with open(f'data/budget/orders-{nxt}.json', 'w') as f:
            json.dump(book, f)
        if args.publish_orders:
            os.makedirs(f'data/eoddata/orders', exist_ok=True)
            shutil.copy(f'data/budget/orders-{nxt}.json', f'data/eoddata/orders/{args.market}.json')
        with open('report/budget.json', 'w') as f:
            json.dump({'market': args.market, 'date': days[-1] if days else '', 'score': score, 'share': share, 'total': total, 'currency': CURRENCY.get(args.market), 'slots': slots, 'next': nxt, 'amount': total / nxt,
                       'chosen': {str(k): picks.count(k) for k in slots}, 'annual': {str(k): stats([series[k][d] for d in days])[0] for k in slots},
                       'auto': {'annual': a, 'sharpe': sh},
                       'expected': {'daily': ed, 'yearly': ey, 'ordersPerDay': opd, 'filled': fr, 'lastYearDaily': ed1, 'lastYearYearly': ey1}}, f, indent=1)
        try:  # the page's live report carries the same expected return
            rj = json.load(open('report/ofm.json'))
            rj['expected'] = {'daily': ed, 'yearly': ey, 'ordersPerDay': opd, 'filled': fr, 'lastYearDaily': ed1, 'lastYearYearly': ey1}
            # the money the market has for the next session (its share of the account's daily limit): with
            # none, the bridge places nothing, and the page says the orders are only what the model would place
            rj['budget'] = {'total': total, 'trades': nxt, 'amount': total / nxt, 'share': share, 'currency': CURRENCY.get(args.market)}
            json.dump(rj, open('report/ofm.json', 'w'))
        except (OSError, ValueError):
            pass
        print(f"strategies: {args.market} live {s['name']} (budget {total:g}, {nxt} trades of {total / nxt:g}): {' '.join(f'{x}={v}' for x, v in s.get('env', {}).items()) or 'defaults'}", flush=True)
        sys.stdout.write(open('report/ofm.txt').read())
        index['strategies'].append({'name': s['name'], 'role': 'live', 'env': s.get('env', {}), 'ref': s.get('ref'), 'only': s.get('only'), 'app': s.get('app'),
                                    'report': 'ofm.json', 'budget': {'total': total, 'trades': nxt, 'amount': total / nxt}})
    for s, live, stem, p, log in jobs:
        code = p.wait()
        log.close()
        print(f"strategies: {args.market} {s['role']} {s['name']}" + (f" (at {s['ref']})" if s.get('ref') else '') + (' (AIM shares only)' if s.get('only') == 'aim' else '') + ({'single': ' (single-share model)', 'hold': ' (multi-day single-share model)', 'uk': ' (UK model)'}.get(s.get('app'), '')) + f": {' '.join(f'{k}={v}' for k, v in s.get('env', {}).items()) or 'defaults'}", flush=True)
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
