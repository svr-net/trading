"""Runs a market's strategies (config/markets.json) on the day's data: the live one, whose orders the
bridge places, and its shadows, whose orders are kept in the private store and never placed. Every
run is checked as the live one always was (X_VERIFY: the kernel's trades against the trade path).

  python3 tools/scan/strategies.py --market EU [--publish-orders] -- <ofm_research's arguments>

Writes report/ofm.json and report/ofm.txt (the live strategy, as before), report/shadow-<name>.json
and .txt for each shadow, and report/strategies.json (the market's strategies, results only). With
--publish-orders (the main branch): the orders to data/eoddata/orders/<MARKET>.json (live) and
data/eoddata/shadow/<MARKET>-<name>.json (shadows). A strategy with a ref is built at that version
first (in .pins/<ref>); one with "only": "aim" sees only the UK's AIM shares (no stamp duty;
tools/eoddata/aim.txt). A shadow that fails is reported and skipped; the live one failing fails the run.
"""
import argparse
import json
import os
import re
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
    return out


def binary(ref):
    """The model's binary: this checkout's, or one built at ref."""
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
    for s in strategies(args.config, args.market):
        live = s['role'] == 'live'
        stem = 'ofm' if live else f"shadow-{s['name']}"
        args_m = list(model)
        if s.get('only') == 'aim' and '--csv' in args_m:
            k = args_m.index('--csv') + 1
            args_m[k] = only_aim(args_m[k])
        cmd = [binary(s.get('ref')), *args_m, '--json', f'report/{stem}.json']
        if args.publish_orders:
            path = f'data/eoddata/orders/{args.market}.json' if live else f"data/eoddata/shadow/{args.market}-{s['name']}.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cmd += ['--orders', path]
        env = {k: v for k, v in os.environ.items() if not k.startswith('X_')}
        env.update(s.get('env', {}), X_VERIFY='1')
        log = open(f'report/{stem}.txt', 'w')
        jobs.append((s, live, stem, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    for s, live, stem, p, log in jobs:
        code = p.wait()
        log.close()
        print(f"strategies: {args.market} {s['role']} {s['name']}" + (f" (at {s['ref']})" if s.get('ref') else '') + (' (AIM shares only)' if s.get('only') == 'aim' else '') + f": {' '.join(f'{k}={v}' for k, v in s.get('env', {}).items()) or 'defaults'}", flush=True)
        sys.stdout.write(open(f'report/{stem}.txt').read())
        if code:
            if live:
                for _, _, _, q, _ in jobs:
                    q.kill()
                sys.exit(f"strategies: the live strategy {s['name']} failed ({code})")
            print(f"strategies: the shadow {s['name']} failed ({code}): skipped")
            continue
        index['strategies'].append({'name': s['name'], 'role': s['role'], 'env': s.get('env', {}), 'ref': s.get('ref'), 'only': s.get('only'), 'report': f'{stem}.json'})
    with open('report/strategies.json', 'w') as f:
        json.dump(index, f, indent=1)


if __name__ == '__main__':
    main()
