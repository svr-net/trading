"""Runs a market's strategies (config/markets.json) on the day's data: the live one, whose orders the
bridge places, and its shadows, whose orders are kept in the private store and never placed. Every
run is checked as the live one always was (X_VERIFY: the kernel's trades against the trade path).

  python3 tools/scan/strategies.py --market EU [--publish-orders] -- <ofm_research's arguments>

Writes report/ofm.json and report/ofm.txt (the live strategy, as before), report/shadow-<name>.json
and .txt for each shadow, and report/strategies.json (the market's strategies, results only). With
--publish-orders (the main branch): the orders to data/eoddata/orders/<MARKET>.json (live) and
data/eoddata/shadow/<MARKET>-<name>.json (shadows). A strategy with a ref is built at that version
first (in .pins/<ref>). A shadow that fails is reported and skipped; the live one failing fails the run.
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
    for s in strategies(args.config, args.market):
        live = s['role'] == 'live'
        stem = 'ofm' if live else f"shadow-{s['name']}"
        cmd = [binary(s.get('ref')), *model, '--json', f'report/{stem}.json']
        if args.publish_orders:
            path = f'data/eoddata/orders/{args.market}.json' if live else f"data/eoddata/shadow/{args.market}-{s['name']}.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cmd += ['--orders', path]
        env = {k: v for k, v in os.environ.items() if not k.startswith('X_')}
        env.update(s.get('env', {}), X_VERIFY='1')
        print(f"strategies: {args.market} {s['role']} {s['name']}" + (f" (at {s['ref']})" if s.get('ref') else '') + f": {' '.join(f'{k}={v}' for k, v in s.get('env', {}).items()) or 'defaults'}", flush=True)
        with open(f'report/{stem}.txt', 'w') as log:
            p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log.write(p.stdout)
        sys.stdout.write(p.stdout)
        if p.returncode:
            if live:
                sys.exit(f"strategies: the live strategy {s['name']} failed ({p.returncode})")
            print(f"strategies: the shadow {s['name']} failed ({p.returncode}): skipped")
            continue
        index['strategies'].append({'name': s['name'], 'role': s['role'], 'env': s.get('env', {}), 'ref': s.get('ref'), 'report': f'{stem}.json'})
    with open('report/strategies.json', 'w') as f:
        json.dump(index, f, indent=1)


if __name__ == '__main__':
    main()
