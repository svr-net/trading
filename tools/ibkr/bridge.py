"""A local bridge between the page and IB Gateway: http://127.0.0.1:8765, on your own computer only.

The page (svr-net.github.io/trading) looks for it and, when it answers, shows a "Your broker" panel:
the day's orders with their prices (fetched here from the private svr-net/eod, never published),
and buttons that place them or close the positions through IB Gateway (tools/ibkr/submit.py).

  GET  /status                     the bridge and whether IB Gateway's API port answers
  GET  /orders?market=UK           the day's orders, with each trade's quantity for --capital
  POST /place  {"market": "UK"}    place them (submit.py)
  POST /close  {"market": "UK"}    cancel the unfilled ones and sell at the close (submit.py --close)
  GET  /diag                       checks: the token, the orders, IB Gateway, the account, the data, the schedule
  POST /bars   {"market": "UK"}    fetch the pool's daily bars from IBKR into the private store (background)
  POST /paths  {"market": "UK"}    record today's candidates' prices through the day (background)

Only the page's origin may call it (CORS, with Chrome's private-network permission), and every call
must carry the key (header X-Bridge-Key) that you set in BRIDGE_KEY and enter once on the page; a
wrong key costs a second. On a server it sits behind HTTPS (docker/ibkr/docker-compose.vps.yml).

Automatic trading (AUTO_MARKETS, e.g. UK or UK,US; empty: off): on weekdays the bridge places each
market's orders before its open and closes what is left before its close, by the exchange's clock
(UK 07:45 and 16:20 London, EU 08:45 and 17:20 Paris, US 09:20 and 15:40 New York, APAC 09:45 and
15:50 Sydney; AUTO_UK_PLACE, AUTO_UK_CLOSE, AUTO_US_PLACE,
AUTO_US_CLOSE change them), each once a day, and skips an orders file more than 4 days old. The
times are when steps become due, not the only moments they can run: every 30 seconds the scheduler
does whatever step of the day is due and not yet done (marked in HISTORY_DIR/state, which outlives
restarts and redeploys), so a restart, a redeploy or a failed step is caught up, not lost. A
watchdog asks IB Gateway's API for its clock every minute and, when its port is open but its API
silent for 3 minutes (hung), has the restarter container (restarter.py) restart it.

Opening auctions (AUCTION=1, the default; 0 off): on the same days, at UK 07:52 London and US 09:24
New York (AUTO_UK_AUCTION, AUTO_US_AUCTION), the bridge records the opening auction of the day's 60
highest ranked candidates (submit.py --auction; it places nothing): the indicative price, volume and
imbalance every 15 seconds until the open, then the open. Each day's recording is kept in
HISTORY_DIR/auction and written to the private store as auction/<MARKET>/<date>.json, where the
research reads it. On Saturdays at AUTO_BARS (06:00; empty: off), and on POST /bars, it fetches
IBKR's daily bars for the pool into the private store (ibkr/<MARKET>/daily.csv): IBKR's opens are the
opening auctions', to check the opens the model learns from.

The day's paths (PATHS=1, the default; 0 off): on the same days, at the auction's time (or, without
it, the place time), the bridge records in the background the day's candidates' prices every 30
seconds until 20 minutes after the close, from IBKR's free delayed data (submit.py --paths), and
writes them to the private store as paths/<MARKET>/<date>.csv: the intraday paths the research needs
to test a buy-back after a stop exactly (daily bars cannot tell whether the high came first).

Buy-backs: on the same days, right after placing, the bridge watches the day's fills until the
close's time (submit.py --rebuy, in the background): when one of the model's stops has sold, it
places that order's buy-back (a buy-stop part of the way back up to the price paid, with a limit
sale at the price paid and a round trip's costs), as the backtest trades it.

History (HISTORY_DIR, default the orders cache): every placing writes the orders placed, the model's
own result for each day's orders arrives with the next orders file (the backtest's fill, entry, exit
and return, to set beside the broker's), and after
each close (automatically at UK 16:45 London, US 16:10 New York, or with POST /report) the day's
fills from IBKR (submit.py --report); GET /history returns them by day for the page's History.

Environment: BRIDGE_KEY, EOD_TOKEN (the daily job's token, svr-net/eod contents; EOD_READ_TOKEN also read), IB_HOST, IB_PORT
(4002 paper), CAPITAL (a trade's amount, default 5000), ALLOWED_ORIGIN, BIND (127.0.0.1; 0.0.0.0
inside Docker, published on 127.0.0.1 only), PORT (8765).
"""
import base64
import hmac
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get('BRIDGE_KEY', '')
TOKEN = os.environ.get('EOD_TOKEN') or os.environ.get('EOD_READ_TOKEN', '')  # the same token as the daily job's secret
IB_HOST = os.environ.get('IB_HOST', '127.0.0.1')
IB_PORT = int(os.environ.get('IB_PORT', '4002'))
CAPITAL = float(os.environ.get('CAPITAL', '5000'))
ORIGIN = os.environ.get('ALLOWED_ORIGIN', 'https://svr-net.github.io')
CACHE = os.environ.get('ORDERS_DIR', '/tmp/ofm-orders')
MARKETS = ('UK', 'EU', 'US', 'APAC')
AUTO = [m.strip().upper() for m in os.environ.get('AUTO_MARKETS', '').split(',') if m.strip().upper() in MARKETS]
ZONE = {'UK': 'Europe/London', 'EU': 'Europe/Paris', 'US': 'America/New_York', 'APAC': 'Australia/Sydney'}
TIMES = {'UK': (os.environ.get('AUTO_UK_PLACE', '07:45'), os.environ.get('AUTO_UK_CLOSE', '16:20')),
         'US': (os.environ.get('AUTO_US_PLACE', '09:20'), os.environ.get('AUTO_US_CLOSE', '15:40')),
         'EU': (os.environ.get('AUTO_EU_PLACE', '08:45'), os.environ.get('AUTO_EU_CLOSE', '17:20')),
         'APAC': (os.environ.get('AUTO_APAC_PLACE', '09:45'), os.environ.get('AUTO_APAC_CLOSE', '15:50'))}
HISTORY = os.environ.get('HISTORY_DIR', os.environ.get('ORDERS_DIR', '/tmp/ofm-orders'))
REPORT = {'UK': os.environ.get('AUTO_UK_REPORT', '16:45'), 'US': os.environ.get('AUTO_US_REPORT', '16:10'), 'APAC': os.environ.get('AUTO_APAC_REPORT', '16:30'), 'EU': os.environ.get('AUTO_EU_REPORT', '17:50')}
# The opening auction's recording (submit.py --auction; places nothing), after the orders are in: the
# LSE auction runs 07:50 to 08:00, the US opening cross at 09:30. AUCTION=0 turns it off.
BARS_AT = os.environ.get('AUTO_BARS', '06:00')  # Saturdays (the exchange's clock): IBKR's daily bars for the pool; empty: off
AUCTION = {'UK': os.environ.get('AUTO_UK_AUCTION', '07:52'), 'US': os.environ.get('AUTO_US_AUCTION', '09:24'), 'APAC': os.environ.get('AUTO_APAC_AUCTION', '09:52'), 'EU': os.environ.get('AUTO_EU_AUCTION', '08:52')} if os.environ.get('AUCTION', '1') != '0' else {}
PATHS = os.environ.get('PATHS', '1') != '0'
STARTED = datetime.now().astimezone()
VERSION = os.environ.get('BRIDGE_VERSION', 'unknown')  # the commit it was built from
TRADES = {}  # market -> (when, the live state): asked of IBKR at most every 30 seconds
LAST = []  # the latest automatic actions, newest first: (when, market, action, ok, summary)


def gateway_up():
    try:
        with socket.create_connection((IB_HOST, IB_PORT), timeout=2):
            return True
    except OSError:
        return False


def fetch_orders(market):
    """The day's orders file from svr-net/eod (or a local file when ORDERS_FILE_<MARKET> is set)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.environ.get(f'ORDERS_FILE_{market}') or os.path.join(CACHE, f'{market}.json')
    if not os.environ.get(f'ORDERS_FILE_{market}'):
        if not TOKEN:
            raise RuntimeError('no EOD_TOKEN in the bridge (.env next to docker-compose.yml)')
        # The contents API (not the raw CDN): no cache, and it tells a bad token from a missing grant.
        req = urllib.request.Request(f'https://api.github.com/repos/svr-net/eod/contents/orders/{market}.json',
                                     headers={'Authorization': f'Bearer {TOKEN}', 'Accept': 'application/vnd.github.raw',
                                              'User-Agent': 'ofm-bridge', 'Cache-Control': 'no-cache'})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError('EOD_TOKEN rejected by GitHub (expired or revoked): put the current token in .env') from None
            if e.code == 404:
                raise RuntimeError(f'orders/{market}.json not readable: the EOD_TOKEN has no access to svr-net/eod '
                                   f'(Repository access, Contents read), or the {market} job has not written it yet') from None
            raise
        with open(path, 'wb') as f:
            f.write(body)
        record_check(market, path)
    return path


def record_check(market, path):
    """The model's own result for the latest day's orders (the orders file's check: filled or not,
    the entry, the exit and the net return, as the backtest trades them), kept once a day in the
    history beside the broker's fills."""
    try:
        check = json.load(open(path)).get('check') or {}
    except ValueError:
        return
    if not check.get('date') or not check.get('trades'):
        return
    os.makedirs(HISTORY, exist_ok=True)
    mark = os.path.join(HISTORY, f"model-{market}-{check['date']}")
    if os.path.exists(mark):
        return
    history_add({'kind': 'model', 'date': check['date'], 'market': market, 'trades': check['trades']})
    open(mark, 'w').close()


def run_submit(args, timeout=300):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'submit.py'), *args, '--host', IB_HOST, '--port', str(IB_PORT)],
                       capture_output=True, text=True, timeout=timeout)
    # The summaries read the last line: the script's own (its standard output) when it succeeds, its
    # error when it fails; ib_async's log lines on standard error never stand in for a result.
    out = (p.stdout + p.stderr) if p.returncode else (p.stderr + p.stdout)
    return p.returncode, out.strip()


def history_add(record):
    """Appends one record (JSON per line) to the history."""
    os.makedirs(HISTORY, exist_ok=True)
    with open(os.path.join(HISTORY, 'history.jsonl'), 'a') as f:
        f.write(json.dumps(record) + '\n')


def placed(market, path, dry):
    if not dry:
        book = json.load(open(path))
        history_add({'kind': 'placed', 'date': date.today().isoformat(), 'market': market, 'asOf': book.get('asOf'),
                     'capital': CAPITAL, 'orders': book.get('orders', [])})


def report(market):
    """Today's fills from IBKR into the history; returns (ok, summary)."""
    free_client(market)
    code, out = run_submit(['--market', market, '--report'])
    line = next((l for l in reversed(out.splitlines()) if l.startswith('{')), '')
    if code != 0 or not line:
        return False, out.splitlines()[-1] if out else 'no report'
    rec = json.loads(line)
    rec['kind'] = 'result'
    history_add(rec)
    return True, f"{len(rec['trades'])} shares traded"


def store_put(path, data, message):
    """Writes one file to the private store (svr-net/eod) with EOD_TOKEN (Contents read and write)."""
    url = f'https://api.github.com/repos/svr-net/eod/contents/{path}'
    head = {'Authorization': f'Bearer {TOKEN}', 'Accept': 'application/vnd.github+json', 'User-Agent': 'ofm-bridge'}
    sha = None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=head), timeout=20) as r:
            sha = json.load(r).get('sha')
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    body = {'message': message, 'content': base64.b64encode(data).decode()}
    if sha:
        body['sha'] = sha
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**head, 'Content-Type': 'application/json'}, method='PUT')
    with urllib.request.urlopen(req, timeout=30):
        pass


def record_auction(market):
    """The candidates' opening auction (submit.py --auction), kept with the history and in the private
    store as auction/<MARKET>/<date>.json; returns (ok, summary)."""
    path = fetch_orders(market)
    code, out = run_submit(['--market', market, '--orders', path, '--auction'], timeout=1500)
    line = next((l for l in reversed(out.splitlines()) if l.startswith('{')), '')
    if code != 0 or not line:
        return False, out.splitlines()[-1] if out else 'no recording'
    rec = json.loads(line)
    os.makedirs(os.path.join(HISTORY, 'auction'), exist_ok=True)
    with open(os.path.join(HISTORY, 'auction', f"{market}-{rec['date']}.json"), 'w') as f:
        f.write(line + '\n')
    summary = next((l for l in out.splitlines() if 'auction:' in l), f"{len(rec['shares'])} shares")
    try:
        store_put(f"auction/{market}/{rec['date']}.json", (line + '\n').encode(), f"{market} opening auction {rec['date']}")
    except Exception as e:  # noqa: BLE001
        return True, f'{summary}; kept here, not stored in svr-net/eod ({e})'
    return True, summary


BARS_RUNNING = set()
PATHS_RUNNING = set()


def record_paths(market):
    """Today's candidates' prices through the day (submit.py --paths), written to the private store
    as paths/<MARKET>/<date>.csv; returns (ok, summary). Runs for the whole session: in a thread."""
    if market in PATHS_RUNNING:
        return False, 'already running'
    PATHS_RUNNING.add(market)
    try:
        day = datetime.now(ZoneInfo(ZONE[market])).date().isoformat()
        out_file = os.path.join(HISTORY, f'paths-{market}-{day}.csv')
        os.makedirs(HISTORY, exist_ok=True)
        code, out = run_submit(['--market', market, '--orders', fetch_orders(market), '--paths', '--out', out_file], timeout=12 * 3600)
        summary = out.splitlines()[-1] if out else ''
        if code != 0 or not os.path.exists(out_file):
            return False, summary or 'failed'
        with open(out_file, 'rb') as f:
            store_put(f'paths/{market}/{day}.csv', f.read(), f'{market} intraday paths {day}')
        return True, summary
    finally:
        PATHS_RUNNING.discard(market)


REBUY_RUNNING = set()
START_LOCK = threading.Lock()


def place_and_watch(market):
    """The day's orders (submit.py; up to 30 minutes: each order's price checks), then the buy-back
    watch until the close's time in the same thread, so the watch connects only after the placing
    has let go of the market's client. Returns (ok, summary)."""
    path = fetch_orders(market)
    digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    as_of = json.load(open(path)).get('asOf', '')
    if not as_of or date.fromisoformat(as_of) < date.today() - timedelta(days=4):
        return False, f'orders from {as_of or "?"} are too old: not placed'
    code, out = run_submit(['--market', market, '--orders', path, '--capital', str(CAPITAL)], timeout=1800)
    lines = out.splitlines()
    placed_n = sum(1 for x in lines if ': buy ' in x)
    skipped = [x.strip() for x in lines if 'skipped' in x]
    verdict = next((x.strip() for x in reversed(lines) if 'accepted by IBKR' in x), '')
    rejected = [x.strip() for x in lines if 'REJECTED' in x]
    summary = f'{placed_n} sent; {verdict}' + (f'; rejected: ' + '; '.join(rejected) if rejected else '') + (f'; {len(skipped)} skipped: ' + '; '.join(skipped) if skipped else '')
    if code != 0:
        return False, summary or 'failed'
    placed(market, path, False)
    mark_done(market, 'place')
    with open(placed_digest_file(market), 'w') as f:  # the file placed today: a newer one is placed too (scheduler)
        f.write(digest)
    in_background(market, 'rebuy', watch_rebuy)
    return True, summary


def placed_digest_file(market):
    return os.path.join(CACHE, f'placed-{market}-{datetime.now(ZoneInfo(ZONE[market])).date().isoformat()}.sha256')


def orders_changed(market):
    """Whether the day's orders file differs from the orders placed today (a rerun of the market's scan
    during the day): its orders (share and limit) against today's last 'placed' record in the history,
    which every bridge version writes, so a bridge redeployed mid-day still sees it."""
    day = datetime.now(ZoneInfo(ZONE[market])).date().isoformat()
    last = None
    try:
        for line in open(os.path.join(HISTORY, 'history.jsonl')).read().splitlines()[-400:]:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('kind') == 'placed' and r.get('market') == market and r.get('date') == day:
                last = r
    except OSError:
        return False
    if not last:
        return False
    key = lambda orders: sorted((o.get('ticker'), round(float(o.get('limit') or 0), 6)) for o in orders)  # noqa: E731
    book = json.load(open(fetch_orders(market)))
    return key(book.get('orders', [])) != key(last.get('orders', []))


def replace_orders(market):
    """A newer orders file during the day: the watch lets go of the market's client, the orders not
    yet placed go in (those already placed are kept), then the watch starts again on the new file."""
    free_client(market)
    for _ in range(30):  # the watch's thread notices its process ended
        if market not in REBUY_RUNNING:
            break
        time.sleep(1)
    return place_and_watch(market)


def watch_rebuy(market):
    """The day's buy-backs after stops (submit.py --rebuy) until the close's time; returns (ok, summary)."""
    with START_LOCK:  # the scheduler and a placing may both start it: one watch only
        if market in REBUY_RUNNING:
            return False, 'already watching'
        REBUY_RUNNING.add(market)
    try:
        day = datetime.now(ZoneInfo(ZONE[market])).date().isoformat()
        state = os.path.join(HISTORY, f'rebuy-{market}-{day}.txt')
        os.makedirs(HISTORY, exist_ok=True)
        # Until two minutes before the close's time: the close then runs as the same client (one connection at a time).
        h, mi = (int(x) for x in TIMES[market][1].split(':'))
        until = (datetime(2000, 1, 1, h, mi) - timedelta(minutes=2)).strftime('%H:%M')
        code, out = run_submit(['--market', market, '--orders', fetch_orders(market), '--rebuy', '--until', until, '--out', state], timeout=12 * 3600)
        return code == 0, (out.splitlines()[-1] if out else '')
    finally:
        REBUY_RUNNING.discard(market)


def in_background(market, action, job):
    def run():
        try:
            ok, summary = job(market)
        except Exception as e:  # noqa: BLE001
            ok, summary = False, str(e)
        LAST.insert(0, (datetime.now(ZoneInfo(ZONE[market])).strftime('%Y-%m-%d %H:%M %Z'), market, action, ok, summary))
        del LAST[10:]
        sys.stderr.write(f'bridge: {action} {market}: {"ok" if ok else "FAILED"} {summary}\n')
    threading.Thread(target=run, daemon=True).start()


def fetch_bars(market):
    """IBKR's daily bars for the pool (submit.py --bars, about 75 minutes for the UK's), written to the
    private store as ibkr/<MARKET>/daily.csv; returns (ok, summary)."""
    if market in BARS_RUNNING:
        return False, 'already running'
    BARS_RUNNING.add(market)
    try:
        out_file = os.path.join(HISTORY, f'ibkr-{market}.csv')
        code, out = run_submit(['--market', market, '--orders', fetch_orders(market), '--bars', '--out', out_file], timeout=6 * 3600)
        summary = out.splitlines()[-1] if out else ''
        if code != 0:
            return False, summary or 'failed'
        with open(out_file, 'rb') as f:
            store_put(f'ibkr/{market}/daily.csv', f.read(), f'{market} daily bars from IBKR {date.today().isoformat()}')
        return True, summary
    finally:
        BARS_RUNNING.discard(market)


def diag():
    """The checks behind GET /diag, each {check, ok (true, false, or null: a warning), detail}."""
    out = []
    add = lambda check, ok, detail: out.append({'check': check, 'ok': ok, 'detail': detail})  # noqa: E731
    add('Bridge', True, f'version {VERSION}, running since {STARTED:%Y-%m-%d %H:%M %Z}; automatic: {",".join(AUTO) or "off"}; capital {CAPITAL:g} a trade')
    # The token: read the store, and whether it may write (the auctions and bars go there).
    if not TOKEN:
        add('EOD_TOKEN', False, 'not set in .env')
    else:
        try:
            req = urllib.request.Request('https://api.github.com/repos/svr-net/eod', headers={'Authorization': f'Bearer {TOKEN}', 'User-Agent': 'ofm-bridge'})
            with urllib.request.urlopen(req, timeout=15) as r:
                push = (json.load(r).get('permissions') or {}).get('push')
            add('EOD_TOKEN', True if push else None, 'reads svr-net/eod' + ('' if push else '; may not write (recordings stay on the NAS)'))
        except urllib.error.HTTPError as e:
            add('EOD_TOKEN', False, 'rejected (expired or revoked)' if e.code == 401 else 'no access to svr-net/eod' if e.code == 404 else f'HTTP {e.code}')
        except Exception as e:  # noqa: BLE001
            add('EOD_TOKEN', False, f'GitHub not reachable ({e})')
    for m in AUTO or ['UK']:
        try:
            book = json.load(open(fetch_orders(m)))
            age = (date.today() - date.fromisoformat(book.get('asOf', '1970-01-01'))).days
            add(f'{m} orders', age <= 4 or None, f"after the close of {book.get('asOf')} ({age} day{'s' if age != 1 else ''} ago): {len(book.get('orders', []))} orders, "
                f"{len(book.get('candidates', []))} candidates, pool {len(book.get('pool', []))}")
        except Exception as e:  # noqa: BLE001
            add(f'{m} orders', False, str(e))
    add('IB Gateway port', gateway_up(), f'{IB_HOST}:{IB_PORT}')
    m = (AUTO or ['UK'])[0]
    try:
        code, text = run_submit(['--market', m, '--diag'], timeout=90)
        line = next((l for l in reversed(text.splitlines()) if l.startswith('{')), '')
        d = json.loads(line) if line else {'api': False, 'error': text.splitlines()[-1] if text else 'no answer'}
    except Exception as e:  # noqa: BLE001
        d = {'api': False, 'error': str(e)}
    if not d.get('api'):
        add('IB Gateway API', False, d.get('error', 'no answer'))
    else:
        add('IB Gateway API', True, f"logged in, {'paper' if d.get('paper') else 'LIVE'} account; IBKR time {d.get('serverTime', '?')}")
        live = d.get('dataType') == 1 and not any(e.startswith(('354', '10089', '10168')) for e in d.get('errors', []))
        quote = ', '.join(f'{k} {d[k]}' for k in ('bid', 'ask', 'last', 'close') if d.get(k))
        add(f'{m} real-time data', live or None, f"{d.get('probe', '?')}: " + (quote or 'no prices (outside hours, or no subscription)') +
            ('' if live else '; the auction recording needs the exchange\'s real-time subscription'))
        add(f'{m} historical data', bool(d.get('historyBars')), f"{d.get('historyBars', 0)} daily bars" + (f" ({d['historyError']})" if d.get('historyError') else ''))
        if d.get('errors'):
            add('IBKR messages', None, ' | '.join(d['errors']))
    plan = '; '.join(f"{m}: place {TIMES[m][0]}" + (f", auction {AUCTION[m]}" if AUCTION.get(m) else '') + ', buy-back watch' + (', paths' if PATHS else '') + f", close {TIMES[m][1]}, report {REPORT[m]} ({ZONE[m]})" for m in AUTO)
    add('Schedule', True if AUTO else None, (plan or 'automatic trading off (AUTO_MARKETS)') + (f'; IBKR bars Saturdays {BARS_AT}' if BARS_AT else ''))
    for m in AUTO:
        add(f'{m} today', None, ', '.join(f"{a} {'done' if is_done(m, a) else '-'}" for a in ('leftovers', 'place', 'paths', 'auction', 'close', 'flat', 'report'))
            + f"; running: {', '.join(sorted(a for (mk, a) in RUNNING if mk == m) + (['rebuy'] if m in REBUY_RUNNING else [])) or 'nothing'}")
    g = GATEWAY
    add('IB Gateway watchdog', g['ok'], ('not checked yet' if g['ok'] is None else 'API answers' if g['ok'] else 'API does not answer')
        + f" (checked {g['checked']}); restarts {g['restarts']}" + (f", last {g['restarted']}: {g['note']}" if g['restarted'] else ''))
    failed = [x for x in LAST if not x[3]]
    add('Last automatic actions', None if failed else True, '; '.join(f'{w} {mk} {a}: {"ok" if ok else "FAILED"} {sm}' for w, mk, a, ok, sm in LAST[:8]) or 'none yet')
    adir = os.path.join(HISTORY, 'auction')
    files = sorted(os.listdir(adir)) if os.path.isdir(adir) else []
    if files:
        rec = json.load(open(os.path.join(adir, files[-1])))
        n = sum(1 for r in rec['shares'] if any(x.get('price') for x in r['samples']))
        add('Auction recordings', n > 0 or None, f"{len(files)} day{'s' if len(files) != 1 else ''}, latest {files[-1][:-5]}: an indicative price for {n} of {len(rec['shares'])} shares")
    else:
        add('Auction recordings', None, 'none yet')
    for m in AUTO or ['UK']:
        f = os.path.join(HISTORY, f'ibkr-{m}.csv')
        if m in BARS_RUNNING:
            add(f'{m} IBKR bars', None, 'fetching now')
        elif os.path.exists(f):
            with open(f) as fh:
                rows = sum(1 for _ in fh) - 1
            add(f'{m} IBKR bars', rows > 0 or None, f'{rows} bars, fetched {datetime.fromtimestamp(os.path.getmtime(f)):%Y-%m-%d %H:%M}')
        else:
            add(f'{m} IBKR bars', None, 'not fetched yet')
    days = history()
    add('History', True, f'{len(days)} market-days recorded' + (f", latest {days[0]['date']} {days[0]['market']}" if days else ''))
    return out


def history():
    """The history by day and market: the orders placed and what was traded."""
    f = os.path.join(HISTORY, 'history.jsonl')
    days = {}
    if os.path.exists(f):
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            d = days.setdefault((r['date'], r['market']), {'date': r['date'], 'market': r['market'], 'orders': [], 'trades': [], 'model': []})
            if r['kind'] == 'placed':
                d['orders'] = r['orders']
            elif r['kind'] == 'result':
                d['trades'] = r['trades']  # the latest report of the day wins
            elif r['kind'] == 'model':
                d['model'] = r['trades']  # the model's own result for the day (record_check)
    return sorted(days.values(), key=lambda d: (d['date'], d['market']), reverse=True)


def auto_run(market, action):
    """One automatic action; returns (ok, summary)."""
    if action == 'place':  # in the background: the placing (minutes: each order's price checks), then the buy-back watch
        in_background(market, 'place', place_and_watch)
        return True, 'placing the orders, then watching the stops for buy-backs'
    elif action == 'report':
        return report(market)
    elif action == 'auction':  # up to 25 minutes: in the background, so the schedule's other steps keep their times
        in_background(market, 'auction', record_auction)
        return True, 'recording the opening auction'
    elif action == 'bars':
        return fetch_bars(market)
    elif action == 'rebuy':  # until the close: in the background, logged when it ends
        in_background(market, 'rebuy', watch_rebuy)
        return True, 'watching the stops for buy-backs until the close'
    elif action == 'paths':  # the whole session: in the background, logged when it ends
        in_background(market, 'paths', record_paths)
        return True, 'recording the day\'s paths until after the close'
    else:
        code, out = run_submit(['--market', market, '--close'])
    return code == 0, out.splitlines()[-1] if out else ''


def state_path(market, action, day):
    return os.path.join(HISTORY, 'state', day, f'{market}-{action}')


def is_done(market, action):
    return os.path.exists(state_path(market, action, datetime.now(ZoneInfo(ZONE[market])).date().isoformat()))


def mark_done(market, action):
    """A step of today done, kept in the history folder (it outlives restarts and redeploys)."""
    f = state_path(market, action, datetime.now(ZoneInfo(ZONE[market])).date().isoformat())
    os.makedirs(os.path.dirname(f), exist_ok=True)
    open(f, 'w').close()


def seed_done():
    """Today's steps already in the history (placed, reported) marked done at start, so a bridge
    upgraded mid-day never places or reports a day twice."""
    try:
        lines = open(os.path.join(HISTORY, 'history.jsonl')).read().splitlines()
    except OSError:
        return
    for line in lines[-200:]:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        m, step = r.get('market'), {'placed': 'place', 'result': 'report'}.get(r.get('kind'))
        if m in ZONE and step and r.get('date') == datetime.now(ZoneInfo(ZONE[m])).date().isoformat() and not is_done(m, step):
            mark_done(m, step)


def at(now, hhmm, minutes=0):
    h, mi = (int(x) for x in hhmm.split(':'))
    return now.replace(hour=h, minute=mi, second=0, microsecond=0) + timedelta(minutes=minutes)


RUNNING = set()     # (market, action) jobs in flight
ATTEMPTS = {}       # (market, action, day) -> (tries, last try)
GATEWAY = {'ok': None, 'fails': 0, 'checked': None, 'restarts': 0, 'restarted': None, 'note': ''}


def job(market, action, fn, done_on_ok=True):
    """Runs fn(market) in the background once at a time; marks the step done when it succeeds."""
    if (market, action) in RUNNING:
        return
    day = datetime.now(ZoneInfo(ZONE[market])).date().isoformat()
    tries, _ = ATTEMPTS.get((market, action, day), (0, None))
    ATTEMPTS[(market, action, day)] = (tries + 1, time.monotonic())
    RUNNING.add((market, action))

    def wrapped(m):
        try:
            ok, summary = fn(m)
            if ok and done_on_ok:
                mark_done(m, action)
            return ok, summary
        finally:
            RUNNING.discard((m, action))
    in_background(market, action, wrapped)


def may_try(market, action, every_min, most=None):
    day = datetime.now(ZoneInfo(ZONE[market])).date().isoformat()
    tries, last = ATTEMPTS.get((market, action, day), (0, None))
    if most is not None and tries >= most:
        return False
    return last is None or time.monotonic() - last >= 60 * every_min


def free_client(market):
    """Ends any of this container's submit.py runs still holding the market's orders' client (a
    buy-back watch past its time, or one started by hand) before the close or the report needs it:
    IBKR allows one connection per client id, and a held one made the close fail (3 Nov, EU)."""
    import signal
    ended = []
    for pid in (x for x in os.listdir('/proc') if x.isdigit() and int(x) != os.getpid()):
        try:
            argv = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')
        except OSError:
            continue
        a = [x.decode(errors='replace') for x in argv if x]
        if not any(x.endswith('submit.py') for x in a) or market not in a or '--rebuy' not in a:
            continue
        try:
            os.kill(int(pid), signal.SIGTERM)
            ended.append(int(pid))
        except OSError:
            pass
    for _ in range(10):
        if not any(os.path.exists(f'/proc/{p}') for p in ended):
            break
        time.sleep(1)
    for p in ended:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    return len(ended)


def close_market(market):
    n = free_client(market)
    code, out = run_submit(['--market', market, '--close'], timeout=600)
    last = out.splitlines()[-1] if out else ''
    return code == 0, (f'ended {n} buy-back watch still connected; ' if n else '') + last


def model_tickers(market, days=10):
    """The shares the model placed in this market lately (the bridge's history): the only holdings
    the leftovers step may sell."""
    since = (date.today() - timedelta(days=days)).isoformat()
    out = set()
    try:
        for line in open(os.path.join(HISTORY, 'history.jsonl')):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('kind') == 'placed' and r.get('market') == market and (r.get('date') or '') >= since:
                out |= {o['ticker'] for o in r.get('orders', []) if o.get('ticker')}
    except OSError:
        pass
    return sorted(out)


def sell_leftovers(market):
    """Before the day's placing: the model's shares still held from an earlier day sold at the open."""
    tickers = model_tickers(market)
    if not tickers:
        return True, 'none placed lately'
    code, out = run_submit(['--market', market, '--leftovers', '--tickers', ','.join(tickers)], timeout=600)
    return code == 0, out.splitlines()[-1] if out else ''


def flat_check(market):
    """After the close: is any of the model's shares still held? Held is a failure (the monitor turns
    red and GitHub emails); the next morning's leftovers step sells it."""
    tickers = model_tickers(market)
    if not tickers:
        mark_done(market, 'flat')
        return True, 'none placed lately'
    code, out = run_submit(['--market', market, '--leftovers', '--check', '--tickers', ','.join(tickers)], timeout=300)
    last = out.splitlines()[-1] if out else ''
    if code in (0, 1) and ('held' in last):
        mark_done(market, 'flat')
    return code == 0, last + (': sold at the next open' if code == 1 else '')


def scheduler():
    """Each automatic market's day as steps that should be done by now, checked every 30 seconds and
    done when missing, so a restart, a redeploy or a failed step is caught up rather than lost: the
    model's shares still held from an earlier day sold at the open (from 30 minutes before the
    placing; the placing waits for it), the orders placed (retried every 5 minutes, at most 4 times, until 15 minutes before the close;
    already placed shares are skipped), the buy-back watch running from then until the close, the
    day's paths recorded, the opening auction recorded (only at its time), the close done (retried
    every minute for 9 minutes after its time; a rerun never sells twice), the check 30 minutes after
    the close that none of the model's shares is still held, and the report (retried every 10 minutes). Steps
    that need IB Gateway wait while it does not answer."""
    os.makedirs(CACHE, exist_ok=True)
    while True:
        for m in AUTO:
            try:
                now = datetime.now(ZoneInfo(ZONE[m]))
                if now.weekday() == 5 and BARS_AT and at(now, BARS_AT) <= now < at(now, BARS_AT, 60) and not is_done(m, 'bars'):
                    job(m, 'bars', fetch_bars)
                if now.weekday() >= 5 and not os.environ.get('AUTO_WEEKENDS'):
                    continue
                place, close = TIMES[m]
                up = GATEWAY['ok'] is not False
                if up and at(now, place, -30) <= now < at(now, close, -15) and not is_done(m, 'leftovers') and may_try(m, 'leftovers', 5, 3):
                    job(m, 'leftovers', sell_leftovers)
                tried = ATTEMPTS.get((m, 'leftovers', now.date().isoformat()), (0, None))[0] > 0
                settled = is_done(m, 'leftovers') or (tried and (m, 'leftovers') not in RUNNING)  # the placing waits for it to finish
                if up and settled and at(now, place) <= now < at(now, close, -15) and not is_done(m, 'place') and may_try(m, 'place', 5, 4):
                    job(m, 'place', place_and_watch, done_on_ok=False)  # place_and_watch marks it and starts the watch
                if up and is_done(m, 'place') and now < at(now, close, -3) and m not in REBUY_RUNNING and (m, 'place') not in RUNNING \
                        and (m, 'replace') not in RUNNING and may_try(m, 'rebuy', 5):
                    job(m, 'rebuy', watch_rebuy, done_on_ok=False)
                # A newer orders file during the session (the scan run again): placed too, without anyone pressing anything.
                if up and is_done(m, 'place') and at(now, place) <= now < at(now, close, -15) and (m, 'place') not in RUNNING \
                        and (m, 'replace') not in RUNNING and may_try(m, 'replace-check', 3) and orders_changed(m):
                    job(m, 'replace', replace_orders, done_on_ok=False)
                start = AUCTION.get(m) or place
                if PATHS and up and at(now, start) <= now < at(now, close, 15) and m not in PATHS_RUNNING and not is_done(m, 'paths') \
                        and may_try(m, 'paths', 5):
                    job(m, 'paths', record_paths)
                if AUCTION.get(m) and up and at(now, AUCTION[m]) <= now < at(now, AUCTION[m], 5) and not is_done(m, 'auction') \
                        and may_try(m, 'auction', 10, 1):
                    job(m, 'auction', record_auction)
                if at(now, close) <= now < at(now, close, 9) and not is_done(m, 'close') and may_try(m, 'close', 1):
                    job(m, 'close', close_market)
                if up and at(now, close, 30) <= now < at(now, close, 240) and not is_done(m, 'flat') and may_try(m, 'flat', 10):
                    job(m, 'flat', flat_check, done_on_ok=False)
                if at(now, REPORT[m]) <= now < at(now, REPORT[m], 180) and not is_done(m, 'report') and may_try(m, 'report', 10):
                    job(m, 'report', report)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f'bridge: scheduler {m}: {e}\n')
        time.sleep(30)


def restart_gateway():
    """Asks the restarter (tools/ibkr/restarter.py, the only container with Docker's socket) to restart
    IB Gateway; returns a note."""
    url = os.environ.get('GATEWAY_RESTARTER', 'http://restarter:8766') + '/restart'
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=b'', method='POST'), timeout=150) as r:
            return json.loads(r.read()).get('note', 'restarted')
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b'{}').get('note', f'restarter answered {e.code}')
    except Exception as e:  # noqa: BLE001
        return f'no restarter ({type(e).__name__}): restart IB Gateway by hand'


def watchdog():
    """Every minute, whether IB Gateway's API answers (a connection and its clock, not only the port);
    after 3 minutes in a row with its port open but its API silent (hung), IB Gateway is restarted,
    then given 4 minutes to log in."""
    while True:
        try:
            code, out = run_submit(['--market', 'UK', '--ping'], timeout=45)
            ok = code == 0
        except Exception:  # noqa: BLE001
            ok = False
        # A failure counts only while the gateway's port is open: a closed port is the gateway starting
        # or restarting (its own nightly restart too), which Docker's restart policy already covers.
        hung = not ok and gateway_up()
        GATEWAY.update(ok=ok, checked=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z'), fails=GATEWAY['fails'] + 1 if hung else 0)
        if GATEWAY['fails'] >= 3:
            note = restart_gateway()
            GATEWAY.update(fails=0, restarts=GATEWAY['restarts'] + 1, restarted=GATEWAY['checked'], note=note)
            LAST.insert(0, (GATEWAY['checked'], 'IB', 'gateway restart', 'restarted' in note, f'IB Gateway did not answer 3 times: {note}'))
            del LAST[10:]
            sys.stderr.write(f'bridge: IB Gateway did not answer 3 times: {note}\n')
            time.sleep(240)
            continue
        time.sleep(60)


class Handler(BaseHTTPRequestHandler):
    def cors(self):
        if self.headers.get('Origin') == ORIGIN:
            self.send_header('Access-Control-Allow-Origin', ORIGIN)
            self.send_header('Vary', 'Origin')

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):  # the browser's preflight (with Chrome's private-network permission)
        self.send_response(204)
        self.cors()
        self.send_header('Access-Control-Allow-Methods', 'GET, POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Bridge-Key')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.end_headers()

    def allowed(self):
        if self.headers.get('Origin') not in (None, ORIGIN):
            self.reply(403, {'error': 'origin not allowed'})
            return False
        if not KEY or not hmac.compare_digest(self.headers.get('X-Bridge-Key', ''), KEY):
            time.sleep(1.0)  # a wrong key costs a second: guessing a long key is hopeless
            self.reply(401, {'error': 'key missing or wrong'})
            return False
        return True

    def market(self, value):
        m = (value or '').upper()
        if m not in MARKETS:
            self.reply(400, {'error': 'market must be UK, EU, US or APAC'})
            return None
        return m

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/status':  # no key needed: tells the page the bridge is there
            return self.reply(200, {'bridge': True, 'version': VERSION, 'started': STARTED.strftime('%Y-%m-%d %H:%M %Z'), 'gateway': gateway_up(), 'port': IB_PORT, 'capital': CAPITAL, 'keyed': bool(KEY),
                                    'auto': {m: {'place': TIMES[m][0], 'auction': AUCTION.get(m), 'close': TIMES[m][1], 'report': REPORT[m], 'zone': ZONE[m],
                                                 'today': {a: is_done(m, a) for a in ('leftovers', 'place', 'paths', 'auction', 'close', 'flat', 'report')},
                                                 'running': sorted(a for (mk, a) in RUNNING if mk == m) + (['rebuy'] if m in REBUY_RUNNING else [])} for m in AUTO},
                                    'gatewayApi': GATEWAY,
                                    'last': [{'when': w, 'market': mk, 'action': a, 'ok': ok, 'summary': sm} for w, mk, a, ok, sm in LAST]})
        if not self.allowed():
            return
        if u.path == '/diag':
            return self.reply(200, {'checks': diag()})
        if u.path == '/history':
            return self.reply(200, {'days': history()})
        if u.path == '/trades':  # the live state of today's trades and the model's holdings, from IBKR
            m = self.market(parse_qs(u.query).get('market', [''])[0])
            if not m:
                return
            hit = TRADES.get(m)
            if hit and time.monotonic() - hit[0] < 30:
                return self.reply(200, hit[1])
            code, out = run_submit(['--market', m, '--state', '--capital', str(CAPITAL), '--tickers', ','.join(model_tickers(m))], timeout=90)
            line = next((x for x in reversed(out.splitlines()) if x.startswith('{')), '')
            if code != 0 or not line:
                return self.reply(502, {'error': (out.splitlines() or ['no answer from IBKR'])[-1]})
            TRADES[m] = (time.monotonic(), json.loads(line))
            return self.reply(200, TRADES[m][1])
        if u.path == '/orders':
            m = self.market(parse_qs(u.query).get('market', [''])[0])
            if not m:
                return
            try:
                book = json.load(open(fetch_orders(m)))
            except Exception as e:  # noqa: BLE001
                return self.reply(502, {'error': f'orders not available: {e}'})
            for o in book['orders']:
                share = o['limit'] / 100.0 if m == 'UK' else o['limit']  # LSE prices are in pence
                o['quantity'] = math.floor(float(o.get('capital') or CAPITAL) / share) if share > 0 else 0  # the order's own amount (the model's)
            book.update(market=m, capital=CAPITAL)
            return self.reply(200, book)
        self.reply(404, {'error': 'not found'})

    def do_POST(self):
        if not self.allowed():
            return
        n = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except ValueError:
            return self.reply(400, {'error': 'bad JSON'})
        m = self.market(body.get('market'))
        if not m:
            return
        path = urlparse(self.path).path
        if path == '/place':
            if not body.get('dryRun'):  # in the background (minutes), then the buy-back watch; the status shows it when done
                in_background(m, 'place', replace_orders)  # the running watch lets go of the client first
                return self.reply(202, {'ok': True, 'log': f'{m}: placing the orders (a few minutes; already placed shares are skipped), then watching the stops for buy-backs. Diagnostics shows the result.'})
            try:
                code, out = run_submit(['--market', m, '--orders', fetch_orders(m), '--capital', str(CAPITAL)] + (['--dry-run'] if body.get('dryRun') else []))
            except Exception as e:  # noqa: BLE001
                return self.reply(502, {'error': str(e)})
            if code == 0:
                placed(m, fetch_orders(m) if not os.environ.get(f'ORDERS_FILE_{m}') else os.environ[f'ORDERS_FILE_{m}'], bool(body.get('dryRun')))
            return self.reply(200 if code == 0 else 500, {'ok': code == 0, 'log': out})
        if path == '/bars':  # in the background: it takes over an hour
            if m in BARS_RUNNING:
                return self.reply(409, {'ok': False, 'log': f'{m} bars: already running'})

            def run():
                ok, summary = fetch_bars(m)
                LAST.insert(0, (datetime.now(ZoneInfo(ZONE[m])).strftime('%Y-%m-%d %H:%M %Z'), m, 'bars', ok, summary))
                del LAST[10:]
            threading.Thread(target=run, daemon=True).start()
            return self.reply(202, {'ok': True, 'log': f'{m}: fetching the pool\'s daily bars from IBKR (about an hour); the status shows when done'})
        if path == '/paths':
            if m in PATHS_RUNNING:
                return self.reply(409, {'ok': False, 'log': f'{m} paths: already recording'})
            in_background(m, 'paths', record_paths)
            return self.reply(202, {'ok': True, 'log': f'{m}: recording the candidates\' prices until 20 minutes after the close'})
        if path == '/report':
            ok, summary = report(m)
            return self.reply(200 if ok else 500, {'ok': ok, 'log': summary})
        if path == '/close':
            code, out = run_submit(['--market', m, '--close'])
            return self.reply(200 if code == 0 else 500, {'ok': code == 0, 'log': out})
        self.reply(404, {'error': 'not found'})

    def log_message(self, fmt, *args):
        sys.stderr.write('bridge: ' + fmt % args + '\n')


if __name__ == '__main__':
    if not KEY:
        sys.exit('Set BRIDGE_KEY (a long random string; you enter it once on the page).')
    bind, port = os.environ.get('BIND', '127.0.0.1'), int(os.environ.get('PORT', '8765'))
    print(f'bridge on http://{bind}:{port} for {ORIGIN}; IB Gateway at {IB_HOST}:{IB_PORT}; automatic: {",".join(AUTO) or "off"}')
    if AUTO:
        seed_done()
        threading.Thread(target=scheduler, daemon=True).start()
        threading.Thread(target=watchdog, daemon=True).start()
    ThreadingHTTPServer((bind, port), Handler).serve_forever()
