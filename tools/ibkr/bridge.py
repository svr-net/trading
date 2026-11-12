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
IBKR's daily bars for the pool (two years) and every trading day after the report (BARS_DAILY=1, the
default; 0 off) the latest month of them, each merged into the private store's ibkr/<MARKET>/daily.csv
(a fetch never removes bars), so the evening's and the morning's scans learn and backtest on the prices that traded:
IBKR's open and close are the auctions', its low one the exchange's book traded at (EODData's low for
a thin UK share was a median 1.3% under IBKR's: dip limits the backtest filled that IBKR would not).
One fetch at a time (IBKR's pacing).

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
MARKETS = ('UK', 'EU', 'US', 'APAC', 'INDEX')  # INDEX: US-listed index ETFs, on New York's clock
AUTO = [m.strip().upper() for m in os.environ.get('AUTO_MARKETS', '').split(',') if m.strip().upper() in MARKETS]
ZONE = {'UK': 'Europe/London', 'EU': 'Europe/Paris', 'US': 'America/New_York', 'APAC': 'Australia/Sydney', 'INDEX': 'America/New_York'}
TIMES = {'UK': (os.environ.get('AUTO_UK_PLACE', '07:45'), os.environ.get('AUTO_UK_CLOSE', '16:20')),
         'US': (os.environ.get('AUTO_US_PLACE', '09:20'), os.environ.get('AUTO_US_CLOSE', '15:40')),
         'EU': (os.environ.get('AUTO_EU_PLACE', '08:45'), os.environ.get('AUTO_EU_CLOSE', '17:20')),
         'APAC': (os.environ.get('AUTO_APAC_PLACE', '09:45'), os.environ.get('AUTO_APAC_CLOSE', '15:50')),
         'INDEX': (os.environ.get('AUTO_INDEX_PLACE', '09:21'), os.environ.get('AUTO_INDEX_CLOSE', '15:41'))}
HISTORY = os.environ.get('HISTORY_DIR', os.environ.get('ORDERS_DIR', '/tmp/ofm-orders'))
REPORT = {'INDEX': os.environ.get('AUTO_INDEX_REPORT', '16:20'), 'UK': os.environ.get('AUTO_UK_REPORT', '16:45'), 'US': os.environ.get('AUTO_US_REPORT', '16:10'), 'APAC': os.environ.get('AUTO_APAC_REPORT', '16:30'), 'EU': os.environ.get('AUTO_EU_REPORT', '17:50')}
# The opening auction's recording (submit.py --auction; places nothing), after the orders are in: the
# LSE auction runs 07:50 to 08:00, the US opening cross at 09:30. AUCTION=0 turns it off.
BARS_AT = os.environ.get('AUTO_BARS', '06:00')  # Saturdays (the exchange's clock): IBKR's daily bars for the pool; empty: off
BARS_DAILY = os.environ.get('BARS_DAILY', '1') != '0'  # every trading day after the report: the latest month merged in
# The multi-day books (HOLD_BOOKS, MARKET:SHADOW,...; empty: none): a shadow's orders (the store's
# shadow/<MARKET>-<SHADOW>.json) placed after the day's orders and kept over the days (submit.py --holds):
# offered at each close only above their entry price, sold at the fifth; the day's close and the
# leftovers step never sell them (their own orderRef; their shares left out of the model's).
HOLD_BOOKS = dict(x.split(':', 1) for x in os.environ.get('HOLD_BOOKS', 'UK:hold-up').split(',') if ':' in x)
AUCTION = {'INDEX': os.environ.get('AUTO_INDEX_AUCTION', '09:25'), 'UK': os.environ.get('AUTO_UK_AUCTION', '07:52'), 'US': os.environ.get('AUTO_US_AUCTION', '09:24'), 'APAC': os.environ.get('AUTO_APAC_AUCTION', '09:52'), 'EU': os.environ.get('AUTO_EU_AUCTION', '08:52')} if os.environ.get('AUCTION', '1') != '0' else {}
PATHS = os.environ.get('PATHS', '1') != '0'
STARTED = datetime.now().astimezone()
VERSION = os.environ.get('BRIDGE_VERSION', 'unknown')  # the commit it was built from
TRADES = {}  # market -> (when, the live state): read from IBKR in the background (refresher), served from here
TRADES_NOTE = {}  # market -> the last read's failure, if it failed
IBDIAG = {'at': None, 'd': None}  # the IBKR part of /diag (submit.py --diag), refreshed in the background
READING = set()  # the background reads in flight ('trades-UK', 'diag', ...): one of each at a time
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


# A daily-bars fetch (hundreds of historical-data requests, an hour or more) slowed the gateway's other
# answers past their time limit: on 11 Nov the US report (21:10 UTC) timed out during the EU's fetch. So
# every other step's run pauses the fetch first: the bridge writes BARS_PAUSE, the fetch (submit.py --bars
# --pause) stops between two shares and answers with BARS_PAUSE.ack, and the step runs on a quiet gateway;
# the fetch goes on when no step needs the gateway. The all-day watches (--rebuy, --paths) do not pause it.
BARS_PAUSE = os.path.join(CACHE, 'bars.pause')
QUIET = {'n': 0}
QUIET_LOCK = threading.Lock()
LONG_RUNS = ('--bars', '--rebuy', '--paths')


STEPS = ('leftovers', 'place', 'close', 'flat', 'report', 'hold-place', 'hold-close')


def steps_pending():
    """Whether a step that needs the gateway is running, or a market's report is due and not done yet (a
    fetch started now would run into it: on 11 Nov the US report and the US fetch started in the same
    minute after a restart, the report before the fetch could be paused). The fetch waits for them."""
    if any(a in STEPS for _, a in RUNNING):
        return True
    for m in AUTO:
        now = datetime.now(ZoneInfo(ZONE[m]))
        if now.weekday() < 5 and at(now, REPORT[m]) <= now < at(now, REPORT[m], 180) and not is_done(m, 'report'):
            return True
    return False


def bars_quiet(wait=150):
    """Pauses a running bars fetch (one more step needs the gateway) and waits, at most `wait` seconds,
    for it to confirm it stopped between shares (its own request then done). True when quiet."""
    with QUIET_LOCK:
        QUIET['n'] += 1
        os.makedirs(CACHE, exist_ok=True)
        open(BARS_PAUSE, 'w').close()
    if not BARS_RUNNING:
        return True
    for _ in range(wait):
        if os.path.exists(BARS_PAUSE + '.ack'):
            return True
        time.sleep(1)
    sys.stderr.write('bridge: the bars fetch did not confirm its pause in time; the step runs anyway\n')
    return False


def bars_resume():
    """One step less needs the gateway: with none left, the bars fetch goes on."""
    with QUIET_LOCK:
        QUIET['n'] = max(0, QUIET['n'] - 1)
        if QUIET['n'] == 0:
            for p in (BARS_PAUSE, BARS_PAUSE + '.ack'):
                try:
                    os.remove(p)
                except OSError:
                    pass


def run_submit(args, timeout=300):
    if not any(a in args for a in LONG_RUNS):  # a step: the bars fetch waits for it
        bars_quiet()
        try:
            return run_submit_now(args, timeout)
        finally:
            bars_resume()
    return run_submit_now(args, timeout)


def run_submit_now(args, timeout=300):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'submit.py'), *args, '--host', IB_HOST, '--port', str(IB_PORT)],
                       capture_output=True, text=True, timeout=timeout)
    # The summaries read the last line: the script's own result (its standard output), success or
    # failure, so ib_async's log lines on standard error never stand in for it (5 Nov: an EU close's
    # failure showed only 'Error 300: Can't find EId'); its error last only when it crashed. A failure's
    # whole output goes to the bridge's log (docker logs) too.
    crashed = 'Traceback' in p.stderr or not p.stdout.strip()
    out = (p.stdout + p.stderr) if crashed else (p.stderr + p.stdout)
    if p.returncode:
        sys.stderr.write(f"bridge: submit.py {' '.join(args)} exited {p.returncode}:\n{p.stdout}{p.stderr}\n")
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


def store_get(path):
    """One file of the private store (svr-net/eod), or None when it is not there."""
    url = f'https://api.github.com/repos/svr-net/eod/contents/{path}'
    head = {'Authorization': f'Bearer {TOKEN}', 'Accept': 'application/vnd.github.raw', 'User-Agent': 'ofm-bridge'}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=head), timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


# IBKR's market data lines (a paper account streams about 100 shares at once; more answer "101: Max
# number of tickers has been reached" and those shares record nothing): the recordings share a budget
# (MARKET_LINES, 90 by default, the rest kept for the orders' checks and /diag). Each asks for its share
# of the budget among the automatic markets whose sessions overlap (UK, EU and US do, APAC trades
# alone), the paths at least the day's orders; it gets what is free when it starts and gives it back
# when it ends.
LINES_BUDGET = int(os.environ.get('MARKET_LINES', '90'))
LINES_USED = {}  # recording ('paths-UK', 'auction-EU', ...) -> the lines it holds
OVERLAP = {'UK': ('UK', 'EU', 'US', 'INDEX'), 'EU': ('UK', 'EU', 'US', 'INDEX'), 'US': ('UK', 'EU', 'US', 'INDEX'), 'INDEX': ('UK', 'EU', 'US', 'INDEX'), 'APAC': ('APAC',)}


def lines_grant(name, want):
    with START_LOCK:
        free = LINES_BUDGET - sum(n for k, n in LINES_USED.items() if k != name)
        LINES_USED[name] = max(0, min(want, free))
        return LINES_USED[name]


def lines_release(name):
    with START_LOCK:
        LINES_USED.pop(name, None)


def lines_want(market, orders=0):
    together = sum(1 for m in OVERLAP[market] if m in AUTO) or 1
    return max(LINES_BUDGET // together, orders)


def record_auction(market):
    """The candidates' opening auction (submit.py --auction), kept with the history and in the private
    store as auction/<MARKET>/<date>.json; returns (ok, summary). Its lines come after the paths'."""
    path = fetch_orders(market)
    name = f'auction-{market}'
    n = lines_grant(name, lines_want(market))
    if n == 0:
        lines_release(name)
        return True, 'skipped: no market data lines free (the paths hold them; without real-time data on the account it records nothing anyway)'
    try:
        code, out = run_submit(['--market', market, '--orders', path, '--auction', '--lines', str(n)], timeout=1500)
    finally:
        lines_release(name)
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
    name = f'paths-{market}'
    try:
        day = datetime.now(ZoneInfo(ZONE[market])).date().isoformat()
        out_file = os.path.join(HISTORY, f'paths-{market}-{day}.csv')
        os.makedirs(HISTORY, exist_ok=True)
        path = fetch_orders(market)
        n = lines_grant(name, lines_want(market, len(json.load(open(path)).get('orders', []))))
        if n == 0:
            return False, 'no market data lines free'
        code, out = run_submit(['--market', market, '--orders', path, '--paths', '--out', out_file, '--lines', str(n)], timeout=12 * 3600)
        summary = out.splitlines()[-1] if out else ''
        if code != 0 or not os.path.exists(out_file):
            return False, summary or 'failed'
        with open(out_file, 'rb') as f:
            store_put(f'paths/{market}/{day}.csv', f.read(), f'{market} intraday paths {day}')
        return True, summary
    finally:
        lines_release(name)
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
    book = json.load(open(path))
    if book.get('book') == 'hold':  # a multi-day book (apps/ofm_uk --hold): the hold book places it (hold-place), nothing within the day
        LIVE_HOLD[market] = (datetime.now(ZoneInfo(ZONE[market])).date().isoformat(), True)
        mark_done(market, 'place')
        return True, f"a multi-day book ({len(book.get('orders', []))} orders, held up to {book.get('days', 5)} days): placed by hold-place"
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


BACKGROUND = [0]  # background jobs in flight (any kind): the update waits for none


def in_background(market, action, job):
    def run():
        BACKGROUND[0] += 1
        try:
            ok, summary = job(market)
        except Exception as e:  # noqa: BLE001
            ok, summary = False, str(e)
        finally:
            BACKGROUND[0] -= 1
        LAST.insert(0, (datetime.now(ZoneInfo(ZONE[market])).strftime('%Y-%m-%d %H:%M %Z'), market, action, ok, summary))
        del LAST[10:]
        sys.stderr.write(f'bridge: {action} {market}: {"ok" if ok else "FAILED"} {summary}\n')
    threading.Thread(target=run, daemon=True).start()


def fetch_bars(market, daily=False):
    """IBKR's daily bars for the pool (submit.py --bars, about 75 minutes for the UK's), merged into the
    file kept here and written to the private store as ibkr/<MARKET>/daily.csv; returns (ok,
    summary). daily: only the latest month is fetched. Every fetch is merged in (a share-day IBKR
    answers again replaces the kept one; the rest are kept), so a fetch IBKR answered badly never
    loses bars: on 7 Nov the Saturday refresh got no EU bars at all and replaced the file with an
    empty one. Without a kept file, the store's is the start; a fetch with no bars changes nothing."""
    import csv
    if BARS_RUNNING:
        return False, f"waiting: {', '.join(sorted(BARS_RUNNING))} bars fetching"
    BARS_RUNNING.add(market)
    try:
        out_file = os.path.join(HISTORY, f'ibkr-{market}.csv')
        if not (os.path.exists(out_file) and os.path.getsize(out_file) > 100):
            kept = store_get(f'ibkr/{market}/daily.csv')
            if kept and len(kept) > 100:
                with open(out_file, 'wb') as f:
                    f.write(kept)
        have = os.path.exists(out_file) and os.path.getsize(out_file) > 100
        target = out_file + '.new'
        with QUIET_LOCK:  # a pause no step holds (one left by a restart) would hold the fetch for good
            if QUIET['n'] == 0:
                for p in (BARS_PAUSE, BARS_PAUSE + '.ack'):
                    if os.path.exists(p):
                        os.remove(p)
        code, out = run_submit(['--market', market, '--orders', fetch_orders(market), '--bars', '--out', target, '--pause', BARS_PAUSE,
                                *(['--duration', '1 M'] if daily and have else [])], timeout=6 * 3600)
        summary = out.splitlines()[-1] if out else ''
        if code != 0:
            return False, summary or 'failed'
        with open(target, newline='') as f:
            new = list(csv.reader(f))
        if len(new) < 2:
            return False, f'{summary}: no bars from IBKR; the kept file unchanged'
        rows = []
        if have:
            fresh = {(r[0][:10], r[1]) for r in new[1:] if len(r) >= 2}
            with open(out_file, newline='') as f:
                rows = [r for r in list(csv.reader(f))[1:] if len(r) >= 2 and (r[0][:10], r[1]) not in fresh]
        rows += new[1:]
        rows.sort(key=lambda r: (r[0][:10], r[1]))
        with open(out_file + '.tmp', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(new[0])
            w.writerows(rows)
        os.replace(out_file + '.tmp', out_file)
        summary += f'; {len(rows)} rows kept, {len({r[1] for r in rows})} shares, the latest {rows[-1][0][:10]}'
        with open(out_file, 'rb') as f:
            store_put(f'ibkr/{market}/daily.csv', f.read(), f'{market} daily bars from IBKR {date.today().isoformat()}')
        return True, summary
    finally:
        BARS_RUNNING.discard(market)


def read_once(name, fn):
    """Runs fn in the background unless the same read is already in flight; never blocks a request."""
    with START_LOCK:
        if name in READING:
            return
        READING.add(name)

    def run():
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f'bridge: background read {name}: {e}\n')
        finally:
            READING.discard(name)
    threading.Thread(target=run, daemon=True).start()


def read_trades(m):
    """Today's trades and the model's holdings from IBKR (submit.py --state), into TRADES."""
    # the model's shares: the day's book's lately and the multi-day book's held (the page shows both)
    tickers = sorted(set(model_tickers(m)) | hold_symbols(m))
    code, out = run_submit(['--market', m, '--state', '--capital', str(CAPITAL), '--tickers', ','.join(tickers)], timeout=180)
    line = next((x for x in reversed(out.splitlines()) if x.startswith('{')), '')
    if code != 0 or not line:
        TRADES_NOTE[m] = (out.splitlines() or ['no answer from IBKR'])[-1]
        return
    TRADES[m] = (time.monotonic(), json.loads(line))
    TRADES_NOTE.pop(m, None)


def read_ibdiag():
    """The API, the account and the data on one share (submit.py --diag), into IBDIAG."""
    m = (AUTO or ['UK'])[0]
    try:
        code, text = run_submit(['--market', m, '--diag'], timeout=180)
        line = next((l for l in reversed(text.splitlines()) if l.startswith('{')), '')
        d = json.loads(line) if line else {'api': False, 'error': text.splitlines()[-1] if text else 'no answer'}
    except Exception as e:  # noqa: BLE001
        d = {'api': False, 'error': str(e)}
    IBDIAG['at'], IBDIAG['d'] = time.monotonic(), d


def refresher():
    """Keeps the slow reads fresh so that requests answer at once from the latest snapshot: each
    automatic market's trades every minute from 30 minutes before its placing to 30 minutes after its
    close (weekdays), the IBKR check every 5 minutes. Each read is a short-lived client of its own."""
    while True:
        for m in AUTO:
            try:
                now = datetime.now(ZoneInfo(ZONE[m]))
                place, close = TIMES[m]
                if now.weekday() < 5 and at(now, place, -30) <= now < at(now, close, 30):
                    hit = TRADES.get(m)
                    if not hit or time.monotonic() - hit[0] >= 55:
                        read_once(f'trades-{m}', lambda m=m: read_trades(m))
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f'bridge: refresher {m}: {e}\n')
        if IBDIAG['at'] is None or time.monotonic() - IBDIAG['at'] >= 300:
            read_once('diag', read_ibdiag)
        time.sleep(20)


def diag():
    """The checks behind GET /diag, each {check, ok (true, false, or null: a warning), detail}."""
    out = []
    add = lambda check, ok, detail: out.append({'check': check, 'ok': ok, 'detail': detail})  # noqa: E731
    add('Bridge', True, f'version {VERSION}, running since {STARTED:%Y-%m-%d %H:%M %Z}; automatic: {",".join(AUTO) or "off"}; capital {CAPITAL:g} a trade')
    add('Bridge update', None if UPDATE['note'] == 'not checked yet' else not UPDATE['note'].startswith(('no restarter', 'failed')),
        f"newest image checked when no market is near its day: {UPDATE['note']}")
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
    # The IBKR part from the background (refresher): a request never waits on IBKR.
    if IBDIAG['at'] is None or time.monotonic() - IBDIAG['at'] >= 300:
        read_once('diag', read_ibdiag)
    d = IBDIAG['d']
    ago = f" (checked {int(time.monotonic() - IBDIAG['at'])} s ago)" if IBDIAG['at'] is not None else ''
    if d is None:
        add('IB Gateway API', None, 'being checked in the background: ask again in a minute')
        d = {}
    elif not d.get('api'):
        add('IB Gateway API', False, d.get('error', 'no answer') + ago)
    else:
        add('IB Gateway API', True, f"logged in, {'paper' if d.get('paper') else 'LIVE'} account; IBKR time {d.get('serverTime', '?')}{ago}")
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
        add(f'{m} today', None, ', '.join(f"{a} {'done' if is_done(m, a) else '-'}" for a in ('leftovers', 'place', 'paths', 'auction', 'close', 'flat', 'report') + (('hold-place', 'hold-close') if hold_book(m) else ()))
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
    """The day's close: today's positions sold at the close (submit.py --close), then any of the
    model's shares still held from an earlier day sold too (submit.py --leftovers: a sale confirmed
    with IBKR), so the day ends flat rather than waiting for the next morning (6 Nov: three UK
    shares held from earlier days were not part of the close and would have stayed over the weekend)."""
    n = free_client(market)
    code, out = run_submit(['--market', market, '--close'], timeout=600)
    last = out.splitlines()[-1] if out else ''
    tickers = model_tickers(market)
    if tickers:
        code2, out2 = run_submit(['--market', market, '--leftovers', '--tickers', ','.join(tickers)], timeout=600)
        last2 = out2.splitlines()[-1] if out2 else ''
        if 'held from earlier days' in last2 and not last2.split(':')[1].strip().startswith('0 held'):
            last += f'; {last2}'
        code = code or code2
    return code == 0, (f'ended {n} buy-back watch still connected; ' if n else '') + last


def hold_state(market):
    return os.path.join(HISTORY, f'holds-{market}.json')


def hold_symbols(market):
    """The multi-day book's shares held now (its state file): never the day's leftovers."""
    try:
        return set(json.load(open(hold_state(market))).get('positions', {}))
    except (OSError, ValueError):
        return set()


LIVE_HOLD = {}  # market -> (day, whether its live orders are a multi-day book)


def live_is_hold(market):
    """Whether the market's live orders (orders/<MARKET>.json) are a multi-day book ("book": "hold",
    apps/ofm_uk --hold): then the hold book places them (submit.py --holds), not the day's placing.
    Read once a day (a failed read is tried again later)."""
    day = datetime.now(ZoneInfo(ZONE[market])).date().isoformat()
    c = LIVE_HOLD.get(market)
    if c and c[0] == day:
        return c[1]
    try:
        v = json.load(open(fetch_orders(market))).get('book') == 'hold'
    except Exception:  # noqa: BLE001
        return False
    LIVE_HOLD[market] = (day, v)
    return v


def hold_book(market):
    """Whether the market runs a multi-day book: its live orders (a hold book), or a shadow's (HOLD_BOOKS)."""
    return live_is_hold(market) or market in HOLD_BOOKS


def place_holds(market):
    """The multi-day book's orders for the day (after the day's orders): submit.py --holds place. The
    live orders when they are a hold book, else the shadow's named in HOLD_BOOKS."""
    if live_is_hold(market):
        path = fetch_orders(market)
    else:
        name = HOLD_BOOKS[market]
        body = store_get(f'shadow/{market}-{name}.json')
        if not body:
            return False, f'no shadow/{market}-{name}.json in the store yet'
        path = os.path.join(CACHE, f'{market}-{name}.json')
        with open(path, 'wb') as f:
            f.write(body)
    code, out = run_submit(['--market', market, '--holds', 'place', '--orders', path, '--hold-state', hold_state(market)], timeout=900)
    return code == 0, out.splitlines()[-1] if out else ''


def close_holds(market):
    """The multi-day book's positions offered in the closing auction: submit.py --holds close."""
    code, out = run_submit(['--market', market, '--holds', 'close', '--hold-state', hold_state(market)], timeout=600)
    return code == 0, out.splitlines()[-1] if out else ''


def model_tickers(market, days=10):
    """The shares the model placed in this market lately (the bridge's history): the only holdings
    the leftovers step may sell (never the multi-day book's)."""
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
    keep = hold_symbols(market)
    return sorted(t for t in out if t.split(':')[-1] not in keep)


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
                if now.weekday() == 5 and BARS_AT and at(now, BARS_AT) <= now < at(now, BARS_AT, 240) and not is_done(m, 'bars') and not BARS_RUNNING \
                        and may_try(m, 'bars', 10):
                    job(m, 'bars', fetch_bars)
                if now.weekday() >= 5 and not os.environ.get('AUTO_WEEKENDS'):
                    continue
                place, close = TIMES[m]
                up = GATEWAY['ok'] is not False
                # IBKR's daily bars, the latest month merged in, after the day's report (one market at a time).
                if BARS_DAILY and up and at(now, REPORT[m], 15) <= now < at(now, REPORT[m], 360) and not is_done(m, 'bars') and not BARS_RUNNING \
                        and not steps_pending() and may_try(m, 'bars', 20, 4):
                    job(m, 'bars', lambda mk: fetch_bars(mk, daily=True))
                if up and at(now, place, -30) <= now < at(now, close, -15) and not is_done(m, 'leftovers') and may_try(m, 'leftovers', 5, 3):
                    job(m, 'leftovers', sell_leftovers)
                tried = ATTEMPTS.get((m, 'leftovers', now.date().isoformat()), (0, None))[0] > 0
                settled = is_done(m, 'leftovers') or (tried and (m, 'leftovers') not in RUNNING)  # the placing waits for it to finish
                if up and settled and at(now, place) <= now < at(now, close, -15) and not is_done(m, 'place') and may_try(m, 'place', 5, 4):
                    job(m, 'place', place_and_watch, done_on_ok=False)  # place_and_watch marks it and starts the watch
                if up and is_done(m, 'place') and not live_is_hold(m) and now < at(now, close, -3) and m not in REBUY_RUNNING and (m, 'place') not in RUNNING \
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
                if hold_book(m) and up and is_done(m, 'place') and at(now, place) <= now < at(now, close, -15) and not is_done(m, 'hold-place') \
                        and (m, 'place') not in RUNNING and may_try(m, 'hold-place', 5, 3):
                    job(m, 'hold-place', place_holds)
                if hold_book(m) and up and at(now, close) <= now < at(now, close, 9) and not is_done(m, 'hold-close') and may_try(m, 'hold-close', 2, 3):
                    job(m, 'hold-close', close_holds)
                if up and at(now, close, 30) <= now < at(now, close, 240) and not is_done(m, 'flat') and may_try(m, 'flat', 10):
                    job(m, 'flat', flat_check, done_on_ok=False)
                if at(now, REPORT[m]) <= now < at(now, REPORT[m], 180) and not is_done(m, 'report') and may_try(m, 'report', 10):
                    job(m, 'report', report)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f'bridge: scheduler {m}: {e}\n')
        try:
            if quiet() and time.monotonic() - UPDATE['last'] >= 3600:
                UPDATE['last'] = time.monotonic()
                UPDATE['note'] = ask_restarter('/update', 900)
                sys.stderr.write(f"bridge: update: {UPDATE['note']}\n")
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f'bridge: update: {e}\n')
        time.sleep(30)


UPDATE = {'last': -1e9, 'note': 'not checked yet'}


def quiet():
    """No job in flight and no automatic market within an hour of its day (from an hour before the
    placing to half an hour after the report, weekdays): the time the bridge may be replaced by a newer
    image without touching a trading day."""
    if RUNNING or BACKGROUND[0] or REBUY_RUNNING or PATHS_RUNNING or BARS_RUNNING:
        return False
    for m in AUTO:
        now = datetime.now(ZoneInfo(ZONE[m]))
        if now.weekday() < 5 or os.environ.get('AUTO_WEEKENDS'):
            if at(now, TIMES[m][0], -60) <= now < at(now, REPORT[m], 30):
                return False
    return True


def ask_restarter(path, timeout):
    """POSTs to the restarter (tools/ibkr/restarter.py, the only container with Docker's socket);
    returns its note."""
    url = os.environ.get('GATEWAY_RESTARTER', 'http://restarter:8766') + path
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=b'', method='POST'), timeout=timeout) as r:
            return json.loads(r.read()).get('note', 'done')
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b'{}').get('note', f'restarter answered {e.code}')
    except Exception as e:  # noqa: BLE001
        return f'no restarter ({type(e).__name__})'


def restart_gateway():
    """Asks the restarter (tools/ibkr/restarter.py, the only container with Docker's socket) to restart
    IB Gateway; returns a note."""
    note = ask_restarter('/restart', 150)
    return note + ': restart IB Gateway by hand' if note.startswith('no restarter') else note


def watchdog():
    """Every minute, whether IB Gateway's API answers and still reaches IBKR's servers (a connection,
    its clock and one contract lookup: submit.py --ping; not only the port);
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
            # From the background's latest read (refresher), at once; an old or missing one is read again
            # in the background and the request answers with what there is.
            hit = TRADES.get(m)
            if not hit or time.monotonic() - hit[0] >= 55:
                read_once(f'trades-{m}', lambda: read_trades(m))
            if not hit:
                return self.reply(202, {'error': TRADES_NOTE.get(m) or 'being read from IBKR: ask again in a minute', 'shares': []})
            return self.reply(200, {**hit[1], 'age': int(time.monotonic() - hit[0]), 'note': TRADES_NOTE.get(m, '')})
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
    threading.Thread(target=refresher, daemon=True).start()
    ThreadingHTTPServer((bind, port), Handler).serve_forever()
