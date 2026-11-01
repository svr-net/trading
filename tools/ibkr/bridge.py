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

Only the page's origin may call it (CORS, with Chrome's private-network permission), and every call
must carry the key (header X-Bridge-Key) that you set in BRIDGE_KEY and enter once on the page; a
wrong key costs a second. On a server it sits behind HTTPS (docker/ibkr/docker-compose.vps.yml).

Automatic trading (AUTO_MARKETS, e.g. UK or UK,US; empty: off): on weekdays the bridge places each
market's orders before its open and closes what is left before its close, by the exchange's clock
(UK 07:45 and 16:20 London, US 09:20 and 15:40 New York, APAC 09:45 and 15:50 Sydney; AUTO_UK_PLACE, AUTO_UK_CLOSE, AUTO_US_PLACE,
AUTO_US_CLOSE change them), each once a day, and skips an orders file more than 4 days old.

Opening auctions (AUCTION=1, the default; 0 off): on the same days, at UK 07:52 London and US 09:24
New York (AUTO_UK_AUCTION, AUTO_US_AUCTION), the bridge records the opening auction of the day's 60
highest ranked candidates (submit.py --auction; it places nothing): the indicative price, volume and
imbalance every 15 seconds until the open, then the open. Each day's recording is kept in
HISTORY_DIR/auction and written to the private store as auction/<MARKET>/<date>.json, where the
research reads it. On Saturdays at AUTO_BARS (06:00; empty: off), and on POST /bars, it fetches
IBKR's daily bars for the pool into the private store (ibkr/<MARKET>/daily.csv): IBKR's opens are the
opening auctions', to check the opens the model learns from.

History (HISTORY_DIR, default the orders cache): every placing writes the orders placed, and after
each close (automatically at UK 16:45 London, US 16:10 New York, or with POST /report) the day's
fills from IBKR (submit.py --report); GET /history returns them by day for the page's History.

Environment: BRIDGE_KEY, EOD_TOKEN (the daily job's token, svr-net/eod contents; EOD_READ_TOKEN also read), IB_HOST, IB_PORT
(4002 paper), CAPITAL (a trade's amount, default 5000), ALLOWED_ORIGIN, BIND (127.0.0.1; 0.0.0.0
inside Docker, published on 127.0.0.1 only), PORT (8765).
"""
import base64
import hmac
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
MARKETS = ('UK', 'US', 'APAC')
AUTO = [m.strip().upper() for m in os.environ.get('AUTO_MARKETS', '').split(',') if m.strip().upper() in MARKETS]
ZONE = {'UK': 'Europe/London', 'US': 'America/New_York', 'APAC': 'Australia/Sydney'}
TIMES = {'UK': (os.environ.get('AUTO_UK_PLACE', '07:45'), os.environ.get('AUTO_UK_CLOSE', '16:20')),
         'US': (os.environ.get('AUTO_US_PLACE', '09:20'), os.environ.get('AUTO_US_CLOSE', '15:40')),
         'APAC': (os.environ.get('AUTO_APAC_PLACE', '09:45'), os.environ.get('AUTO_APAC_CLOSE', '15:50'))}
HISTORY = os.environ.get('HISTORY_DIR', os.environ.get('ORDERS_DIR', '/tmp/ofm-orders'))
REPORT = {'UK': os.environ.get('AUTO_UK_REPORT', '16:45'), 'US': os.environ.get('AUTO_US_REPORT', '16:10'), 'APAC': os.environ.get('AUTO_APAC_REPORT', '16:30')}
# The opening auction's recording (submit.py --auction; places nothing), after the orders are in: the
# LSE auction runs 07:50 to 08:00, the US opening cross at 09:30. AUCTION=0 turns it off.
BARS_AT = os.environ.get('AUTO_BARS', '06:00')  # Saturdays (the exchange's clock): IBKR's daily bars for the pool; empty: off
AUCTION = {'UK': os.environ.get('AUTO_UK_AUCTION', '07:52'), 'US': os.environ.get('AUTO_US_AUCTION', '09:24'), 'APAC': os.environ.get('AUTO_APAC_AUCTION', '09:52')} if os.environ.get('AUCTION', '1') != '0' else {}
STARTED = datetime.now().astimezone()
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
    return path


def run_submit(args, timeout=300):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'submit.py'), *args, '--host', IB_HOST, '--port', str(IB_PORT)],
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


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
    add('Bridge', True, f'running since {STARTED:%Y-%m-%d %H:%M %Z}; automatic: {",".join(AUTO) or "off"}; capital {CAPITAL:g} a trade')
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
    plan = '; '.join(f"{m}: place {TIMES[m][0]}" + (f", auction {AUCTION[m]}" if AUCTION.get(m) else '') + f", close {TIMES[m][1]}, report {REPORT[m]} ({ZONE[m]})" for m in AUTO)
    add('Schedule', True if AUTO else None, (plan or 'automatic trading off (AUTO_MARKETS)') + (f'; IBKR bars Saturdays {BARS_AT}' if BARS_AT else ''))
    failed = [x for x in LAST if not x[3]]
    add('Last automatic actions', None if failed else True, '; '.join(f'{w} {mk} {a}: {"ok" if ok else "FAILED"} {sm}' for w, mk, a, ok, sm in LAST[:4]) or 'none yet')
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
            d = days.setdefault((r['date'], r['market']), {'date': r['date'], 'market': r['market'], 'orders': [], 'trades': []})
            if r['kind'] == 'placed':
                d['orders'] = r['orders']
            elif r['kind'] == 'result':
                d['trades'] = r['trades']  # the latest report of the day wins
    return sorted(days.values(), key=lambda d: (d['date'], d['market']), reverse=True)


def auto_run(market, action):
    """One automatic action; returns (ok, summary)."""
    if action == 'place':
        path = fetch_orders(market)
        as_of = json.load(open(path)).get('asOf', '')
        if not as_of or date.fromisoformat(as_of) < date.today() - timedelta(days=4):
            return False, f'orders from {as_of or "?"} are too old: not placed'
        code, out = run_submit(['--market', market, '--orders', path, '--capital', str(CAPITAL)])
        if code == 0:
            placed(market, path, False)
    elif action == 'report':
        return report(market)
    elif action == 'auction':
        return record_auction(market)
    elif action == 'bars':
        return fetch_bars(market)
    else:
        code, out = run_submit(['--market', market, '--close'])
    return code == 0, out.splitlines()[-1] if out else ''


def scheduler():
    """Places and closes each automatic market's orders at its times, weekdays, once a day each."""
    done = set()
    os.makedirs(CACHE, exist_ok=True)
    while True:
        for m in AUTO:
            now = datetime.now(ZoneInfo(ZONE[m]))
            weekend = now.weekday() >= 5 and not os.environ.get('AUTO_WEEKENDS')  # AUTO_WEEKENDS: for testing only
            plan = [('bars', BARS_AT)] if now.weekday() == 5 and BARS_AT else []  # Saturdays: IBKR's daily bars
            if not weekend:
                plan += [('place', TIMES[m][0]), ('auction', AUCTION.get(m)), ('close', TIMES[m][1]), ('report', REPORT[m])]
            if not plan:
                continue
            for action, hhmm in ((a, t) for a, t in plan if t):
                h, mi = (int(x) for x in hhmm.split(':'))
                due = now.replace(hour=h, minute=mi, second=0, microsecond=0)
                mark = os.path.join(CACHE, f'done-{m}-{action}-{now.date().isoformat()}')
                # Within 10 minutes after its time (a restart just after still catches it), once.
                if due <= now < due + timedelta(minutes=10) and mark not in done and not os.path.exists(mark):
                    done.add(mark)
                    open(mark, 'w').close()
                    try:
                        ok, summary = auto_run(m, action)
                    except Exception as e:  # noqa: BLE001
                        ok, summary = False, str(e)
                    LAST.insert(0, (now.strftime('%Y-%m-%d %H:%M %Z'), m, action, ok, summary))
                    del LAST[10:]
                    sys.stderr.write(f'bridge: automatic {action} {m}: {"ok" if ok else "FAILED"} {summary}\n')
        time.sleep(20)


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
            self.reply(400, {'error': 'market must be UK, US or APAC'})
            return None
        return m

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/status':  # no key needed: tells the page the bridge is there
            return self.reply(200, {'bridge': True, 'gateway': gateway_up(), 'port': IB_PORT, 'capital': CAPITAL, 'keyed': bool(KEY),
                                    'auto': {m: {'place': TIMES[m][0], 'auction': AUCTION.get(m), 'close': TIMES[m][1], 'report': REPORT[m], 'zone': ZONE[m]} for m in AUTO},
                                    'last': [{'when': w, 'market': mk, 'action': a, 'ok': ok, 'summary': sm} for w, mk, a, ok, sm in LAST]})
        if not self.allowed():
            return
        if u.path == '/diag':
            return self.reply(200, {'checks': diag()})
        if u.path == '/history':
            return self.reply(200, {'days': history()})
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
                o['quantity'] = math.floor(CAPITAL / share) if share > 0 else 0
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
        threading.Thread(target=scheduler, daemon=True).start()
    ThreadingHTTPServer((bind, port), Handler).serve_forever()
