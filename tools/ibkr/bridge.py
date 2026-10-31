"""A local bridge between the page and IB Gateway: http://127.0.0.1:8765, on your own computer only.

The page (svr-net.github.io/trading) looks for it and, when it answers, shows a "Your broker" panel:
the day's orders with their prices (fetched here from the private svr-net/eod, never published),
and buttons that place them or close the positions through IB Gateway (tools/ibkr/submit.py).

  GET  /status                     the bridge and whether IB Gateway's API port answers
  GET  /orders?market=UK           the day's orders, with each trade's quantity for --capital
  POST /place  {"market": "UK"}    place them (submit.py)
  POST /close  {"market": "UK"}    cancel the unfilled ones and sell at the close (submit.py --close)

Only the page's origin may call it (CORS, with Chrome's private-network permission), and every call
must carry the key (header X-Bridge-Key) that you set in BRIDGE_KEY and enter once on the page; a
wrong key costs a second. On a server it sits behind HTTPS (docker/ibkr/docker-compose.vps.yml).

Automatic trading (AUTO_MARKETS, e.g. UK or UK,US; empty: off): on weekdays the bridge places each
market's orders before its open and closes what is left before its close, by the exchange's clock
(UK 07:45 and 16:20 London, US 09:20 and 15:40 New York; AUTO_UK_PLACE, AUTO_UK_CLOSE, AUTO_US_PLACE,
AUTO_US_CLOSE change them), each once a day, and skips an orders file more than 4 days old.

Environment: BRIDGE_KEY, EOD_READ_TOKEN (read-only, svr-net/eod contents), IB_HOST, IB_PORT
(4002 paper), CAPITAL (a trade's amount, default 5000), ALLOWED_ORIGIN, BIND (127.0.0.1; 0.0.0.0
inside Docker, published on 127.0.0.1 only), PORT (8765).
"""
import hmac
import json
import math
import os
import socket
import subprocess
import sys
import time
import threading
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get('BRIDGE_KEY', '')
TOKEN = os.environ.get('EOD_READ_TOKEN', '')
IB_HOST = os.environ.get('IB_HOST', '127.0.0.1')
IB_PORT = int(os.environ.get('IB_PORT', '4002'))
CAPITAL = float(os.environ.get('CAPITAL', '5000'))
ORIGIN = os.environ.get('ALLOWED_ORIGIN', 'https://svr-net.github.io')
CACHE = os.environ.get('ORDERS_DIR', '/tmp/ofm-orders')
MARKETS = ('UK', 'US')
AUTO = [m.strip().upper() for m in os.environ.get('AUTO_MARKETS', '').split(',') if m.strip().upper() in MARKETS]
ZONE = {'UK': 'Europe/London', 'US': 'America/New_York'}
TIMES = {'UK': (os.environ.get('AUTO_UK_PLACE', '07:45'), os.environ.get('AUTO_UK_CLOSE', '16:20')),
         'US': (os.environ.get('AUTO_US_PLACE', '09:20'), os.environ.get('AUTO_US_CLOSE', '15:40'))}
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
        req = urllib.request.Request(f'https://raw.githubusercontent.com/svr-net/eod/main/orders/{market}.json',
                                     headers={'Authorization': f'Bearer {TOKEN}', 'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=20) as r, open(path, 'wb') as f:
            f.write(r.read())
    return path


def run_submit(args):
    p = subprocess.run([sys.executable, os.path.join(HERE, 'submit.py'), *args, '--host', IB_HOST, '--port', str(IB_PORT)],
                       capture_output=True, text=True, timeout=300)
    return p.returncode, (p.stdout + p.stderr).strip()


def auto_run(market, action):
    """One automatic action; returns (ok, summary)."""
    if action == 'place':
        path = fetch_orders(market)
        as_of = json.load(open(path)).get('asOf', '')
        if not as_of or date.fromisoformat(as_of) < date.today() - timedelta(days=4):
            return False, f'orders from {as_of or "?"} are too old: not placed'
        code, out = run_submit(['--market', market, '--orders', path, '--capital', str(CAPITAL)])
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
            if now.weekday() >= 5 and not os.environ.get('AUTO_WEEKENDS'):  # AUTO_WEEKENDS: for testing only
                continue
            for action, hhmm in zip(('place', 'close'), TIMES[m]):
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
            self.reply(400, {'error': 'market must be UK or US'})
            return None
        return m

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/status':  # no key needed: tells the page the bridge is there
            return self.reply(200, {'bridge': True, 'gateway': gateway_up(), 'port': IB_PORT, 'capital': CAPITAL, 'keyed': bool(KEY),
                                    'auto': {m: {'place': TIMES[m][0], 'close': TIMES[m][1], 'zone': ZONE[m]} for m in AUTO},
                                    'last': [{'when': w, 'market': mk, 'action': a, 'ok': ok, 'summary': sm} for w, mk, a, ok, sm in LAST]})
        if not self.allowed():
            return
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
            return self.reply(200 if code == 0 else 500, {'ok': code == 0, 'log': out})
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
