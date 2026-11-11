"""The UK model's orders through tools/ibkr/submit.py against a stand-in IBKR (no gateway, no network):
a day of orders for the open, from placing them to the watch's levels.

  python3 tools/ibkr/test/levels.py

Checked: each order for the open ("entry": "open") is one limit buy good only until the opening
auction is over (GTD), with no levels attached; the watch places its stop-loss and take-profit from
the price it was bought at (not the last close), one cancelling the other, for the shares bought,
placed again only when more were bought, never twice; a buy whose good-till time IBKR refuses goes
again as a day order and is cancelled once the auction is over; a stop that sells cancels the
take-profit; a share never bought gets no levels.
"""
import datetime as _dt
import io
import json
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from zoneinfo import ZoneInfo

LONDON = ZoneInfo('Europe/London')
CLOCK = [_dt.datetime.combine(_dt.date.today(), _dt.time(7, 45), LONDON)]


class FakeDateTime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        t = CLOCK[0]
        return t.astimezone(tz) if tz else t.replace(tzinfo=None)


fake_dt = types.ModuleType('datetime')
fake_dt.__dict__.update({k: getattr(_dt, k) for k in dir(_dt) if not k.startswith('__')})
fake_dt.datetime = FakeDateTime
sys.modules['datetime'] = fake_dt

# The stand-in ib_async.
ib_async = types.ModuleType('ib_async')


class Order:
    def __init__(self, action, qty, orderType, **kw):
        self.action, self.totalQuantity, self.orderType = action, qty, orderType
        self.lmtPrice = self.auxPrice = 0.0
        self.orderRef = self.tif = self.ocaGroup = self.goodTillDate = self.goodAfterTime = ''
        self.transmit, self.parentId, self.ocaType, self.orderId, self.permId = True, 0, 0, 0, 0
        for k, v in kw.items():
            setattr(self, k, v)


def LimitOrder(action, qty, price, **kw):
    return Order(action, qty, 'LMT', lmtPrice=price, **kw)


def StopOrder(action, qty, price, **kw):
    return Order(action, qty, 'STP', auxPrice=price, **kw)


def StopLimitOrder(action, qty, lmt, stop, **kw):
    return Order(action, qty, 'STP LMT', lmtPrice=lmt, auxPrice=stop, **kw)


def MarketOrder(action, qty, **kw):
    return Order(action, qty, 'MKT', **kw)


class Contract:
    def __init__(self, symbol='', exchange='', currency='', conId=0, primaryExchange=''):
        self.symbol, self.exchange, self.currency, self.conId, self.primaryExchange = symbol, exchange, currency, conId, primaryExchange


def Stock(symbol, exchange, currency, primaryExchange=''):
    return Contract(symbol, exchange, currency, 0, primaryExchange)


NS = types.SimpleNamespace
CONIDS = {'AAA': 1, 'BBB': 2, 'CCC': 3}
CLOSES = {'AAA': 350.0, 'BBB': 120.0, 'CCC': 80.0}


class Event:
    def __init__(self):
        self.fns = []

    def __iadd__(self, f):
        self.fns.append(f)
        return self

    def emit(self, *a):
        for f in self.fns:
            f(*a)


class Trade:
    def __init__(self, contract, order):
        self.contract, self.order = contract, order
        self.orderStatus = NS(status='Submitted', filled=0.0)
        self.log = []


class Broker:
    """IBKR as the day goes: AAA's opening auction at 351 (under its limit) buys 60 shares then 39;
    BBB's auction is above its limit (nothing bought; the order expires); CCC's good-till time is
    refused. At 09:00 AAA trades down through its stop."""

    def __init__(self):
        self._trades, self.fills, self.next_id = [], [], 100
        self.errorEvent, self.execDetailsEvent = Event(), Event()
        self.events = [((8, 0, 30), self.auction), ((8, 0, 40), self.auction_rest), ((8, 2, 0), self.gtd_expiry), ((9, 0, 0), self.fall)]

    # connection
    def connect(self, *a, **k):
        pass

    def isConnected(self):
        return True

    def disconnect(self):
        pass

    # the clock
    def sleep(self, s=0):
        CLOCK[0] += _dt.timedelta(seconds=s)
        while self.events and CLOCK[0] >= CLOCK[0].replace(hour=self.events[0][0][0], minute=self.events[0][0][1], second=self.events[0][0][2]):
            self.events.pop(0)[1]()

    # contracts and data
    def qualifyContracts(self, c):
        sym = c.symbol or next((s for s, i in CONIDS.items() if i == c.conId), '')
        if sym not in CONIDS:
            return [None]
        c.symbol, c.conId, c.primaryExchange, c.currency = sym, CONIDS[sym], 'LSE', 'GBP'
        c.exchange = c.exchange or 'LSE'
        return [c]

    def reqContractDetails(self, c):
        return [NS(minTick=0.5, marketRuleIds='', validExchanges='LSE')]

    def reqMarketRule(self, rid):
        return []

    def reqHistoricalData(self, c, *a, **k):
        return [NS(close=CLOSES[c.symbol], low=CLOSES[c.symbol] * 0.97)]

    # orders
    def placeOrder(self, contract, order):
        if not order.orderId:
            self.next_id += 1
            order.orderId = order.permId = self.next_id
        t = next((x for x in self._trades if x.order.orderId == order.orderId), None)
        if t is None:
            t = Trade(contract, order)
            self._trades.append(t)
        if contract.symbol == 'CCC' and order.tif == 'GTD':
            t.orderStatus.status = 'Cancelled'
            self.errorEvent.emit(order.orderId, 10052, 'Invalid time in force: GTD is not allowed for this order', contract)
        return t

    def cancelOrder(self, order):
        for t in self._trades:
            if t.order.orderId == order.orderId and t.orderStatus.status not in ('Filled', 'Cancelled'):
                t.orderStatus.status = 'Cancelled'

    def live(self):
        return [t for t in self._trades if t.orderStatus.status not in ('Filled', 'Cancelled', 'Inactive')]

    def reqAllOpenOrders(self):
        return self.live()

    def openTrades(self):
        return self.live()

    def trades(self):
        return list(self._trades)

    def reqCompletedOrders(self, *_):
        return []

    def reqExecutions(self):
        return list(self.fills)

    def reqPositions(self):
        return []

    # the market
    def fill(self, t, qty, price):
        t.orderStatus.filled += qty
        if t.orderStatus.filled >= t.order.totalQuantity:
            t.orderStatus.status = 'Filled'
        side = 'BOT' if t.order.action == 'BUY' else 'SLD'
        self.fills.append(NS(execution=NS(orderRef=t.order.orderRef, side=side, shares=qty, price=price, permId=t.order.permId), contract=t.contract))
        if t.order.ocaGroup:  # one cancels the other (reduced with block: ocaType 2)
            for o in self.live():
                if o is not t and o.order.ocaGroup == t.order.ocaGroup:
                    o.orderStatus.status = 'Cancelled'

    def buy(self, sym):
        return next(t for t in self.live() if t.contract.symbol == sym and t.order.action == 'BUY')

    def auction(self):
        t = self.buy('AAA')
        assert t.order.lmtPrice >= 351, t.order.lmtPrice
        self.fill(t, 60, 351.0)

    def auction_rest(self):
        self.fill(self.buy('AAA'), 39, 351.0)

    def gtd_expiry(self):  # IBKR ends a GTD order at its time
        for t in self.live():
            if t.order.tif == 'GTD':
                t.orderStatus.status = 'Cancelled'

    def fall(self):  # AAA trades down through its stop
        for t in self.live():
            if t.contract.symbol == 'AAA' and t.order.orderType == 'STP':
                self.fill(t, t.order.totalQuantity, t.order.auxPrice)


ib_async.__dict__.update(IB=Broker, LimitOrder=LimitOrder, StopOrder=StopOrder, StopLimitOrder=StopLimitOrder, MarketOrder=MarketOrder, Stock=Stock, Contract=Contract)
sys.modules['ib_async'] = ib_async

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import submit  # noqa: E402

SHARED = []  # one broker for every connection of the day: IBKR keeps the orders between them


def _broker(cls, *a, **k):
    if not SHARED:
        b = object.__new__(cls)
        Broker._setup(b)
        SHARED.append(b)
    return SHARED[0]


Broker._setup = Broker.__init__
Broker.__init__ = lambda self, *a, **k: None
Broker.__new__ = _broker


def run(argv):
    sys.argv = ['submit.py', *argv]
    out = io.StringIO()
    with redirect_stdout(out):
        try:
            submit.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                raise AssertionError(f'submit.py {argv} exited {e.code}:\n{out.getvalue()}')
    return out.getvalue()


def day_on_one_broker():
    """The placing, then the watch, on the same stand-in (as IBKR keeps the orders between connections)."""
    if True:
        d = tempfile.mkdtemp()
        book = {'model': 'uk', 'asOf': _dt.date.today().isoformat(), 'slots': 5, 'orders': [
            {'ticker': 'AAA', 'close': 350.0, 'entry': 'open', 'limit': 353.5, 'stop': 339.5, 'take': 364.0, 'stopRel': -0.03, 'takeRel': 0.04, 'capital': 350.0, 'maxCapital': 400.0},
            {'ticker': 'BBB', 'close': 120.0, 'entry': 'open', 'limit': 121.0, 'stop': 115.0, 'take': None, 'stopRel': -0.04, 'takeRel': None, 'capital': 300.0, 'maxCapital': 300.0},
            {'ticker': 'CCC', 'close': 80.0, 'entry': 'open', 'limit': 80.5, 'stop': 77.0, 'take': None, 'stopRel': -0.04, 'takeRel': None, 'capital': 200.0, 'maxCapital': 200.0},
        ]}
        path = os.path.join(d, 'UK.json')
        json.dump(book, open(path, 'w'))
        placed = run(['--market', 'UK', '--orders', path])
        T = SHARED[0]._trades
        assert not [t for t in T if t.order.action == 'SELL'], 'levels attached before any fill:\n' + placed
        a = next(t for t in T if t.contract.symbol == 'AAA' and t.order.action == 'BUY')
        # GBP 400 at most (maxCapital over the capital asked), at 3.535 a share: 113 shares; asked 350: 99
        assert a.order.tif == 'GTD' and a.order.lmtPrice == 353.5 and a.order.totalQuantity == 99, vars(a.order)
        until = _dt.datetime.strptime(a.order.goodTillDate, '%Y%m%d-%H:%M:%S').replace(tzinfo=_dt.timezone.utc).astimezone(LONDON)
        assert (until.hour, until.minute) == (8, submit.AUCTION_MINUTES), until
        c = [t for t in T if t.contract.symbol == 'CCC' and t.order.action == 'BUY']
        assert [x.order.tif for x in c] == ['GTD', 'DAY'], [vars(x.order) for x in c]
        watched = run(['--market', 'UK', '--orders', path, '--rebuy', '--until', '16:15', '--out', os.path.join(d, 'rebuy.txt')])
        sells = [t for t in T if t.order.action == 'SELL' and t.contract.symbol == 'AAA']
        assert 'bought 60 at 351' in watched and 'bought 99 at 351' in watched, watched
        # first set for the 60 bought, then for the 99 once the rest came (the first pair cancelled)
        groups = sorted({t.order.ocaGroup for t in sells})
        assert len(groups) == 2 and all(t.orderStatus.status == 'Cancelled' for t in sells if t.order.ocaGroup.endswith('-60')), (groups, watched)
        last = [t for t in sells if t.order.ocaGroup.endswith('-99')]
        stop = next(t for t in last if t.order.orderType == 'STP')
        take = next(t for t in last if t.order.orderType == 'LMT')
        # from the fill at 351 (not the close at 350): 351 x 0.97 = 340.47 -> 340.0; 351 x 1.04 = 365.04 -> 365.5
        assert stop.order.auxPrice == 340.0 and take.order.lmtPrice == 365.5, (stop.order.auxPrice, take.order.lmtPrice)
        assert stop.order.totalQuantity == 99 and take.order.totalQuantity == 99 and stop.order.ocaType == 2, vars(stop.order)
        assert stop.orderStatus.status == 'Filled' and take.orderStatus.status == 'Cancelled', (stop.orderStatus.status, take.orderStatus.status)
        assert not [t for t in T if t.order.action == 'SELL' and t.contract.symbol in ('BBB', 'CCC')], 'levels for a share never bought'
        assert all(t.orderStatus.status == 'Cancelled' for t in c), 'the day order for the open outlived the auction'
        live_sells = [t for t in sells if t.orderStatus.status not in ('Filled', 'Cancelled')]
        assert not live_sells, 'levels left after the stop sold'
        return placed, watched


if __name__ == '__main__':
    placed, watched = day_on_one_broker()
    print(placed + watched)
    print('levels: orders for the open placed for the auction only; stop and take-profit from the fill, one cancelling the other; OK')
