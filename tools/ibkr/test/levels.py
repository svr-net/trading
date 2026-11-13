"""The UK model's orders through tools/ibkr/submit.py against a stand-in IBKR (no gateway, no network):
a day of orders for the open, from placing them to the watch's levels.

  python3 tools/ibkr/test/levels.py

Checked: each order for the open ("entry": "open") is one limit buy good only until the opening
auction is over (GTD), with no levels attached; the watch places its stop-loss and take-profit from
the price it was bought at (not the last close), one cancelling the other, for the shares bought,
placed again only when more were bought, never twice; a buy whose good-till time IBKR refuses goes
again as a day order and is cancelled once the auction is over; a stop that sells cancels the
take-profit; a share never bought gets no levels; a trailing stop goes to IBKR as its own trailing
order (TRAIL, its percent and starting level from the fill), a fixed stop where IBKR refuses it; a
break-even stop stays put until the share is up its trigger since the fill, then moves to the price
paid (and sells there when the price falls back).
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
    def __init__(self, action='', qty=0, orderType='', **kw):
        self.action, self.totalQuantity, self.orderType = action, qty, orderType
        self.lmtPrice = self.auxPrice = self.trailingPercent = self.trailStopPrice = 0.0
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
CONIDS = {'AAA': 1, 'BBB': 2, 'CCC': 3, 'DDD': 4, 'EEE': 5, 'XXX': 6, 'MC': 7, 'BRK B': 8, 'BHP': 9, 'D05': 10, 'SPY': 11}
CLOSES = {'AAA': 350.0, 'BBB': 120.0, 'CCC': 80.0, 'DDD': 100.0, 'EEE': 50.0, 'XXX': 100.0, 'MC': 500.0, 'BRK B': 480.0, 'BHP': 45.0, 'D05': 50.0, 'SPY': 600.0}


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
        self.quotes = {}
        self.seen = {}  # what the test checks along the day
        self.events = [((8, 0, 30), self.auction), ((8, 0, 40), self.auction_rest), ((8, 2, 0), self.gtd_expiry), ((9, 0, 0), self.fall),
                       ((9, 29, 0), self.before_rise), ((9, 30, 0), self.rise), ((11, 0, 0), self.drop)]

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

    def reqMarketDataType(self, *_):
        pass

    def reqMktData(self, c, *a):
        return self.quotes.setdefault(c.symbol, NS(contract=c, high=float('nan'), last=float('nan'), bid=float('nan'), close=CLOSES[c.symbol]))

    def cancelMktData(self, c):
        pass

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
        if contract.symbol == 'EEE' and order.orderType == 'TRAIL':  # a share where IBKR takes no trailing stop
            t.orderStatus.status = 'Cancelled'
            self.errorEvent.emit(order.orderId, 387, 'Unsupported order type for this exchange and security type', contract)
        if order.orderType in ('LOC', 'MOC') and contract.symbol in getattr(self, 'no_close_orders', ()):  # LSE through the API (13 Nov)
            t.orderStatus.status = 'Cancelled'
            self.errorEvent.emit(order.orderId, 387, 'Unsupported order type for this exchange and security type.', contract)
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

    def run(self, aw):
        import asyncio
        return asyncio.new_event_loop().run_until_complete(aw)

    def reqCompletedOrdersAsync(self, *_):
        async def done():
            return []
        return done()

    def reqCompletedOrders(self, *_):
        return []

    def reqExecutions(self):
        return list(self.fills)

    def reqPositions(self):
        held = {}
        for f in self.fills:
            c = f.contract
            q = held.get(c.conId, (c, 0.0))[1] + (f.execution.shares if f.execution.side == 'BOT' else -f.execution.shares)
            held[c.conId] = (c, q)
        return [NS(contract=c, position=q) for c, q in held.values() if q > 0]


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
        d = self.buy('DDD')
        self.fill(d, d.order.totalQuantity, 100.5)
        e = self.buy('EEE')
        self.fill(e, e.order.totalQuantity, 50.0)

    def auction_rest(self):
        self.fill(self.buy('AAA'), 39, 351.0)

    def gtd_expiry(self):  # IBKR ends a GTD order at its time
        for t in self.live():
            if t.order.tif == 'GTD':
                t.orderStatus.status = 'Cancelled'

    def fall(self):  # AAA trades down through its trailing stop
        for t in self.live():
            if t.contract.symbol == 'AAA' and t.order.orderType in ('STP', 'TRAIL'):
                self.fill(t, t.order.totalQuantity, t.order.trailStopPrice if t.order.orderType == 'TRAIL' else t.order.auxPrice)

    def stop_of(self, sym):
        return next(t for t in self._trades if t.contract.symbol == sym and t.order.action == 'SELL' and t.order.orderType == 'STP')

    def before_rise(self):  # DDD only up 1%: its stop not moved yet
        q = self.reqMktData(self.stop_of('DDD').contract)
        q.high, q.last, q.bid = 101.5, 101.4, 101.3
        self.seen['before'] = self.stop_of('DDD').order.auxPrice

    def rise(self):  # DDD up 2.5% since bought at 100.5: past its break-even trigger (2%)
        q = self.quotes['DDD']
        q.high, q.last, q.bid = 103.0, 102.8, 102.7
        self.seen['rise'] = self.stop_of('DDD').order.auxPrice

    def drop(self):  # DDD falls back through the entry: the moved stop sells it there
        self.seen['before-drop'] = self.stop_of('DDD').order.auxPrice
        q = self.quotes['DDD']
        q.last, q.bid = 100.0, 99.5
        t = self.stop_of('DDD')
        if t.orderStatus.status not in ('Filled', 'Cancelled') and t.order.auxPrice >= 100.0:
            self.fill(t, t.order.totalQuantity, t.order.auxPrice)


ib_async.__dict__.update(IB=Broker, Order=Order, LimitOrder=LimitOrder, StopOrder=StopOrder, StopLimitOrder=StopLimitOrder, MarketOrder=MarketOrder, Stock=Stock, Contract=Contract)
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
            {'ticker': 'AAA', 'close': 350.0, 'entry': 'open', 'limit': 353.5, 'stop': 339.5, 'take': 364.0, 'stopRel': -0.03, 'takeRel': 0.04, 'stopMode': 'trail', 'trailRel': 0.03,
             'capital': 350.0, 'maxCapital': 400.0},
            {'ticker': 'EEE', 'close': 50.0, 'entry': 'open', 'limit': 50.5, 'stop': 48.0, 'take': None, 'stopRel': -0.04, 'takeRel': None, 'stopMode': 'trail', 'trailRel': 0.04,
             'capital': 200.0, 'maxCapital': 200.0},
            {'ticker': 'DDD', 'close': 100.0, 'entry': 'open', 'limit': 101.0, 'stop': 96.0, 'take': None, 'stopRel': -0.04, 'takeRel': None, 'stopMode': 'breakeven', 'breakevenRel': 0.02,
             'capital': 300.0, 'maxCapital': 300.0},
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
        stop = next(t for t in last if t.order.orderType == 'TRAIL')
        take = next(t for t in last if t.order.orderType == 'LMT')
        # IBKR's trailing stop, 3% under the high, from the fill at 351 (not the close at 350): 351 x 0.97 = 340.47 -> 340.0;
        # the take-profit 351 x 1.04 = 365.04 -> 365.5
        assert stop.order.trailingPercent == 3.0 and stop.order.trailStopPrice == 340.0 and take.order.lmtPrice == 365.5, (vars(stop.order), take.order.lmtPrice)
        assert stop.order.totalQuantity == 99 and take.order.totalQuantity == 99 and stop.order.ocaType == 2, vars(stop.order)
        assert stop.orderStatus.status == 'Filled' and take.orderStatus.status == 'Cancelled', (stop.orderStatus.status, take.orderStatus.status)
        assert not [t for t in T if t.order.action == 'SELL' and t.contract.symbol in ('BBB', 'CCC')], 'levels for a share never bought'
        # DDD: a fixed stop from the fill at 100.5 (100.5 x 0.96 = 96.48 -> 96.0), left there while up only 1%, moved to the
        # entry (100.5) once up 2.5% (past its 2%), and sold there when the price fell back; no take-profit
        b = SHARED[0]
        dstops = [t for t in T if t.contract.symbol == 'DDD' and t.order.action == 'SELL']
        assert len(dstops) == 1 and dstops[0].order.orderType == 'STP', [vars(t.order) for t in dstops]
        assert b.seen == {'before': 96.0, 'rise': 96.0, 'before-drop': 100.5}, b.seen
        assert dstops[0].orderStatus.status == 'Filled' and 'stop moved to 100.5 (break-even)' in watched, watched
        # EEE: IBKR refused its trailing stop: a fixed stop from the fill (50 x 0.96 = 48.0) instead, placed once
        estops = [t for t in T if t.contract.symbol == 'EEE' and t.order.action == 'SELL']
        assert [t.order.orderType for t in estops] == ['TRAIL', 'STP'] and estops[1].order.auxPrice == 48.0, [vars(t.order) for t in estops]
        assert estops[1].orderStatus.status not in ('Cancelled',) and 'refused the trailing stop' in watched, watched
        assert all(t.orderStatus.status == 'Cancelled' for t in c), 'the day order for the open outlived the auction'
        live_sells = [t for t in sells if t.orderStatus.status not in ('Filled', 'Cancelled')]
        assert not live_sells, 'levels left after the stop sold'
        return placed, watched


def hold_book():
    """The UK model's multi-day book (apps/ofm_uk --hold: "book": "hold") through submit.py --holds: the
    buy good only through the opening auction (GTD), the position offered at each close only at or above
    its exit level (LOC at the entry times 1 + exitAbove, on the price step), and at its last day's close
    sold at any price (MOC)."""
    SHARED.clear()
    CLOCK[0] = _dt.datetime.combine(_dt.date.today(), _dt.time(7, 50), LONDON)
    d = tempfile.mkdtemp()
    book = {'model': 'uk', 'book': 'hold', 'days': 5, 'slots': 5, 'asOf': _dt.date.today().isoformat(), 'orders': [
        {'ticker': 'XXX', 'close': 100.0, 'entry': 'open', 'open': True, 'limit': 101.0, 'stop': None, 'take': None, 'exitAbove': 0.012, 'days': 5,
         'capital': 1000.0, 'maxCapital': 1000.0}]}
    path, state = os.path.join(d, 'UK.json'), os.path.join(d, 'holds-UK.json')
    json.dump(book, open(path, 'w'))
    _broker(Broker).events = []  # this day's events are the test's own
    placed = run(['--market', 'UK', '--holds', 'place', '--orders', path, '--hold-state', state])
    b = SHARED[0]
    buy = next(t for t in b._trades if t.contract.symbol == 'XXX' and t.order.action == 'BUY')
    # GBP 1,000 at 1.01 a share (pence): 990 shares, at most 101, the opening auction only
    assert buy.order.orderType == 'LMT' and buy.order.tif == 'GTD' and buy.order.lmtPrice == 101.0 and buy.order.totalQuantity == 990, (vars(buy.order), placed)
    b.fill(buy, 990, 100.0)  # the auction at 100
    CLOCK[0] = CLOCK[0].replace(hour=16, minute=30)
    closed = run(['--market', 'UK', '--holds', 'close', '--hold-state', state])
    sells = [t for t in b._trades if t.contract.symbol == 'XXX' and t.order.action == 'SELL']
    assert sells, closed
    sell = sells[0]
    # its first close: only at or above 100 x 1.012 = 101.2, on the 0.5 price step: 101.5
    assert sell.order.orderType == 'LOC' and sell.order.lmtPrice == 101.5 and sell.order.totalQuantity == 990, (vars(sell.order), closed)
    st = json.load(open(state))
    pos = st['positions']['XXX']
    assert pos['entry'] == 100.0 and pos['exitAbove'] == 0.012 and pos['days'] == 5 and pos['closes'] == 1, pos
    # its fifth close (four closes later, not reached its level): sold at any price
    sell.orderStatus.status = 'Cancelled'  # the LOC did not trade (the close under 101.5)
    pos['closes'], pos['offered'] = 4, '2000-01-01'
    json.dump(st, open(state, 'w'))
    last = run(['--market', 'UK', '--holds', 'close', '--hold-state', state])
    moc = [t for t in b._trades if t.contract.symbol == 'XXX' and t.order.action == 'SELL' and t.order.orderType == 'MOC']
    assert len(moc) == 1 and moc[0].order.totalQuantity == 990 and 'its last day (5)' in last, (last, [vars(t.order) for t in b._trades if t.order.action == 'SELL'])
    # The daily update (the next morning's orders file): kept, its exit level set again; then, a morning
    # later, no longer wanted: sold at the open, and not offered again at that close.
    moc[0].orderStatus.status = 'Cancelled'
    st = json.load(open(state))
    st['positions']['XXX'].update(closes=2, offered='2000-01-01')
    json.dump(st, open(state, 'w'))
    book['orders'] = []
    book['updates'] = [{'ticker': 'XXX', 'keep': True, 'exitAbove': 0.02, 'expected': 0.004}]
    json.dump(book, open(path, 'w'))
    kept = run(['--market', 'UK', '--holds', 'place', '--orders', path, '--hold-state', state])
    assert json.load(open(state))['positions']['XXX']['exitAbove'] == 0.02 and '+1.20% -> +2.00%' in kept, kept
    book['updates'] = [{'ticker': 'XXX', 'keep': False, 'exitAbove': 0.02, 'expected': -0.001}]
    json.dump(book, open(path, 'w'))
    dropped = run(['--market', 'UK', '--holds', 'place', '--orders', path, '--hold-state', state])
    out = [t for t in b._trades if t.contract.symbol == 'XXX' and t.order.action == 'SELL' and t.order.orderRef.endswith('-exit')]
    assert len(out) == 1 and out[0].order.orderType == 'MKT' and out[0].order.totalQuantity == 990 and 'no longer holds it' in dropped, (dropped, [vars(t.order) for t in out])
    again = run(['--market', 'UK', '--holds', 'place', '--orders', path, '--hold-state', state])
    assert 'already sold at the open today' in again and len([t for t in b._trades if t.order.orderRef.endswith('-exit')]) == 1, again
    n = len([t for t in b._trades if t.order.action == 'SELL'])
    closed2 = run(['--market', 'UK', '--holds', 'close', '--hold-state', state])
    assert len([t for t in b._trades if t.order.action == 'SELL']) == n and 'being sold since the open' in closed2, closed2
    return placed + closed + last + kept + dropped + closed2


def eu_book():
    """Europe's hold book (apps/ofm_uk --costs EU --hold 5): a ticker EXCHANGE:CODE, priced in euros (not
    pence), its buy good only through Paris's opening auction (09:00 Paris, GTD two minutes later)."""
    SHARED.clear()
    CLOCK[0] = _dt.datetime.combine(_dt.date.today(), _dt.time(8, 50), ZoneInfo('Europe/Paris'))
    d = tempfile.mkdtemp()
    book = {'model': 'uk', 'book': 'hold', 'days': 5, 'slots': 5, 'asOf': _dt.date.today().isoformat(), 'orders': [
        {'ticker': 'PAR:MC', 'close': 500.0, 'entry': 'open', 'open': True, 'limit': 503.0, 'stop': None, 'take': None, 'exitAbove': 0.01, 'days': 5,
         'capital': 6000.0, 'maxCapital': 8000.0}]}
    path, state = os.path.join(d, 'EU.json'), os.path.join(d, 'holds-EU.json')
    json.dump(book, open(path, 'w'))
    _broker(Broker).events = []
    placed = run(['--market', 'EU', '--holds', 'place', '--orders', path, '--hold-state', state])
    buy = next(t for t in SHARED[0]._trades if t.contract.symbol == 'MC' and t.order.action == 'BUY')
    until = _dt.datetime.strptime(buy.order.goodTillDate, '%Y%m%d-%H:%M:%S').replace(tzinfo=_dt.timezone.utc).astimezone(ZoneInfo('Europe/Paris'))
    # EUR 6,000 at EUR 503 a share: 11 shares (euros, not pence); on Paris's exchange (SBF); until 09:02 Paris
    assert buy.order.tif == 'GTD' and buy.order.lmtPrice == 503.0 and buy.order.totalQuantity == 11 and buy.contract.exchange == 'SBF', (vars(buy.order), placed)
    assert (until.hour, until.minute) == (9, 2), until
    return placed


def us_book():
    """The US hold book (apps/ofm_uk --costs US --hold 5): EODData's BRK.B is IBKR's "BRK B", routed SMART,
    priced in dollars, its buy good only through the opening auction (09:30 New York, GTD two minutes later)."""
    SHARED.clear()
    CLOCK[0] = _dt.datetime.combine(_dt.date.today(), _dt.time(9, 20), ZoneInfo('America/New_York'))
    d = tempfile.mkdtemp()
    book = {'model': 'uk', 'book': 'hold', 'days': 5, 'slots': 5, 'asOf': _dt.date.today().isoformat(), 'orders': [
        {'ticker': 'BRK.B', 'close': 480.0, 'entry': 'open', 'open': True, 'limit': 484.0, 'stop': None, 'take': None, 'exitAbove': 0.004, 'days': 5,
         'capital': 10000.0, 'maxCapital': 20000.0}]}
    path, state = os.path.join(d, 'US.json'), os.path.join(d, 'holds-US.json')
    json.dump(book, open(path, 'w'))
    _broker(Broker).events = []
    placed = run(['--market', 'US', '--holds', 'place', '--orders', path, '--hold-state', state])
    buy = next(t for t in SHARED[0]._trades if t.contract.symbol == 'BRK B' and t.order.action == 'BUY')
    until = _dt.datetime.strptime(buy.order.goodTillDate, '%Y%m%d-%H:%M:%S').replace(tzinfo=_dt.timezone.utc).astimezone(ZoneInfo('America/New_York'))
    # USD 10,000 at USD 484 a share: 20 shares; routed SMART; until 09:32 New York
    assert buy.order.tif == 'GTD' and buy.order.lmtPrice == 484.0 and buy.order.totalQuantity == 20 and buy.contract.exchange == 'SMART', (vars(buy.order), placed)
    assert (until.hour, until.minute) == (9, 32), until
    return placed


def apac_book():
    """Asia-Pacific's hold book: each order good only through its own exchange's opening auction: the ASX's
    (alphabetical groups from 10:00 Sydney: until 10:12), the SGX's (09:00 Singapore: until 09:02)."""
    SHARED.clear()
    CLOCK[0] = _dt.datetime.combine(_dt.date.today(), _dt.time(9, 45), ZoneInfo('Australia/Sydney'))
    d = tempfile.mkdtemp()
    book = {'model': 'uk', 'book': 'hold', 'days': 5, 'slots': 5, 'asOf': _dt.date.today().isoformat(), 'orders': [
        {'ticker': 'ASX:BHP', 'close': 45.0, 'entry': 'open', 'open': True, 'limit': 45.5, 'stop': None, 'take': None, 'exitAbove': 0.01, 'days': 5, 'capital': 5000.0, 'maxCapital': 9000.0},
        {'ticker': 'SGX:D05', 'close': 50.0, 'entry': 'open', 'open': True, 'limit': 50.4, 'stop': None, 'take': None, 'exitAbove': 0.01, 'days': 5, 'capital': 5000.0, 'maxCapital': 9000.0}]}
    path, state = os.path.join(d, 'APAC.json'), os.path.join(d, 'holds-APAC.json')
    json.dump(book, open(path, 'w'))
    _broker(Broker).events = []
    placed = run(['--market', 'APAC', '--holds', 'place', '--orders', path, '--hold-state', state])
    ends = {}
    for sym, tz in (('BHP', 'Australia/Sydney'), ('D05', 'Asia/Singapore')):
        buy = next(t for t in SHARED[0]._trades if t.contract.symbol == sym and t.order.action == 'BUY')
        u = _dt.datetime.strptime(buy.order.goodTillDate, '%Y%m%d-%H:%M:%S').replace(tzinfo=_dt.timezone.utc).astimezone(ZoneInfo(tz))
        ends[sym] = (u.hour, u.minute)
        assert buy.order.tif == 'GTD', vars(buy.order)
    assert ends == {'BHP': (10, 12), 'D05': (9, 2)}, (ends, placed)
    return placed


def index_book():
    """The INDEX market's hold book: a UCITS index ETF on Xetra (FRA:SPY here: the stand-in's share) on IBIS in
    euros, good only through Frankfurt's opening auction, on the INDEX market's own IBKR client (57)."""
    SHARED.clear()
    CLOCK[0] = _dt.datetime.combine(_dt.date.today(), _dt.time(8, 46), ZoneInfo('Europe/Berlin'))
    d = tempfile.mkdtemp()
    book = {'model': 'uk', 'book': 'hold', 'days': 5, 'slots': 5, 'asOf': _dt.date.today().isoformat(), 'orders': [
        {'ticker': 'FRA:SPY', 'close': 600.0, 'entry': 'open', 'open': True, 'limit': 603.0, 'stop': None, 'take': None, 'exitAbove': 0.003, 'days': 5,
         'capital': 7000.0, 'maxCapital': 1e9}]}
    path, state = os.path.join(d, 'INDEX.json'), os.path.join(d, 'holds-INDEX.json')
    json.dump(book, open(path, 'w'))
    b = _broker(Broker)
    b.events = []
    ids = []
    real = b.connect
    b.connect = lambda *a, **k: (ids.append(k.get('clientId')), real(*a, **k))[1]
    placed = run(['--market', 'INDEX', '--holds', 'place', '--orders', path, '--hold-state', state])
    del b.connect
    buy = next(t for t in SHARED[0]._trades if t.contract.symbol == 'SPY' and t.order.action == 'BUY')
    until = _dt.datetime.strptime(buy.order.goodTillDate, '%Y%m%d-%H:%M:%S').replace(tzinfo=_dt.timezone.utc).astimezone(ZoneInfo('Europe/Berlin'))
    # EUR 7,000 at EUR 603: 11 shares; IBIS; until 09:02 Frankfurt; the INDEX client (17 + 40 + 7 for the hold book)
    assert buy.contract.exchange == 'IBIS' and buy.order.totalQuantity == 11 and (until.hour, until.minute) == (9, 2), (vars(buy.order), placed)
    assert ids and ids[0] == 17 + 40 + 7, ids
    return placed


def legacy_positions():
    """Positions of an earlier book in the state (no exit plan of their own: no 'days', as the UK's old
    hold-up book's) do not take the live book's slots: with 2 of them held and a book of 2 slots, both of
    the day's orders are placed (12 Nov: 6 old positions left the UK's new book no slot)."""
    SHARED.clear()
    CLOCK[0] = _dt.datetime.combine(_dt.date.today(), _dt.time(7, 45), LONDON)
    d = tempfile.mkdtemp()
    b = _broker(Broker)
    b.events = []
    for sym, cid in (('AAA', 1), ('BBB', 2)):  # held at IBKR from the old book
        b.fills.append(NS(execution=NS(orderRef='ofm-UKH-2026-11-05', side='BOT', shares=10, price=100.0, permId=0), contract=Contract(sym, 'LSE', 'GBP', cid, 'LSE')))
    state = os.path.join(d, 'holds-UK.json')
    json.dump({'positions': {'AAA': {'conId': 1, 'qty': 10, 'entry': 100.0, 'since': '2026-11-05', 'closes': 2},
                             'BBB': {'conId': 2, 'qty': 10, 'entry': 100.0, 'since': '2026-11-05', 'closes': 2}}, 'pending': {}}, open(state, 'w'))
    book = {'model': 'uk', 'book': 'hold', 'days': 5, 'slots': 2, 'asOf': _dt.date.today().isoformat(), 'orders': [
        {'ticker': t, 'close': 100.0, 'entry': 'open', 'open': True, 'limit': 101.0, 'stop': None, 'take': None, 'exitAbove': 0.01, 'days': 5,
         'capital': 1000.0, 'maxCapital': 1000.0} for t in ('XXX', 'DDD')]}
    path = os.path.join(d, 'UK.json')
    json.dump(book, open(path, 'w'))
    placed = run(['--market', 'UK', '--holds', 'place', '--orders', path, '--hold-state', state])
    buys = sorted(t.contract.symbol for t in b._trades if t.order.action == 'BUY')
    assert buys == ['DDD', 'XXX'] and '2 positions of an earlier book held' in placed, (buys, placed)
    return placed


def close_orders_refused():
    """An exchange that takes no LOC (387: the LSE through the API, 13 Nov): the position is offered with a
    limit at the same exit level for the rest of the day instead (it rests into the closing auction), and
    on its last day a refused MOC becomes a limit at the 20-day low (sell_out)."""
    SHARED.clear()
    CLOCK[0] = _dt.datetime.combine(_dt.date.today(), _dt.time(16, 20), LONDON)
    d = tempfile.mkdtemp()
    b = _broker(Broker)
    b.events, b.no_close_orders = [], {'XXX'}
    today = _dt.date.today().isoformat()
    b.fills.append(NS(execution=NS(orderRef=f'ofm-UKH-{today}', side='BOT', shares=990, price=100.0, permId=0), contract=Contract('XXX', 'LSE', 'GBP', 6, 'LSE')))
    state = os.path.join(d, 'holds-UK.json')
    json.dump({'positions': {}, 'pending': {'XXX': {'conId': 6, 'qty': 990, 'exitAbove': 0.012, 'days': 5}}}, open(state, 'w'))
    closed = run(['--market', 'UK', '--holds', 'close', '--hold-state', state])
    sells = [t for t in b._trades if t.order.action == 'SELL']
    assert [t.order.orderType for t in sells] == ['LOC', 'LMT'] and sells[1].order.lmtPrice == 101.5 and sells[1].order.tif == 'DAY', ([vars(t.order) for t in sells], closed)
    assert sells[1].orderStatus.status == 'Submitted' and 'for the rest of the day instead: live' in closed, closed
    st = json.load(open(state))
    sells[1].orderStatus.status = 'Cancelled'
    st['positions']['XXX'].update(closes=4, offered='2000-01-01')
    json.dump(st, open(state, 'w'))
    last = run(['--market', 'UK', '--holds', 'close', '--hold-state', state])
    sells = [t for t in b._trades if t.order.action == 'SELL']
    assert [t.order.orderType for t in sells[2:]] == ['MOC', 'LMT'] and 'limit at the 20-day low instead: live' in last, ([vars(t.order) for t in sells], last)
    return closed + last


def leftovers_keep_book():
    """The leftovers step sells only the daily book's shares from earlier days: never one the multi-day book
    bought (orderRef ofm-<MARKET>H-), also before the book's close has recorded it (13 Nov: the EU close sold
    WKL, bought by the book that morning, as the old daily book had traded it within its 10 days)."""
    SHARED.clear()
    CLOCK[0] = _dt.datetime.combine(_dt.date.today(), _dt.time(16, 21), LONDON)
    b = _broker(Broker)
    b.events = []
    today = _dt.date.today().isoformat()
    b.fills.append(NS(execution=NS(orderRef=f'ofm-UKH-{today}', side='BOT', shares=4, price=100.0, permId=0), contract=Contract('XXX', 'LSE', 'GBP', 6, 'LSE')))
    b.fills.append(NS(execution=NS(orderRef='ofm-UK-2026-11-05', side='BOT', shares=10, price=100.0, permId=0), contract=Contract('AAA', 'LSE', 'GBP', 1, 'LSE')))
    try:
        run(['--market', 'UK', '--leftovers', '--check', '--tickers', 'XXX,AAA'])
        checked = ''
    except AssertionError as e:  # exit 1: one of the model's shares held
        checked = str(e)
    assert 'exited 1' in checked and 'AAA: 10 shares still held' in checked and 'XXX' not in checked.split('\n', 1)[1], checked
    out = run(['--market', 'UK', '--leftovers', '--tickers', 'XXX,AAA'])
    sold = sorted(t.contract.symbol for t in b._trades if t.order.action == 'SELL')
    assert sold == ['AAA'], (sold, out)
    return out


if __name__ == '__main__':
    placed, watched = day_on_one_broker()
    print(placed + watched)
    print('levels: orders for the open placed for the auction only; stop and take-profit from the fill, one cancelling the other; '
          'a trailing stop at IBKR (a fixed one where refused); a break-even stop moved by the watch; OK')
    print(hold_book())
    print('hold book: bought in the opening auction only; offered at each close at or above its exit level; sold at any price on its last day; '
          'updated every morning (kept with its new exit level, or sold at the open); OK')
    print(eu_book())
    print('Europe: an EXCHANGE:CODE order in euros, good only through Paris\'s opening auction; OK')
    print(us_book())
    print('US: BRK.B as BRK B, routed SMART, in dollars, good only through New York\'s opening auction; OK')
    print(apac_book())
    print('Asia-Pacific: each order good only through its own exchange\'s opening auction (ASX until 10:12 Sydney, SGX until 09:02 Singapore); OK')
    print(index_book())
    print('INDEX: a Xetra ETF on IBIS in euros, its own client, good only through Frankfurt\'s opening auction; OK')
    print(legacy_positions())
    print('hold book: an earlier book\'s positions do not take the live book\'s slots; OK')
    print(close_orders_refused())
    print('hold book: where the exchange takes no LOC or MOC, a day limit at the exit level (the last day: at the 20-day low); OK')
    print(leftovers_keep_book())
    print('leftovers: never the multi-day book\'s shares; OK')
