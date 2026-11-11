"""Places the model's next-open orders with Interactive Brokers (IB Gateway or TWS), and exits at the close.

The daily job writes each market's orders to the private store (svr-net/eod, orders/<MARKET>.json:
the last close, the buy limit, the stop-loss and the take-profit of every trade, in the exchange's
prices). This script runs next to a logged-in IB Gateway:

  before the open:  python tools/ibkr/submit.py --market UK --orders UK.json --capital 1000
                    each trade: a day limit buy at the limit (it takes part in the opening auction,
                    then rests during the day, as in the backtest), with its take-profit (limit
                    sell) and stop-loss (stop sell) attached, one cancelling the other;
  before the close: python tools/ibkr/submit.py --market UK --close
                    cancels today's unfilled orders and sells what is left at the close (MOC);
  the auction:      python tools/ibkr/submit.py --market UK --orders UK.json --auction
                    records (places nothing) the opening auction of the file's candidates: every 15
                    seconds until the open, the indicative auction price, volume and imbalance and
                    the bid and ask; after it, the open; one JSON line on standard output;
  IBKR's bars:      python tools/ibkr/submit.py --market UK --orders UK.json --bars --out ibkr-UK.csv
                    the pool's daily bars over --years (2) from IBKR, paced at one request every 10.5
                    seconds (IBKR allows 60 in 10 minutes), as date,ticker,open,high,low,close,volume.

  the day's paths:  python tools/ibkr/submit.py --market UK --orders UK.json --paths --out paths.csv
                    records (places nothing) the candidates' prices through the trading day, every 30
                    seconds from the open to 20 minutes after the close, from IBKR's free delayed data
                    (15 minutes late; real time where subscribed), as time,ticker,last,bid,ask,type: the
                    intraday paths the research needs to test a buy-back after a stop exactly.

  the buy-back:     python tools/ibkr/submit.py --market UK --orders UK.json --rebuy --until 16:20 --out state.txt
                    watches the day's fills until --until (the exchange's clock): when one of the
                    model's stops has sold, places the order's buy-back (its "rebuy" fraction of the way
                    from the stop back up to the price paid: a buy-stop there for the shares sold) with
                    its sale at the price paid and a round trip's costs (a limit sell attached), as the
                    backtest trades it; whatever is left is sold at the close as before (--close).

--capital is the amount per trade (the account's currency); --port 4002 is IB Gateway's paper
account (4001 live, 7497/7496 TWS). --dry-run prints the orders without connecting. Every order
carries orderRef ofm-<market>-<date>, so --close touches only this model's orders.

Requires: pip install ib_async
"""
import argparse
import os
import json
import math
import re
import sys
from datetime import date


# Each market's exchange and currency at IBKR (US shares routed SMART); in markets of several
# exchanges (EU, APAC) a ticker is EXCHANGE:CODE, EODData's exchange, and takes that exchange's.
VENUE = {'UK': ('LSE', 'GBP'), 'US': ('SMART', 'USD'), 'APAC': ('ASX', 'AUD'), 'EU': ('SBF', 'EUR')}
EXCHANGE = {'PAR': ('SBF', 'EUR'), 'AMS': ('AEB', 'EUR'), 'BRU': ('ENEXT.BE', 'EUR'), 'LIS': ('BVL', 'EUR'), 'MSE': ('BM', 'EUR'),
            'FRA': ('IBIS', 'EUR'), 'OSL': ('OSE', 'NOK'), 'ASX': ('ASX', 'AUD'), 'SGX': ('SGX', 'SGD')}


def qualified(ib, contract):
    """Whether IBKR knows the share (and the contract now carries its conId). ib_async returns a list
    with None for an unknown contract, so the list itself is not the answer: an unknown share used
    later (as a key, or for market data) stopped the whole run (3 Nov: OSL:JAREN)."""
    try:
        r = ib.qualifyContracts(contract)
    except Exception:  # noqa: BLE001
        return False
    return bool(r) and r[0] is not None and bool(getattr(contract, 'conId', 0))


def contract_of(ticker, market):
    from ib_async import Stock
    if ':' in ticker:
        ex, code = ticker.split(':', 1)
        return Stock(code, *EXCHANGE[ex])
    return Stock(ib_symbol(ticker, market), *VENUE[market])


def reference_close(ib, contract):
    """IBKR's last close for the share: its daily bars (the exchange's data subscription), else its
    delayed quote (free; the close, else the last); None when neither answers."""
    try:
        bars = ib.reqHistoricalData(contract, '', '2 D', '1 day', 'TRADES', useRTH=True, timeout=20)
        if bars and bars[-1].close > 0:
            return float(bars[-1].close)
    except Exception:  # noqa: BLE001
        pass
    ib.reqMarketDataType(4)  # delayed, frozen when the market is shut
    k = ib.reqMktData(contract, '', False, False)
    for _ in range(16):
        ib.sleep(0.5)
        for x in (k.close, k.last):
            if isinstance(x, (int, float)) and math.isfinite(x) and x > 0:
                ib.cancelMktData(contract)
                return float(x)
    ib.cancelMktData(contract)
    return None


def tick_rule(ib, details, exchange):
    """The share's price increments by price band (IBKR's market rule for its exchange; the LSE's
    depend on the price), as [(low edge, increment)], or None."""
    ids = [x for x in (details.marketRuleIds or '').split(',') if x]
    exs = (details.validExchanges or '').split(',')
    rid = ids[exs.index(exchange)] if exchange in exs and len(ids) == len(exs) else (ids[0] if ids else None)
    try:
        rules = ib.reqMarketRule(int(rid)) if rid else None
    except Exception:  # noqa: BLE001
        rules = None
    return sorted(((r.lowEdge, r.increment) for r in rules), key=lambda r: r[0]) if rules else None


def tick_at(rule, min_tick, price):
    inc = min_tick
    for low, step in rule or ():
        if price >= low - 1e-12:
            inc = step
    return inc or 0.01


def capital_of(o, args):
    # The order's own amount (the model's, in its exchange's currency), else --capital; never more than
    # its maxCapital (apps/ofm_uk: a share of what the share trades in a day).
    c = float(o.get('capital') or args.capital)
    return min(c, float(o['maxCapital'])) if o.get('maxCapital') else c


AUCTION_MINUTES = 2  # an order for the open ("entry": "open") works until this long after the open: the opening auction only


# Exchanges whose opening auction is not the market's open: the end of an order-for-the-open there (zone,
# hour, minute). The ASX opens in five alphabetical groups from 10:00 to about 10:09 Sydney time (an order
# good until 10:12 meets every group's auction); the SGX at 09:00 Singapore time, three hours after it.
AUCTION_END = {'ASX': ('Australia/Sydney', 10, 12), 'SGX': ('Asia/Singapore', 9, 2)}


def auction_end_utc(market, ticker=''):
    """The end of an order for the open (IBKR's good-till time, UTC, yyyymmdd-hh:mm:ss): a few minutes
    after its exchange's open (ticker EXCHANGE:CODE; else the market's), so it buys in the opening auction
    or not at all (never on a dip later)."""
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo
    ex = ticker.split(':', 1)[0] if ':' in ticker else ''
    if ex in AUCTION_END:
        tz_name, h, m = AUCTION_END[ex]
        return datetime.now(ZoneInfo(tz_name)).replace(hour=h, minute=m, second=0, microsecond=0).astimezone(timezone.utc).strftime('%Y%m%d-%H:%M:%S')
    tz_name, oh, om = OPEN[market]
    opens = datetime.now(ZoneInfo(tz_name)).replace(hour=oh, minute=om, second=0, microsecond=0)
    return (opens + timedelta(minutes=AUCTION_MINUTES)).astimezone(timezone.utc).strftime('%Y%m%d-%H:%M:%S')


def levels_from_fill(paid, o, ticks):
    """An order's stop-loss and take-profit from the price it was bought at (its stopRel and takeRel,
    fractions of that price), on the share's price increments: the stop rounded down, the take-profit up."""
    stop = take = None
    if o.get('stopRel') is not None:
        x = paid * (1 + float(o['stopRel']))
        stop = round_to(x, tick_at(ticks[0], ticks[1], x), 'down')
    if o.get('takeRel') is not None:
        x = paid * (1 + float(o['takeRel']))
        take = round_to(x, tick_at(ticks[0], ticks[1], x), 'up')
    return stop, take


def ib_symbol(ticker, market):
    # EODData's class shares (BRK.B) are "BRK B" at IBKR in the US; LSE keeps the dot (BT.A).
    return ticker.replace('.', ' ') if market == 'US' else ticker


def per_share(o, market):
    # The price of one share in the account's currency: EODData quotes LSE shares in pence.
    return o['limit'] / 100.0 if market == 'UK' else o['limit']


def round_to(x, tick, how):
    n = x / tick
    n = math.floor(n + 1e-9) if how == 'down' else math.ceil(n - 1e-9) if how == 'up' else round(n)
    return round(n * tick, 10)


DEAD = ('Cancelled', 'ApiCancelled', 'Inactive')
REQUEST_TIMEOUT = 60  # seconds: the longest any single request to IBKR may take (connect())


def connect(ib, args, client_id, tries=3):
    """Connects, retrying: a client id still held (a process ending, a gateway restarting) is often
    free again within a minute."""
    for k in range(tries):
        try:
            ib.connect(args.host, args.port, clientId=client_id, timeout=20)
            # ib_async can log "API connection failed" and return without a connection (IB Gateway
            # still logging in, 4 Nov): that is a failure, never a run against nothing.
            if not ib.isConnected():
                raise ConnectionError('IB Gateway did not accept the API connection')
            # Every blocking request answers within a minute or fails (ib_async waits forever by
            # default): a gateway that accepts connections but no longer reaches IBKR's servers (its
            # session taken over by a login elsewhere with the same user, 6 Nov) made steps hang for hours.
            ib.RequestTimeout = REQUEST_TIMEOUT
            return
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            print(f'  connecting as client {client_id}: {type(e).__name__}; again in 20 seconds')
            import time
            time.sleep(20)


def exit_contract(ib, contract):
    """The share's contract for a sale: its conId qualified, routed to its home exchange (the
    auctions), with its price increments."""
    from ib_async import Contract
    c = Contract(conId=contract.conId)
    if not qualified(ib, c):
        return None, None
    c.exchange = c.primaryExchange or c.exchange
    details = ib.reqContractDetails(c)[0]
    return c, (tick_rule(ib, details, c.exchange), details.minTick)


def floor_price(ib, contract, ticks):
    """The lowest price a forced sale accepts, on the share's price increments: the lowest low of its
    last 20 days (its own recent range, nothing fixed); without IBKR's daily bars (no data
    subscription for the exchange: 162, ALITL on 4 Nov), the lowest of today's low, the last trade,
    the bid and the previous close in IBKR's free delayed quote; None when neither answers."""
    lows = []
    try:
        bars = ib.reqHistoricalData(contract, '', '1 M', '1 day', 'TRADES', useRTH=True, timeout=20) or []
        lows = [b.low for b in bars[-20:] if b.low and b.low > 0]
    except Exception:  # noqa: BLE001
        pass
    if not lows:
        ib.reqMarketDataType(4)  # delayed, frozen when the market is shut: free, no subscription
        k = ib.reqMktData(contract, '', False, False)
        for _ in range(20):
            ib.sleep(0.5)
            lows = [x for x in (k.low, k.last, k.bid, k.close) if isinstance(x, (int, float)) and math.isfinite(x) and x > 0]
            if lows:
                ib.sleep(1.0)  # the rest of the quote's fields
                lows = [x for x in (k.low, k.last, k.bid, k.close) if isinstance(x, (int, float)) and math.isfinite(x) and x > 0]
                break
        ib.cancelMktData(contract)
    if not lows:
        return None
    lo = min(lows)
    return round_to(lo, tick_at(ticks[0], ticks[1], lo), 'down')


def order_fate(ib, trade, errors, wait=20):
    """Where an order stands at IBKR after up to `wait` seconds: 'live' (working: Submitted or
    PreSubmitted), 'filled', or 'dead' (cancelled, inactive, or no longer among the open orders).
    ib_async marks an order 'ValidationError' when IBKR answers with a warning while the order is
    still live (wrapper.error, isWarning): that is not an answer, so it waits for the next status,
    and if none comes, asks IBKR for its open orders. Returns (fate, IBKR's messages, numbers masked)."""
    for _ in range(wait * 2):
        ib.sleep(0.5)
        st = trade.orderStatus.status
        if st == 'Filled':
            break
        if st in DEAD or st in ('Submitted', 'PreSubmitted'):
            break
    st = trade.orderStatus.status
    notes = list(errors.get(trade.order.orderId, [])) + [e.message for e in trade.log if getattr(e, 'message', '') and e.errorCode]
    seen, msgs = set(), []
    for m in notes:
        m = re.sub(r'\d+\.\d+', '#', str(m))[:160]
        if m not in seen:
            seen.add(m), msgs.append(m)
    if st == 'Filled':
        return 'filled', msgs
    if st in DEAD:
        return 'dead', msgs
    if st in ('Submitted', 'PreSubmitted'):
        return 'live', msgs
    # No clear status (ValidationError, PendingSubmit, ...): is it among IBKR's open orders?
    ids = {(t.order.permId, t.order.orderId) for t in ib.reqAllOpenOrders() if t.orderStatus.status not in DEAD}
    if any(trade.order.permId and pid == trade.order.permId or oid == trade.order.orderId for pid, oid in ids):
        return 'live', msgs
    if any(f.execution.orderId == trade.order.orderId for f in ib.reqExecutions()):
        return 'filled', msgs
    return 'dead', msgs


def sell_out(ib, contract, qty, ref, errors, at_close):
    """Sells qty: at the close (MOC) or at the next price (market; at the open when placed before
    it), and confirms with IBKR that the order is working or filled (order_fate: a warning is not a
    confirmation). An order that did not survive (refused, cancelled by the exchange, an order type
    it does not take) is replaced by a limit at the share's 20-day low, which trades at the auction's
    or the market's price above it, and that is confirmed the same way. Returns (ok, what happened)."""
    from ib_async import MarketOrder, LimitOrder
    c, ticks = exit_contract(ib, contract)
    if c is None:
        return False, f'{contract.symbol}: not found at IBKR, NOT SOLD'
    o = MarketOrder('SELL', qty, orderRef=ref, tif='DAY')
    if at_close:
        o.orderType = 'MOC'
    kind = 'at the close' if at_close else 'at market'
    fate, msgs = order_fate(ib, ib.placeOrder(c, o), errors)
    said = f" [IBKR: {' | '.join(msgs)}]" if msgs else ''
    if fate != 'dead':
        return True, f'{c.symbol}: sell {qty:g} {kind}: {fate}{said}'
    lim = floor_price(ib, c, ticks)
    if lim is None:
        return False, f'{c.symbol}: sell {kind} did not survive{said}; no price for a limit: NOT SOLD'
    fate2, msgs2 = order_fate(ib, ib.placeOrder(c, LimitOrder('SELL', qty, lim, orderRef=ref, tif='DAY')), errors)
    said2 = f" [IBKR: {' | '.join(msgs2)}]" if msgs2 else ''
    if fate2 == 'dead':
        return False, f'{c.symbol}: sell {kind} did not survive{said}; limit at the 20-day low did not survive either{said2}: NOT SOLD'
    return True, f'{c.symbol}: sell {kind} did not survive{said}; limit at the 20-day low instead: {fate2}{said2}'


def leftovers(args):
    """The model's shares still held from an earlier day (--tickers: the shares it placed lately, from
    the bridge's history; never any other holding): with --check, listed (exit 1 when there are any);
    otherwise sold at the next price (before the open: in the opening auction), unless a sale is
    already waiting. Its own client (+5), so a busy orders' client never blocks it."""
    from ib_async import IB

    mine = {ib_symbol(t.split(':')[-1], args.market) for t in (args.tickers or '').split(',') if t}
    cur = {'UK': {'GBP'}, 'US': {'USD'}, 'EU': {'EUR', 'NOK'}, 'APAC': {'AUD', 'SGD'}}[args.market]
    ib = IB()
    errors = {}
    ib.errorEvent += lambda req, code, msg, contract: errors.setdefault(req, []).append(f'{code}: {msg}') if code not in (2104, 2106, 2107, 2108, 2119, 2158) else None
    connect(ib, args, args.client_id + 5)
    held = [p for p in ib.reqPositions() if p.position > 0 and p.contract.symbol in mine and p.contract.currency in cur]
    if args.check:
        for p in held:
            print(f'  {p.contract.symbol}: {p.position:g} shares still held')
        print(f'{args.market}: {len(held)} of the model\'s shares still held' if held else f'{args.market}: none of the model\'s shares held')
        ib.disconnect()
        sys.exit(1 if held else 0)
    waiting = {t.contract.conId for t in ib.reqAllOpenOrders() if t.order.action == 'SELL' and t.orderStatus.status not in DEAD}
    # Shares bought today by today's orders are the day's trades, not leftovers: never sold here.
    today = f'ofm-{args.market}-{date.today().isoformat()}'
    bought = {}
    for f in ib.reqExecutions():
        if f.execution.orderRef.startswith(today) and f.execution.side == 'BOT':
            bought[f.contract.conId] = bought.get(f.contract.conId, 0) + f.execution.shares
    held = [(p, p.position - bought.get(p.contract.conId, 0)) for p in held]
    held = [(p, q) for p, q in held if q > 0]
    ok_all, lines = True, []
    for p, qty in held:
        if p.contract.conId in waiting:
            lines.append(f'{p.contract.symbol}: a sale is already waiting')
            continue
        ok, line = sell_out(ib, p.contract, qty, f'ofm-{args.market}-leftover-{date.today().isoformat()}', errors, at_close=False)
        ok_all &= ok
        lines.append(line)
    ib.sleep(2)
    ib.disconnect()
    for line in lines:
        print('  ' + line)
    print(f'{args.market} leftovers: {len(held)} held from earlier days' + (f": {'; '.join(lines)}" if lines else ''))
    sys.exit(0 if ok_all else 1)


def state(args):
    """The live state of the market's trades at IBKR, as one JSON line (the page's "Today at IBKR"):
    each of today's orders (waiting to buy, not bought, refused, bought and held, stopped, took its
    profit, sold at the close) with what was bought and sold, and every holding of the model's shares
    (--tickers) with IBKR's price and profit; read only, its own client (+6)."""
    from ib_async import IB

    ref = f'ofm-{args.market}-{date.today().isoformat()}'
    mine = {ib_symbol(t.split(':')[-1], args.market) for t in (args.tickers or '').split(',') if t}
    ib = IB()
    connect(ib, args, args.client_id + 6, tries=2)
    num = lambda x: float(x) if isinstance(x, (int, float)) and math.isfinite(x) and abs(x) < 1e300 else None  # noqa: E731
    rows = {}

    def row(c):
        return rows.setdefault(c.symbol, {'ticker': c.symbol, 'currency': c.currency, 'state': '', 'limit': None, 'stop': None, 'take': None,
                                          'bought': 0.0, 'buy': None, 'sold': 0.0, 'sell': None, 'soldBy': '', 'held': 0.0, 'price': None,
                                          'pnl': 0.0, 'open': None, 'fees': 0.0, 'feesKnown': False, 'ordered': None, 'note': ''})
    orders = {}
    for t in list(ib.reqAllOpenOrders()) + list(ib.reqCompletedOrders(False)):
        o = t.order
        if not (o.orderRef or '').startswith(ref):
            continue
        orders[o.permId] = (t.contract, o, t.orderStatus.status)
        r = row(t.contract)
        if o.action == 'BUY' and not o.parentId and o.orderType == 'LMT':
            r['limit'], r['buyStatus'] = num(o.lmtPrice), t.orderStatus.status
            r['ordered'] = (r['ordered'] or 0) + float(o.totalQuantity or 0)  # partial fills show against it
        elif o.action == 'SELL' and o.orderType in ('STP', 'TRAIL'):
            r['stop'] = num(o.auxPrice) if o.orderType == 'STP' else num(getattr(o, 'trailStopPrice', None))
        elif o.action == 'SELL' and o.orderType == 'LMT' and (o.parentId or o.ocaGroup):
            r['take'] = r['take'] or num(o.lmtPrice)
    bv = {}
    for f in ib.reqExecutions():
        e = f.execution
        if not (e.orderRef or '').startswith(ref):
            continue
        r = row(f.contract)
        k = 'bought' if e.side == 'BOT' else 'sold'
        r[k] += e.shares
        bv[(r['ticker'], k)] = bv.get((r['ticker'], k), 0.0) + e.shares * e.price
        if k == 'sold':
            o = orders.get(e.permId, (None, None, None))[1]
            by = 'close' if (e.orderRef or '').endswith('-close') else {'STP': 'stop', 'TRAIL': 'stop', 'LMT': 'take-profit'}.get(getattr(o, 'orderType', ''), 'sale')
            r['soldBy'] = ', '.join(sorted(set(filter(None, r['soldBy'].split(', '))) | {by}))
        cr = f.commissionReport
        if cr is not None and num(cr.commission) is not None:
            r['fees'] += num(cr.commission)
            r['feesKnown'] = True
    for (tk, k), v in bv.items():
        r = rows[tk]
        r['buy' if k == 'bought' else 'sell'] = v / r[k] if r[k] else None
    # The profit from the fills themselves (IBKR's realised figure comes only with its commission
    # reports, which a reconnected client may not get again: every row showed 0 on 4 Nov): the shares
    # sold times the sale less the purchase price, in money (IBKR's price magnifier: LSE prices are in
    # pence, 100 to the pound), less the commissions IBKR reported.
    mags = {}
    for r in rows.values():
        if r['sold'] and r['buy'] and r['sell']:
            c = next((x.contract for x in ib.reqExecutions() if x.contract.symbol == r['ticker']), None)
            if c is not None and c.conId not in mags:
                try:
                    m = float(ib.reqContractDetails(c)[0].priceMagnifier or 1) or 1.0
                except Exception:  # noqa: BLE001
                    m = 1.0
                if m == 1.0 and r['ordered'] and r['limit'] and args.capital > 0:
                    # Not reported: the scale the order was sized with (its shares times its limit
                    # against the amount a trade, pence or pounds), the nearer of 1 and 100.
                    k = r['ordered'] * r['limit'] / args.capital
                    m = 100.0 if k > 0 and abs(math.log(k / 100.0)) < abs(math.log(k)) else 1.0
                mags[c.conId] = m
            mag = mags.get(c.conId, 1.0) if c is not None else 1.0
            r['pnl'] = min(r['sold'], r['bought']) * (r['sell'] - r['buy']) / mag - r['fees']
    for it in ib.portfolio():
        c = it.contract
        if it.position <= 0 or (c.symbol not in mine and c.symbol not in rows):
            continue
        r = row(c)
        r['held'], r['price'], r['open'] = it.position, num(it.marketPrice), num(it.unrealizedPNL)
        if not r['bought']:
            r['buy'] = num(it.averageCost)
    waiting = {t.contract.symbol for t in ib.reqAllOpenOrders() if t.order.action == 'SELL' and t.orderStatus.status not in DEAD}
    for r in rows.values():
        st = r.pop('buyStatus', '')
        if r['held'] and not r['bought']:
            r['state'] = 'held from an earlier day' + (', sale waiting' if r['ticker'] in waiting else '')
        elif r['held']:
            r['state'] = 'bought, held' + (', sale placed' if r['ticker'] in waiting else '')
        elif r['bought'] and r['sold'] >= r['bought']:
            r['state'] = f"sold ({r['soldBy'] or 'sale'})"
        elif st in DEAD:
            r['state'] = 'not bought (expired)' if st == 'Cancelled' else 'refused by IBKR'
        elif st == 'Filled':
            r['state'] = 'bought'
        elif st:
            r['state'] = 'waiting to buy'
        else:
            r['state'] = 'no order'
    ib.disconnect()
    print(json.dumps({'market': args.market, 'date': date.today().isoformat(), 'at': __import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds'),
                      'shares': sorted(rows.values(), key=lambda r: (not r['held'], not r['bought'], r['ticker']))}))


def holds(args):
    """The multi-day book (--holds place|close; the scan's shadow orders, e.g. the UK's exit on strength):
    its own orders (orderRef ofm-<MARKET>H-<date>: the day's --close and the leftovers step never touch
    them) and its positions kept in --hold-state (the bridge's history folder):
      place  the positions no longer held are dropped (sold); each order for a share not held is placed:
             a limit buy at its limit, or bought in the opening auction (open: true), for --capital (or
             the order's own amount)
      close  today's fills become positions (their entry price and day); today's unfilled buys are
             cancelled; each position is offered in the closing auction: at its last close (the order's
             days, else the fifth) sold at any price (MOC), before that only at or above its exit level
             (LOC at the entry times 1 + exitAbove, at least the entry and a tick: sold at the first close
             that reaches it, as the backtest).
    An order with "entry": "open" (apps/ofm_uk --hold) is a limit good only through the opening auction
    (GTD a few minutes after the open; a day order if IBKR refuses that), so it never buys a dip later."""
    from ib_async import IB, LimitOrder, MarketOrder
    path = args.hold_state
    st = json.load(open(path)) if path and os.path.exists(path) else {'positions': {}, 'pending': {}}
    ref = f'ofm-{args.market}H-{date.today().isoformat()}'
    ib = IB()
    errors = {}
    ib.errorEvent += lambda req, code, msg, contract: errors.setdefault(req, []).append(f'{code}: {msg}') if code not in (2104, 2106, 2107, 2108, 2119, 2158) else None
    connect(ib, args, args.client_id + 7)
    lines = []
    held = {p.contract.symbol: p for p in ib.reqPositions() if p.position > 0}
    if args.holds == 'place':
        for sym in [s_ for s_ in st['positions'] if s_ not in held]:
            lines.append(f'{sym}: sold (no longer held)')
            st['positions'].pop(sym)
        st['pending'] = {}
        book = json.load(open(args.orders)) if args.orders else {'orders': []}
        # The daily update (apps/ofm_uk --hold: "updates", every share on the day's list): each position the
        # model would no longer buy is sold now (before the open: in the opening auction); the others keep,
        # their exit level set again from the day's. A share without an update keeps its plan.
        today = date.today().isoformat()
        upd = {ib_symbol(u['ticker'].split(':')[-1], args.market): u for u in book.get('updates', []) if u.get('ticker')}
        for sym, pos in st['positions'].items():
            u = upd.get(sym)
            if not u or sym not in held:
                continue
            if not u.get('keep', True):
                if pos.get('exiting') == today:
                    lines.append(f'{sym}: already sold at the open today')
                    continue
                qty = min(float(pos.get('qty', held[sym].position)), held[sym].position)
                ok, line = sell_out(ib, held[sym].contract, qty, ref + '-exit', errors, at_close=False)
                if ok:
                    pos['exiting'] = today
                lines.append(f"{sym}: the model no longer holds it ({100 * float(u.get('expected') or 0):+.2f}% expected), sold at the open: {line}")
            elif u.get('exitAbove') is not None and abs(float(u['exitAbove']) - float(pos.get('exitAbove') or 0)) > 1e-9:
                lines.append(f"{sym}: kept; its exit level {100 * float(pos.get('exitAbove') or 0):+.2f}% -> {100 * float(u['exitAbove']):+.2f}% above the price paid")
                pos['exitAbove'] = float(u['exitAbove'])
        # The day's own orders (the daily strategy) keep their shares: the book never buys one of them.
        daily = {t.contract.symbol for t in ib.reqAllOpenOrders() if (t.order.orderRef or '').startswith(f'ofm-{args.market}-{date.today().isoformat()}')}
        free = max(0, int(book.get('slots', 10)) - len(st['positions']))  # the book's slots not held
        for o in book.get('orders', []):
            sym = ib_symbol(o['ticker'].split(':')[-1], args.market)
            if free <= 0:
                lines.append(f'{sym}: every slot of the book held, not bought')
                continue
            if sym in held or sym in st['positions'] or sym in daily:
                lines.append(f'{sym}: held already or in the day\'s orders, not bought')
                continue
            c = contract_of(o['ticker'], args.market)
            if not qualified(ib, c):
                lines.append(f'{sym}: not found at IBKR')
                continue
            px = o.get('limit') or o.get('close')
            qty = math.floor(capital_of(o, args) / per_share(dict(o, limit=px), args.market)) if px and px > 0 else 0
            if qty < 1:
                lines.append(f'{sym}: the amount buys no share')
                continue
            if o.get('entry') == 'open' and o.get('limit') and o.get('close'):  # the auction only, at most its limit
                ibkr_close = reference_close(ib, c)
                r = ibkr_close / o['close'] if ibkr_close else (None if args.market == 'UK' else 1.0)
                scale = next((x for x in (1.0, 0.01, 100.0) if r and abs(math.log(r / x)) < 0.05), None)
                c2, ticks = exit_contract(ib, c)
                if scale is None or c2 is None:
                    lines.append(f'{sym}: no IBKR price to check the price scale against; skipped')
                    continue
                lim = round_to(o['limit'] * scale, tick_at(ticks[0], ticks[1], o['limit'] * scale), 'down')
                order = LimitOrder('BUY', qty, lim, orderRef=ref, tif='GTD')
                order.goodTillDate = auction_end_utc(args.market, o['ticker'])
                how = f'in the opening auction, at most {lim:g}'
            elif o.get('open') or not o.get('limit'):
                order, how = MarketOrder('BUY', qty, orderRef=ref, tif='DAY'), 'in the opening auction'
            else:  # the limit in IBKR's price scale (LSE: pence or pounds), as the day's orders are placed
                ibkr_close = reference_close(ib, c)
                r = ibkr_close / o['close'] if ibkr_close and o.get('close') else (None if args.market == 'UK' else 1.0)
                scale = next((x for x in (1.0, 0.01, 100.0) if r and abs(math.log(r / x)) < 0.05), None)
                c2, ticks = exit_contract(ib, c)
                if scale is None or c2 is None:
                    lines.append(f'{sym}: no IBKR price to check the price scale against; skipped')
                    continue
                lim = round_to(o['limit'] * scale, tick_at(ticks[0], ticks[1], o['limit'] * scale), 'down')
                order, how = LimitOrder('BUY', qty, lim, orderRef=ref, tif='DAY'), f'at {lim:g}'
            fate, msgs = order_fate(ib, ib.placeOrder(c, order), errors)
            if fate == 'dead' and order.tif == 'GTD' and any('GTD' in m or 'time in force' in m.lower() or 'TIF' in m for m in msgs):
                order = LimitOrder('BUY', qty, order.lmtPrice, orderRef=ref, tif='DAY')  # GTD refused: a day order (the close cancels it if unfilled)
                fate, msgs = order_fate(ib, ib.placeOrder(c, order), errors)
                how += ' (a day order: IBKR refused the good-till time)'
            st['pending'][sym] = {'conId': c.conId, 'qty': qty, 'exit': o.get('exit', 'up'), 'exitAbove': float(o.get('exitAbove') or 0), 'days': int(o.get('days') or 5)}
            free -= 1
            lines.append(f"{sym}: buy {qty} {how}: {fate}" + (f" [IBKR: {' | '.join(msgs)}]" if msgs else ''))
    else:
        today = date.today().isoformat()
        got = {}
        for f in ib.reqExecutions():
            if f.execution.orderRef == ref and f.execution.side == 'BOT':
                g = got.setdefault(f.contract.symbol, [0.0, 0.0, f.contract])
                g[0] += f.execution.shares
                g[1] += f.execution.shares * f.execution.price
        for t in ib.reqAllOpenOrders():
            if t.order.orderRef == ref and t.order.action == 'BUY' and t.orderStatus.status not in DEAD:
                ib.cancelOrder(t.order)
        pend = st.get('pending', {})  # the day's orders' exit plans (exitAbove, days), for the positions they opened
        for sym, (q, v, c) in got.items():
            if st['positions'].get(sym, {}).get('since') == today:  # recorded by an earlier run today
                continue
            plan = pend.get(sym, {})
            st['positions'][sym] = {'conId': c.conId, 'qty': q, 'entry': v / q, 'since': today, 'closes': 0,
                                    'exitAbove': float(plan.get('exitAbove') or 0), 'days': int(plan.get('days') or 5)}
            lines.append(f'{sym}: bought {q:g} at {v / q:g}')
        st['pending'] = {}
        for sym, pos in st['positions'].items():
            if sym not in held:
                continue
            p = held[sym]
            qty = min(float(pos.get('qty', p.position)), p.position)  # its own shares only (never the day's trades in the same share)
            if pos.get('offered') == today:  # a rerun the same day: never offered twice
                lines.append(f'{sym}: already offered at this close')
                continue
            if pos.get('exiting') == today:  # sold at the open by the daily update (still held: its sale is working)
                lines.append(f'{sym}: being sold since the open (the daily update): not offered at the close')
                continue
            pos['closes'] = int(pos.get('closes', 0)) + 1  # this close counts (the fill's day is the first)
            pos['offered'] = today
            c, ticks = exit_contract(ib, p.contract)
            if c is None:
                lines.append(f'{sym}: not found at IBKR, NOT OFFERED')
                continue
            last = int(pos.get('days') or 5)
            if pos['closes'] >= last:  # its last day: sold at the close whatever the price (a limit at the 20-day low if the exchange refuses a MOC)
                ok, line = sell_out(ib, c, qty, ref + '-close', errors, at_close=True)
                lines.append(f'{sym}: its last day ({last}): {line}')
                continue
            else:
                tick = tick_at(ticks[0], ticks[1], pos['entry'])
                level = pos['entry'] * (1 + float(pos.get('exitAbove') or 0))
                lim = max(round_to(pos['entry'] + tick, tick, 'up'), round_to(level, tick_at(ticks[0], ticks[1], level), 'up'))
                o = LimitOrder('SELL', qty, lim, orderRef=ref + '-close', tif='DAY')
                o.orderType = 'LOC'
                how = f"in the closing auction only at or above {lim:g} (LOC, close {pos['closes']} of {last})"
            fate, msgs = order_fate(ib, ib.placeOrder(c, o), errors)
            lines.append(f"{sym}: sell {qty:g} {how}: {fate}" + (f" [IBKR: {' | '.join(msgs)}]" if msgs else ''))
    ib.sleep(2)
    ib.disconnect()
    if path:
        with open(path + '.tmp', 'w') as f:
            json.dump(st, f)
        os.replace(path + '.tmp', path)
    for line in lines:
        print('  ' + line)
    print(f"{args.market} multi-day book ({args.holds}): {len(st['positions'])} held" + (f": {'; '.join(lines)}" if lines else ''))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--market', choices=['UK', 'EU', 'US', 'APAC'], required=True)
    a.add_argument('--orders', help='the orders file (orders/<MARKET>.json from svr-net/eod)')
    a.add_argument('--capital', type=float, default=1000.0, help='amount per trade, in the account currency')
    a.add_argument('--host', default='127.0.0.1')
    a.add_argument('--port', type=int, default=4002, help='4002 IB Gateway paper, 4001 live')
    a.add_argument('--client-id', type=int, default=17)
    a.add_argument('--close', action='store_true', help='cancel unfilled orders and sell what is left at the close')
    a.add_argument('--report', action='store_true', help="print today's fills of these orders as JSON (after the close)")
    a.add_argument('--dry-run', action='store_true', help='print the orders; do not connect')
    a.add_argument('--auction', action='store_true', help="record the candidates' opening auction (places nothing)")
    a.add_argument('--bars', action='store_true', help="fetch the pool's daily bars from IBKR into --out")
    a.add_argument('--out', help='the CSV file for --bars or --paths')
    a.add_argument('--pause', help="with --bars: while this file exists the fetch waits between shares (the bridge's other steps need the gateway), "
                   "and says so with the file .ack")
    a.add_argument('--rebuy', action='store_true', help="watch the day's stops and place their buy-backs")
    a.add_argument('--until', help='HH:MM, the exchange clock: --rebuy stops watching then')
    a.add_argument('--paths', action='store_true', help="record the candidates' prices through the day (places nothing)")
    a.add_argument('--lines', type=int, default=0, help='with --paths or --auction: at most this many shares streamed at once (IBKR market data lines; 0: all)')
    a.add_argument('--years', type=int, default=2, help='years of daily bars for --bars')
    a.add_argument('--duration', default='', help="IBKR's duration for --bars in place of --years (e.g. '1 M': the latest month, merged into the full file by the bridge)")
    a.add_argument('--diag', action='store_true', help="check the API, the account and the market's data; one JSON line")
    a.add_argument('--leftovers', action='store_true', help="sell the model's shares still held from earlier days (--tickers)")
    a.add_argument('--tickers', help='with --leftovers: the shares the model placed lately, comma separated')
    a.add_argument('--state', action='store_true', help="the live state of today's trades and the model's holdings (--tickers), as JSON")
    a.add_argument('--check', action='store_true', help='with --leftovers: only list them; exit 1 when there are any')
    a.add_argument('--ping', action='store_true', help="exit 0 when IB Gateway's API answers within 15 seconds, else 1")
    a.add_argument('--holds', choices=['place', 'close'], help="the multi-day book: place its orders (--orders), or offer its positions at the close")
    a.add_argument('--hold-state', help="with --holds: the book's positions (a JSON file the bridge keeps)")
    args = a.parse_args()
    # Each market its own API clients (IBKR allows one connection per client id): the orders' client
    # (place, buy-back watch, close, report) and its helpers (+1 auction, +2 bars, +3 diag, +4 paths)
    # are 17.. for the UK, 27.. for the EU, 37.. for the US, 47.. for Asia-Pacific, so one market's
    # all-day buy-back watch never blocks another market's orders.
    args.client_id += 10 * ['UK', 'EU', 'US', 'APAC'].index(args.market)
    ref = f'ofm-{args.market}-{date.today().isoformat()}'

    if args.ping:
        return ping(args)
    if args.holds:
        return holds(args)
    if args.leftovers:
        return leftovers(args)
    if args.state:
        return state(args)
    if args.diag:
        return diag(args)
    if args.auction:
        return auction(args)
    if args.bars:
        return daily_bars(args)
    if args.paths:
        return paths(args)
    if args.rebuy:
        return rebuy(args, ref)
    if not args.close and not args.report:
        if not args.orders:
            sys.exit('--orders is required (or --close)')
        book = json.load(open(args.orders))
        print(f'{args.market}: {len(book["orders"])} orders for the open after the close of {book["asOf"]}')
        if args.dry_run:
            for o in book['orders']:
                q = math.floor(capital_of(o, args) / per_share(o, args.market)) if o['limit'] > 0 else 0
                if o.get('entry') == 'open':
                    kind = {'trail': f"a trailing stop {100 * float(o.get('trailRel') or 0):.2f}% under the high, from ",
                            'breakeven': f"a break-even stop (to the entry once up {100 * float(o.get('breakevenRel') or 0):.2f}%), from "}.get(o.get('stopMode'), 'a fixed ')
                    take = f", take-profit {100 * o['takeRel']:+.2f}%" if o.get('takeRel') is not None else ', no take-profit'
                    print(f"  {ib_symbol(o['ticker'], args.market):8s} buy {q:6d} at the open, at most {o['limit']:.4f} (until {AUCTION_MINUTES} minutes after the open); then "
                          f"{kind}stop {100 * o['stopRel']:+.2f}%{take} from the price bought at, one cancelling the other; the rest sold at the close")
                    continue
                print(f"  {ib_symbol(o['ticker'], args.market):8s} buy {q:6d} limit {o['limit']:.4f}  stop {o['stop']}  take {o['take']}"
                      + (f"  buy-back {100 * o['rebuy']:.0f}% back after a stop" if (o.get('rebuy') or 0) > 0 else ''))
            return

    from ib_async import IB, Stock, LimitOrder, StopOrder, MarketOrder  # imported here so --dry-run needs no install

    ib = IB()
    errors = {}  # IBKR's errors by order id (a rejection arrives after placeOrder returns)
    ib.errorEvent += lambda req, code, msg, contract: errors.setdefault(req, []).append(f'{code}: {msg}') if code not in (2104, 2106, 2107, 2108, 2119, 2158) else None
    connect(ib, args, args.client_id)

    if args.report:
        # Today's fills of this model's orders, a share at a time: bought, sold, average prices, the
        # return, the realised profit and loss IBKR reports for the closing fills (after its
        # commissions), and the commissions.
        from datetime import timezone
        per = {}
        kinds = {t.order.permId: t.order.orderType for t in list(ib.reqAllOpenOrders()) + list(ib.reqCompletedOrders(False))}
        for f in ib.fills():
            if not f.execution.orderRef.startswith(ref):  # the day's orders and the close's sales (ref-close)
                continue
            e = per.setdefault(f.contract.symbol, {'ticker': f.contract.symbol, 'currency': f.contract.currency,
                                                    'bought': 0.0, 'buy_value': 0.0, 'sold': 0.0, 'sell_value': 0.0, 'pnl': 0.0, 'commission': 0.0})
            side = 'bought' if f.execution.side == 'BOT' else 'sold'
            e[side] += f.execution.shares
            if side == 'sold':  # how and when it was sold: the close's sale, else the bracket's stop or take-profit
                ot = kinds.get(f.execution.permId, '')
                e['soldBy'] = 'close' if f.execution.orderRef.endswith('-close') else {'STP': 'stop', 'TRAIL': 'stop', 'LMT': 'take-profit'}.get(ot, 'bracket')
                e['soldAt'] = f.time.astimezone(timezone.utc).strftime('%H:%M') if f.time else None
            e['buy_value' if side == 'bought' else 'sell_value'] += f.execution.shares * f.execution.price
            cr = f.commissionReport
            if cr is not None:
                if cr.commission and abs(cr.commission) < 1e300:
                    e['commission'] += cr.commission
                if cr.realizedPNL and abs(cr.realizedPNL) < 1e300:
                    e['pnl'] += cr.realizedPNL
        out = []
        for e in per.values():
            buy = e['buy_value'] / e['bought'] if e['bought'] else None
            sell = e['sell_value'] / e['sold'] if e['sold'] else None
            out.append({'ticker': e['ticker'], 'currency': e['currency'], 'bought': e['bought'], 'buy': buy, 'sold': e['sold'], 'sell': sell,
                        'ret': (sell / buy - 1) if buy and sell else None, 'pnl': e['pnl'], 'commission': e['commission'],
                        'soldBy': e.get('soldBy'), 'soldAt': e.get('soldAt')})
        print(json.dumps({'date': date.today().isoformat(), 'market': args.market, 'trades': out}))
        ib.disconnect()
        return

    if args.close:
        # Cancel this model's open orders of today (its stops and take-profits too), confirm they are
        # gone, then sell each position they opened at the close and confirm IBKR took each sale; a
        # share whose orders could not be cancelled is not sold (its stop could sell it a second time).
        mine = [t for t in ib.openTrades() if t.order.orderRef == ref]
        for t in mine:
            ib.cancelOrder(t.order)
        for _ in range(20):
            ib.sleep(0.5)
            if all(t.orderStatus.status in DEAD + ('Filled',) for t in mine):
                break
        live = [t for t in ib.reqAllOpenOrders() if t.orderStatus.status not in DEAD + ('Filled', 'PendingCancel')]
        stuck = {t.contract.conId: t.contract.symbol for t in live if t.order.orderRef == ref}
        # A sale an earlier run of the close placed (its own reference): a rerun never sells twice.
        selling = {t.contract.conId for t in live if t.order.orderRef == ref + '-close' and t.order.action == 'SELL'}
        filled = {}
        for f in ib.fills():
            if f.execution.orderRef == ref and f.execution.side == 'BOT':
                filled[f.contract.conId] = (f.contract, filled.get(f.contract.conId, (None, 0))[1] + f.execution.shares)
        held = {p.contract.conId: p.position for p in ib.reqPositions()}
        ok_all, lines = True, []
        for con, sym in stuck.items():
            ok_all = False
            lines.append(f'{sym}: an order of today could not be cancelled: NOT SOLD')
        for con, (contract, bought) in filled.items():
            qty = min(bought, held.get(con, 0))
            if qty > 0 and con in selling:
                lines.append(f'{contract.symbol}: its sale at the close is already placed')
            elif qty > 0 and con not in stuck:
                ok, line = sell_out(ib, contract, qty, ref + '-close', errors, at_close=True)
                ok_all &= ok
                lines.append(line)
        ib.sleep(2)
        ib.disconnect()
        for line in lines:
            print('  ' + line)
        print(f"{args.market} close: " + ('; '.join(lines) if lines else 'nothing held'))
        sys.exit(0 if ok_all else 1)

    # A rerun the same day places only what is not there yet: a share with an order or a fill of
    # today's reference (from any client) is skipped, so an interrupted placing is never doubled.
    dead = ('Cancelled', 'ApiCancelled', 'Inactive', 'PendingCancel')  # an order IBKR no longer works: placed again
    open_now = [t for t in ib.reqAllOpenOrders() if t.order.orderRef == ref and t.orderStatus.status not in dead]
    have = {t.contract.conId for t in open_now}
    bought_ids = {f.contract.conId for f in ib.reqExecutions() if f.execution.orderRef == ref}
    have |= bought_ids
    # A share placed earlier today and not filled at all: a newer orders file moves its prices instead.
    waiting = {t.contract.conId: t for t in open_now if t.order.action == 'BUY' and not t.order.parentId
               and t.orderStatus.filled == 0 and t.contract.conId not in bought_ids}
    resent = set()

    def bracket(contract, sym, qty, lim, take, stop, good_after=None):
        """A day limit buy with its take-profit (limit sell) and stop-loss (stop sell) attached, one
        cancelling the other, active once the buy fills; good_after (UTC, yyyymmdd-hh:mm:ss): the buy
        works only from then."""
        parent = LimitOrder('BUY', qty, lim, orderRef=ref, tif='DAY', transmit=False)
        if good_after:
            parent.goodAfterTime = good_after
        ib.placeOrder(contract, parent)
        children = []
        if take:
            children.append(LimitOrder('SELL', qty, take, orderRef=ref, tif='DAY', parentId=parent.orderId, transmit=False))
        if stop:
            children.append(StopOrder('SELL', qty, stop, orderRef=ref, tif='DAY', parentId=parent.orderId, transmit=False))
        for c in children:
            c.ocaGroup, c.ocaType = f'{ref}-{sym}', 1
        for k, c in enumerate(children):
            c.transmit = k == len(children) - 1
            ib.placeOrder(contract, c)
        if not children:
            parent.transmit = True
            ib.placeOrder(contract, parent)
        print(f"  {sym}: buy {qty} limit {lim}" + ''.join(f', {c.action.lower()} {c.orderType} {c.auxPrice if c.orderType == "STP" else c.lmtPrice}' for c in children))
    def open_buy(contract, sym, qty, lim, tif='GTD', ticker=''):
        """A buy for the open: a limit (no higher than lim) good only through the opening auction (GTD a
        few minutes after the open; with DAY, the watch cancels it then). No levels attached: the watch
        (--rebuy) places the stop-loss and take-profit from the price it was bought at, so a gap never
        sets them from the wrong price."""
        parent = LimitOrder('BUY', qty, lim, orderRef=ref, tif=tif, transmit=True)
        if tif == 'GTD':
            parent.goodTillDate = auction_end_utc(args.market, ticker)
        ib.placeOrder(contract, parent)
        print(f'  {sym}: buy {qty} at the open, at most {lim} ({tif}' + (f' until {parent.goodTillDate} UTC' if tif == 'GTD' else '') + '); its stop and take-profit from its fill')

    # More orders than the money pays for (the orders' slots; the model places more when few fill):
    # the first slots go into the opening auction, the rest only from a minute after the open (IBKR's
    # good-after time), and the watch (--rebuy) cancels whatever is still open once slots have filled.
    slots = int(book.get('slots') or 0)
    after = None
    if slots and len(book['orders']) > slots:
        from datetime import datetime, timedelta, timezone
        from zoneinfo import ZoneInfo
        tz_name, oh, om = OPEN[args.market]
        opens = datetime.now(ZoneInfo(tz_name)).replace(hour=oh, minute=om, second=0, microsecond=0)
        after = (opens.astimezone(timezone.utc) + timedelta(minutes=1)).strftime('%Y%m%d-%H:%M:%S')
        print(f'{args.market}: {len(book["orders"])} orders for {slots} trades: the first {slots} in the opening auction, the rest from {after} UTC')
    for k_order, o in enumerate(book['orders']):
        contract = contract_of(o['ticker'], args.market)
        sym = o['ticker']
        if not qualified(ib, contract):
            print(f'  {sym}: not found at IBKR, skipped')
            continue
        if contract.conId in have and contract.conId not in waiting:
            print(f'  {sym}: already placed today, skipped')
            continue
        details = ib.reqContractDetails(contract)[0]
        rule = tick_rule(ib, details, contract.exchange)
        tick = lambda px: tick_at(rule, details.minTick, px)  # noqa: E731 - the increment at a price
        # The price scale: IBKR's last close against ours (LSE prices can be in pence or pounds). Never
        # assumed: without IBKR's close the order is skipped, as a wrong scale would buy without a limit
        # and stop at once.
        ibkr_close = reference_close(ib, contract)
        if ibkr_close is None and args.market == 'UK':
            print(f"  {sym}: no IBKR price to check the price scale against; skipped")
            continue
        # Only the LSE quotes in pence at EODData and in pounds or pence at IBKR; elsewhere both quote in
        # the exchange's currency, so without IBKR's price (no market data) the scale is 1.
        r = ibkr_close / o['close'] if ibkr_close is not None else 1.0
        scale = next((s for s in (1.0, 0.01, 100.0) if abs(math.log(r / s)) < 0.05), None)
        if scale is None:
            print(f"  {sym}: IBKR's close {ibkr_close} does not match ours {o['close']}; skipped")
            continue
        # The levels must sit around the limit: the stop below it, the take-profit above it.
        if (o.get('stop') and not o['stop'] < o['limit']) or (o.get('take') and not o['take'] > min(o['limit'], o['close'])):
            print(f"  {sym}: levels out of order (limit {o['limit']}, stop {o.get('stop')}, take {o.get('take')}); skipped")
            continue
        lim = round_to(o['limit'] * scale, tick(o['limit'] * scale), 'down')
        qty = math.floor(capital_of(o, args) / per_share(o, args.market)) if lim > 0 else 0
        if qty < 1:
            print(f'  {sym}: capital {capital_of(o, args)} buys no share at {per_share(o, args.market)}; skipped')
            continue
        take = round_to(o['take'] * scale, tick(o['take'] * scale), 'up') if o.get('take') else None
        stop = round_to(o['stop'] * scale, tick(o['stop'] * scale), 'down') if o.get('stop') else None
        if contract.conId in waiting:  # placed earlier today, unfilled: its prices moved to this file's
            t0 = waiting[contract.conId]
            if abs(t0.order.lmtPrice - lim) < 1e-9:
                print(f'  {sym}: already placed today at {lim}, kept')
                continue
            kids = [x for x in ib.openTrades() if x.order.parentId == t0.order.orderId]
            t0.order.lmtPrice = lim
            t0.order.transmit = not kids
            ib.placeOrder(t0.contract, t0.order)
            for k, x in enumerate(kids):
                if x.order.orderType == 'LMT' and take:
                    x.order.lmtPrice = take
                if x.order.orderType == 'STP' and stop:
                    x.order.auxPrice = stop
                x.order.transmit = k == len(kids) - 1
                ib.placeOrder(x.contract, x.order)
            print(f'  {sym}: moved to buy {t0.order.totalQuantity:g} limit {lim}' + (f', take {take}' if take else '') + (f', stop {stop}' if stop else ''))
            continue
        if o.get('entry') == 'open':
            open_buy(contract, sym, qty, lim, ticker=o['ticker'])
            continue
        bracket(contract, sym, qty, lim, take, stop, after if after and k_order >= slots else None)
    # IBKR's answer to each order (a rejection comes after placeOrder returns). An order refused only
    # for being routed directly to the exchange (10311: the gateway's API precautions; BYPASS_WARNING
    # in the NAS's settings lifts it) goes again through IBKR's SMART routing, its exchange as the
    # primary: it then trades on that exchange or its alternatives, not necessarily in its auctions.
    ib.sleep(5)
    for t in list(ib.trades()):
        o2 = t.order
        if o2.orderRef != ref or o2.parentId or o2.action != 'BUY' or t.orderStatus.status not in ('Cancelled', 'ApiCancelled', 'Inactive'):
            continue
        if not any(e.startswith('10311') for e in errors.get(o2.orderId, [])) or t.contract.exchange == 'SMART' or t.contract.conId in resent:
            continue
        resent.add(t.contract.conId)
        smart = Stock(t.contract.symbol, 'SMART', t.contract.currency, primaryExchange=t.contract.exchange)
        if not qualified(ib, smart):
            continue
        kids = [x.order for x in ib.trades() if x.order.parentId == o2.orderId]
        take = next((k.lmtPrice for k in kids if k.orderType == 'LMT'), None)
        stop = next((k.auxPrice for k in kids if k.orderType == 'STP'), None)
        bracket(smart, f'{t.contract.symbol} (SMART)', int(o2.totalQuantity), o2.lmtPrice, take, stop)
    if resent:
        ib.sleep(5)
    # An order IBKR refused for its size (201: "not exceeding N" shares, its own limit, read from its
    # answer) goes again for N shares; one refused for its price step (110) goes again with its prices
    # on the next coarser increment of the share's own market rule. Each refused order at most once per
    # round, a few rounds; the refused one no longer counts.
    import re
    superseded = set()
    for _round in range(4):
        again = 0
        for t in list(ib.trades()):
            o2 = t.order
            if o2.orderRef != ref or o2.parentId or o2.action != 'BUY' or o2.orderId in superseded \
                    or t.orderStatus.status not in ('Cancelled', 'ApiCancelled', 'Inactive'):
                continue
            errs = errors.get(o2.orderId, [])
            kids = [x.order for x in ib.trades() if x.order.parentId == o2.orderId]
            take = next((k.lmtPrice for k in kids if k.orderType == 'LMT'), None)
            stop = next((k.auxPrice for k in kids if k.orderType == 'STP'), None)
            qty, lim = int(o2.totalQuantity), o2.lmtPrice
            if o2.tif == 'GTD' and not kids and any('GTD' in e or 'time in force' in e.lower() or 'TIF' in e for e in errs):
                superseded.add(o2.orderId)  # the open's buy refused for its expiry: a day order, which the watch cancels after the auction
                open_buy(t.contract, f'{t.contract.symbol} (again)', qty, lim, tif='DAY')
                again += 1
                continue
            size = next((re.search(r'exceeding\s+([0-9]+)', e) for e in errs if e.startswith('201') and 'exceeding' in e), None)
            if size and 0 < int(size.group(1)) < qty:
                qty = int(size.group(1))
            elif any(e.startswith('110') for e in errs):
                details = ib.reqContractDetails(t.contract)[0]
                rule = tick_rule(ib, details, t.contract.exchange if t.contract.exchange != 'SMART' else t.contract.primaryExchange)
                now = tick_at(rule, details.minTick, lim)
                coarser = sorted({step for _, step in (rule or ()) if step > now + 1e-12})
                if not coarser:
                    continue
                step = coarser[0]
                lim = round_to(lim, step, 'down')
                take = round_to(take, step, 'up') if take else None
                stop = round_to(stop, step, 'down') if stop else None
            else:
                continue
            superseded.add(o2.orderId)
            bracket(t.contract, f'{t.contract.symbol} (again)', qty, lim, take, stop, o2.goodAfterTime or None)
            again += 1
        if not again:
            break
        ib.sleep(5)
    accepted = rejected = 0
    for t in ib.openTrades() + [t for t in ib.trades() if t.orderStatus.status in ('Cancelled', 'ApiCancelled', 'Inactive')]:
        if t.order.orderRef != ref or t.order.parentId or t.order.orderId in superseded:
            continue
        if t.contract.conId in resent and t.contract.exchange != 'SMART' and t.orderStatus.status in ('Cancelled', 'ApiCancelled', 'Inactive'):
            continue  # its SMART resend is the one that counts
        st = t.orderStatus.status
        why = '; '.join(errors.get(t.order.orderId, [])) or ' '.join(x.message for x in t.log if x.message)
        if not why:  # IBKR gave no message: what its log and status say
            why = ' / '.join(f'{x.status} {x.message}'.strip() for x in t.log if x.status or x.message) or f'status {t.orderStatus.status}, no reason given by IBKR'
        if st in ('Cancelled', 'ApiCancelled', 'Inactive'):
            rejected += 1
            print(f'  {t.contract.symbol}: REJECTED ({st}) {why}'.strip())
        else:
            accepted += 1
            print(f'  {t.contract.symbol}: accepted ({st})')
    print(f'{args.market}: {accepted} accepted by IBKR, {rejected} rejected')
    ib.disconnect()
    if rejected and not accepted:
        sys.exit(3)


OPEN = {'UK': ('Europe/London', 8, 0), 'EU': ('Europe/Paris', 9, 0), 'US': ('America/New_York', 9, 30), 'APAC': ('Australia/Sydney', 10, 0)}


def streamed(book, lines):
    """The shares a recording streams: the day's orders first, then the other candidates in their
    rank, at most lines of them (IBKR's market data lines; 0: all)."""
    cands = book.get('candidates') or book['orders']
    ordered = {o['ticker'] for o in book.get('orders', [])}
    first = [c for c in cands if c['ticker'] in ordered] + [{'ticker': o['ticker'], 'close': o.get('close')} for o in book.get('orders', [])
                                                             if o['ticker'] not in {c['ticker'] for c in cands}]
    rest = [c for c in cands if c['ticker'] not in ordered]
    out = first + rest
    return out[:lines] if lines > 0 else out


def auction(args):
    """The candidates' opening auction, as IBKR streams it (generic tick 225: the indicative price,
    volume and imbalance), sampled every 15 seconds until the open, then the open itself. Needs the
    exchange's real-time data on the account (otherwise the samples stay empty, which is recorded)."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from ib_async import IB, Stock

    book = json.load(open(args.orders))
    cands = streamed(book, args.lines)
    zone, h, mi = OPEN[args.market]
    tz = ZoneInfo(zone)
    now = datetime.now(tz)
    opens = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    if now > opens + timedelta(minutes=5):
        sys.exit(f'{args.market} opened at {opens:%H:%M}: too late to record its auction')
    ib = IB()
    connect(ib, args, args.client_id + 1)  # beside the orders' client
    ib.reqMarketDataType(1)  # real time
    num = lambda x: x if isinstance(x, (int, float)) and math.isfinite(x) and x > -1e300 else None  # noqa: E731
    tickers = {}
    for c in cands:
        contract = contract_of(c['ticker'], args.market)
        if qualified(ib, contract):
            tickers[c['ticker']] = ib.reqMktData(contract, '225', False, False)
    rows = {t: {'ticker': t, 'close': c.get('close'), 'score': c.get('score'), 'samples': []} for t, c in ((c['ticker'], c) for c in cands) if t in tickers}

    def sample(at):
        for t, k in tickers.items():
            rows[t]['samples'].append({'at': at, 'price': num(k.auctionPrice), 'volume': num(k.auctionVolume), 'imbalance': num(k.auctionImbalance),
                                       'bid': num(k.bid), 'ask': num(k.ask), 'bidSize': num(k.bidSize), 'askSize': num(k.askSize), 'last': num(k.last),
                                       'type': k.marketDataType})

    while datetime.now(tz) < opens:  # every 15 seconds until the open (the auction ends then, or up to 30 s later in London)
        ib.sleep(max(0.5, min(15.0, (opens - datetime.now(tz)).total_seconds())))
        sample(datetime.now(tz).strftime('%H:%M:%S'))
    ib.sleep(120)  # the uncross and the first trades
    for t, k in tickers.items():
        rows[t].update(open=num(k.open), last=num(k.last), volume=num(k.volume), high=num(k.high), low=num(k.low))
        ib.cancelMktData(k.contract)
    ib.disconnect()
    filled = sum(1 for r in rows.values() if any(s['price'] for s in r['samples']))
    sys.stderr.write(f'{args.market} auction: {len(rows)} shares, an indicative price for {filled}\n')
    print(json.dumps({'date': opens.date().isoformat(), 'market': args.market, 'asOf': book.get('asOf'), 'shares': list(rows.values())}))


CLOSE = {'UK': ('Europe/London', 16, 35), 'EU': ('Europe/Paris', 17, 35), 'US': ('America/New_York', 16, 0), 'APAC': ('Australia/Sydney', 16, 10)}


def paths(args):
    """The candidates' prices through the day (places nothing): every 30 seconds from the open to 20
    minutes after the close, IBKR's delayed data (type 3: real time where the account is subscribed,
    else 15 minutes late, free), as time,ticker,last,bid,ask,type in --out (the time this script's,
    the exchange's zone; a delayed row's prices are about 15 minutes older)."""
    import csv
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from ib_async import IB

    if not args.out:
        sys.exit('--out is required with --paths')
    book = json.load(open(args.orders))
    cands = streamed(book, args.lines)
    zone, h, mi = OPEN[args.market]
    tz = ZoneInfo(zone)
    now = datetime.now(tz)
    opens = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    _, ch, cm = CLOSE[args.market]
    ends = now.replace(hour=ch, minute=cm, second=0, microsecond=0) + timedelta(minutes=20)
    if now > ends:
        sys.exit(f'{args.market} closed at {ch:02d}:{cm:02d}: too late to record its day')
    ib = IB()
    connect(ib, args, args.client_id + 4)  # beside the orders', the auction's and the bars' clients
    ib.reqMarketDataType(3)  # delayed where not subscribed
    num = lambda x: x if isinstance(x, (int, float)) and math.isfinite(x) and x > 0 else ''  # noqa: E731
    tickers = {}
    for c in cands:
        contract = contract_of(c['ticker'], args.market)
        if qualified(ib, contract):
            tickers[c['ticker']] = ib.reqMktData(contract, '', False, False)
    rows = 0
    fresh = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, 'a', newline='') as f:  # appended: a restarted recording keeps the morning's rows
        w = csv.writer(f)
        if fresh:
            w.writerow(['time', 'ticker', 'last', 'bid', 'ask', 'type'])
        while datetime.now(tz) < opens:
            ib.sleep(max(0.5, min(30.0, (opens - datetime.now(tz)).total_seconds())))
        while datetime.now(tz) < ends:
            at = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
            for t, k in tickers.items():
                if num(k.last) or num(k.bid):
                    w.writerow([at, t, num(k.last), num(k.bid), num(k.ask), k.marketDataType])
                    rows += 1
            f.flush()
            ib.sleep(30)
    for k in tickers.values():
        ib.cancelMktData(k.contract)
    ib.disconnect()
    print(f'{args.market} paths: {len(tickers)} shares, {rows} prices')


def rebuy(args, ref):
    """The buy-backs after the day's stops (places them; see the module's help). It runs as the
    orders' own client (--client-id, the one that placed them and closes them), so it sees their fills
    and its orders are the ones --close cancels; it stops at --until (set before the close's time) and
    then cancels its own buy-backs still open, so nothing it placed outlives the day. Each stop is
    handled once (the handled shares kept in --out, so a restart does not buy back twice)."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from ib_async import IB, LimitOrder, Order, StopLimitOrder, StopOrder

    book = json.load(open(args.orders))
    plan = {ib_symbol(o['ticker'].split(':')[-1], args.market): o for o in book['orders'] if (o.get('rebuy') or 0) > 0}
    slots = int(book.get('slots') or 0)
    capped = bool(slots) and len(book['orders']) > slots  # more orders than the money pays for: watch the fills
    # Orders for the open: their stop-loss and take-profit placed from the price each was bought at.
    at_open = {ib_symbol(o['ticker'].split(':')[-1], args.market): o for o in book['orders'] if o.get('entry') == 'open'}
    if not plan and not capped and not at_open:
        print(f'{args.market} buy-back: none in the orders')
        return
    tz = ZoneInfo(OPEN[args.market][0])
    h, mi = (int(x) for x in (args.until or '16:15').split(':'))
    until = datetime.now(tz).replace(hour=h, minute=mi, second=0, microsecond=0)
    state = args.out or f'/tmp/rebuy-{ref}.txt'
    handled = set(open(state).read().split()) if os.path.exists(state) else set()
    ib = IB()
    connect(ib, args, args.client_id)
    placed, mine = 0, []
    full = [False]

    def cap_check(*_):
        """Once the day's buys (the model's orders, not its buy-backs) have filled slots shares, its
        buy orders still open are cancelled (their stops and take-profits with them): the money is used."""
        if not capped or full[0]:
            return
        bought = {x.contract.conId for x in ib.fills() if x.execution.orderRef == ref and x.execution.side == 'BOT'}
        if len(bought) < slots:
            return
        full[0] = True
        n = 0
        for t in ib.openTrades():
            o = t.order
            if o.orderRef == ref and o.action == 'BUY' and not o.parentId and o.orderType == 'LMT' and t.contract.conId not in bought \
                    and t.orderStatus.filled == 0:
                ib.cancelOrder(o)
                n += 1
        print(f'  {len(bought)} of {slots} trades filled: {n} buy orders still open cancelled')
    _, oh, om = OPEN[args.market]
    auction_over = datetime.now(tz).replace(hour=oh, minute=om, second=0, microsecond=0) + timedelta(minutes=AUCTION_MINUTES + 1)
    ticks_of = {}

    entries = {}       # conId -> (symbol, contract, price paid, the order, ticks): the orders for the open that bought
    mine_cancel = set()  # level orders this watch cancelled itself (re-placed for a grown fill)
    trail_ids = {}     # conId -> the trailing stop's order id
    no_trail = set()   # shares whose trailing stop IBKR refused: a fixed stop instead (the watch does not trail it)
    moved = set()      # break-even stops moved to the entry
    quotes = {}        # conId -> IBKR's quote (delayed where the account has no real-time data)

    def set_levels(fills):
        """Each order for the open that has bought: its stop-loss and take-profit (one cancelling the
        other, reduced together on a part fill) for the shares still held, from the price it was bought
        at. The stop is fixed, IBKR's own trailing stop (stopMode trail: trailRel below the highest price
        since it was placed, moved by IBKR, the watch or not), or fixed until the break-even move
        (stopMode breakeven: move_to_breakeven). Placed once; placed again only when more shares were
        bought (a part fill growing), so a restarted watch never doubles them."""
        live = [t for t in ib.reqAllOpenOrders() if t.order.orderRef == ref and t.orderStatus.status not in DEAD + ('Filled', 'PendingCancel')]
        for sym, o in at_open.items():
            buys = [x for x in fills if x.execution.orderRef == ref and x.execution.side == 'BOT' and x.contract.symbol == sym]
            n = sum(x.execution.shares for x in buys)
            if not n:
                continue
            con = buys[0].contract.conId
            if con not in ticks_of:
                ticks_of[con] = exit_contract(ib, buys[0].contract)
            c, ticks = ticks_of[con]
            if c is None:
                print(f'  {sym}: not found at IBKR for its levels: NO STOP (sold at the close)')
                continue
            paid = sum(x.execution.shares * x.execution.price for x in buys) / n
            entries[con] = (sym, c, paid, o, ticks)
            sold = sum(x.execution.shares for x in fills if x.execution.orderRef == ref and x.execution.side == 'SLD' and x.contract.conId == con)
            left = n - sold
            sells = [t for t in live if t.order.action == 'SELL' and t.contract.conId == con]
            if con in trail_ids and con not in no_trail and not any(t.order.orderId == trail_ids[con] for t in sells) and trail_ids[con] not in mine_cancel and left > 0:
                dead = next((t for t in ib.trades() if t.order.orderId == trail_ids[con]), None)
                if dead is not None and dead.orderStatus.status in DEAD:
                    no_trail.add(con)  # refused (not filled, not cancelled by this watch): a fixed stop
                    print(f'  {sym}: IBKR refused the trailing stop; a fixed stop instead')
            have = max((float(t.order.totalQuantity) - float(t.orderStatus.filled or 0) for t in sells), default=0.0)
            if left <= 0 or (sells and abs(have - left) < 1e-9):
                continue
            if any(t.order.action == 'BUY' and t.contract.conId == con for t in live) and sells:
                continue  # still buying in the auction: set once it is done
            for t in sells:
                mine_cancel.add(t.order.orderId)
                ib.cancelOrder(t.order)
            if sells:
                ib.sleep(1)
            stop, take = levels_from_fill(paid, o, ticks)
            group = f'{ref}-{con}-{int(n)}'
            qty = int(left)
            trail = o.get('stopMode') == 'trail' and o.get('trailRel') and stop and con not in no_trail
            kids = []
            if trail:  # IBKR's own trailing stop: its trigger follows the highest price at trailingPercent below it
                kids.append(Order(action='SELL', totalQuantity=qty, orderType='TRAIL', trailingPercent=round(100 * float(o['trailRel']), 2), trailStopPrice=stop, orderRef=ref, tif='DAY'))
            elif stop:
                kids.append(StopOrder('SELL', qty, stop, orderRef=ref, tif='DAY'))
            if take:
                kids.append(LimitOrder('SELL', qty, take, orderRef=ref, tif='DAY'))
            for k in kids:
                k.ocaGroup, k.ocaType, k.transmit = group, 2, True
                ib.placeOrder(c, k)
            if trail:
                trail_ids[con] = kids[0].orderId
            how = (f'trailing stop {100 * float(o["trailRel"]):.2f}% below the high (from {stop})' if trail else
                   f'stop {stop}' + (f' (to the entry once up {100 * float(o["breakevenRel"]):.2f}%)' if o.get('stopMode') == 'breakeven' else ''))
            print(f'  {sym}: bought {n:g} at {paid:.4f}: {how}' + (f', take-profit {take}' if take else ', no take-profit') + f' for {qty} (one cancelling the other)')

    def move_to_breakeven():
        """A break-even stop (stopMode breakeven) moved up to the price paid once the share has traded
        breakevenRel above it since the fill: the day's high in IBKR's quote (the auction is the day's
        first trade, so the high is after the fill; a delayed quote moves it late, never early)."""
        live = [t for t in ib.reqAllOpenOrders() if t.order.orderRef == ref and t.order.action == 'SELL' and t.order.orderType == 'STP'
                and t.orderStatus.status not in DEAD + ('Filled', 'PendingCancel')]
        for con, (sym, c, paid, o, ticks) in entries.items():
            if o.get('stopMode') != 'breakeven' or not o.get('breakevenRel') or con in moved:
                continue
            stp = next((t for t in live if t.contract.conId == con), None)
            if stp is None:
                continue
            to = round_to(paid, tick_at(ticks[0], ticks[1], paid), 'up')
            if stp.order.auxPrice >= to - 1e-9:
                moved.add(con)
                continue
            if con not in quotes:
                ib.reqMarketDataType(3)  # real time where subscribed, else delayed (free)
                quotes[con] = ib.reqMktData(c, '', False, False)
            q = quotes[con]
            hi = next((x for x in (q.high, q.last) if isinstance(x, (int, float)) and math.isfinite(x) and x > 0), None)
            if hi is None or hi < paid * (1 + float(o['breakevenRel'])):
                continue
            stp.order.auxPrice = to
            ib.placeOrder(stp.contract, stp.order)
            moved.add(con)
            now = next((x for x in (q.last, q.bid) if isinstance(x, (int, float)) and math.isfinite(x) and x > 0), None)
            print(f'  {sym}: up {100 * (hi / paid - 1):.2f}% since bought at {paid:.4f}: its stop moved to {to} (break-even)'
                  + (f'; the price now {now}, under it: it sells' if now is not None and now <= to else ''))

    def expire_open_buys():
        """After the opening auction, an order for the open still working (a day order: its good-till
        time refused) is cancelled: it buys in the auction or not at all."""
        if datetime.now(tz) < auction_over:
            return
        for t in ib.reqAllOpenOrders():
            o = t.order
            if o.orderRef == ref and o.action == 'BUY' and not o.parentId and t.contract.symbol in at_open and t.orderStatus.status not in DEAD + ('Filled', 'PendingCancel'):
                ib.cancelOrder(o)
                print(f'  {t.contract.symbol}: the opening auction is over: its buy order cancelled ({float(t.orderStatus.filled or 0):g} bought)')

    if capped:
        ib.execDetailsEvent += cap_check  # at once on each fill, and each round below
        ib.reqAllOpenOrders()
    while datetime.now(tz) < until:
        cap_check()
        fills = ib.reqExecutions()
        if at_open:
            expire_open_buys()
            set_levels(fills)
            move_to_breakeven()
        for tr in ib.reqCompletedOrders(True):
            o, c = tr.order, tr.contract
            # A completed order's status is all IBKR sends with it: its quantity comes from its fills.
            if o.orderRef != ref or o.action != 'SELL' or o.orderType != 'STP' or tr.orderStatus.status != 'Filled':
                continue
            key = str(o.permId)
            if key in handled or c.symbol not in plan:
                continue
            sold = sum(x.execution.shares for x in fills if x.execution.permId == o.permId)
            buys = [x for x in fills if x.contract.conId == c.conId and x.execution.side == 'BOT' and x.execution.orderRef == ref]
            n = sum(x.execution.shares for x in buys)
            if not sold or not n:
                continue  # the fills not in yet: the next round
            handled.add(key)
            with open(state, 'a') as f:
                f.write(key + '\n')
            p = plan[c.symbol]
            paid = sum(x.execution.shares * x.execution.price for x in buys) / n
            level = float(o.auxPrice)
            back, out = level + p['rebuy'] * (paid - level), paid * (1 + (p.get('rebuyCost') or 0))
            contract = c if qualified(ib, c) else c
            details = ib.reqContractDetails(contract)[0]
            rule = tick_rule(ib, details, contract.exchange)
            spread = math.exp(p.get('spread') or 0.0)  # the share's estimated spread (a factor): the most the buy may pay above its trigger
            back = round_to(back, tick_at(rule, details.minTick, back), 'up')
            cap = round_to(back * spread, tick_at(rule, details.minTick, back * spread), 'up')
            out = round_to(out, tick_at(rule, details.minTick, out), 'up')
            qty = int(sold)
            if not (level < back < out) or qty < 1:
                print(f'  {c.symbol}: buy-back levels out of order (stop {level}, back {back}, sale {out}); none')
                continue
            # A stop-limit: triggered when the price climbs back to its level, it buys at most a spread
            # above it (a plain stop would buy at any price in a thin market); unfilled, no buy-back.
            parent = StopLimitOrder('BUY', qty, cap, back, orderRef=ref, tif='DAY', transmit=False)
            ib.placeOrder(contract, parent)
            ib.placeOrder(contract, LimitOrder('SELL', qty, out, orderRef=ref, tif='DAY', parentId=parent.orderId, transmit=True))
            mine.append(parent)
            placed += 1
            print(f'  {c.symbol}: stopped at {level} (paid {paid:.4f}): buy-back {qty} from {back} (at most {cap}), sale at {out}')
        # every 5 seconds around the open (the auction's fills get their levels at once), else every 20
        ib.sleep(5 if at_open and datetime.now(tz) < auction_over + timedelta(minutes=10) else 20)
    for q in quotes.values():
        ib.cancelMktData(q.contract)
    for t in ib.openTrades():  # its buy-backs not triggered: cancelled before the close
        if t.order.orderRef == ref and any(t.order.orderId == p.orderId for p in mine):
            ib.cancelOrder(t.order)
    ib.sleep(2)
    ib.disconnect()
    print(f'{args.market} buy-back: {placed} placed, {len(handled)} stops seen')


def diag(args):
    """The API's answer, the account (paper or live) and whether the market's real-time and historical
    data come through (they need the exchange's subscription), on one liquid share."""
    from ib_async import IB, Stock

    out = {'market': args.market}
    errors = []
    ib = IB()
    ib.errorEvent += lambda req, code, msg, contract: errors.append(f'{code}: {msg}') if code not in (2104, 2106, 2107, 2108, 2158) else None
    try:
        ib.connect(args.host, args.port, clientId=args.client_id + 3, timeout=10)
        if not ib.isConnected():
            raise ConnectionError('IB Gateway did not accept the API connection')
        ib.RequestTimeout = REQUEST_TIMEOUT
    except Exception as e:  # noqa: BLE001
        print(json.dumps({**out, 'api': False, 'error': str(e) or type(e).__name__}))
        return
    accounts = ib.managedAccounts()
    out.update(api=True, accounts=len(accounts), paper=all(a.startswith('D') for a in accounts) if accounts else None,
               serverTime=ib.reqCurrentTime().isoformat())
    contract = Stock({'UK': 'VOD', 'US': 'AAPL', 'APAC': 'BHP', 'EU': 'AIR'}[args.market], *VENUE[args.market])
    if qualified(ib, contract):
        ib.reqMarketDataType(1)
        k = ib.reqMktData(contract, '', False, False)
        ib.sleep(4)
        num = lambda x: x if isinstance(x, (int, float)) and math.isfinite(x) and x > 0 else None  # noqa: E731
        out.update(probe=contract.symbol, dataType=k.marketDataType, bid=num(k.bid), ask=num(k.ask), last=num(k.last), close=num(k.close))
        ib.cancelMktData(contract)
        try:
            out['historyBars'] = len(ib.reqHistoricalData(contract, '', '5 D', '1 day', 'TRADES', useRTH=True, timeout=20) or [])
        except Exception as e:  # noqa: BLE001
            out['historyBars'], out['historyError'] = 0, str(e)
    out['errors'] = errors[-5:]
    ib.disconnect()
    print(json.dumps(out))


def ping(args):
    """The bridge's watchdog: does IB Gateway answer, and does it still reach IBKR's servers? Its
    clock alone is not enough: a gateway whose session was taken over by a login elsewhere with the
    same user (6 Nov) still answers the clock but nothing that needs IBKR (contracts, orders,
    positions), so it also looks up one share's contract, within 20 seconds. Its own client (9), so it
    never meets the markets' clients."""
    from ib_async import IB, Stock

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=9, timeout=15)
        ib.RequestTimeout = 20
        ok = ib.reqCurrentTime() is not None
        if ok:
            ok = bool(ib.reqContractDetails(Stock('VOD', 'SMART', 'GBP', primaryExchange='LSE')))
            if not ok:
                print('no answer from IBKR\'s servers: the contract lookup returned nothing')
    except Exception as e:  # noqa: BLE001 - any failure means the API does not answer
        print(f'no answer: {type(e).__name__}')
        ok = False
    finally:
        ib.disconnect()
    print('API answers' if ok else 'API does not answer')
    sys.exit(0 if ok else 1)


def daily_bars(args):
    """IBKR's own daily bars (trades, regular hours) for the file's pool, to check the opens the model
    learns from against the exchange's: IBKR's open is the opening auction's price. A share its
    exchange gives no bars for is asked again through SMART (the exchange as its primary); when the
    first 3 of an exchange get none that way either, the rest of it is not asked again. The summary
    counts the shares without bars by IBKR's error (6 Nov: none of Paris's or Brussels's shares had
    bars, Amsterdam's had)."""
    import collections
    import csv
    import time
    from ib_async import IB, Stock

    if not args.out:
        sys.exit('--out is required with --bars')
    book = json.load(open(args.orders))
    pool = book.get('pool') or [c['ticker'] for c in book.get('candidates', book['orders'])]
    if args.pause:  # a step holds the gateway already: wait before even connecting
        while os.path.exists(args.pause):
            open(args.pause + '.ack', 'w').close()
            time.sleep(2)
        if os.path.exists(args.pause + '.ack'):
            os.remove(args.pause + '.ack')
    ib = IB()
    connect(ib, args, args.client_id + 2)
    last_error = {}
    ib.errorEvent += lambda req, code, msg, *a: last_error.update(code=code, msg=str(msg)[:80]) if code not in (2104, 2106, 2107, 2108, 2158, 2119) else None
    got = missing = via_smart = 0
    why = collections.Counter()
    smart_tries = collections.Counter()  # exchange -> SMART retries that got nothing
    smart_works = set()

    def fetch(contract):
        last_error.clear()
        try:
            return ib.reqHistoricalData(contract, '', args.duration or f'{args.years} Y', '1 day', 'TRADES', useRTH=True, timeout=60) or []
        except Exception as e:  # noqa: BLE001
            last_error.update(code='exception', msg=type(e).__name__)
            return []

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
        paused_for = 0.0
        for k, t in enumerate(pool):
            if args.pause and os.path.exists(args.pause):  # another step needs the gateway: wait, quiet, between two shares
                f.flush()
                t0 = time.monotonic()
                open(args.pause + '.ack', 'w').close()
                while os.path.exists(args.pause):
                    ib.sleep(2)
                try:
                    os.remove(args.pause + '.ack')
                except OSError:
                    pass
                paused_for += time.monotonic() - t0
            start = time.monotonic()
            contract = contract_of(t, args.market)
            bars = []
            if qualified(ib, contract):
                bars = fetch(contract)
                ex = contract.exchange
                if not bars and contract.exchange != 'SMART' and (ex in smart_works or smart_tries[ex] < 3):
                    time.sleep(max(0.0, 10.5 - (time.monotonic() - start)))
                    alt = Stock(contract.symbol, 'SMART', contract.currency, primaryExchange=ex)
                    if qualified(ib, alt):
                        bars = fetch(alt)
                    if bars:
                        smart_works.add(ex)
                        via_smart += 1
                    else:
                        smart_tries[ex] += 1
                if not bars:
                    why[f"{ex} {last_error.get('code', 'no error')}: {last_error.get('msg', 'no bars')}"] += 1
            else:
                why['not found at IBKR'] += 1
            for b in bars:
                w.writerow([b.date.isoformat(), t, b.open, b.high, b.low, b.close, b.volume])
            got, missing = got + bool(bars), missing + (not bars)
            if (k + 1) % 50 == 0:
                sys.stderr.write(f'  {k + 1} of {len(pool)}\n')
                f.flush()
            if k + 1 < len(pool):
                time.sleep(max(0.0, 10.5 - (time.monotonic() - start)))
    ib.disconnect()
    print(f'{args.market} bars: {got} shares from IBKR ({via_smart} through SMART), {missing} without bars'
          + (f'; paused {paused_for / 60:.0f} minutes for the bridge\'s other steps' if paused_for >= 30 else '')
          + (': ' + '; '.join(f'{n} {k}' for k, n in why.most_common(4)) if why else ''))


if __name__ == '__main__':
    main()
