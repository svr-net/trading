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
    # The order's own amount (the model's, in its exchange's currency), else --capital.
    return float(o.get('capital') or args.capital)


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
    """The lowest price a forced sale accepts: the lowest low of the share's last 20 days (its own
    recent range, nothing fixed), on its price increments; None without IBKR's bars."""
    try:
        bars = ib.reqHistoricalData(contract, '', '1 M', '1 day', 'TRADES', useRTH=True, timeout=20) or []
    except Exception:  # noqa: BLE001
        bars = []
    lows = [b.low for b in bars[-20:] if b.low and b.low > 0]
    if not lows:
        return None
    lo = min(lows)
    return round_to(lo, tick_at(ticks[0], ticks[1], lo), 'down')


def sell_out(ib, contract, qty, ref, errors, at_close):
    """Sells qty: at the close (MOC) or at the next price (market; at the open when placed before
    it), and confirms IBKR took the order. Refused (a share too thin for a market order, 202; an
    order type the exchange does not take), it sells with a limit at the share's 20-day low instead,
    which trades at the auction's or the market's price above it. Returns (ok, what happened)."""
    from ib_async import MarketOrder, LimitOrder
    c, ticks = exit_contract(ib, contract)
    if c is None:
        return False, f'{contract.symbol}: not found at IBKR, NOT SOLD'
    o = MarketOrder('SELL', qty, orderRef=ref, tif='DAY')
    if at_close:
        o.orderType = 'MOC'
    t = ib.placeOrder(c, o)
    for _ in range(16):
        ib.sleep(0.5)
        if t.orderStatus.status in DEAD or t.orderStatus.status in ('Submitted', 'PreSubmitted', 'Filled'):
            break
    kind = 'at the close' if at_close else 'at market'
    if t.orderStatus.status not in DEAD:
        return True, f'{c.symbol}: sell {qty:g} {kind} ({t.orderStatus.status})'
    why = '; '.join(errors.get(o.orderId, [])) or t.orderStatus.status
    lim = floor_price(ib, c, ticks)
    if lim is None:
        return False, f'{c.symbol}: sell {kind} refused ({why}); no price for a limit: NOT SOLD'
    t2 = ib.placeOrder(c, LimitOrder('SELL', qty, lim, orderRef=ref, tif='DAY'))
    for _ in range(16):
        ib.sleep(0.5)
        if t2.orderStatus.status in DEAD or t2.orderStatus.status in ('Submitted', 'PreSubmitted', 'Filled'):
            break
    if t2.orderStatus.status in DEAD:
        return False, f"{c.symbol}: sell {kind} refused ({why}); limit {lim} refused ({'; '.join(errors.get(t2.order.orderId, [])) or t2.orderStatus.status}): NOT SOLD"
    return True, f'{c.symbol}: sell {kind} refused ({why}); sell {qty:g} limit {lim} instead ({t2.orderStatus.status})'


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
                                          'pnl': 0.0, 'open': None, 'fees': 0.0, 'note': ''})
    orders = {}
    for t in list(ib.reqAllOpenOrders()) + list(ib.reqCompletedOrders(False)):
        o = t.order
        if not (o.orderRef or '').startswith(ref):
            continue
        orders[o.permId] = (t.contract, o, t.orderStatus.status)
        r = row(t.contract)
        if o.action == 'BUY' and not o.parentId and o.orderType == 'LMT':
            r['limit'], r['buyStatus'] = num(o.lmtPrice), t.orderStatus.status
        elif o.action == 'SELL' and o.orderType == 'STP':
            r['stop'] = num(o.auxPrice)
        elif o.action == 'SELL' and o.orderType == 'LMT' and o.parentId:
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
            by = 'close' if (e.orderRef or '').endswith('-close') else {'STP': 'stop', 'LMT': 'take-profit'}.get(getattr(o, 'orderType', ''), 'sale')
            r['soldBy'] = ', '.join(sorted(set(filter(None, r['soldBy'].split(', '))) | {by}))
        cr = f.commissionReport
        if cr is not None:
            r['fees'] += num(cr.commission) or 0.0
            r['pnl'] += num(cr.realizedPNL) or 0.0
    for (tk, k), v in bv.items():
        r = rows[tk]
        r['buy' if k == 'bought' else 'sell'] = v / r[k] if r[k] else None
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
    a.add_argument('--rebuy', action='store_true', help="watch the day's stops and place their buy-backs")
    a.add_argument('--until', help='HH:MM, the exchange clock: --rebuy stops watching then')
    a.add_argument('--paths', action='store_true', help="record the candidates' prices through the day (places nothing)")
    a.add_argument('--years', type=int, default=2, help='years of daily bars for --bars')
    a.add_argument('--diag', action='store_true', help="check the API, the account and the market's data; one JSON line")
    a.add_argument('--leftovers', action='store_true', help="sell the model's shares still held from earlier days (--tickers)")
    a.add_argument('--tickers', help='with --leftovers: the shares the model placed lately, comma separated')
    a.add_argument('--state', action='store_true', help="the live state of today's trades and the model's holdings (--tickers), as JSON")
    a.add_argument('--check', action='store_true', help='with --leftovers: only list them; exit 1 when there are any')
    a.add_argument('--ping', action='store_true', help="exit 0 when IB Gateway's API answers within 15 seconds, else 1")
    args = a.parse_args()
    # Each market its own API clients (IBKR allows one connection per client id): the orders' client
    # (place, buy-back watch, close, report) and its helpers (+1 auction, +2 bars, +3 diag, +4 paths)
    # are 17.. for the UK, 27.. for the EU, 37.. for the US, 47.. for Asia-Pacific, so one market's
    # all-day buy-back watch never blocks another market's orders.
    args.client_id += 10 * ['UK', 'EU', 'US', 'APAC'].index(args.market)
    ref = f'ofm-{args.market}-{date.today().isoformat()}'

    if args.ping:
        return ping(args)
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
        per = {}
        for f in ib.fills():
            if not f.execution.orderRef.startswith(ref):  # the day's orders and the close's sales (ref-close)
                continue
            e = per.setdefault(f.contract.symbol, {'ticker': f.contract.symbol, 'currency': f.contract.currency,
                                                    'bought': 0.0, 'buy_value': 0.0, 'sold': 0.0, 'sell_value': 0.0, 'pnl': 0.0, 'commission': 0.0})
            side = 'bought' if f.execution.side == 'BOT' else 'sold'
            e[side] += f.execution.shares
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
                        'ret': (sell / buy - 1) if buy and sell else None, 'pnl': e['pnl'], 'commission': e['commission']})
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
    have = {t.contract.conId for t in ib.reqAllOpenOrders() if t.order.orderRef == ref and t.orderStatus.status not in dead}
    have |= {f.contract.conId for f in ib.reqExecutions() if f.execution.orderRef == ref}
    resent = set()

    def bracket(contract, sym, qty, lim, take, stop):
        """A day limit buy with its take-profit (limit sell) and stop-loss (stop sell) attached, one
        cancelling the other, active once the buy fills."""
        parent = LimitOrder('BUY', qty, lim, orderRef=ref, tif='DAY', transmit=False)
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
    for o in book['orders']:
        contract = contract_of(o['ticker'], args.market)
        sym = o['ticker']
        if not qualified(ib, contract):
            print(f'  {sym}: not found at IBKR, skipped')
            continue
        if contract.conId in have:
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
        bracket(contract, sym, qty, lim, take, stop)
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
    accepted = rejected = 0
    for t in ib.openTrades() + [t for t in ib.trades() if t.orderStatus.status in ('Cancelled', 'ApiCancelled', 'Inactive')]:
        if t.order.orderRef != ref or t.order.parentId:
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


def auction(args):
    """The candidates' opening auction, as IBKR streams it (generic tick 225: the indicative price,
    volume and imbalance), sampled every 15 seconds until the open, then the open itself. Needs the
    exchange's real-time data on the account (otherwise the samples stay empty, which is recorded)."""
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from ib_async import IB, Stock

    book = json.load(open(args.orders))
    cands = book.get('candidates') or book['orders']
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
    rows = {t: {'ticker': t, 'close': c['close'], 'score': c.get('score'), 'samples': []} for t, c in ((c['ticker'], c) for c in cands) if t in tickers}

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
    cands = book.get('candidates') or book['orders']
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
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from ib_async import IB, LimitOrder, StopLimitOrder

    book = json.load(open(args.orders))
    plan = {ib_symbol(o['ticker'].split(':')[-1], args.market): o for o in book['orders'] if (o.get('rebuy') or 0) > 0}
    if not plan:
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
    while datetime.now(tz) < until:
        fills = ib.reqExecutions()
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
        ib.sleep(20)
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
    """The bridge's watchdog: does IB Gateway's API answer? Its own client (9), so it never meets the
    markets' clients."""
    from ib_async import IB

    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=9, timeout=15)
        ok = ib.reqCurrentTime() is not None
    except Exception as e:  # noqa: BLE001 - any failure means the API does not answer
        print(f'no answer: {type(e).__name__}')
        ok = False
    finally:
        ib.disconnect()
    print('API answers' if ok else 'API does not answer')
    sys.exit(0 if ok else 1)


def daily_bars(args):
    """IBKR's own daily bars (trades, regular hours) for the file's pool, to check the opens the model
    learns from against the exchange's: IBKR's open is the opening auction's price."""
    import csv
    import time
    from ib_async import IB, Stock

    if not args.out:
        sys.exit('--out is required with --bars')
    book = json.load(open(args.orders))
    pool = book.get('pool') or [c['ticker'] for c in book.get('candidates', book['orders'])]
    ib = IB()
    connect(ib, args, args.client_id + 2)
    got = missing = 0
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
        for k, t in enumerate(pool):
            start = time.monotonic()
            contract = contract_of(t, args.market)
            bars = []
            if qualified(ib, contract):
                try:
                    bars = ib.reqHistoricalData(contract, '', f'{args.years} Y', '1 day', 'TRADES', useRTH=True, timeout=60) or []
                except Exception as e:  # noqa: BLE001
                    sys.stderr.write(f'  {t}: {e}\n')
            for b in bars:
                w.writerow([b.date.isoformat(), t, b.open, b.high, b.low, b.close, b.volume])
            got, missing = got + bool(bars), missing + (not bars)
            if (k + 1) % 50 == 0:
                sys.stderr.write(f'  {k + 1} of {len(pool)}\n')
                f.flush()
            if k + 1 < len(pool):
                time.sleep(max(0.0, 10.5 - (time.monotonic() - start)))
    ib.disconnect()
    print(f'{args.market} bars: {got} shares from IBKR, {missing} without bars')


if __name__ == '__main__':
    main()
