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

--capital is the amount per trade (the account's currency); --port 4002 is IB Gateway's paper
account (4001 live, 7497/7496 TWS). --dry-run prints the orders without connecting. Every order
carries orderRef ofm-<market>-<date>, so --close touches only this model's orders.

Requires: pip install ib_async
"""
import argparse
import json
import math
import sys
from datetime import date


# Each market's exchange and currency at IBKR (US shares routed SMART); in markets of several
# exchanges (EU, APAC) a ticker is EXCHANGE:CODE, EODData's exchange, and takes that exchange's.
VENUE = {'UK': ('LSE', 'GBP'), 'US': ('SMART', 'USD'), 'APAC': ('ASX', 'AUD'), 'EU': ('SBF', 'EUR')}
EXCHANGE = {'PAR': ('SBF', 'EUR'), 'AMS': ('AEB', 'EUR'), 'BRU': ('ENEXT.BE', 'EUR'), 'LIS': ('BVL', 'EUR'), 'MSE': ('BM', 'EUR'),
            'FRA': ('IBIS', 'EUR'), 'OSL': ('OSE', 'NOK'), 'ASX': ('ASX', 'AUD'), 'SGX': ('SGX', 'SGD')}


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
    a.add_argument('--out', help='the CSV file for --bars')
    a.add_argument('--years', type=int, default=2, help='years of daily bars for --bars')
    a.add_argument('--diag', action='store_true', help="check the API, the account and the market's data; one JSON line")
    args = a.parse_args()
    ref = f'ofm-{args.market}-{date.today().isoformat()}'

    if args.diag:
        return diag(args)
    if args.auction:
        return auction(args)
    if args.bars:
        return daily_bars(args)
    if not args.close and not args.report:
        if not args.orders:
            sys.exit('--orders is required (or --close)')
        book = json.load(open(args.orders))
        print(f'{args.market}: {len(book["orders"])} orders for the open after the close of {book["asOf"]}')
        if args.dry_run:
            for o in book['orders']:
                q = math.floor(capital_of(o, args) / per_share(o, args.market)) if o['limit'] > 0 else 0
                print(f"  {ib_symbol(o['ticker'], args.market):8s} buy {q:6d} limit {o['limit']:.4f}  stop {o['stop']}  take {o['take']}")
            return

    from ib_async import IB, Stock, LimitOrder, StopOrder, MarketOrder  # imported here so --dry-run needs no install

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)

    if args.report:
        # Today's fills of this model's orders, a share at a time: bought, sold, average prices, the
        # return, the realised profit and loss IBKR reports for the closing fills (after its
        # commissions), and the commissions.
        per = {}
        for f in ib.fills():
            if f.execution.orderRef != ref:
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
        # Cancel this model's open orders of today, then sell each position they opened at the close.
        mine = [t for t in ib.openTrades() if t.order.orderRef == ref]
        for t in mine:
            ib.cancelOrder(t.order)
        filled = {}
        for f in ib.fills():
            if f.execution.orderRef == ref and f.execution.side == 'BOT':
                filled[f.contract.conId] = (f.contract, filled.get(f.contract.conId, (None, 0))[1] + f.execution.shares)
        held = {p.contract.conId: p.position for p in ib.positions()}
        for con, (contract, bought) in filled.items():
            qty = min(bought, held.get(con, 0))
            if qty > 0:
                o = MarketOrder('SELL', qty, orderRef=ref, tif='DAY')
                o.orderType = 'MOC'
                ib.placeOrder(contract, o)
                print(f'  {contract.symbol}: sell {qty} at the close')
        ib.sleep(2)
        ib.disconnect()
        return

    for o in book['orders']:
        contract = contract_of(o['ticker'], args.market)
        sym = o['ticker']
        if not ib.qualifyContracts(contract):
            print(f'  {sym}: not found at IBKR, skipped')
            continue
        details = ib.reqContractDetails(contract)[0]
        rule = tick_rule(ib, details, contract.exchange)
        tick = lambda px: tick_at(rule, details.minTick, px)  # noqa: E731 - the increment at a price
        # The price scale: IBKR's last close against ours (LSE prices can be in pence or pounds). Never
        # assumed: without IBKR's close the order is skipped, as a wrong scale would buy without a limit
        # and stop at once.
        ibkr_close = reference_close(ib, contract)
        if ibkr_close is None:
            print(f"  {sym}: no IBKR price to check the price scale against; skipped")
            continue
        r = ibkr_close / o['close']
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
        parent = LimitOrder('BUY', qty, lim, orderRef=ref, tif='DAY', transmit=False)
        ib.placeOrder(contract, parent)
        children = []
        if o.get('take'):
            children.append(LimitOrder('SELL', qty, round_to(o['take'] * scale, tick(o['take'] * scale), 'up'), orderRef=ref, tif='DAY', parentId=parent.orderId, transmit=False))
        if o.get('stop'):
            children.append(StopOrder('SELL', qty, round_to(o['stop'] * scale, tick(o['stop'] * scale), 'down'), orderRef=ref, tif='DAY', parentId=parent.orderId, transmit=False))
        if children:
            oca = f'{ref}-{sym}'
            for c in children:
                c.ocaGroup, c.ocaType = oca, 1
        for k, c in enumerate(children):
            c.transmit = k == len(children) - 1
            ib.placeOrder(contract, c)
        if not children:
            parent.transmit = True
            ib.placeOrder(contract, parent)
        print(f"  {sym}: buy {qty} limit {lim}" + ''.join(f', {c.action.lower()} {c.orderType} {getattr(c, "lmtPrice", None) or c.auxPrice}' for c in children))
    ib.sleep(2)
    ib.disconnect()


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
    ib.connect(args.host, args.port, clientId=args.client_id + 1)  # beside the orders' client
    ib.reqMarketDataType(1)  # real time
    num = lambda x: x if isinstance(x, (int, float)) and math.isfinite(x) and x > -1e300 else None  # noqa: E731
    tickers = {}
    for c in cands:
        contract = contract_of(c['ticker'], args.market)
        if ib.qualifyContracts(contract):
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
    except Exception as e:  # noqa: BLE001
        print(json.dumps({**out, 'api': False, 'error': str(e) or type(e).__name__}))
        return
    accounts = ib.managedAccounts()
    out.update(api=True, accounts=len(accounts), paper=all(a.startswith('D') for a in accounts) if accounts else None,
               serverTime=ib.reqCurrentTime().isoformat())
    contract = Stock({'UK': 'VOD', 'US': 'AAPL', 'APAC': 'BHP', 'EU': 'AIR'}[args.market], *VENUE[args.market])
    if ib.qualifyContracts(contract):
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
    ib.connect(args.host, args.port, clientId=args.client_id + 2)
    got = missing = 0
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
        for k, t in enumerate(pool):
            start = time.monotonic()
            contract = contract_of(t, args.market)
            bars = []
            if ib.qualifyContracts(contract):
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
