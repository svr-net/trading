"""Places the model's next-open orders with Interactive Brokers (IB Gateway or TWS), and exits at the close.

The daily job writes each market's orders to the private store (svr-net/eod, orders/<MARKET>.json:
the last close, the buy limit, the stop-loss and the take-profit of every trade, in the exchange's
prices). This script runs next to a logged-in IB Gateway:

  before the open:  python tools/ibkr/submit.py --market UK --orders UK.json --capital 1000
                    each trade: a day limit buy at the limit (it takes part in the opening auction,
                    then rests during the day, as in the backtest), with its take-profit (limit
                    sell) and stop-loss (stop sell) attached, one cancelling the other;
  before the close: python tools/ibkr/submit.py --market UK --close
                    cancels today's unfilled orders and sells what is left at the close (MOC).

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
    a.add_argument('--market', choices=['UK', 'US'], required=True)
    a.add_argument('--orders', help='the orders file (orders/<MARKET>.json from svr-net/eod)')
    a.add_argument('--capital', type=float, default=1000.0, help='amount per trade, in the account currency')
    a.add_argument('--host', default='127.0.0.1')
    a.add_argument('--port', type=int, default=4002, help='4002 IB Gateway paper, 4001 live')
    a.add_argument('--client-id', type=int, default=17)
    a.add_argument('--close', action='store_true', help='cancel unfilled orders and sell what is left at the close')
    a.add_argument('--report', action='store_true', help="print today's fills of these orders as JSON (after the close)")
    a.add_argument('--dry-run', action='store_true', help='print the orders; do not connect')
    args = a.parse_args()
    ref = f'ofm-{args.market}-{date.today().isoformat()}'

    if not args.close and not args.report:
        if not args.orders:
            sys.exit('--orders is required (or --close)')
        book = json.load(open(args.orders))
        print(f'{args.market}: {len(book["orders"])} orders for the open after the close of {book["asOf"]}')
        if args.dry_run:
            for o in book['orders']:
                q = math.floor(args.capital / per_share(o, args.market)) if o['limit'] > 0 else 0
                print(f"  {ib_symbol(o['ticker'], args.market):8s} buy {q:6d} limit {o['limit']:.4f}  stop {o['stop']}  take {o['take']}")
            return

    from ib_async import IB, Stock, LimitOrder, StopOrder, MarketOrder  # imported here so --dry-run needs no install

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)
    exch, cur = ('LSE', 'GBP') if args.market == 'UK' else ('SMART', 'USD')

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
        sym = ib_symbol(o['ticker'], args.market)
        contract = Stock(sym, exch, cur)
        if not ib.qualifyContracts(contract):
            print(f'  {sym}: not found at IBKR, skipped')
            continue
        details = ib.reqContractDetails(contract)[0]
        tick = details.minTick or 0.01
        # The price scale: IBKR's last close against ours (LSE prices can be in pence or pounds).
        bars = ib.reqHistoricalData(contract, '', '2 D', '1 day', 'TRADES', useRTH=True)
        scale = 1.0
        if bars:
            r = bars[-1].close / o['close']
            scale = next((s for s in (1.0, 0.01, 100.0) if abs(math.log(r / s)) < 0.05), None)
            if scale is None:
                print(f"  {sym}: IBKR's close {bars[-1].close} does not match ours {o['close']}; skipped")
                continue
        lim = round_to(o['limit'] * scale, tick, 'down')
        qty = math.floor(args.capital / per_share(o, args.market)) if lim > 0 else 0
        if qty < 1:
            print(f'  {sym}: capital {args.capital} buys no share at {per_share(o, args.market)}; skipped')
            continue
        parent = LimitOrder('BUY', qty, lim, orderRef=ref, tif='DAY', transmit=False)
        ib.placeOrder(contract, parent)
        children = []
        if o.get('take'):
            children.append(LimitOrder('SELL', qty, round_to(o['take'] * scale, tick, 'up'), orderRef=ref, tif='DAY', parentId=parent.orderId, transmit=False))
        if o.get('stop'):
            children.append(StopOrder('SELL', qty, round_to(o['stop'] * scale, tick, 'down'), orderRef=ref, tif='DAY', parentId=parent.orderId, transmit=False))
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


if __name__ == '__main__':
    main()
