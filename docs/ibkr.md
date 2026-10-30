# Trading the model's orders with Interactive Brokers

Research only, not investment advice. Start with the paper account, and stay there until its
fills have been compared with the backtest for a few weeks (above all the UK opening-auction fills,
which the UK backtest depends on).

## 1. Open the account

1. Apply at interactivebrokers.co.uk (a UK resident is served by Interactive Brokers (U.K.)
   Limited). Choose an **individual** account. Have a passport or driving licence, a proof of
   address (bank statement or utility bill under 3 months old) and your National Insurance number.
2. **Cash or margin.** A *cash* account is enough: every trade is bought and sold the same day
   with one unit each, and nothing is borrowed. (A *margin* account under USD 25,000 is limited by
   the US pattern-day-trader rule to 3 day trades in 5 days on US shares.)
3. **Trading permissions:** stocks in the **United Kingdom** and the **United States**.
4. **Tax form:** the **W-8BEN** (non-US person), offered during the application, so US dividends
   are taxed at the treaty rate.
5. **Market data** (Settings, Market Data Subscriptions): live prices are not needed to place the
   orders, but the script checks each order's price against IBKR's last close, which works with
   the free delayed data. Optional: *UK LSE Equities* and *US Securities Snapshot and Futures
   Value Bundle* for live quotes.
6. **Paper account:** created with the live one (Settings, Paper Trading Account); it has its own
   username and starts with simulated money.
7. **Costs to expect** (the backtest assumes none): IBKR's commission on each order, and UK
   stamp duty of 0.5% on buying UK shares (not on AIM shares).

## 2. Connect the script

The orders are placed through **IB Gateway**, a small program that stays logged in (IBKR requires
an interactive login with two-factor approval, so it runs on your own computer or a small server,
not on GitHub).

1. Install IB Gateway (stable) from IBKR's website and log in with the **paper** username.
2. Configure, API, Settings: tick *Enable ActiveX and Socket Clients*, untick *Read-Only API*,
   socket port **4002** (paper; 4001 is live), and allow connections from 127.0.0.1 only.
3. On the same machine: Python 3.10+ and `pip install ib_async`.
4. The orders file: the daily job writes `orders/UK.json` and `orders/US.json` into the private
   repository svr-net/eod after each close (prices stay private under EODData's licence). Create a
   fine-grained GitHub token with **read-only Contents** access to svr-net/eod and fetch it with
   ```
   curl -fsS -H "Authorization: Bearer $EOD_READ_TOKEN" \
     https://raw.githubusercontent.com/svr-net/eod/main/orders/UK.json -o UK.json
   ```

## 3. Each trading day

| | UK (London time) | US (New York time) |
|---|---|---|
| Orders written by the daily job | about 19:15, the evening before | about 18:45, the evening before |
| Place them (before the opening auction) | by 07:50 | by 09:25 |
| Exit what is left at the close | 16:20 to 16:25 | 15:40 to 15:45 |

```
python tools/ibkr/submit.py --market UK --orders UK.json --capital 1000 --dry-run   # look first
python tools/ibkr/submit.py --market UK --orders UK.json --capital 1000             # place
python tools/ibkr/submit.py --market UK --close                                     # before the close
```

What it places for each trade, as in the backtest:

- a **limit buy at the model's buy limit**, valid for the day: it takes part in the opening auction
  (filled at the open when the share opens at or below the limit) and then rests (filled during the
  day if the price comes down to it);
- attached to it, the **take-profit** (a limit sell) and the **stop-loss** (a stop sell), one
  cancelling the other, active only once the buy has filled;
- with `--close`, the day's unfilled orders are cancelled and whatever is still held is **sold at
  the close** (market-on-close).

`--capital` is the amount per trade in the account's currency (one unit each, as in the backtest).
Before each order the script compares IBKR's last close with the model's: an order whose prices do
not match (a different share, a split IBKR has and EODData not) is skipped and reported. Every order
carries the reference `ofm-<market>-<date>`, so `--close` touches only the model's own orders.

## 4. Before going live

- Run the paper account for at least a few weeks and keep the fills: compare each day's entries,
  exits and results with the page's backtest assumptions (fills at the open price, at the limit,
  the stop, the take-profit and the close).
- Then start live with a small `--capital`, and raise it only if the live fills still match.
