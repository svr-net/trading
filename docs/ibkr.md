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

## 2. The bridge: one container on a NAS

The orders are placed through **IB Gateway**, which stays logged in (IBKR requires an interactive
login, so it runs on your own NAS or Linux box, not on GitHub). One container holds it all
(`ghcr.io/svr-net/trading-ibkr`, `docker/ibkr/Dockerfile.ibkr`): IB Gateway logged in by IBC (the
community image `ghcr.io/gnzsnz/ib-gateway`) and the broker bridge beside it, in Java
(`tools/ibkr/java`): IBKR's official Java API, IBC's command server, HTTPS for the page, and the
automatic day. There is no Python in it.

On a NAS with Docker (Synology Container Manager, QNAP Container Station, Unraid, TrueNAS, or any
Linux with Docker; x86-64 or 64-bit ARM), over SSH:

```
curl -fsSL https://raw.githubusercontent.com/svr-net/trading/main/docker/ibkr/deploy-nas.sh -o deploy-nas.sh
sudo sh deploy-nas.sh
```

It clones the repository (default `/volume1/DEV/trading`), asks for the paper login, the GitHub token
`EOD_TOKEN` (the daily job's: repository svr-net/eod, Contents read and write), the amount a trade,
the automatic markets and how to reach the bridge over HTTPS, makes a key, keeps the answers in
`docker/ibkr/nas/.env` (readable only by you; delete it to answer again) and starts the container
(`docker/ibkr/ibkr-nas.sh`). The page cannot call a plain http address on your network, so the
bridge always answers HTTPS:

- **lan** (default): private to your network, with a certificate made for the NAS's address
  (`docker/ibkr/nas/tls/cert.pem`); trust it once on each device you open the page on.
- **synology**: the certificate DSM already holds for your name (e.g. `name.synology.me`, Control
  Panel, Security, Certificate); the router forwards one port (default 8765) to the NAS. Works from
  anywhere with no certificate to install, if the home connection takes incoming connections (not
  behind the provider's CGNAT). DSM renews it in place; the bridge loads the renewed one by itself.

It ends with the page's address for your bridge (`https://svr-net.github.io/trading/?bridge=...`) and
the key to enter in *Your broker*. The bridge answers only the page's address and only with the key (a
wrong key costs a second, so a long random key cannot be guessed). A paper account normally logs in
without two-factor approval; a live account needs the approval on your phone at each login.

Nothing is ever built on the NAS. GitHub builds the image on every change (stamped with its commit,
made only when the bridge's checks on a stand-in IBKR pass) and publishes it in its registry and as
one file per processor in the repository's `nas-images` release. The NAS takes the registry's (at most
3 minutes), else downloads the file over IPv4 and loads it. Updating the NAS is one command:
`sudo sh /volume1/DEV/trading/docker/ibkr/ibkr-nas.sh`. The monitor shows the version the NAS runs and
fails when it is not the latest.

## 2e. Automatic trading

The bridge trades by itself on weekdays (`AUTO_MARKETS` in `.env`: `UK`, `EU`, `US`, `APAC`, `INDEX`,
comma separated, or none). Every market's orders are a **multi-day book** (`"book": "hold"`, the
strategy engine `apps/ofm_strategy`): positions held over the days with a stop at IBKR. Every 30
seconds the bridge does whichever of the day's steps is due and not yet done:

- **leftovers** (from 30 minutes before the placing): the model's shares still held from an earlier
  day, outside the book, sold at the open unless a sale is already waiting; only shares the model's
  own orders placed in the last 10 days count, nothing else in the account is touched;
- **hold-place** (UK 07:45 London, `AUTO_<MARKET>_PLACE`; retried every 5 minutes, at most 3 times):
  the book's updates (each position kept with its new stop, or sold at the open) and its buys for the
  shares not held, good only through the opening auction; an orders file more than 4 days old is not
  placed; `config/nas-buys.json` turns the buying off;
- **hold-protect** (every 5 minutes through the session): a buy-back filled after a stop becomes a
  position with its own stop and buy-back at once;
- **close** and **hold-close** (UK 16:20, `AUTO_<MARKET>_CLOSE`; retried): the day's own orders
  closed, and the book's positions offered in the closing auction (sold at any price on their last
  day; where the exchange takes no LOC or MOC, a day limit instead);
- **flat** (30 minutes after the close): none of the model's shares held outside the book, or the
  step fails (the monitor turns red);
- **report** (UK 16:45, `AUTO_<MARKET>_REPORT`): the day's fills into the history.

Done steps are marked in `history/state`, so a restart or an update in the middle of the day catches
up rather than losing or repeating a step. A watchdog asks IB Gateway every minute for its clock and
one contract; when the port is open but the API has been silent for 3 minutes, the bridge restarts IB
Gateway through IBC's command server. A refusal by IBKR of the orders themselves (2139: paper trading
not acknowledged) restarts it too, at most every 20 minutes. `/status` and the diagnostics show
today's steps, the watchdog and the last actions; `docker logs trading-ibkr` shows every one.

## 2e2. State and data in the private store

The NAS folder is disposable. The bridge keeps its state in svr-net/eod too (`bridge/state.json`): the order history,
each multi-day book's positions and stops (`holds-<MARKET>.json`), the buying switch's last value, the recorded model
checks and the step marks of the last 7 days. It is uploaded within a minute of any change (and when the container
stops); a container started on an empty history folder restores it from there. `POST /reset {"confirm": "reset"}`
archives the records (into `history/archive-<time>/`; IBKR's bars stay) and empties the store's copy, so a reset is
never undone by a restore. `STATE_MIRROR=0` turns the mirror off. Every read from the store (the orders, the store's
bars and profiles for the charts, the shadow books) goes through a disk cache with GitHub's ETags: an unchanged file
is a 304, and when GitHub does not answer the last good copy is used (diagnostics: *State in the store*, *Store cache*).

## 2f. History of orders and fills

The bridge keeps a history on the NAS (`docker/ibkr/nas/history/history.jsonl`, kept across updates):
the orders each time they are placed, and after each close (automatically at 16:45 London for the
UK, 16:10 New York for the US, or with *Record results*) the day's fills from IBKR: what was bought
and sold, at what prices, the return, IBKR's realised profit and loss after its commissions, and the
commissions. *Your broker* shows it as a table with the days, the fill rate, the share of winning
trades and the total. It is shown only where the bridge answers (your own devices); nothing of it is
published.

## 2g. Opening auctions and IBKR's daily bars (research)

The bridge also collects data for improving the model; it places nothing.

- **Opening auctions**: each weekday (UK 07:52 London, US 09:24 New York) it records the opening
  auction of the day's 60 highest ranked candidates: the indicative price, volume and imbalance every
  15 seconds until the open, then the open. Kept in `history/auction` and in the private store as
  `auction/<MARKET>/<date>.json`. `AUCTION=0` in `.env` turns it off.
- **Daily bars**: on Saturdays (06:00) it fetches IBKR's own daily bars for the model's pool (two
  years; IBKR allows 60 requests in 10 minutes, so about 75 minutes for the UK) into
  `ibkr/<MARKET>/daily.csv` in the private store, to check the opens the model learns from. Start it
  by hand with `curl -X POST -H "X-Bridge-Key: <key>" -d '{"market":"UK"}' https://<bridge>/bars`.

Both need the exchange's real-time data on the account (for the UK: the LSE level 1 subscription,
a few pounds a month; a paper account shares the live account's subscriptions). Without it the
auction samples stay empty and the bars are refused.

## 2h. Before the first live day: settings the model assumes

- **Fixed pricing.** The backtest charges IBKR's fixed commissions, the account's (UK: GBP 3 an
  order up to GBP 6,000, 0.05% above, the exchange's and clearing fees included; US: USD 0.005 a
  share, at least USD 1). At GBP 5,000 a UK trade that is 0.06% each way. An account on tiered
  pricing runs the model with `--pricing tiered`.
- **The orders as the backtest trades them:** each buy good only through the opening auction, its stop
  a good-till-cancelled stop sell at the Parabolic SAR level, moved each morning; its buy-back coupled
  to the stop (a buy stop over the level the stop sold at); sold in the closing auction on its last day
  or when the strategy drops it.
- **Price units:** before placing, each order's price is checked against IBKR's close (its daily
  bars with the exchange's data, else its free delayed quote); an order whose units cannot be
  confirmed is skipped, never sent.

- **The day's paths (research):** from the auction's time to 20 minutes after the close, the bridge
  records the day's candidates' prices every 30 seconds from IBKR's free delayed data and writes them
  to the private store (`paths/<MARKET>/<date>.csv`; `PATHS=0` turns it off). Daily bars cannot tell
  whether a day's high came before or after its low; these paths settle, on the market's own shares,
  whether buying back after a stop pays (`build/ofm_paths paths/UK/*.csv`).

## 3. Each trading day

The daily job writes each market's orders into the private store after its close; the bridge places
them before the next open. On the page, *Your broker* shows them with their quantities, **Check (dry
run)** lists what the book would place (nothing is sent), **Place orders** places the book again
(buys already working are kept), **Close at the close** and **Record results** run those steps now,
and *Today at IBKR* shows every order and holding. Every order carries the reference
`ofm-<market>H-<date>` (its stop `ofm-<market>S-...`), so the bridge never touches anything else in
the account.

## 4. Before going live

- Run the paper account for at least a few weeks and keep the fills: compare each day's entries,
  exits and results with the page's backtest assumptions (fills at the open price, at the limit,
  the stop, the take-profit and the close).
- Then start live with a small `CAPITAL`, and raise it only if the live fills still match.
