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
   the same GitHub token as the daily job's `EOD_TOKEN` secret (repository svr-net/eod, Contents read
   and write) and fetch it with
   ```
   curl -fsS -H "Authorization: Bearer $EOD_TOKEN" \
     https://raw.githubusercontent.com/svr-net/eod/main/orders/UK.json -o UK.json
   ```

## 2b. Or: IB Gateway in Docker, with buttons on the page

`docker/ibkr` runs IB Gateway (paper, logged in automatically by the community image
`ghcr.io/gnzsnz/ib-gateway`) and a small bridge (`tools/ibkr/bridge.py`) on your own computer. The
page then shows a **Your broker** panel (only on that computer): the day's orders with their prices
and quantities, and buttons to check (dry run), place them and close at the close.

1. Install Docker Desktop (Windows, Mac) or Docker Engine (Linux).
2. In the repository: `cd docker/ibkr && cp .env.example .env`, then fill in `.env`: the paper
   account's username and password, a long random `BRIDGE_KEY`, and the GitHub token `EOD_TOKEN`
   `EOD_TOKEN` (step 2.4).
3. `docker compose up -d --build`. The gateway takes a minute or two to log in; `docker compose logs
   -f ib-gateway` shows it (with `VNC_SERVER_PASSWORD` set, a VNC viewer at 127.0.0.1:5900 shows its
   screen).
4. Open https://svr-net.github.io/trading/ on the same computer: the **Your broker** panel appears
   under the backtest. Enter the `BRIDGE_KEY` once (kept in this browser), switch to UK or US, check
   with **Check (dry run)**, then **Place orders** before the open and **Close at the close** before
   the close.

Both containers listen on 127.0.0.1 only; the bridge answers only the page's address and only with the
key. A paper account normally logs in without two-factor approval; a live account needs the approval
on your phone at each login (the image waits for it and restarts if it times out).

## 2c. Or: on a server (VPS), always on, with HTTPS

The same containers on a small Linux server, with **Caddy** in front of the bridge for HTTPS (a free
Let's Encrypt certificate, obtained and renewed automatically). Only Caddy is public; IB Gateway and
the bridge stay private on the server.

1. **A server**: any small Linux VPS (1 vCPU, 2 GB memory is enough; about GBP 4-6 a month at
   Hetzner, DigitalOcean, Vultr or AWS Lightsail), Ubuntu 24.04. A location in London or Western
   Europe keeps it near both IBKR and the LSE.
2. **A name for it**: an `A` record of a domain you own (e.g. `bridge.example.com`) pointing to the
   server's IP address, or a free subdomain from DuckDNS (`yourname.duckdns.org`).
3. **On the server** (as root or with sudo):
   ```
   apt update && apt install -y docker.io docker-compose-v2 git ufw
   ufw allow 22 && ufw allow 80 && ufw allow 443 && ufw --force enable
   git clone https://github.com/svr-net/trading.git && cd trading/docker/ibkr
   cp .env.example .env && nano .env      # paper login, BRIDGE_KEY, EOD_TOKEN, BRIDGE_DOMAIN
   docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d --build
   ```
   `docker compose logs -f caddy` shows the certificate being obtained (a minute after the DNS
   record is live); `docker compose logs -f ib-gateway` shows the login.
4. **The page**: open `https://svr-net.github.io/trading/?bridge=https://bridge.example.com` once
   (or *Broker bridge…* at the bottom of the page, then the address), enter the `BRIDGE_KEY`, Save.
   The browser keeps both; the panel then appears whenever the page is opened there.

Security: the bridge answers only the page's address and only with the key (a wrong key costs a
second, so a long random key cannot be guessed); use a key of 32 or more random characters
(`openssl rand -hex 24`). Keep the server updated (`apt upgrade`), and log in with an SSH key rather
than a password. With a **live** account IB Gateway needs the approval on your phone at each login
(at least weekly); start with the paper account.

## 2d. Or: on a NAS (setup script)

On a NAS with Docker (Synology Container Manager, QNAP Container Station, Unraid, TrueNAS, or any
Linux with Docker; x86-64 or 64-bit ARM), over SSH:

```
curl -fsSL https://raw.githubusercontent.com/svr-net/trading/main/docker/ibkr/setup-nas.sh -o setup-nas.sh
sh setup-nas.sh            # sudo sh setup-nas.sh if this user may not use Docker
```

It asks for the paper login, the GitHub token `EOD_TOKEN`, the amount a trade and how to reach the
bridge over HTTPS, makes a key, writes `ibkr-bridge/` (docker-compose.yml, Caddyfile, .env readable
only by you) and starts IB Gateway, the bridge and Caddy. The page cannot call a plain http address on
your network, so the bridge is always behind HTTPS:

- **lan** (default): private to your network. Caddy's own certificate authority signs a certificate
  for the NAS's address; install `ibkr-bridge/caddy-root.crt` once as a trusted root on each device
  you open the page on (the script prints how for Windows, macOS, iPhone/iPad and Android).
- **public**: a domain or DuckDNS name with a Let's Encrypt certificate; the router forwards outside
  ports 80 and 443 to the NAS (by default to 8080 and 8443, as NAS often use 80 and 443 themselves).

It ends with the page's address for your bridge (`https://svr-net.github.io/trading/?bridge=...`) and
the key to enter in *Your broker*. Run it again to change anything (delete `ibkr-bridge/.env` to answer
the questions again).

## 2e. Automatic placing and closing

The bridge can place and close the orders by itself on weekdays (`AUTO_MARKETS` in `.env`, asked by
the setup script: `UK`, `US`, `UK,US` or `none`): at 07:45 and 16:20 London time for the UK, 09:20 and
15:40 New York time for the US (`AUTO_UK_PLACE`, `AUTO_UK_CLOSE`, `AUTO_US_PLACE`, `AUTO_US_CLOSE`
change them), each once a day; an orders file more than 4 days old is not placed. The page's *Your
broker* line shows the schedule and the last automatic action; `docker compose logs bridge` shows
every one. IB Gateway must stay logged in (the container restarts it daily).

## 2f. History of orders and fills

The bridge keeps a history on the NAS (`docker/ibkr/nas/history/history.jsonl`, kept across updates):
the orders each time they are placed, and after each close (automatically at 16:45 London for the
UK, 16:10 New York for the US, or with *Record results*) the day's fills from IBKR: what was bought
and sold, at what prices, the return, IBKR's realised profit and loss after its commissions, and the
commissions. *Your broker* shows it as a table with the days, the fill rate, the share of winning
trades and the total. It is shown only where the bridge answers (your own devices); nothing of it is
published.

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
