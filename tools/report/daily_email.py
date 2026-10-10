"""Daily report of the top-10 same-day trades: the day's booked trades against its bars, the
running record, and the trades to book at the next open. Results only (no prices).

  python tools/report/daily_email.py track  report/ofm.json web/data/track.json
      adds the scan's last completed day to the live track record (once per date)
  python tools/report/daily_email.py send   report/ofm.json web/data/track.json
      sends the report by SMTP; settings from the environment (repository secrets):
      SMTP_SERVER, SMTP_PORT (465: SSL, otherwise STARTTLS), SMTP_USERNAME, SMTP_PASSWORD,
      MAIL_TO (comma separated), MAIL_FROM (optional, defaults to SMTP_USERNAME).
      Without SMTP_SERVER it prints the report and sends nothing.
  python tools/report/daily_email.py print  report/ofm.json web/data/track.json
"""
import html
import json
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage


def pct(x, d=2):
    return "–" if x is None else f"{100 * x:+.{d}f}%"


def prob(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def strategy(scan):
    return next(s for s in scan["top10"]["strategies"] if s.get("dayTrades"))


def booked_last_day(s):
    return [c for c in s.get("lastDay", []) if c["exit"] != "still open"]


def update_track(scan, track):
    s = strategy(scan)
    date = s.get("lastDate")
    if not date or any(d["date"] == date for d in track["days"]):
        return False
    day = next((x for x in s.get("recent", []) if x["date"] == date), None)
    track["days"].append({
        "date": date,
        "ret": day["ret"] if day else 0.0,
        "trades": [{"ticker": c["ticker"], "exit": c["exit"], "ret": c["ret"], "stop": c["stop"], "take": c["take"]}
                   for c in booked_last_day(s)],
    })
    track["days"].sort(key=lambda d: d["date"])
    return True


def compound(rets):
    w = 1.0
    for r in rets:
        w *= 1 + r
    return w - 1


def build(scan, track):
    s = strategy(scan)
    costs = scan["costs"]
    date, as_of = s.get("lastDate"), scan["asOf"]
    today = booked_last_day(s)
    recent = s.get("recent", [])
    day = next((x for x in recent if x["date"] == date), None)
    plan = [p for p in s["plan"] if p["action"] == 1]
    whole = s["parts"][0]["rows"]
    policy, market = whole[0], next(r for r in whole if r["name"].startswith("market"))
    live = track["days"]
    live_ret = compound(d["ret"] for d in live)
    last5, last20 = compound(x["ret"] for x in recent[-5:]), compound(x["ret"] for x in recent[-20:])

    subject = (f"UK top-10 · {date}: {pct(day['ret'] if day else None)} ({len(today)} trade{'s' if len(today) != 1 else ''})"
               f" · {len(plan)} to book for the next open")

    lines = [subject, ""]
    lines.append(f"Trades of {date} (planned at the close before), against the day's bars, relative to the open:")
    for c in today:
        lines.append(f"  {c['ticker']:<6} stop {pct(c['stop'])} take {pct(c['take'])} | high {pct(c['high'])} low {pct(c['low'])}"
                     f" close {pct(c['close'])} | {c['exit']}, net {pct(c['ret'])}")
    if not today:
        lines.append("  none booked")
    lines.append(f"Day: {pct(day['ret'] if day else None)} (the 10 slots, cash slots earning nothing, after costs)")
    lines.append(f"Last 5 days {pct(last5)}, last 20 days {pct(last20)}; live record since {live[0]['date'] if live else '–'}:"
                 f" {pct(live_ret)} over {len(live)} days")
    lines.append(f"Backtest {s['parts'][0]['from']} – {s['parts'][0]['to']}: {pct(policy['annualReturn'], 1)} a year, Sharpe"
                 f" {policy['sharpe']:.2f}, max drawdown {100 * policy['maxDrawdown']:.1f}%; market {pct(market['annualReturn'], 1)} a year")
    lines.append("")
    lines.append(f"To book at the next open (after the close of {as_of}): buy at the open, sell at the stop, the take-profit or the close")
    for p in plan:
        lines.append(f"  {p['rank']:>2} {p['ticker']:<6} stop {pct(p['stop'])} take {pct(p['take'])} expected close {pct(p['close'])}"
                     f" | P(stop) {prob(p['pStopDay'])} P(take) {prob(p['pTakeDay'])}")
    if not plan:
        lines.append("  none: no stock's learnt levels beat cash after costs")
    lines.append("")
    lines.append(f"Costs {costs['buyBps'] / 100:.2f}% on purchases, {costs['sellBps'] / 100:.2f}% on sales. Levels relative to the open."
                 " Research, not investment advice. https://svr-net.github.io/trading/")
    text = "\n".join(lines)

    def table(head, rows):
        th = "".join(f"<th style='text-align:left;padding:4px 8px;border-bottom:1px solid #ccc'>{html.escape(h)}</th>" for h in head)
        body = "".join("<tr>" + "".join(f"<td style='padding:4px 8px;border-bottom:1px solid #eee'>{html.escape(str(v))}</td>" for v in r)
                       + "</tr>" for r in rows)
        return f"<table style='border-collapse:collapse;font:13px monospace'><tr>{th}</tr>{body}</table>"

    page = [f"<h2 style='font-family:sans-serif'>{html.escape(subject)}</h2>",
            f"<h3 style='font-family:sans-serif'>Trades of {date} against the day's bars</h3>",
            table(["Stock", "Stop", "Take", "High", "Low", "Close", "Exit", "Net"],
                  [[c["ticker"], pct(c["stop"]), pct(c["take"]), pct(c["high"]), pct(c["low"]), pct(c["close"]), c["exit"], pct(c["ret"])]
                   for c in today] or [["none booked", "", "", "", "", "", "", ""]]),
            f"<p style='font-family:sans-serif'>Day {pct(day['ret'] if day else None)} · last 5 days {pct(last5)} · last 20 days {pct(last20)}"
            f" · live record {pct(live_ret)} over {len(live)} days<br>Backtest: {pct(policy['annualReturn'], 1)} a year, Sharpe"
            f" {policy['sharpe']:.2f}, max drawdown {100 * policy['maxDrawdown']:.1f}% (market {pct(market['annualReturn'], 1)} a year)</p>",
            f"<h3 style='font-family:sans-serif'>To book at the next open (close of {as_of})</h3>",
            table(["Rank", "Stock", "Stop", "Take", "Expected close", "P(stop)", "P(take)"],
                  [[p["rank"], p["ticker"], pct(p["stop"]), pct(p["take"]), pct(p["close"]), prob(p["pStopDay"]), prob(p["pTakeDay"])]
                   for p in plan] or [["", "none", "", "", "", "", ""]]),
            "<p style='font-family:sans-serif;color:#666'>Buy at the open, sell at the stop, the take-profit or the close."
            f" Costs {costs['buyBps'] / 100:.2f}% on purchases, {costs['sellBps'] / 100:.2f}% on sales. Levels relative to the open."
            " Research, not investment advice. <a href='https://svr-net.github.io/trading/'>svr-net.github.io/trading</a></p>"]
    return subject, text, "".join(page)


def send(subject, text, page):
    server = os.environ.get("SMTP_SERVER")
    if not server:
        print("SMTP_SERVER not set: nothing sent\n")
        print(text)
        return
    port = int(os.environ.get("SMTP_PORT") or 465)
    user, password = os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"]
    msg = EmailMessage()
    msg["Subject"], msg["From"] = subject, os.environ.get("MAIL_FROM") or user
    msg["To"] = os.environ["MAIL_TO"]
    msg.set_content(text)
    msg.add_alternative(page, subtype="html")
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(server, port, context=context) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(server, port) as smtp:
            smtp.starttls(context=context)
            smtp.login(user, password)
            smtp.send_message(msg)
    print(f"sent: {subject}")


def main():
    mode, scan_path, track_path = sys.argv[1:4]
    with open(scan_path) as f:
        scan = json.load(f)
    try:
        with open(track_path) as f:
            track = json.load(f)
    except FileNotFoundError:
        track = {"about": "Live record of the top-10 same-day trades, one entry per day (results only).", "days": []}
    if mode == "track":
        if update_track(scan, track):
            with open(track_path, "w") as f:
                json.dump(track, f, indent=1)
            print(f"track: added {track['days'][-1]['date']}")
        else:
            print("track: unchanged")
        return
    subject, text, page = build(scan, track)
    if mode == "send":
        send(subject, text, page)
    else:
        print(text)


if __name__ == "__main__":
    main()
