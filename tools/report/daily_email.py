"""Daily report of the model portfolio: the day's result against the market, the running record,
and the orders for the next open. Results only (no prices).

  python tools/report/daily_email.py track  report/ofm.json web/data/track.json
      adds the scan's last day to the live track record (once per date)
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


def bp(x, d=2):
    return "–" if x is None else f"{1e4 * x:+.{d}f} bp"


def update_track(scan, track):
    recent = scan.get("recent", [])
    if not recent:
        return False
    day = recent[-1]
    if any(d["date"] == day["date"] for d in track["days"]):
        return False
    track["days"].append({"date": day["date"], "ret": day["ret"], "market": day.get("market")})
    track["days"].sort(key=lambda d: d["date"])
    return True


def compound(rets):
    w = 1.0
    for r in rets:
        w *= 1 + r
    return w - 1


def build(scan, track):
    recent = scan.get("recent", [])
    day = recent[-1] if recent else None
    date = day["date"] if day else scan["asOf"]
    orders = [e for e in scan["expected"] if e["order"] != 0]
    buys = [e for e in orders if e["order"] > 0]
    sales = [e for e in orders if e["order"] < 0]
    held = sum(1 for e in scan["expected"] if (e["held"] and e["order"] >= 0) or e["order"] > 0)
    whole = scan["parts"][0]["rows"]
    model = next(r for r in whole if r["name"] == "model")
    market = next(r for r in whole if r["name"].startswith("market"))
    live = track["days"]
    last5 = compound(x["ret"] for x in recent[-5:])
    last20 = compound(x["ret"] for x in recent[-20:])
    costs = scan["costs"]

    subject = (f"UK model portfolio · {date}: {pct(day['ret'] if day else None)} (market {pct(day.get('market') if day else None)})"
               f" · next open: {len(buys)} buy, {len(sales)} sell")
    lines = [subject, ""]
    lines.append(f"Day {date}: portfolio {pct(day['ret'] if day else None)}, market {pct(day.get('market') if day else None)} (net of costs)")
    lines.append(f"Last 5 days {pct(last5)}, last 20 days {pct(last20)}; live record since {live[0]['date'] if live else '–'}:"
                 f" {pct(compound(d['ret'] for d in live))} over {len(live)} days")
    lines.append(f"Backtest {scan['parts'][0]['from']} – {scan['parts'][0]['to']}: {pct(model['annualReturn'], 1)} a year, Sharpe"
                 f" {model['sharpe']:.2f}, max drawdown {100 * model['maxDrawdown']:.1f}%; market {pct(market['annualReturn'], 1)} a year,"
                 f" Sharpe {market['sharpe']:.2f}")
    lines.append("")
    lines.append(f"Orders at the next open (after the close of {scan['asOf']}); {held} stocks held after them:")
    for e in orders:
        lines.append(f"  {'BUY ' if e['order'] > 0 else 'SELL'} {e['ticker']:<6} expected {bp(e['daily'])} a day")
    if not orders:
        lines.append("  none: keep the holdings")
    lines.append("")
    lines.append(f"Costs {costs['buyBps'] / 100:.2f}% on purchases, {costs['sellBps'] / 100:.2f}% on sales. Fills at the open."
                 " Research, not investment advice. https://svr-net.github.io/trading/")
    text = "\n".join(lines)

    def table(head, rows):
        th = "".join(f"<th style='text-align:left;padding:4px 8px;border-bottom:1px solid #ccc'>{html.escape(h)}</th>" for h in head)
        body = "".join("<tr>" + "".join(f"<td style='padding:4px 8px;border-bottom:1px solid #eee'>{html.escape(str(v))}</td>" for v in r)
                       + "</tr>" for r in rows)
        return f"<table style='border-collapse:collapse;font:13px monospace'><tr>{th}</tr>{body}</table>"

    page = [f"<h2 style='font-family:sans-serif'>{html.escape(subject)}</h2>",
            "<p style='font-family:sans-serif'>" + "<br>".join(html.escape(x) for x in lines[2:5]) + "</p>",
            f"<h3 style='font-family:sans-serif'>Orders at the next open ({held} stocks held after them)</h3>",
            table(["Order", "Stock", "Expected, a day"], [["BUY" if e["order"] > 0 else "SELL", e["ticker"], bp(e["daily"])] for e in orders]
                  or [["none", "keep the holdings", ""]]),
            "<p style='font-family:sans-serif;color:#666'>" + html.escape(lines[-1]) + "</p>"]
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
        track = {"about": "Live record of the model portfolio, one entry per day (results only).", "days": []}
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
