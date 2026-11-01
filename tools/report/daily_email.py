"""Daily report of the daily trades: the day's result, the running record,
and the trades for the next open. Results only (no prices).

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
    entry = {"date": day["date"], "ret": day.get("ret", day.get("day"))}
    if day.get("day") is not None:
        entry["day"] = day["day"]
    track["days"].append(entry)
    track["days"].sort(key=lambda d: d["date"])
    return True


def compound(rets):
    w = 1.0
    for r in rets:
        w *= 1 + r
    return w - 1


def build_model(scan, track):
    """The email of the predictive model's scan (model 'v2'): the day, the next open's trades, the backtest."""
    recent = scan.get("recent", [])
    day = recent[-1] if recent else {}
    date = day.get("date", scan["asOf"])
    live = [d["day"] for d in track["days"] if d.get("day") is not None]
    orders = scan.get("orders", [])
    B = scan["backtest"]
    subject = (f"{os.environ.get('MARKET', 'UK')} daily trades · {date}: {pct(day.get('day'))}"
               f" · next open: {len(orders)} trade{'' if len(orders) == 1 else 's'}")
    lines = [f"Day {date}: daily trades {pct(day.get('day'))}",
             f"Live record: {pct(compound(live))} over {len(live)} days",
             f"Backtest {B['from']} – {B['to']}: {pct(B['annualReturn'], 1)} a year, Sharpe {B['sharpe']:.2f}, max drawdown"
             f" {100 * B['maxDrawdown']:.1f}%", "",
             f"Trades for the next open (after the close of {scan['asOf']}): buy at the open, sell at the close"]
    lines += [f"  BUY {o['ticker']:<6} expected {pct(o['expected'])}" for o in orders] or ["  none: stay in cash"]
    html = "<pre>" + "\n".join(lines) + "</pre>"
    return subject, "\n".join(lines), html


def build(scan, track):
    return build_model(scan, track)


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
        track = {"about": "Live record, one entry per day: ret the model portfolio, day the daily trades (results only).", "days": []}
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
