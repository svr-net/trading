"""Daily backtest report with stop-loss and take-profit (results only, no prices).

    python model/run_daily.py data/LSE.csv [--sl-atr 2] [--tp-atr 4] [--max-positions 20]
                              [--cooldown 5] [--buy-bps 60] [--sell-bps 10]

Pass mark, fixed before any result: the configured run must beat the market (equal weight,
bought once and held) on Sharpe ratio over the whole period and in each half. The sensitivity
table of other stop / take-profit levels is reported for information and decides nothing.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive import run_adaptive  # noqa: E402
from daily import DailySpec, prepare_daily, simulate_daily  # noqa: E402
from orthofactor import load_bars, metrics  # noqa: E402


def block(res, lo, hi, title):
    print(f"\n{title}: {res.dates[lo].date()} .. {res.dates[hi - 1].date()} ({(hi - lo) / 252:.1f} years)")
    print(f"  {'strategy':<26} {'ann.ret':>8} {'vol':>7} {'Sharpe':>7} {'max DD':>7}")
    out = {}
    for name, r in res.returns.items():
        m = metrics(r[lo:hi])
        out[name] = m
        print(f"  {name:<26} {m['ann_return']:>7.1%} {m['ann_vol']:>6.1%} {m['sharpe']:>7.2f} {m['max_dd']:>6.1%}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sl-atr", type=float, default=2.0)
    ap.add_argument("--tp-atr", type=float, default=4.0)
    ap.add_argument("--max-positions", type=int, default=20)
    ap.add_argument("--cooldown", type=int, default=5)
    ap.add_argument("--buy-bps", type=float, default=60.0)
    ap.add_argument("--sell-bps", type=float, default=10.0)
    ap.add_argument("--no-grid", action="store_true")
    a = ap.parse_args()
    spec = DailySpec(a.sl_atr, a.tp_atr, a.max_positions, a.cooldown, a.buy_bps, a.sell_bps)

    bars = load_bars(a.csv)
    c = bars["close"]
    print(f"data: {c.shape[1]} stocks, {c.shape[0]} days, {c.index[0].date()} .. {c.index[-1].date()}")
    print(f"rules: stop-loss {spec.sl_atr} x ATR(14), take-profit {spec.tp_atr} x ATR(14), up to {spec.max_positions} positions of "
          f"1/{spec.max_positions} equity, {spec.cooldown}-day cool-down after a stop; fills at the next open; "
          f"costs {spec.buy_bps:.0f} bp on purchases, {spec.sell_bps:.0f} bp on sales")
    pr = prepare_daily(bars, spec.min_months)
    res = simulate_daily(pr, spec)
    n = len(res.dates)
    full = block(res, 0, n, "Whole period")
    first = block(res, 0, n // 2, "First half")
    second = block(res, n // 2, n, "Second half")

    t = res.trades
    print(f"\nTrades: {len(t)} closed ({len(t) / (n / 252):.0f} a year); invested {res.exposure:.0%} of the time on average, "
          f"{res.positions:.1f} positions; traded {res.turnover:.1f}x equity a year")
    if t:
        r = np.array([x.ret for x in t])
        hold = np.array([x.exit_day - x.entry_day for x in t])
        print(f"  win rate {np.mean(r > 0):.0%}, mean win {r[r > 0].mean() if (r > 0).any() else 0:+.2%}, "
              f"mean loss {r[r <= 0].mean() if (r <= 0).any() else 0:+.2%}, mean holding {hold.mean():.0f} days")
        for reason in ["stop", "take", "signal"]:
            sel = np.array([x.reason == reason for x in t])
            if sel.any():
                print(f"  exits by {reason:<6}: {sel.sum():>5}  mean trade {r[sel].mean():+.2%}")

    print("\nCalendar years (net return)")
    names = list(res.returns)
    print("  year  " + "  ".join(f"{s:>26}" for s in names))
    for y in sorted(set(res.dates.year)):
        sel = res.dates.year == y
        print(f"  {y}  " + "  ".join(f"{np.prod(1 + res.returns[s][sel]) - 1:>26.1%}" for s in names))

    ok = all(x["model (daily, SL/TP)"]["sharpe"] > x["market (buy and hold)"]["sharpe"] for x in (full, first, second))
    print(f"\nPass mark (beats the market on Sharpe ratio over the whole period and in both halves): {'PASS' if ok else 'FAIL'}")

    if not a.no_grid:
        print("\nSensitivity (information only; does not decide the pass mark): whole-period Sharpe and annual return")
        for sl, tp in [(1.5, 3.0), (2.0, 4.0), (3.0, 6.0), (2.0, 1e9), (1e9, 1e9)]:
            r = simulate_daily(pr, replace(spec, sl_atr=sl, tp_atr=tp))
            m = metrics(r.returns["model (daily, SL/TP)"])
            label = f"stop {'none' if sl > 1e8 else f'{sl:g} ATR'}, take {'none' if tp > 1e8 else f'{tp:g} ATR'}"
            print(f"  {label:<28} Sharpe {m['sharpe']:5.2f}  ann.ret {m['ann_return']:6.1%}  max DD {m['max_dd']:5.1%}  trades {len(r.trades)}")
    print("\nAdaptive settings: each day follow the stop / take-profit / position-count book (or the market) with the best"
          " shrunk record so far; switching pays the trades between books")
    ad = run_adaptive(pr, spec)
    mk = res.returns["market (buy and hold)"]
    for title, lo, hi in [("whole period", 0, n), ("first half", 0, n // 2), ("second half", n // 2, n)]:
        m, b = metrics(ad.returns[lo:hi]), metrics(mk[lo:hi])
        print(f"  {title:<13} adaptive Sharpe {m['sharpe']:5.2f} ({m['ann_return']:6.1%} a year)   market {b['sharpe']:5.2f} ({b['ann_return']:6.1%})")
    used = np.bincount(ad.choice, minlength=len(ad.names)) / len(ad.choice)
    print(f"  {ad.switches} switches costing {ad.switch_cost:.1%} of equity in total; time in each book:")
    for i in np.argsort(-used)[:6]:
        if used[i] > 0:
            print(f"    {ad.names[i]:<34} {used[i]:5.0%}")
    ok_ad = all(metrics(ad.returns[lo:hi])["sharpe"] > metrics(mk[lo:hi])["sharpe"] for lo, hi in [(0, n), (0, n // 2), (n // 2, n)])
    print(f"  Pass mark for the adaptive version: {'PASS' if ok_ad else 'FAIL'}")
    print("Caveats: the universe is today's most traded shares that have the history (survivorship bias); prices exclude"
          " dividends; fills at the open and at stop levels assume no extra slippage; past results do not predict future ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
