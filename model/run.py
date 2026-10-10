"""Backtest report for the orthonormal shrinkage factor model (results only, no prices).

    python model/run.py data/LSE.csv [--buy-bps 60] [--sell-bps 10]

Pass mark, fixed before any result: the model must beat the market (equal weight, bought once
and held) after costs on Sharpe ratio over the whole period and in each half.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orthofactor import SIGNALS, load_csv, metrics, run  # noqa: E402


def table(res, lo: int, hi: int, title: str) -> dict[str, dict[str, float]]:
    print(f"\n{title}: {res.dates[lo].date()} .. {res.dates[hi - 1].date()} ({(hi - lo) / 252:.1f} years)")
    print(f"  {'strategy':<32} {'ann.ret':>8} {'vol':>7} {'Sharpe':>7} {'max DD':>7}")
    out = {}
    for name, r in res.returns.items():
        m = metrics(r[lo:hi])
        out[name] = m
        print(f"  {name:<32} {m['ann_return']:>7.1%} {m['ann_vol']:>6.1%} {m['sharpe']:>7.2f} {m['max_dd']:>6.1%}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--buy-bps", type=float, default=60.0, help="dealing plus stamp duty on purchases")
    ap.add_argument("--sell-bps", type=float, default=10.0)
    a = ap.parse_args()

    close, volume = load_csv(a.csv)
    print(f"data: {close.shape[1]} stocks, {close.shape[0]} days, {close.index[0].date()} .. {close.index[-1].date()}")
    print(f"costs: {a.buy_bps:.0f} bp on purchases, {a.sell_bps:.0f} bp on sales")
    res = run(close, volume, a.buy_bps, a.sell_bps)
    n = len(res.dates)
    full = table(res, 0, n, "Whole period")
    first = table(res, 0, n // 2, "First half")
    second = table(res, n // 2, n, "Second half")

    print(f"\n  {'strategy':<32} {'turnover/yr':>11} {'stocks held':>11}")
    for name in res.returns:
        print(f"  {name:<32} {res.turnover[name]:>10.2f}x {res.holdings[name]:>11.1f}")

    print("\nCalendar years (net return)")
    names = list(res.returns)
    print("  year  " + "  ".join(f"{s[:22]:>22}" for s in names))
    years = res.dates.year
    for y in sorted(set(years)):
        sel = years == y
        print(f"  {y}  " + "  ".join(f"{np.prod(1 + res.returns[s][sel]) - 1:>22.1%}" for s in names))

    print("\nFactors at the last month-end (orthonormal; mean monthly factor return, t, shrunk premium)")
    last = res.raw_premia.index[-1]
    for k in SIGNALS:
        used = (res.premia[k] != 0).mean()
        print(f"  {k:<16} mean {res.raw_premia.loc[last, k]:+.4f}  t {res.t_stats.loc[last, k]:+5.2f}  "
              f"shrunk {res.premia.loc[last, k]:+.4f}  (tilted in {used:.0%} of months)")
    if res.persistence:
        print(f"  persistence of expected returns month to month: {np.mean(res.persistence):.2f}")

    ok = all(full["model"]["sharpe"] > full["market (buy and hold)"]["sharpe"] and True for _ in [0]) and \
        first["model"]["sharpe"] > first["market (buy and hold)"]["sharpe"] and \
        second["model"]["sharpe"] > second["market (buy and hold)"]["sharpe"]
    print(f"\nPass mark (beats the market on Sharpe ratio over the whole period and in both halves): {'PASS' if ok else 'FAIL'}")
    print("Caveats: the universe is today's most traded shares that have the history (survivorship bias,"
          " which flatters long-only results); prices exclude dividends; past results do not predict future ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
