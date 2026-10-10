"""Factor model with futures inputs: report (results only, no prices).

    python model/run_futures.py data/LSE.csv data/futures.csv

Variants, all fixed before any result:
  A. the model as before;
  B. premia conditioned on market states from futures and currency series (index future trend,
     index future volatility, sterling trend);
  C. A with the index futures trend hedge;
  D. B with the hedge;
plus the market bought and held, with and without the hedge.
Pass mark, per variant: beat the market (bought and held, unhedged) on Sharpe ratio over the
whole period and in both halves. All variants are shown; none is picked after the fact.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from futures import STERLING, align, load_series, pick_index, states_at, trend_hedge  # noqa: E402
from orthofactor import load_csv, metrics, run  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stocks")
    ap.add_argument("futures")
    a = ap.parse_args()
    close, volume = load_csv(a.stocks)
    fut = load_series(a.futures)
    al = align(fut, close.index)
    print(f"stocks: {close.shape[1]}, {close.index[0].date()} .. {close.index[-1].date()}")
    for c in fut.columns:
        s = fut[c].dropna()
        print(f"futures {c:<14} {len(s):>5} days  {s.index[0].date() if len(s) else '-'} .. {s.index[-1].date() if len(s) else '-'}")
    index = pick_index(fut)
    print(f"index future used: {index or 'none (no index state, no hedge)'}; sterling: {'yes' if STERLING in fut.columns else 'no'}")

    with np.errstate(invalid="ignore", divide="ignore"):
        daily = close.to_numpy()[1:] / close.to_numpy()[:-1] - 1.0
    mkt_daily = np.concatenate([[np.nan], np.nanmean(daily, axis=1)])

    base = run(close, volume)
    cond = run(close, volume, states=lambda ends: states_at(al, ends, index))
    first_day = close.index.get_loc(base.dates[0]) - 1
    n = len(base.dates)
    series = {
        "market (bought and held)": base.returns["market (buy and hold)"],
        "market + futures hedge": trend_hedge(base.returns["market (buy and hold)"], first_day, al, mkt_daily, index)[0],
        "A. model": base.returns["model"],
        "B. model, futures states": cond.returns["model"],
    }
    hedged_a, h = trend_hedge(base.returns["model"], first_day, al, mkt_daily, index)
    series["C. model + futures hedge"] = hedged_a
    series["D. model, states + hedge"] = trend_hedge(cond.returns["model"], first_day, al, mkt_daily, index)[0]
    print(f"\nhedge on {h.mean():.0%} of days")

    ref = "market (bought and held)"
    results = {}
    for title, lo, hi in [("Whole period", 0, n), ("First half", 0, n // 2), ("Second half", n // 2, n)]:
        print(f"\n{title}: {base.dates[lo].date()} .. {base.dates[hi - 1].date()}")
        print(f"  {'strategy':<30} {'ann.ret':>8} {'vol':>7} {'Sharpe':>7} {'max DD':>7}")
        for name, r in series.items():
            m = metrics(r[lo:hi])
            results.setdefault(name, []).append(m["sharpe"])
            print(f"  {name:<30} {m['ann_return']:>7.1%} {m['ann_vol']:>6.1%} {m['sharpe']:>7.2f} {m['max_dd']:>6.1%}")

    print("\nState-conditioned premia at the last month-end (shrunk; 0 = no evidence)")
    print("  " + "  ".join(f"{k}: {v:+.4f}" for k, v in cond.premia.iloc[-1].items()))
    print("\nPass mark (beats the unhedged market on Sharpe ratio over the whole period and in both halves)")
    for name in [k for k in series if k[0] in "ABCD"]:
        ok = all(results[name][i] > results[ref][i] for i in range(3))
        print(f"  {name:<30} {'PASS' if ok else 'FAIL'}")
    print("Caveats: survivorship bias in the stock universe; prices exclude dividends; futures continuation series"
          " with roll days replaced by the stock market's return; past results do not predict future ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
