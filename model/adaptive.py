"""Adaptive parametrisation of the daily rules: the settings are chosen by their own records.

Every combination of stop-loss (2 or 3 x ATR, or none), take-profit (4 or 6 x ATR, or none) and
position count (10 or 20) runs side by side as a shadow book, with the market (equal weight,
bought once and held) as one more book. At each close the strategy follows the book whose daily
net returns so far have the best shrunk mean: the mean times max(0, 1 - 1/t^2), t being the
mean's t-statistic over the book's whole record (no window). Weak evidence shrinks to zero, and
ties go to the market, so with no evidence the strategy holds the market. Switching from one
book to another pays the actual trades between their holdings (purchases `buy_bps`, sales
`sell_bps`).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from daily import DailySpec, Prepared, simulate_daily

GRID = [(sl, tp, n) for sl in (2.0, 3.0, 1e9) for tp in (4.0, 6.0, 1e9) for n in (10, 20)]


def label(sl: float, tp: float, n: int) -> str:
    s = "no stop" if sl > 1e8 else f"stop {sl:g} ATR"
    t = "no take" if tp > 1e8 else f"take {tp:g} ATR"
    return f"{s}, {t}, {n} pos"


@dataclass
class AdaptiveResult:
    returns: np.ndarray
    choice: np.ndarray              # book followed each day (index into names)
    names: list[str]
    books: dict[str, np.ndarray]    # each book's own daily net returns
    switches: int
    switch_cost: float              # total cost of switching, as a fraction of equity


def shrunk_score(sum1: float, sum2: float, n: int) -> float:
    if n < 2:
        return 0.0
    m = sum1 / n
    var = max(sum2 / n - m * m, 0.0) * n / (n - 1)
    if var <= 0:
        return 0.0
    t2 = m * m / (var / n)
    return m * max(0.0, 1.0 - 1.0 / t2) if t2 > 0 else 0.0


def run_adaptive(pr: Prepared, base: DailySpec = DailySpec()) -> AdaptiveResult:
    buy, sell = base.buy_bps * 1e-4, base.sell_bps * 1e-4
    names, rets, weights = [], [], []
    market = None
    for sl, tp, n in GRID:
        r = simulate_daily(pr, replace(base, sl_atr=sl, tp_atr=tp, max_positions=n), keep_weights=True)
        names.append(label(sl, tp, n))
        rets.append(r.returns["model (daily, SL/TP)"])
        weights.append(r.weights)
        market = r.returns["market (buy and hold)"]
    # The market book: equal weight in the eligible stocks of the first day, drifting.
    T = len(market)
    first = np.isfinite(pr.E[pr.start])
    w = first / first.sum()
    mw = []
    for k in range(T):
        d = pr.start + 1 + k
        r = np.nan_to_num(pr.C[d] / pr.C[d - 1] - 1.0)
        w = w * (1 + r)
        mw.append(w / w.sum())
    names.append("market (buy and hold)")
    rets.append(market)
    weights.append(np.array(mw))
    R = np.array(rets)  # books x days
    W = weights
    B = len(names)
    market_idx = B - 1

    s1, s2 = np.zeros(B), np.zeros(B)
    held = market_idx  # no evidence yet: the market (its purchase is in its own first-day return)
    pending = 0.0      # cost of a switch decided at the last close, paid on the day it is made
    out, choice = [], []
    switches, cost_total = 0, 0.0
    for k in range(T):
        # Follow today the book chosen at yesterday's close.
        out.append(R[held, k] - pending)
        choice.append(held)
        pending = 0.0
        s1 += R[:, k]
        s2 += R[:, k] ** 2
        # Choose at today's close, from the records so far.
        scores = np.array([shrunk_score(s1[b], s2[b], k + 1) for b in range(B)])
        best = int(np.argmax(scores))
        if scores[best] <= scores[market_idx]:
            best = market_idx
        if best != held:
            d = W[best][k] - W[held][k]
            pending = buy * np.clip(d, 0, None).sum() + sell * np.clip(-d, 0, None).sum()
            cost_total += pending
            switches += 1
            held = best
    return AdaptiveResult(np.array(out), np.array(choice), names, {n: R[i] for i, n in enumerate(names)}, switches, cost_total)
