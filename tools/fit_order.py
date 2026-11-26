#!/usr/bin/env python3
"""Learns, from the store's 5-minute bars, which of a day's two bracket levels a price reaches first.

A daily bar does not say whether a day that touched both a stop under its open and a take-profit over it reached
the stop first. The 5-minute bars do: on each day with them, levels are drawn on both sides of the open (a fraction
u of the way from the open to the day's low, v of the way to its high), and the bars say which was reached first (a
5-minute bar that reached both says nothing, and is left out). A logistic regression learns the chance the stop came
first from what a daily bar shows:
    z = b0 + b1 * (open - low) / range + b2 * (close - low) / range + b3 * u + b4 * v
It is fitted walk-forward, once a year, on the days before that year only, so the engine (apps/ofm_strategy,
--order-model) uses for a decision only what was known then; a year with no earlier days has no model (an even
chance). Only the coefficients are written: no prices (the data licence).

    python3 tools/fit_order.py STORE_INTRADAY_DIR OUT.json
"""
import json
import math
import os
import random
import sys

import numpy as np


def days_of(path):
    with open(path) as f:
        d = json.load(f).get("days", {})
    for day, bars in d.items():
        if len(bars) >= 12:
            yield day, bars


def samples(bars, rng, k=6):
    hi = [b[1] for b in bars]
    lo = [b[2] for b in bars]
    H, L, C = max(hi), min(lo), bars[-1][3]
    O = (bars[0][1] + bars[0][2]) / 2  # the open: the first bar's middle (the bars carry no open)
    if not (H > O > L > 0):
        return
    for _ in range(k):
        u, v = rng.uniform(0.05, 1.0), rng.uniform(0.05, 1.0)
        sl, tk = O - u * (O - L), O + v * (H - O)
        js = next((j for j, x in enumerate(lo) if x <= sl), None)
        jt = next((j for j, x in enumerate(hi) if x >= tk), None)
        if js is None or jt is None or js == jt:
            continue
        yield [(O - L) / (H - L), (C - L) / (H - L), u, v], 1.0 if js < jt else 0.0


def fit(X, y, iters=50, ridge=1e-4):
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + ridge * w
        Hm = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-X @ w))
    ll = float(np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
    return w, ll


def main():
    root, out = sys.argv[1], sys.argv[2]
    rng = random.Random(7)
    by_year = {}
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if f.endswith(".json"))
    for path in files:
        for day, bars in days_of(path):
            yr = int(day[:4])
            for x, lab in samples(bars, rng):
                by_year.setdefault(yr, []).append((x, lab))
    years = sorted(by_year)
    model = {"about": __doc__.strip().splitlines()[0], "features": ["1", "(open-low)/range", "(close-low)/range", "stop depth u", "take depth v"], "years": {}}
    for y in range(years[0] + 1, years[-1] + 2):
        rows = [r for yy in years if yy < y for r in by_year[yy]]
        if len(rows) < 1000:
            continue
        X = np.array([r[0] for r in rows])
        lab = np.array([r[1] for r in rows])
        w, ll = fit(X, lab)
        model["years"][str(y)] = {"w": [round(float(a), 6) for a in w], "samples": len(rows), "stopFirst": round(float(lab.mean()), 4), "loglik": round(ll, 4)}
        print(f"{y}: {len(rows)} samples (days before {y}), stop first {lab.mean():.3f}, log-lik {ll:.4f} (an even chance: {math.log(0.5):.4f}), w {np.round(w, 3).tolist()}")
    with open(out, "w") as f:
        json.dump(model, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
