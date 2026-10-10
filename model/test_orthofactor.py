"""Checks of the model on synthetic data:  python model/test_orthofactor.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orthofactor import loewdin, run, shrunk_premia, zscores  # noqa: E402


def synthetic(n_stocks=80, n_days=252 * 8, momentum=0.0, seed=1):
    """Prices with a market factor and, optionally, a planted 12-1 momentum premium."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    r = 0.0003 + 0.01 * rng.standard_normal((n_days, 1)) + 0.015 * rng.standard_normal((n_days, n_stocks))
    if momentum:
        # Next-day drift proportional to the past year's (excluding last month) cross-sectional rank.
        logp = np.zeros(n_stocks)
        hist = [logp.copy()]
        for t in range(n_days):
            if t > 252:
                past = hist[t - 21] - hist[t - 252]
                rank = (np.argsort(np.argsort(past)) / (n_stocks - 1)) - 0.5
                r[t] += momentum * rank
            logp = logp + np.log1p(r[t])
            hist.append(logp.copy())
    close = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=dates, columns=[f"S{i}" for i in range(n_stocks)])
    volume = pd.DataFrame(1e5 * (1 + rng.random((n_days, n_stocks))), index=dates, columns=close.columns)
    return close, volume


def test_loewdin_orthonormal_and_closest():
    rng = np.random.default_rng(0)
    s = zscores(rng.standard_normal((50, 6)) @ rng.standard_normal((6, 6)))
    o = loewdin(s)
    assert np.allclose(o.T @ o, np.eye(6), atol=1e-9)
    # Order independence: permuting the signals permutes the result.
    perm = [3, 1, 5, 0, 2, 4]
    assert np.allclose(loewdin(s[:, perm]), o[:, perm], atol=1e-9)


def test_shrinkage():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((200, 2)) * 0.05
    noise[:, 0] += 0.05  # strong premium in factor 0, none in factor 1
    p = shrunk_premia(noise)
    assert p[0] > 0.03 and abs(p[1]) < abs(noise[:, 1].mean()) + 1e-12
    assert np.all(shrunk_premia(noise[:5]) == 0)  # too few months: no tilt


def test_no_lookahead():
    close, volume = synthetic(40, 252 * 4, seed=3)
    a = run(close, volume)
    # Change the last 100 days: every return before them must be unchanged.
    c2 = close.copy()
    c2.iloc[-100:] *= np.linspace(1, 3, 100)[:, None]
    b = run(c2, volume)
    cut = len(a.returns["model"]) - 100
    for k in a.returns:
        assert np.allclose(a.returns[k][: cut - 1], b.returns[k][: cut - 1]), k


def test_finds_planted_momentum_and_stays_put_without_it():
    close, volume = synthetic(momentum=0.0015, seed=5)
    res = run(close, volume)
    assert res.premia["momentum_12_1"].iloc[-1] > 0 or res.premia["trend_quality"].iloc[-1] > 0
    gain = res.returns["model"].mean() - res.returns["market (buy and hold)"].mean()
    assert gain > 0, gain
    # No premium anywhere: the model should mostly stay in the market.
    close, volume = synthetic(momentum=0.0, seed=6)
    res = run(close, volume)
    assert res.turnover["model"] < 1.0, res.turnover["model"]


def _bars(close, volume):
    rng = np.random.default_rng(9)
    spread = 1 + 0.004 * rng.random(close.shape)
    return {"close": close, "open": close.shift(1).fillna(close) * (1 + 0.002 * rng.standard_normal(close.shape)),
            "high": close * spread, "low": close / spread, "volume": volume}


def test_daily_no_lookahead_and_stops_fill_correctly():
    from daily import DailySpec, run_daily
    close, volume = synthetic(30, 252 * 3 + 40, momentum=0.0015, seed=4)
    a = run_daily(_bars(close, volume), DailySpec())
    c2 = close.copy()
    c2.iloc[-30:] *= 1.5
    b = run_daily(_bars(c2, volume), DailySpec())
    cut = len(a.returns["model (daily, SL/TP)"]) - 30
    for k in a.returns:
        assert np.allclose(a.returns[k][: cut - 1], b.returns[k][: cut - 1]), k
    assert len(a.trades) > 0
    for t in a.trades:
        # A stop never fills above its entry, a take-profit never below it.
        if t.reason == "stop":
            assert t.exit <= t.entry
        if t.reason == "take":
            assert t.exit >= t.entry


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
