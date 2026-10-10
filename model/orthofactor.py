"""Orthonormal shrinkage factor model with cost-hurdle trading.

Each month-end, for every stock with a year of history:

1. Six signals from the factor literature, from prices and volumes up to that close only:
   12-1 momentum, 1-month reversal, low volatility, nearness to the 52-week high, small size
   (traded value), and trend quality (12-1 return over its volatility).
2. Forced orthonormality: the cross-sectional z-scores of the six signals are made exactly
   orthonormal by symmetric (Loewdin) orthogonalisation, S -> S (S'S)^(-1/2): the orthonormal
   set closest to the raw signals, independent of their order. The factors are then separate
   bets, and a stock's return projects onto them without double counting.
3. Self-adaptive premia: each factor's return per month (its orthonormal portfolio's return over
   the following month) is recorded once that month is over. Its premium is the mean of that
   record shrunk by its own evidence, mean * max(0, 1 - 1/t^2) (positive-part James-Stein), so
   weak evidence gives no tilt. The record is expanding: no window to set.
4. Expected excess return of each stock for the next month: E = S_orthonormal @ premia.
5. Cost-hurdle trading, starting from the market: the first portfolio holds every eligible stock
   in equal weight (no evidence = the market). Afterwards a held stock is sold when its expected
   return over its remaining life falls below the market's by more than a round trip costs, and
   a stock is bought when its expected return beats the market's by more than that. The
   life of an expected return is 1 / (1 - rho), rho being the measured month-to-month rank
   persistence of E. Bought stocks get an equal share; the rest keep their drifting weights.

Costs: `sell_bps` on sales, `buy_bps` on purchases (UK: 10 bp dealing + 50 bp stamp duty).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SIGNALS = ["momentum_12_1", "reversal_1m", "low_volatility", "near_52w_high", "small_size", "trend_quality"]
YEAR, MONTH, QUARTER = 252, 21, 63


def load_csv(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long CSV (date,ticker,open,high,low,close,volume) -> close and volume panels (dates x tickers)."""
    df = pd.read_csv(path, usecols=["date", "ticker", "close", "volume"])
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = df.pivot(index="date", columns="ticker", values="volume").reindex(close.index)
    close.index = pd.to_datetime(close.index)
    volume.index = close.index
    return close.astype(float), volume.astype(float)


def month_ends(index: pd.DatetimeIndex) -> np.ndarray:
    """Positions of the last trading day of each month."""
    s = pd.Series(np.arange(len(index)), index=index)
    return s.groupby([index.year, index.month]).max().to_numpy()


def signals_at(close: np.ndarray, volume: np.ndarray, p: int) -> np.ndarray:
    """Raw signals (stocks x 6) at day position p, NaN where a stock lacks the history."""
    c = close[p - YEAR : p + 1]  # 253 closes: one year of returns
    r = c[1:] / c[:-1] - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = c[-1 - MONTH] / c[0] - 1.0
        rev = -(c[-1] / c[-1 - MONTH] - 1.0)
        vol = np.nanstd(r, axis=0, ddof=1) if r.shape[0] > 1 else np.full(c.shape[1], np.nan)
        high = c[-1] / np.nanmax(c, axis=0)
        value = (close[p - QUARTER + 1 : p + 1] * volume[p - QUARTER + 1 : p + 1])
        size = -np.log(np.nanmedian(np.where(value > 0, value, np.nan), axis=0))
        vol_mom = np.nanstd(r[: -MONTH], axis=0, ddof=1) * np.sqrt(YEAR - MONTH)
        tq = mom / vol_mom
    out = np.column_stack([mom, rev, -vol, high, size, tq])
    # Every price in the year and enough trading: at most 20% of the last quarter without volume.
    complete = np.isfinite(c).all(axis=0)
    traded = (np.nan_to_num(volume[p - QUARTER + 1 : p + 1]) > 0).mean(axis=0) >= 0.8
    out[~(complete & traded)] = np.nan
    return out


def zscores(x: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks scaled to mean 0, sd 1, per column."""
    n = x.shape[0]
    ranks = np.argsort(np.argsort(x, axis=0), axis=0).astype(float)
    z = ranks - (n - 1) / 2.0
    sd = z.std(axis=0, ddof=0)
    return z / np.where(sd > 0, sd, 1.0)


def loewdin(s: np.ndarray) -> np.ndarray:
    """Symmetric orthonormalisation: the orthonormal columns closest to s."""
    g = s.T @ s
    w, v = np.linalg.eigh(g)
    w = np.clip(w, 1e-12 * max(w.max(), 1e-300), None)
    return s @ (v @ np.diag(w ** -0.5) @ v.T)


def shrunk_premia(history: np.ndarray, min_months: int = 12) -> np.ndarray:
    """Positive-part James-Stein shrinkage of each factor's mean return by its own t-statistic."""
    k = history.shape[1] if history.ndim == 2 else len(SIGNALS)
    if history.ndim != 2 or history.shape[0] < min_months:
        return np.zeros(k)
    m = history.mean(axis=0)
    se = history.std(axis=0, ddof=1) / np.sqrt(history.shape[0])
    t2 = np.where(se > 0, (m / np.where(se > 0, se, 1.0)) ** 2, 0.0)
    return m * np.clip(1.0 - 1.0 / np.where(t2 > 0, t2, np.inf), 0.0, None)


@dataclass
class Result:
    dates: pd.DatetimeIndex
    returns: dict[str, np.ndarray]           # daily net returns per strategy
    turnover: dict[str, float]               # sum of |trades| per year
    holdings: dict[str, float]               # mean number of stocks held
    premia: pd.DataFrame                     # shrunk premium per factor at each month-end
    raw_premia: pd.DataFrame                 # unshrunk mean per factor
    t_stats: pd.DataFrame
    persistence: list[float] = field(default_factory=list)


def run(close: pd.DataFrame, volume: pd.DataFrame, buy_bps: float = 60.0, sell_bps: float = 10.0, min_months: int = 12) -> Result:
    """Backtests the model and its benchmarks over the same days; everything decided at a
    month-end uses data up to that close only."""
    C, V = close.to_numpy(), volume.to_numpy()
    T, N = C.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = np.vstack([np.full((1, N), np.nan), C[1:] / C[:-1] - 1.0])
    ends = month_ends(close.index)
    ends = ends[ends >= YEAR]
    if len(ends) < min_months + 3:
        raise ValueError("not enough history: need a year for signals and a year of factor records")
    buy, sell = buy_bps * 1e-4, sell_bps * 1e-4

    # Pass 1: signals, orthonormal exposures and each factor's realised return per month.
    exposures, factor_returns = [], []
    for m, p in enumerate(ends):
        raw = signals_at(C, V, p)
        ok = np.isfinite(raw).all(axis=1)
        S = np.full((N, len(SIGNALS)), np.nan)
        if ok.sum() > 2 * len(SIGNALS):
            S[ok] = loewdin(zscores(raw[ok]))
        exposures.append(S)
        if m + 1 < len(ends):
            q = ends[m + 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                fwd = C[q] / C[p] - 1.0
            use = ok & np.isfinite(fwd)
            factor_returns.append(np.nan_to_num(S[use]).T @ fwd[use] if use.any() else np.zeros(len(SIGNALS)))

    # Pass 2: month by month, premia from the months already over, expected returns, trades.
    start_m = min_months  # the first month-end with min_months resolved factor returns
    first_day = ends[start_m]
    names = ["model", "market (buy and hold)", "momentum top decile (monthly)"]
    rets = {n: [] for n in names}
    turn = {n: 0.0 for n in names}
    held_count = {n: [] for n in names}
    w = {n: np.zeros(N) for n in names}
    premia_rows, raw_rows, t_rows, rho_hist = [], [], [], []
    prev_E = None

    def trade(name: str, target: np.ndarray) -> float:
        d = target - w[name]
        cost = sell * np.clip(-d, 0, None).sum() + buy * np.clip(d, 0, None).sum()
        turn[name] += np.abs(d).sum()
        w[name] = target
        return cost

    for m in range(start_m, len(ends)):
        p = ends[m]
        hist = np.array(factor_returns[:m])  # months that ended by this close
        prem = shrunk_premia(hist, min_months)
        premia_rows.append(prem)
        raw_rows.append(hist.mean(axis=0))
        se = hist.std(axis=0, ddof=1) / np.sqrt(len(hist))
        t_rows.append(np.where(se > 0, hist.mean(axis=0) / np.where(se > 0, se, 1.0), 0.0))
        S = exposures[m]
        ok = np.isfinite(S).all(axis=1)
        E = np.where(ok, np.nan_to_num(S) @ prem, 0.0)
        if prev_E is not None:
            both = ok & np.isfinite(prev_E)
            if both.sum() > 10 and np.std(E[both]) > 0 and np.std(prev_E[both]) > 0:
                rho_hist.append(float(np.corrcoef(zscores(E[both][:, None])[:, 0], zscores(prev_E[both][:, None])[:, 0])[0, 1]))
        prev_E = np.where(ok, E, np.nan)
        rho = float(np.clip(np.mean(rho_hist), 0.0, 0.95)) if rho_hist else 0.0
        life = 1.0 / (1.0 - rho)
        hurdle = buy + sell

        costs = {}
        if m == start_m:
            mk = ok / ok.sum()
            costs["model"] = trade("model", mk.astype(float))
            costs["market (buy and hold)"] = trade("market (buy and hold)", mk.astype(float))
        else:
            # Model: swaps that pay for themselves over the expected life of the signal.
            # E is each stock's expected return over the market (the cross-sectional mean), so
            # the comparison is with the market: holding a stock is kept unless it is expected to
            # lag the market by more than a round trip over its expected life, and a stock is
            # bought only if it is expected to beat the market by more than that.
            cur = w["model"] / max(w["model"].sum(), 1e-12)
            held = cur > 0
            exits = held & ok & (-E * life > hurdle)
            entries = ~held & ok & (E * life > hurdle)
            keep = held & ~exits
            n_new = int(keep.sum() + entries.sum())
            target = np.zeros(N)
            if n_new > 0:
                share = entries.sum() / n_new
                if keep.any():
                    target[keep] = cur[keep] / cur[keep].sum() * (1.0 - share)
                target[entries] = 1.0 / n_new
            else:
                target = cur
            w["model"] = cur
            costs["model"] = trade("model", target)
            w["market (buy and hold)"] = w["market (buy and hold)"] / max(w["market (buy and hold)"].sum(), 1e-12)
        # Momentum benchmark: equal weight in the top tenth by 12-1 momentum, rebalanced monthly.
        raw_mom = signals_at(C, V, p)[:, 0]
        good = np.isfinite(raw_mom) & ok
        mom_t = np.zeros(N)
        if good.sum() >= 10:
            cut = np.quantile(raw_mom[good], 0.9)
            top = good & (raw_mom >= cut)
            mom_t[top] = 1.0 / top.sum()
        cur_m = w["momentum top decile (monthly)"]
        w["momentum top decile (monthly)"] = cur_m / cur_m.sum() if cur_m.sum() > 0 else cur_m
        costs["momentum top decile (monthly)"] = trade("momentum top decile (monthly)", mom_t)

        # Hold until the next month-end (or the last day), weights drifting with prices.
        q = ends[m + 1] if m + 1 < len(ends) else T - 1
        for n in names:
            held_count[n].append(int((w[n] > 0).sum()))
            val = w[n].copy()
            first = True
            for d in range(p + 1, q + 1):
                r = np.nan_to_num(daily[d])
                before = val.sum()
                val = val * (1.0 + r)
                ret = val.sum() / before - 1.0 if before > 0 else 0.0
                if first:
                    ret -= costs.get(n, 0.0)
                    first = False
                rets[n].append(ret)
            w[n] = val
        if q == T - 1:
            break

    days = close.index[first_day + 1 : first_day + 1 + len(rets["model"])]
    years = len(days) / YEAR
    cols = SIGNALS
    idx = close.index[ends[start_m : start_m + len(premia_rows)]]
    return Result(
        dates=days,
        returns={n: np.array(v) for n, v in rets.items()},
        turnover={n: turn[n] / years for n in names},
        holdings={n: float(np.mean(held_count[n])) for n in names},
        premia=pd.DataFrame(premia_rows, index=idx, columns=cols),
        raw_premia=pd.DataFrame(raw_rows, index=idx, columns=cols),
        t_stats=pd.DataFrame(t_rows, index=idx, columns=cols),
        persistence=rho_hist,
    )


def metrics(r: np.ndarray) -> dict[str, float]:
    if len(r) < 2:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    wealth = np.cumprod(1.0 + r)
    years = len(r) / YEAR
    ann = wealth[-1] ** (1.0 / years) - 1.0
    vol = r.std(ddof=1) * np.sqrt(YEAR)
    peak = np.maximum.accumulate(wealth)
    return {"ann_return": ann, "ann_vol": vol, "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(YEAR) if r.std() > 0 else 0.0,
            "max_dd": float((1 - wealth / peak).max())}
