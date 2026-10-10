"""Futures inputs for the factor model: market states and a stamp-duty-free index hedge.

Series (from tools/eoddata/fetch_series.mjs, tickers EXCHANGE:CODE). EODData keeps history for
continuation series (".C") but not for front-month codes, so the index future is the first of
INDEX_CHOICES with data: Stoxx Europe 600 (which includes UK shares), MSCI Europe, Euro Stoxx 50.

States at a month-end, from closes up to that day only (each 1 or 0, or -1 when unknown):
  - index trend: the index future's return over the past year is positive;
  - volatility: its 21-day realised volatility is above its median so far (no VIX history);
  - sterling: GBP/USD rose over the past quarter (no gilt history).

Hedge: when the index future's past-year return is negative at a close, the next day the stock
book is hedged one-for-one with a short index future (time-series momentum; Moskowitz, Ooi and
Pedersen, 2012). A European index future hedges UK shares only in part (basis risk). Changing the hedge costs `cost_bps` per unit of notional; futures pay
no stamp duty. Continuation series jump at contract rolls: a day whose futures return differs
from the stock market's by more than 4% is treated as a roll and given the stock market's return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INDEX_CHOICES = ["EUREX:FY.C", "EUREX:JD.C", "LIFFE:L7.C", "EUREX:FX.C"]
STERLING = "FOREX:GBPUSD"
INDEX = INDEX_CHOICES[0]  # replaced by pick_index()


def pick_index(series: pd.DataFrame) -> str | None:
    for c in INDEX_CHOICES:
        if c in series.columns and series[c].notna().sum() > 2 * YEAR:
            return c
    return None
YEAR, QUARTER = 252, 63


def load_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["date", "ticker", "close"])
    out = df.pivot(index="date", columns="ticker", values="close").sort_index().astype(float)
    out.index = pd.to_datetime(out.index)
    return out


def align(series: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Each series on the stock dates, carrying the last close forward (never backward)."""
    return series.reindex(series.index.union(dates)).sort_index().ffill().reindex(dates)


def clean_returns(fut: np.ndarray, market: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = fut[1:] / fut[:-1] - 1.0
    r = np.concatenate([[np.nan], r])
    roll = np.isfinite(r) & np.isfinite(market) & (np.abs(r - market) > 0.04)
    r[roll] = market[roll]
    return r


def states_at(aligned: pd.DataFrame, positions: np.ndarray, index: str | None) -> np.ndarray:
    """States (len(positions) x 3) at the given day positions."""
    out = np.full((len(positions), 3), -1, dtype=int)
    idx = aligned[index].to_numpy() if index in aligned.columns else None
    gbp = aligned[STERLING].to_numpy() if STERLING in aligned.columns else None
    vol = None
    if idx is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.concatenate([[np.nan], idx[1:] / idx[:-1] - 1.0])
        vol = pd.Series(r).rolling(21, min_periods=15).std().to_numpy()
    for j, p in enumerate(positions):
        if idx is not None and p >= YEAR and np.isfinite(idx[p]) and np.isfinite(idx[p - YEAR]):
            out[j, 0] = int(idx[p] > idx[p - YEAR])
        if vol is not None and np.isfinite(vol[p]):
            past = vol[: p + 1]
            past = past[np.isfinite(past)]
            if len(past) > QUARTER:
                out[j, 1] = int(vol[p] > np.median(past))
        if gbp is not None and p >= QUARTER and np.isfinite(gbp[p]) and np.isfinite(gbp[p - QUARTER]):
            out[j, 2] = int(gbp[p] > gbp[p - QUARTER])
    return out


def trend_hedge(strategy: np.ndarray, first_day: int, aligned: pd.DataFrame, market_daily: np.ndarray,
                index: str | None, cost_bps: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Hedged daily returns of a strategy whose returns start the day after `first_day`, and the
    hedge ratio held each day. market_daily: the equal-weight stock market's daily return per
    day position (used only to detect rolls)."""
    if index not in aligned.columns:
        return strategy.copy(), np.zeros(len(strategy))
    px = aligned[index].to_numpy()
    fr = clean_returns(px, market_daily)
    out, held = np.empty(len(strategy)), np.empty(len(strategy))
    h = 0.0
    for k in range(len(strategy)):
        d = first_day + 1 + k
        p = d - 1  # decided at the previous close
        want = 1.0 if (p >= YEAR and np.isfinite(px[p]) and np.isfinite(px[p - YEAR]) and px[p] < px[p - YEAR]) else 0.0
        cost = abs(want - h) * cost_bps * 1e-4
        h = want
        r_f = fr[d] if np.isfinite(fr[d]) else 0.0
        out[k] = strategy[k] - h * r_f - cost
        held[k] = h
    return out, held
