"""Daily trading of the orthonormal shrinkage factor model, with stop-loss and take-profit.

Every day, at the close of day t (data up to that close only):
  - each stock's six signals are recomputed and made orthonormal (Loewdin), and its expected
    excess return over the market is E = S @ premia, the premia being the factors' shrunk
    monthly records resolved by that day (orthofactor.shrunk_premia);
  - orders for day t + 1 are decided:
      * sell a held stock whose expected excess return over the signal's life, E / (1 - rho), is
        below minus a round trip (it is expected to lag the market by more than the trade costs);
      * buy, best first, stocks whose E / (1 - rho) beats a round trip, while position slots are
        free, skipping stocks stopped out within the cool-down.
Execution on day t + 1: orders fill at the open. Each position carries a stop-loss and a
take-profit set at entry from the 14-day average true range (ATR) at the decision close:
stop = entry - sl_atr * ATR, take = entry + tp_atr * ATR. During the day, if the low reaches the
stop the position is sold at the stop (at the open if it gapped below); if the high reaches the
take-profit it is sold there (at the open if it gapped above); if both, the stop is assumed to
have come first. Positions are sized at 1 / max_positions of equity; the rest is cash (no
interest). Costs: `buy_bps` on purchases (dealing plus stamp duty), `sell_bps` on sales.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from orthofactor import YEAR, loewdin, month_ends, shrunk_premia, signals_at, zscores


@dataclass
class DailySpec:
    sl_atr: float = 2.0
    tp_atr: float = 4.0
    max_positions: int = 20
    cooldown: int = 5
    buy_bps: float = 60.0
    sell_bps: float = 10.0
    min_months: int = 12


@dataclass
class Trade:
    ticker: str
    entry_day: int
    exit_day: int
    entry: float
    exit: float
    reason: str
    ret: float  # net of costs


@dataclass
class DailyResult:
    dates: pd.DatetimeIndex
    returns: dict[str, np.ndarray]
    trades: list[Trade] = field(default_factory=list)
    exposure: float = 0.0
    turnover: float = 0.0
    positions: float = 0.0
    weights: np.ndarray | None = None  # daily end-of-day weights (with keep_weights)


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    prev = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    tr = np.nanmax(np.stack([high - low, np.abs(high - prev), np.abs(low - prev)]), axis=0)
    out = np.full_like(close, np.nan)
    c = np.nancumsum(np.nan_to_num(tr), axis=0)
    out[n:] = (c[n:] - c[:-n]) / n
    return out


def monthly_records(C: np.ndarray, V: np.ndarray, ends: np.ndarray):
    """Each month-end's orthonormal exposures and the factors' realised return over the next month."""
    k = 6
    records = []
    for m, p in enumerate(ends[:-1]):
        raw = signals_at(C, V, p)
        ok = np.isfinite(raw).all(axis=1)
        if ok.sum() <= 2 * k:
            records.append(np.zeros(k))
            continue
        S = loewdin(zscores(raw[ok]))
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd = C[ends[m + 1], ok] / C[p, ok] - 1.0
        use = np.isfinite(fwd)
        records.append(S[use].T @ fwd[use])
    return np.array(records)


@dataclass
class Prepared:
    """Everything the daily rules need that does not depend on their settings."""
    C: np.ndarray
    Cf: np.ndarray
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    A: np.ndarray
    E: np.ndarray      # expected excess return per day and stock, decided at that close (NaN = ineligible)
    life: np.ndarray   # expected life of an expected return (months), per day
    start: int
    dates: pd.DatetimeIndex
    tickers: list


def prepare_daily(bars: dict[str, pd.DataFrame], min_months: int = 12) -> Prepared:
    C, O, H, L, V = (bars[k].to_numpy() for k in ["close", "open", "high", "low", "volume"])
    Cf = bars["close"].ffill().to_numpy()  # valuation at the last traded price on days a stock has no bar
    T, N = C.shape
    ends = month_ends(bars["close"].index)
    ends = ends[ends >= YEAR]
    if len(ends) < min_months + 3:
        raise ValueError("not enough history: need a year for signals and a year of factor records")
    rec = monthly_records(C, V, ends)  # rec[m] resolves at ends[m + 1]
    A = atr(H, L, C)
    start = int(ends[min_months])
    E_all = np.full((T, N), np.nan)
    life = np.ones(T)
    end_set = set(int(e) for e in ends)
    prev_E, rho_hist = None, []
    for d in range(start, T):
        m_res = int(np.searchsorted(ends, d, side="right")) - 1  # months whose end <= d
        prem = shrunk_premia(rec[: max(m_res, 0)], min_months) if m_res > 0 else np.zeros(6)
        raw = signals_at(C, V, d)
        ok = np.isfinite(raw).all(axis=1) & np.isfinite(A[d]) & (A[d] > 0)
        if ok.sum() > 12:
            E_all[d, ok] = loewdin(zscores(raw[ok])) @ prem
        E = E_all[d]
        if d in end_set:
            if prev_E is not None:
                both = np.isfinite(E) & np.isfinite(prev_E)
                if both.sum() > 10 and np.nanstd(E[both]) > 0 and np.nanstd(prev_E[both]) > 0:
                    rho_hist.append(float(np.corrcoef(zscores(E[both][:, None])[:, 0], zscores(prev_E[both][:, None])[:, 0])[0, 1]))
            prev_E = E.copy()
        rho = float(np.clip(np.mean(rho_hist), 0.0, 0.95)) if rho_hist else 0.0
        life[d] = 1.0 / (1.0 - rho)
    return Prepared(C, Cf, O, H, L, A, E_all, life, start, bars["close"].index, list(bars["close"].columns))


def simulate_daily(pr: Prepared, spec: DailySpec, keep_weights: bool = False) -> DailyResult:
    C, Cf, O, H, L, A = pr.C, pr.Cf, pr.O, pr.H, pr.L, pr.A
    T, N = C.shape
    buy, sell = spec.buy_bps * 1e-4, spec.sell_bps * 1e-4
    hurdle = buy + sell
    start = pr.start
    cash, shares = 1.0, np.zeros(N)
    stop, take, entry_px, entry_day = np.full(N, np.nan), np.full(N, np.nan), np.zeros(N), np.zeros(N, int)
    last_stop = np.full(N, -10**9)
    trades: list[Trade] = []
    rets, mkt, weights = [], [], []
    orders_buy, orders_sell = [], []
    traded_value, exposure_sum, pos_sum = 0.0, 0.0, 0.0
    mkt_w = None
    eq_ref = 1.0  # equity at the previous close: the base of the turnover figure

    def equity(d):
        px = np.where(np.isfinite(Cf[d]), Cf[d], 0.0)
        return cash + float((shares * px).sum())

    def close_position(i, d, px, reason):
        nonlocal cash, traded_value
        value = shares[i] * px
        cash += value * (1 - sell)
        traded_value += value / eq_ref
        cost_in = shares[i] * entry_px[i] * (1 + buy)
        trades.append(Trade(pr.tickers[i], int(entry_day[i]), d, float(entry_px[i]), float(px), reason, float(value * (1 - sell) / cost_in - 1)))
        if reason == "stop":
            last_stop[i] = d
        shares[i] = 0.0
        stop[i] = take[i] = np.nan

    for d in range(start, T):
        if d > start:
            eq_before = eq_ref = equity(d - 1)
            # Orders decided at yesterday's close, filled at today's open.
            for i in orders_sell:
                if shares[i] > 0:
                    px = O[d, i] if np.isfinite(O[d, i]) else Cf[d - 1, i]
                    close_position(i, d, px, "signal")
            for i, sl_dist, tp_dist in orders_buy:
                px = O[d, i]
                if not np.isfinite(px) or px <= 0:
                    continue
                budget = min(eq_ref / spec.max_positions, cash / (1 + buy))
                if budget <= 0:
                    break
                shares[i] = budget / px
                cash -= budget * (1 + buy)
                traded_value += budget / eq_ref
                entry_px[i], entry_day[i] = px, d
                stop[i], take[i] = px - sl_dist, px + tp_dist
            # Stops and take-profits during the day.
            for i in np.flatnonzero(shares > 0):
                lo, hi, op = L[d, i], H[d, i], O[d, i]
                if not np.isfinite(lo) or not np.isfinite(hi):
                    continue
                if lo <= stop[i]:
                    px = op if np.isfinite(op) and op < stop[i] else stop[i]
                    close_position(i, d, px, "stop")
                elif hi >= take[i]:
                    px = op if np.isfinite(op) and op > take[i] else take[i]
                    close_position(i, d, px, "take")
            eq_after = equity(d)
            rets.append(eq_after / eq_before - 1.0)
            exposure_sum += 1.0 - cash / eq_after
            pos_sum += (shares > 0).sum()
            if keep_weights:
                weights.append(shares * np.where(np.isfinite(Cf[d]), Cf[d], 0.0) / eq_after)
            # Market: equal weight in the stocks eligible on the first day, bought once.
            r = np.nan_to_num(C[d] / C[d - 1] - 1.0)
            before = mkt_w.sum()
            mkt_w = mkt_w * (1 + r)
            mkt.append(mkt_w.sum() / before - 1.0 - (buy if d == start + 1 else 0.0))

        # Decide at today's close.
        E = pr.E[d]
        ok = np.isfinite(E)
        if mkt_w is None:
            mkt_w = ok / ok.sum()
        life = pr.life[d]
        held = shares > 0
        orders_sell = [i for i in np.flatnonzero(held & ok) if -E[i] * life > hurdle]
        free = spec.max_positions - (int(held.sum()) - len(orders_sell))
        cand = np.flatnonzero(~held & ok & (E * life > hurdle) & (d - last_stop > spec.cooldown))
        cand = cand[np.argsort(-E[cand])][: max(free, 0)]
        orders_buy = [(i, spec.sl_atr * A[d, i], spec.tp_atr * A[d, i]) for i in cand]

    days = pr.dates[start + 1 : start + 1 + len(rets)]
    years = len(rets) / YEAR
    res = DailyResult(dates=days, returns={"model (daily, SL/TP)": np.array(rets), "market (buy and hold)": np.array(mkt)},
                      trades=trades, exposure=exposure_sum / max(len(rets), 1), turnover=traded_value / max(years, 1e-9),
                      positions=pos_sum / max(len(rets), 1))
    if keep_weights:
        res.weights = np.array(weights)
    return res


def run_daily(bars: dict[str, pd.DataFrame], spec: DailySpec = DailySpec()) -> DailyResult:
    return simulate_daily(prepare_daily(bars, spec.min_months), spec)
