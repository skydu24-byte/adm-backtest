"""Fractional-share close-to-close accounting, without leverage or FX.

A target observed at session t's close executes at t+1's close, AFTER
marking old holdings. Costs apply per side to actual traded notional.
Unavailable target assets remain uninvested, explicitly reported.
"""
import numpy as np
import pandas as pd
from schedule import rebalance_dates

def simulate(prices, signals, cfg):
    cfg.validate()
    frame = pd.DataFrame(prices).sort_index()
    if frame.empty or frame.index.has_duplicates or not frame.index.equals(signals.index):
        raise ValueError("Prices and signals must share a nonempty sorted unique calendar")
    tickers = [t for t in frame if t != cfg.rate_ticker]
    if cfg.benchmark not in tickers:
        raise ValueError("Benchmark prices are required")
    n = len(tickers)
    if "position" in signals:
        weights = pd.DataFrame(0.0, index=signals.index, columns=tickers)
        for dt, ticker in signals["position"].items():
            if pd.isna(ticker):
                continue
            if ticker not in tickers:
                raise ValueError(f"Unknown target ticker: {ticker}")
            weights.loc[dt, ticker] = 1.0
    else:
        unknown = set(signals) - set(tickers) - {"__rebalance__"}
        if unknown:
            raise ValueError(f"Unknown signal columns: {unknown}")
        weights = signals.reindex(columns=tickers, fill_value=0.0).astype(float)
    target_array = weights.to_numpy()
    if not np.isfinite(target_array).all() or (target_array < 0).any() or (target_array.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError("Target weights must be finite, nonnegative and sum to <= 1")
    scheduled = signals["__rebalance__"].fillna(False).to_numpy(dtype=bool) if "__rebalance__" in signals else signals.index.isin(rebalance_dates(signals.index, cfg.rebalance_freq))
    px = frame[tickers].to_numpy(dtype=float)
    qty = np.zeros(n)
    cash = float(cfg.initial_capital)
    previous_nav = cash
    rate = cfg.transaction_cost + cfg.slippage
    records = []
    trade_count = 0
    benchmark_idx = tickers.index(cfg.benchmark)
    for i, dt in enumerate(frame.index):
        valid = np.isfinite(px[i]) & (px[i] > 0)
        if ((qty > 1e-12) & ~valid).any():
            raise ValueError(f"Cannot value a held asset at {dt}; missing/stale price")
        values = qty * np.where(valid, px[i], 0.0)
        before = float(cash + values.sum())
        fee = 0.0
        turnover = 0.0
        order_count = 0
        unavailable = 0.0
        if i > 0 and scheduled[i - 1]:
            target = target_array[i - 1].copy()
            unavailable = float(target[~valid].sum())
            target[~valid] = 0.0
            if rate:
                lo, hi = 0.0, before
                for _ in range(64):
                    mid = (lo + hi) / 2
                    if mid + rate * np.abs(mid * target - values).sum() > before:
                        hi = mid
                    else:
                        lo = mid
                after = (lo + hi) / 2
            else:
                after = before
            desired = after * target
            orders = desired - values
            notional = float(np.abs(orders).sum())
            fee = rate * notional
            order_count = int((np.abs(orders) > max(before * 1e-10, 1e-9)).sum())
            turnover = notional / before if before else 0.0
            if order_count:
                trade_count += 1
            qty = np.divide(desired, px[i], out=np.zeros(n), where=valid)
            values = desired
            cash = before - fee - float(desired.sum())
            if cash < -max(before * 1e-10, 1e-8):
                raise ArithmeticError("Negative cash after trade")
            cash = max(cash, 0.0)
        nav = float(cash + values.sum())
        if nav <= 0 or not np.isfinite(nav):
            raise ArithmeticError("Invalid portfolio value")
        bench_ret = 0.0
        if i:
            b0, b1 = px[i - 1, benchmark_idx], px[i, benchmark_idx]
            if not np.isfinite([b0, b1]).all() or min(b0, b1) <= 0:
                raise ValueError("Invalid benchmark prices")
            bench_ret = b1 / b0 - 1.0
        actual = values / nav
        parts = {t: w for t, w in zip(tickers, actual) if w > 1e-9}
        if cash / nav > 1e-9:
            parts["CASH"] = cash / nav
        position = ", ".join(f"{t}({w:.1%})" for t, w in parts.items())
        record = {"date": dt, "nav": nav, "position": position, "daily_return": nav / previous_nav - 1.0, "bench_return": bench_ret, "trade_count": trade_count, "orders": order_count, "fees": fee, "turnover": turnover, "unavailable_target_weight": unavailable, "weight_CASH": cash / nav}
        record.update({f"weight_{t}": float(w) for t, w in zip(tickers, actual)})
        records.append(record)
        previous_nav = nav
    out = pd.DataFrame(records).set_index("date")
    month = out.index.to_period("M")
    month_start = out["nav"].shift().groupby(month).transform("first")
    month_start.loc[month == month[0]] = cfg.initial_capital
    out["monthly_return"] = out["nav"] / month_start - 1.0
    return out
