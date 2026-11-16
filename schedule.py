"""Actual observed session endpoints; no assumption of calendar-day trading."""
import pandas as pd

def rebalance_dates(index, freq="ME"):
    index = pd.DatetimeIndex(index)
    if index.empty or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Expected a nonempty sorted unique calendar")
    period = {"M": "M", "ME": "M", "W": "W-FRI", "W-FRI": "W-FRI"}.get(freq)
    if period is None:
        raise ValueError(f"Unsupported rebalance frequency: {freq}")
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(index.to_period(period)).last().values)

def month_prices(price):
    dates = rebalance_dates(price.index)
    return price.reindex(dates)

def monthly_returns(price, months):
    return month_prices(price).pct_change(months, fill_method=None).reindex(price.index, method="ffill")

def weight_signals(weights, index, cfg):
    out = weights.reindex(index).ffill().fillna(0.0)
    out["__rebalance__"] = out.index.isin(weights.index)
    return out
