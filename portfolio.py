"""
portfolio.py — Portfolio allocation and trade execution engine.

Supports two signal formats:
  1. Single-asset: DataFrame with a ``"position"`` column (ticker name).
  2. Multi-asset / weight-based: DataFrame with ticker-name columns
     containing target weights (each row sums to 1.0).
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np

from config import ADMConfig

logger = logging.getLogger(__name__)


def simulate(
    prices: Dict[str, pd.Series],
    signals: pd.DataFrame,
    cfg: ADMConfig,
) -> pd.DataFrame:
    """
    Run the portfolio simulation.

    Parameters
    ----------
    prices : dict
        ``{ticker: pd.Series(adj_close)}``
    signals : pd.DataFrame
        Either:
        - A DataFrame with a ``"position"`` column (single-asset format)
        - A DataFrame with ticker columns (weight format, each row sums to 1.0)
    cfg : ADMConfig

    Returns
    -------
    pd.DataFrame with columns:
        date, nav, position, daily_return, monthly_return, bench_return, trade_count
    """
    signal_dates = signals.index

    # Build combined price DataFrame
    all_tickers = list(prices.keys())
    price_df = pd.DataFrame({t: prices[t] for t in all_tickers})

    # Detect signal format
    if "position" in signals.columns:
        return _simulate_single(prices, signals, cfg, price_df, signal_dates)
    else:
        return _simulate_weights(prices, signals, cfg, price_df, signal_dates)


def _simulate_single(
    prices: Dict[str, pd.Series],
    signals: pd.DataFrame,
    cfg: ADMConfig,
    price_df: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Original single-asset simulation (ADM-style)."""
    asset_col = "position"
    nav = cfg.initial_capital
    current_position: str = cfg.cash_asset
    trade_count = 0
    last_nav = nav
    bench_col = cfg.benchmark

    records = []

    for dt in signal_dates:
        target = signals.loc[dt, asset_col]
        if target is None or target not in price_df.columns:
            target = cfg.cash_asset

        # Daily return of current position
        px = price_df.loc[dt, current_position] if current_position in price_df.columns else np.nan
        prev_px = price_df.shift(1).loc[dt, current_position] if current_position in price_df.columns else np.nan

        if pd.notna(px) and pd.notna(prev_px) and prev_px > 0:
            daily_ret = px / prev_px - 1.0
        else:
            daily_ret = 0.0

        nav *= (1.0 + daily_ret)

        # Trade costs
        prev_pos = signals.shift(1).loc[dt, asset_col] if dt != signal_dates[0] else None
        if prev_pos is not None and target != prev_pos:
            total_cost = cfg.transaction_cost + cfg.slippage
            nav *= (1.0 - total_cost)
            trade_count += 1
            current_position = target

        # Monthly return
        monthly_ret = nav / last_nav - 1.0 if last_nav else 0.0

        # Benchmark daily return
        if bench_col in price_df.columns:
            b_px = price_df.loc[dt, bench_col]
            b_prev = price_df.shift(1).loc[dt, bench_col]
            bench_ret = b_px / b_prev - 1.0 if pd.notna(b_px) and pd.notna(b_prev) and b_prev > 0 else 0.0
        else:
            bench_ret = 0.0

        records.append({
            "date": dt, "nav": nav, "position": current_position,
            "daily_return": daily_ret, "monthly_return": monthly_ret,
            "bench_return": bench_ret, "trade_count": trade_count,
        })

        # Month boundary
        if dt == signal_dates[-1] or (
            dt != signal_dates[-1]
            and dt.month != signal_dates[signal_dates.get_loc(dt) + 1].month
        ):
            last_nav = nav

    return pd.DataFrame(records).set_index("date")


def _simulate_weights(
    prices: Dict[str, pd.Series],
    signals: pd.DataFrame,
    cfg: ADMConfig,
    price_df: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Weight-based simulation (multi-asset strategies).

    *signals* has ticker columns with target weights (each row ≈ 1.0).
    """
    weight_tickers = [c for c in signals.columns if c in price_df.columns]
    if not weight_tickers:
        raise ValueError("No valid ticker columns found in weight signals.")

    nav = cfg.initial_capital
    benchmark = cfg.benchmark
    bench_col = benchmark if benchmark in price_df.columns else None

    prev_weights = {t: 0.0 for t in weight_tickers}
    trade_count = 0
    last_nav = nav

    records = []

    for dt in signal_dates:
        target_weights = signals.loc[dt].to_dict()
        curr_weights = {t: target_weights.get(t, 0.0) for t in weight_tickers}
        total_w = sum(curr_weights.values())
        if total_w > 0:
            for t in curr_weights:
                curr_weights[t] /= total_w  # normalise

        # Daily return = weighted average of individual asset returns
        daily_ret = 0.0
        for t in weight_tickers:
            w = curr_weights.get(t, 0.0)
            if w == 0.0:
                continue
            px = price_df.loc[dt, t] if t in price_df.columns else np.nan
            prev_px = price_df.shift(1).loc[dt, t] if t in price_df.columns else np.nan
            if pd.notna(px) and pd.notna(prev_px) and prev_px > 0:
                daily_ret += w * (px / prev_px - 1.0)

        nav *= (1.0 + daily_ret)

        # Trade cost: fee applied to fraction of portfolio that moved
        turnover = sum(abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in weight_tickers) / 2.0
        if turnover > 0.001:
            total_cost = (cfg.transaction_cost + cfg.slippage) * turnover
            nav *= (1.0 - total_cost)
            trade_count += 1
        prev_weights = curr_weights

        # Monthly return
        monthly_ret = nav / last_nav - 1.0 if last_nav else 0.0

        # Benchmark
        if bench_col:
            b_px = price_df.loc[dt, bench_col]
            b_prev = price_df.shift(1).loc[dt, bench_col]
            bench_ret = b_px / b_prev - 1.0 if pd.notna(b_px) and pd.notna(b_prev) and b_prev > 0 else 0.0
        else:
            bench_ret = 0.0

        # Position summary (ticker with highest weight for display)
        top_ticker = max(curr_weights, key=curr_weights.get) if curr_weights else cfg.cash_asset
        top_pct = curr_weights.get(top_ticker, 0.0)
        position_str = f"{top_ticker}({top_pct:.0%})" if top_pct < 0.95 else top_ticker

        records.append({
            "date": dt,
            "nav": nav,
            "position": position_str,
            "daily_return": daily_ret,
            "monthly_return": monthly_ret,
            "bench_return": bench_ret,
            "trade_count": trade_count,
        })

        # Month boundary
        if dt == signal_dates[-1] or (
            dt != signal_dates[-1]
            and dt.month != signal_dates[signal_dates.get_loc(dt) + 1].month
        ):
            last_nav = nav

    return pd.DataFrame(records).set_index("date")
