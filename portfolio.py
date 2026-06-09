"""
portfolio.py — Portfolio allocation and trade execution engine.

Simulates monthly rebalancing: at each rebalance date the portfolio
switches to the target asset (risky or defensive) according to the
signal, paying transaction costs and slippage.
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
        ``{ticker: pd.Series(adj_close)}`` — same as returned by ``data_loader``.
    signals : pd.DataFrame
        Must contain a ``"position"`` column with the target ticker at each date.
    cfg : ADMConfig

    Returns
    -------
    pd.DataFrame with columns:
        date, nav, position, monthly_return, bench_return, trade_count
    """
    asset_col = "position"
    signal_dates = signals.index

    # Build a combined price DataFrame for every possible ticker
    all_tickers = list(prices.keys())
    price_df = pd.DataFrame({t: prices[t] for t in all_tickers})

    nav = cfg.initial_capital
    current_position: str = cfg.cash_asset  # starting placeholder
    trade_count = 0
    last_nav = nav

    records = []
    bench_col = cfg.benchmark

    for dt in signal_dates:
        target = signals.loc[dt, asset_col]
        if target is None or target not in price_df.columns:
            target = cfg.cash_asset

        # Daily return of current position
        current_price = price_df.loc[dt, current_position] if current_position in price_df.columns else np.nan
        prev_price = price_df.shift(1).loc[dt, current_position] if current_position in price_df.columns else np.nan

        if pd.notna(current_price) and pd.notna(prev_price) and prev_price > 0:
            daily_ret = current_price / prev_price - 1.0
        else:
            daily_ret = 0.0

        # Apply daily return
        nav *= (1.0 + daily_ret)

        # Trade costs on rebalance (when the position changes)
        # We detect a change by comparing with *last* day's position
        prev_pos = signals.shift(1).loc[dt, asset_col] if dt != signal_dates[0] else None
        if prev_pos is not None and target != prev_pos:
            total_cost = cfg.transaction_cost + cfg.slippage
            nav *= (1.0 - total_cost)
            trade_count += 1
            current_position = target

        # Monthly return (for reporting — first month is NaN)
        monthly_label = pd.Timestamp(dt.year, dt.month, 1) + pd.offsets.MonthEnd(0)
        monthly_ret = nav / last_nav - 1.0 if last_nav else 0.0

        # Benchmark daily return
        bench_price = price_df.loc[dt, bench_col] if bench_col in price_df.columns else np.nan
        bench_prev = price_df.shift(1).loc[dt, bench_col] if bench_col in price_df.columns else np.nan
        if pd.notna(bench_price) and pd.notna(bench_prev) and bench_prev > 0:
            bench_ret = bench_price / bench_prev - 1.0
        else:
            bench_ret = 0.0

        records.append({
            "date": dt,
            "nav": nav,
            "position": current_position,
            "daily_return": daily_ret,
            "monthly_return": monthly_ret,
            "bench_return": bench_ret,
            "trade_count": trade_count,
        })

        # Update last_nav at month boundaries
        if dt == signal_dates[-1] or dt.month != signal_dates[signal_dates.get_loc(dt) + 1].month if dt != signal_dates[-1] else True:
            last_nav = nav

    result = pd.DataFrame(records).set_index("date")
    return result
