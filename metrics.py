"""
metrics.py — Performance metrics computation.

Produces a dictionary of standard risk/return statistics from the
portfolio equity curve and the benchmark.
"""

import logging
from typing import Dict
import pandas as pd
import numpy as np

from config import ADMConfig

logger = logging.getLogger(__name__)


def compute_metrics(portfolio_df: pd.DataFrame, cfg: ADMConfig) -> Dict:
    """
    Calculate all performance metrics from the portfolio simulation results.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Must have columns ``nav``, ``daily_return``, ``bench_return``,
        ``monthly_return``, ``trade_count``.
    cfg : ADMConfig

    Returns
    -------
    dict with keys:
        total_return, cagr, volatility, sharpe_ratio, sortino_ratio,
        max_drawdown, max_drawdown_duration, num_trades,
        avg_holding_months, win_rate, calmar_ratio,
        bench_total_return, bench_cagr, bench_volatility, bench_max_drawdown
    """
    nav = portfolio_df["nav"]
    daily_ret = portfolio_df["daily_return"]
    bench_ret = portfolio_df["bench_return"]

    total_days = len(nav)
    years = total_days / 252.0  # approximate trading days / year

    # ── Portfolio metrics ─────────────────────────────────────────────────
    total_return = nav.iloc[-1] / cfg.initial_capital - 1.0

    cagr = (nav.iloc[-1] / cfg.initial_capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    # Annualised volatility (daily returns → annual: * sqrt(252))
    vol = daily_ret.std() * np.sqrt(252)

    # Risk-free rate (approximate 0 for Sharpe — user can adjust)
    rf = 0.0

    excess_daily = daily_ret - rf / 252.0
    sharpe = excess_daily.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0.0

    # Sortino (downside deviation only)
    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
    sortino = (daily_ret.mean() - rf / 252.0) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    # More proper Sortino:
    annualised_excess = (daily_ret.mean() - rf / 252.0) * 252
    sortino_proper = annualised_excess / downside_std if downside_std > 0 else 0.0

    # Max drawdown
    peak = nav.expanding().max()
    drawdown = (nav - peak) / peak
    max_dd = drawdown.min()
    max_dd_duration = _max_drawdown_duration(drawdown)

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Trade statistics
    num_trades = portfolio_df["trade_count"].iloc[-1]
    # Average holding period: total days / (trades + 1) → months
    avg_holding_days = total_days / max(num_trades + 1, 1)
    avg_holding_months = avg_holding_days / 21.0

    # Win rate (months where monthly_return > 0)
    monthly_rets = portfolio_df["monthly_return"].dropna()
    win_rate = (monthly_rets > 0).sum() / len(monthly_rets) if len(monthly_rets) > 0 else 0.0

    # ── Benchmark metrics ─────────────────────────────────────────────────
    bench_price = (1.0 + bench_ret).cumprod()
    bench_total_return = bench_price.iloc[-1] - 1.0
    bench_cagr = (bench_price.iloc[-1]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    bench_vol = bench_ret.std() * np.sqrt(252)
    bench_peak = bench_price.expanding().max()
    bench_dd = (bench_price - bench_peak) / bench_peak
    bench_max_dd = bench_dd.min()

    final = {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino_proper,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": max_dd_duration,
        "calmar_ratio": calmar,
        "num_trades": num_trades,
        "avg_holding_months": avg_holding_months,
        "win_rate": win_rate,
        "bench_total_return": bench_total_return,
        "bench_cagr": bench_cagr,
        "bench_volatility": bench_vol,
        "bench_max_drawdown": bench_max_dd,
    }

    return final


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """Longest period (in days) the portfolio stayed below previous peak."""
    is_underwater = drawdown < 0
    if not is_underwater.any():
        return 0
    # Length of each underwater streak
    streaks = (is_underwater != is_underwater.shift()).cumsum()
    durations = is_underwater.groupby(streaks).sum()
    return int(durations.max())
