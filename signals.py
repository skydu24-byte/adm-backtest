"""
signals.py — Momentum computation and ADM scoring logic.

The Accelerating Dual Momentum (ADM) score for each risky asset is
the sum (or average) of total returns over the specified look-back
windows (e.g. 1, 3 and 6 months).
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config import ADMConfig

logger = logging.getLogger(__name__)


def total_return(series: pd.Series, months: int) -> pd.Series:
    """
    Return the total (close-to-close) return over *months* calendar months.

    Uses month-end-to-month-end alignment by resampling to month-end first,
    then shifting.  NaN where history is insufficient.
    """
    # Resample to month-end last available price
    monthly = series.resample("ME").last()
    ret = monthly.pct_change(periods=months)
    # Map back to daily index for alignment with the rebalance dates
    return ret.reindex(series.index, method="ffill")


def adm_score(price: pd.Series, windows_months: List[int]) -> pd.Series:
    """
    Compute the ADM score for a single asset as the *sum* (or equivalently
    the mean, since the ranking is unaffected by scaling) of total returns
    over the specified windows.

    Returns a Series (same index as *price*) with the score.
    """
    returns = [total_return(price, m) for m in windows_months]
    score = pd.concat(returns, axis=1).sum(axis=1, min_count=1)
    # min_count=1 means NaN only if *all* windows are NaN (start of series)
    return score


def pick_risky(
    scores: pd.DataFrame,
) -> pd.Series:
    """
    Given a DataFrame ``scores`` with columns = tickers, index = dates,
    return a Series with the chosen risky ticker (or None) for each date.

    Rules:
      - Pick the asset with the highest score, **only if** that score > 0.
      - If the highest score <= 0, return None → switch to defensive.
      - If all scores are NaN (early history), return None.
    """
    def _best(row: pd.Series) -> Optional[str]:
        if row.isna().all():
            return None
        best_ticker = row.idxmax()
        best_val = row[best_ticker]
        return best_ticker if best_val > 0 else None

    return scores.apply(_best, axis=1)


def pick_best_defensive(
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Pick the defensive asset with the highest 1-month return.

    Returns a Series (same index as *prices*) with the chosen ticker.
    Falls back to the first column if all NaN.
    """
    monthly_ret = prices.resample("ME").last().pct_change(periods=1)
    monthly_ret = monthly_ret.reindex(prices.index, method="ffill")

    def _best(row: pd.Series) -> str:
        if row.isna().all():
            return prices.columns[0]
        return row.idxmax()

    return monthly_ret.apply(_best, axis=1)


def compute_signals(
    prices: Dict[str, pd.Series], cfg: ADMConfig
) -> pd.DataFrame:
    """
    Master signal function.

    Returns a DataFrame with columns:
      - ``date``: index  (daily)
      - ``risky_pick``: the chosen risky ticker (or None)
      - ``defensive_pick``: the chosen defensive ticker
      - ``regime``: ``"risky"`` or ``"defensive"``
      - ``position``: final ticker the strategy holds

    The signal is computed only on rebalance days; it is forward-filled
    (carried forward) until the next rebalance.
    """
    # Build a DataFrame of the risky asset prices
    risky_price_df = pd.DataFrame({t: prices[t] for t in cfg.risky_assets})

    # Compute ADM scores for each risky asset
    score_df = pd.DataFrame(
        {t: adm_score(risky_price_df[t], cfg.momentum_windows_months)
         for t in cfg.risky_assets}
    )

    # Pick risky
    risky_pick = pick_risky(score_df)

    # Determine defensive
    if cfg.best_of_defensives:
        def_prices = pd.DataFrame({t: prices[t] for t in cfg.best_of_defensives})
        defensive_pick = pick_best_defensive(def_prices)
    else:
        defensive_pick = pd.Series(cfg.defensive_asset, index=risky_pick.index)

    # Regime
    regime = risky_pick.notna().map({True: "risky", False: "defensive"})

    # Position
    def _pos(row: Tuple[Optional[str], str, str]) -> str:
        risky, defensive, reg = row
        return str(risky) if reg == "risky" else defensive

    df = pd.DataFrame({
        "risky_pick": risky_pick,
        "defensive_pick": defensive_pick,
        "regime": regime,
    })
    df["position"] = df.apply(_pos, axis=1)

    # ── Snap to rebalance dates, then forward-fill ────────────────────────
    rebal_dates = _rebalance_dates(df.index, cfg)
    df_rebal = df.loc[rebal_dates].copy()
    df_rebal = df_rebal.reindex(df.index).ffill()

    return df_rebal


def _rebalance_dates(date_index: pd.DatetimeIndex, cfg: ADMConfig) -> pd.DatetimeIndex:
    """
    Return the subset of *date_index* that falls on rebalance days.
    For 'M' frequency, these are month-end dates present in the index.
    """
    if not cfg.sleeve_rebalance:
        # Standard month-end
        ends = date_index[date_index.is_month_end]
        return date_index.intersection(ends)

    # Sleeve (half-rebalancing): 1st and 15th of each month.
    first = date_index[date_index.day == 1]
    mid = date_index[date_index.day == 15]
    both = first.union(mid)
    return date_index.intersection(both)
