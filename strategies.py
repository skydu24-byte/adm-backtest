"""
strategies.py — Alternative strategy implementations beyond ADM.

Each strategy function accepts a price dictionary and an ADMConfig,
and returns a DataFrame of **target weights** (one column per ticker,
each row summing to 1.0) sampled at rebalance dates, forward-filled
to every calendar day.

Available strategies (called via backtest.py dispatch):
  - enhanced_dm
  - managed_futures_dm
  - vigilant_dm
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from config import ADMConfig

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _total_return_12m(price: pd.Series) -> pd.Series:
    """12-month total return, month-end aligned, NaN if insufficient history."""
    monthly = price.resample("ME").last()
    ret_12m = monthly.pct_change(periods=12)
    return ret_12m.reindex(price.index, method="ffill")


def _total_return_6m(price: pd.Series) -> pd.Series:
    """6-month total return."""
    monthly = price.resample("ME").last()
    ret_6m = monthly.pct_change(periods=6)
    return ret_6m.reindex(price.index, method="ffill")


def _rebalance_dates(date_index: pd.DatetimeIndex, freq: str = "ME") -> pd.DatetimeIndex:
    """Return the subset of *date_index* falling on rebalance days."""
    if freq == "ME":
        ends = date_index[date_index.is_month_end]
        return date_index.intersection(ends)
    # Fallback: just return month-ends
    ends = date_index[date_index.is_month_end]
    return date_index.intersection(ends)


def _ffill_weights(signal_df: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Forward-fill weights from rebalance days to every day in *full_index*.
    NaN rows (before first signal) are filled with 0.0 weights.
    """
    out = signal_df.reindex(full_index).ffill()
    out = out.fillna(0.0)
    return out


def _single_to_weights(positions: pd.Series, all_tickers: List[str]) -> pd.DataFrame:
    """
    Convert a Series of ticker names (one per date) into a weight DataFrame
    where the chosen ticker has weight 1.0, others 0.0.
    """
    df = pd.DataFrame(0.0, index=positions.index, columns=all_tickers)
    for dt, ticker in positions.items():
        if ticker is not None and pd.notna(ticker) and ticker in all_tickers:
            df.loc[dt, ticker] = 1.0
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: Enhanced Dual Momentum (Multi-Asset, Top-N)
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals_enhanced_dm(
    prices: Dict[str, pd.Series],
    cfg: ADMConfig,
) -> pd.DataFrame:
    """
    Enhanced Dual Momentum — 8-asset universe, 12-month absolute momentum
    filter, equal-weight top *enhanced_top_n*.

    Universe: SPY, QQQ, GLD, DBC, VNQ, KMLM, SHY, BIL
    """
    universe = cfg.multi_asset_universe
    lookback = cfg.momentum_lookback_months  # default 12
    top_n = cfg.enhanced_top_n

    price_df = pd.DataFrame({t: prices[t] for t in universe if t in prices})
    tickers = list(price_df.columns)

    # 12-month return for each asset
    monthly_ret = price_df.resample("ME").last().pct_change(periods=lookback)
    daily_ret_signal = monthly_ret.reindex(price_df.index, method="ffill")

    rebal = _rebalance_dates(price_df.index, cfg.rebalance_freq)

    # Container: rebalance-date-only weights
    weight_records: Dict[str, list] = {t: [] for t in tickers}
    date_list: list = []

    for dt in rebal:
        date_list.append(dt)
        scores = {t: daily_ret_signal.loc[dt, t] for t in tickers}

        # Absolute momentum filter
        qualified = [t for t in tickers
                     if pd.notna(scores[t]) and scores[t] > 0]

        if not qualified:
            # All to cash
            cash = cfg.cash_asset if cfg.cash_asset in tickers else tickers[0]
            for t in tickers:
                weight_records[t].append(1.0 if t == cash else 0.0)
            continue

        # Sort by momentum score descending
        qualified.sort(key=lambda t: scores[t], reverse=True)

        # Pick top N
        selected = qualified[:top_n]
        w = 1.0 / len(selected)
        for t in tickers:
            weight_records[t].append(w if t in selected else 0.0)

    df_rebal = pd.DataFrame({t: weight_records[t] for t in tickers}, index=pd.DatetimeIndex(date_list))

    # Forward-fill to daily
    result = _ffill_weights(df_rebal, price_df.index)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: Managed Futures Dual Momentum
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals_managed_futures_dm(
    prices: Dict[str, pd.Series],
    cfg: ADMConfig,
) -> pd.DataFrame:
    """
    Managed Futures Dual Momentum — replaces TLT with KMLM/DBMF/FLOT.

    If SPY 12m > 0: pick best of SPY vs VEA → 100%.
    Else: pick best of KMLM, DBMF, FLOT, BIL by 12m return → 100%.
    If all defensive negative: 100% BIL.
    """
    risky = cfg.managed_futures_risky
    defensive = cfg.managed_futures_defensive
    all_assets = list(set(risky + defensive + [cfg.cash_asset, cfg.benchmark]))
    price_df = pd.DataFrame({t: prices[t] for t in all_assets if t in prices})
    tickers = list(price_df.columns)

    # 12-month returns
    ret_12m = _total_return_12m(price_df[risky[0]]) if risky[0] in price_df.columns else pd.Series(0.0, index=price_df.index)

    rebal = _rebalance_dates(price_df.index, cfg.rebalance_freq)

    weight_records: Dict[str, list] = {t: [] for t in tickers}
    date_list: list = []

    for dt in rebal:
        date_list.append(dt)

        spy_ret = ret_12m.loc[dt] if dt in ret_12m.index else np.nan

        if pd.notna(spy_ret) and spy_ret > 0:
            # Relative momentum among risky
            best_risky = None
            best_score = -1e9
            for t in risky:
                if t in prices:
                    r = prices[t].reindex(price_df.index, method="ffill")
                    r_12m = r.pct_change(periods=252).loc[dt] if dt in r.index else -1e9
                    if pd.notna(r_12m) and r_12m > best_score:
                        best_score = r_12m
                        best_risky = t
            chosen = best_risky if best_risky else cfg.cash_asset
        else:
            # Defensive mode: rank defensive assets by 12m return
            best_def = None
            best_score = -1e9
            for t in defensive:
                if t in prices:
                    r = prices[t].reindex(price_df.index, method="ffill")
                    r_12m = r.pct_change(periods=252).loc[dt] if dt in r.index else -1e9
                    if pd.notna(r_12m) and r_12m > best_score:
                        best_score = r_12m
                        best_def = t
            chosen = best_def if best_def and best_score > 0 else cfg.cash_asset

        for t in tickers:
            weight_records[t].append(1.0 if t == chosen else 0.0)

    df_rebal = pd.DataFrame({t: weight_records[t] for t in tickers}, index=pd.DatetimeIndex(date_list))
    result = _ffill_weights(df_rebal, price_df.index)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Vigilant Dual Momentum (Rate-Aware Hedge)
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals_vigilant_dm(
    prices: Dict[str, pd.Series],
    cfg: ADMConfig,
) -> pd.DataFrame:
    """
    Vigilant Dual Momentum — dynamically selects safe haven based on
    the 6-month trend of ^TNX (10-year Treasury yield).

    If SPY 12m > 0: relative momentum SPY vs VEA → 100%.
    Else:
      - ^TNX rising over 6m → pick best of SHY, GLD, BIL, USFR
      - ^TNX falling/steady → pick best of TLT, GLD, SHY, IEF
    If all candidates negative: 100% BIL.
    """
    risky = cfg.vigilant_risky
    def_rising = cfg.vigilant_defensive_rising
    def_falling = cfg.vigilant_defensive_falling
    rate_ticker = cfg.rate_ticker  # "^TNX"

    all_assets = list(set(
        risky + def_rising + def_falling
        + [cfg.cash_asset, cfg.benchmark]
    ))
    price_df = pd.DataFrame({t: prices[t] for t in all_assets if t in prices})
    tickers = list(price_df.columns)

    # Rate data
    rate_series = prices.get(rate_ticker)
    if rate_series is not None:
        rate_monthly = rate_series.resample("ME").last()
        rate_6m_chg = rate_monthly.diff(periods=6)
        rate_6m_chg_daily = rate_6m_chg.reindex(price_df.index, method="ffill")
    else:
        rate_6m_chg_daily = pd.Series(0.0, index=price_df.index)

    # 12m SPY return
    spy_price = prices.get(risky[0])
    spy_12m = _total_return_12m(spy_price) if spy_price is not None else pd.Series(0.0, index=price_df.index)

    rebal = _rebalance_dates(price_df.index, cfg.rebalance_freq)

    weight_records: Dict[str, list] = {t: [] for t in tickers}
    date_list: list = []

    for dt in rebal:
        date_list.append(dt)

        spy_val = spy_12m.loc[dt] if dt in spy_12m.index else np.nan

        if pd.notna(spy_val) and spy_val > 0:
            # Risky mode: relative momentum
            best_risky = None
            best_score = -1e9
            for t in risky:
                if t in prices:
                    px = prices[t].reindex(price_df.index, method="ffill")
                    r = px.pct_change(periods=252).loc[dt] if dt in px.index else -1e9
                    if pd.notna(r) and r > best_score:
                        best_score = r
                        best_risky = t
            chosen = best_risky if best_risky else cfg.cash_asset
        else:
            # Defensive mode: rate-aware
            rate_chg = rate_6m_chg_daily.loc[dt] if dt in rate_6m_chg_daily.index else 0.0
            candidates = def_rising if rate_chg > 0 else def_falling

            best_def = None
            best_score = -1e9
            for t in candidates:
                if t in prices:
                    px = prices[t].reindex(price_df.index, method="ffill")
                    r = px.pct_change(periods=252).loc[dt] if dt in px.index else -1e9
                    if pd.notna(r) and r > best_score:
                        best_score = r
                        best_def = t
            chosen = best_def if best_def and best_score > 0 else cfg.cash_asset

        for t in tickers:
            weight_records[t].append(1.0 if t == chosen else 0.0)

    df_rebal = pd.DataFrame({t: weight_records[t] for t in tickers}, index=pd.DatetimeIndex(date_list))
    result = _ffill_weights(df_rebal, price_df.index)
    return result
