"""
data_loader.py — Fetch and prepare price data for ADM backtesting.

Supports two data sources:
  - Yahoo Finance (live via yfinance)
  - Local CSV files (one per ticker, columns: Date, Adj Close…)
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

from config import ADMConfig

logger = logging.getLogger(__name__)


def load_data(cfg: ADMConfig) -> Dict[str, pd.Series]:
    """
    Load adjusted-close prices for all tickers referenced in the config.

    Returns a dict ``{ticker: pd.Series(index=DatetimeIndex, values=adj_close)}``.
    All series share the same date index after outer-join alignment.
    """
    # Collect all unique tickers
    all_tickers = list(
        set(cfg.risky_assets)
        | {cfg.defensive_asset, cfg.benchmark, cfg.cash_asset}
        | set(cfg.best_of_defensives)
    )

    raw: Dict[str, pd.DataFrame] = {}

    for ticker in all_tickers:
        df = _load_single(ticker, cfg)
        if df.empty:
            logger.warning("No data for %s — skipping.", ticker)
            continue

        # Use adjusted close if available, else Close
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        raw[ticker] = df[col]

    if not raw:
        raise RuntimeError("No data could be loaded for any ticker.")

    # Align all series to a common index (union of dates)
    full_index = pd.DatetimeIndex([])
    for s in raw.values():
        full_index = full_index.union(s.dropna().index)
    full_index = pd.DatetimeIndex(sorted(full_index))

    out: Dict[str, pd.Series] = {}
    for ticker, s in raw.items():
        out[ticker] = s.reindex(full_index).ffill().bfill()

    return out


def _load_single(ticker: str, cfg: ADMConfig) -> pd.DataFrame:
    """Load data for a single ticker from the configured source."""
    if cfg.data_source == "yahoo":
        return _from_yahoo(ticker, cfg.start_date, cfg.end_date)
    elif cfg.data_source == "csv":
        return _from_csv(ticker, cfg)
    else:
        raise ValueError(f"Unknown data_source '{cfg.data_source}'. Use 'yahoo' or 'csv'.")


def _from_yahoo(
    ticker: str, start: str, end: Optional[str]
) -> pd.DataFrame:
    """Download from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "yfinance is required for live Yahoo downloads. "
            "Install it with: uv pip install yfinance"
        )

    df = yf.download(ticker, start=start, end=end or None, auto_adjust=False, progress=False)
    if df.empty:
        logger.warning("yfinance returned empty DataFrame for %s", ticker)
        return pd.DataFrame()
    # yfinance multi-index columns -> flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.DatetimeIndex(df.index)
    return df


def _from_csv(ticker: str, cfg: ADMConfig) -> pd.DataFrame:
    """Load from a local CSV file in *data_dir*."""
    path = Path(cfg.data_dir) / f"{ticker}.csv"
    if not path.exists():
        logger.warning("CSV not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df
