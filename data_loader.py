"""Strict input validation; no prices invented before listing or long gaps."""
from pathlib import Path
import numpy as np
import pandas as pd

YIELD_TICKERS = {"^TNX", "^VIX", "^GSPC", "^IXIC"}

def _from_yahoo(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _from_csv(ticker, cfg):
    path = Path(cfg.data_dir) / f"{ticker}.csv"
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")

def _load_single(ticker, cfg):
    return _from_yahoo(ticker, cfg.start_date, cfg.end_date) if cfg.data_source == "yahoo" else _from_csv(ticker, cfg)

def load_data(cfg):
    cfg.validate()
    raw = {}
    for ticker in cfg.get_universe():
        df = _load_single(ticker, cfg)
        if df.empty:
            raise ValueError(f"No data for required ticker {ticker}")
        index = pd.DatetimeIndex(df.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        if index.hasnans or index.has_duplicates:
            raise ValueError(f"Invalid or duplicate dates for {ticker}")
        df = df.copy()
        df.index = index
        df = df.sort_index()
        column = "Close" if ticker in YIELD_TICKERS else "Adj Close"
        if column not in df:
            raise ValueError(f"Required column {column!r} missing for {ticker}; adjusted prices are mandatory")
        series = pd.to_numeric(df[column], errors="raise").astype(float)
        series = series.loc[series.index >= pd.Timestamp(cfg.start_date)]
        if cfg.end_date:
            series = series.loc[series.index < pd.Timestamp(cfg.end_date)]
        finite = series.dropna()
        if finite.empty or not np.isfinite(finite).all() or ((finite <= 0).any() if ticker != cfg.rate_ticker else (finite < 0).any()):
            raise ValueError(f"Invalid prices for {ticker}")
        raw[ticker] = series
    # Benchmark sessions define the executable calendar. End is exclusive.
    calendar = raw[cfg.benchmark].dropna().index
    if len(calendar) < 2:
        raise ValueError("Need at least two benchmark sessions")
    result = {}
    for ticker, series in raw.items():
        aligned = series.reindex(calendar.union(series.index)).sort_index()
        # No backward fill. Limit is measured on benchmark sessions.
        result[ticker] = aligned.reindex(calendar).ffill(limit=cfg.max_ffill_sessions)
        if not result[ticker].notna().any():
            raise ValueError(f"No overlapping prices for {ticker}")
    return result
