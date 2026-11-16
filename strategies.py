"""Momentum strategies using consistent calendar-month lookbacks."""
import pandas as pd
from schedule import monthly_returns, month_prices, rebalance_dates, weight_signals

def _best(row, candidates, positive=True):
    valid = row.reindex(candidates).dropna()
    if valid.empty or (positive and valid.max() <= 0):
        return None
    return valid.idxmax()

def _compute(prices, cfg, kind):
    cfg.validate()
    index = prices[cfg.benchmark].index
    tickers = [t for t in cfg.get_universe() if t != cfg.rate_ticker]
    scores = pd.DataFrame({t: monthly_returns(prices[t], cfg.momentum_lookback_months) for t in tickers})
    dates = rebalance_dates(index, cfg.rebalance_freq)
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    rate_change = None
    if kind == "vigilant_dm":
        rate_change = month_prices(prices[cfg.rate_ticker]).diff(6).reindex(index, method="ffill")
    for dt in dates:
        row = scores.loc[dt]
        selected = []
        if kind == "enhanced_dm":
            valid = row.reindex(cfg.multi_asset_universe).dropna()
            selected = list(valid[valid > 0].sort_values(ascending=False, kind="stable").head(cfg.enhanced_top_n).index)
        else:
            risky = cfg.managed_futures_risky if kind == "managed_futures_dm" else cfg.vigilant_risky
            if pd.notna(row[risky[0]]) and row[risky[0]] > 0:
                chosen = _best(row, risky)
            elif kind == "managed_futures_dm":
                chosen = _best(row, cfg.managed_futures_defensive)
            elif pd.notna(rate_change.loc[dt]):
                candidates = cfg.vigilant_defensive_rising if rate_change.loc[dt] > 0 else cfg.vigilant_defensive_falling
                chosen = _best(row, candidates)
            else:
                chosen = None
            selected = [chosen] if chosen else []
        selected = selected or [cfg.cash_asset]
        weights.loc[dt, selected] = 1.0 / len(selected)
    return weight_signals(weights, index, cfg)

def compute_signals_enhanced_dm(prices, cfg):
    return _compute(prices, cfg, "enhanced_dm")

def compute_signals_managed_futures_dm(prices, cfg):
    return _compute(prices, cfg, "managed_futures_dm")

def compute_signals_vigilant_dm(prices, cfg):
    return _compute(prices, cfg, "vigilant_dm")
