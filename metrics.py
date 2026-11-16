"""Metrics use net returns consistently; undefined ratios are NaN."""
import numpy as np
import pandas as pd

def monthly_returns(portfolio_df):
    returns = portfolio_df["daily_return"]
    # Includes partial first/last months; these are explicitly documented.
    return returns.add(1).groupby(returns.index.to_period("M")).prod().sub(1)

def _max_drawdown_duration(drawdown):
    start = drawdown.index[0]
    longest = 0
    for dt, value in drawdown.items():
        if value >= -1e-14:
            start = dt
        else:
            longest = max(longest, (dt - start).days)
    return int(longest)

def _stats(nav, returns, initial, years):
    total = nav.iloc[-1] / initial - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    # Omit only the initial baseline row, not a potential initial loss.
    r = returns.iloc[1:] if returns.iloc[0] == 0 else returns
    std = r.std(ddof=1)
    vol = std * np.sqrt(252)
    sharpe = r.mean() / std * np.sqrt(252) if std > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252)
    sortino = r.mean() * 252 / downside if downside > 0 else np.nan
    peak = nav.cummax().clip(lower=initial)
    dd = nav / peak - 1
    return {"total_return": total, "cagr": cagr, "volatility": vol, "sharpe_ratio": sharpe, "sortino_ratio": sortino, "max_drawdown": dd.min(), "max_drawdown_duration_days": _max_drawdown_duration(dd), "calmar_ratio": cagr / abs(dd.min()) if dd.min() < 0 else np.nan}

def compute_metrics(portfolio_df, cfg):
    if len(portfolio_df) < 2:
        raise ValueError("At least two sessions are required for metrics")
    nav = portfolio_df["nav"]
    years = (nav.index[-1] - nav.index[0]).total_seconds() / (365.25 * 86400)
    result = _stats(nav, portfolio_df["daily_return"], cfg.initial_capital, years)
    bench = portfolio_df["bench_return"].add(1).cumprod()
    result.update({"bench_" + key: value for key, value in _stats(bench, portfolio_df["bench_return"], 1.0, years).items()})
    monthly = monthly_returns(portfolio_df)
    result["win_rate"] = float((monthly > 0).mean())
    result["num_trades"] = int(portfolio_df["trade_count"].iloc[-1])
    result["total_fees"] = float(portfolio_df["fees"].sum()) if "fees" in portfolio_df else np.nan
    result["unavailable_target_events"] = int((portfolio_df.get("unavailable_target_weight", pd.Series(dtype=float)) > 0).sum())
    return result
