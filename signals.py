"""ADM signals known at the close; execution is handled by portfolio.py."""
import pandas as pd
from schedule import monthly_returns, rebalance_dates

def total_return(series, months):
    return monthly_returns(series, months)

def adm_score(price, windows_months):
    returns = [total_return(price, m) for m in windows_months]
    return pd.concat(returns, axis=1).sum(axis=1, min_count=len(returns))

def pick_risky(scores):
    def best(row):
        valid = row.dropna()
        return valid.idxmax() if not valid.empty and valid.max() > 0 else None
    return scores.apply(best, axis=1)

def pick_best_defensive(prices):
    ret = prices.apply(lambda s: monthly_returns(s, 1))
    return ret.apply(lambda row: row.idxmax() if row.notna().any() else prices.columns[0], axis=1)

def _rebalance_dates(date_index, cfg):
    return rebalance_dates(date_index, cfg.rebalance_freq)

def compute_signals(prices, cfg):
    cfg.validate()
    risky = pd.DataFrame({t: prices[t] for t in cfg.risky_assets})
    scores = risky.apply(lambda s: adm_score(s, cfg.momentum_windows_months))
    chosen = pick_risky(scores)
    defensive = pick_best_defensive(pd.DataFrame({t: prices[t] for t in cfg.best_of_defensives})) if cfg.best_of_defensives else pd.Series(cfg.defensive_asset, index=risky.index)
    position = chosen.where(chosen.notna(), defensive)
    dates = _rebalance_dates(risky.index, cfg)
    out = pd.DataFrame({"risky_pick": chosen, "defensive_pick": defensive, "regime": chosen.notna().map({True: "risky", False: "defensive"}), "position": position})
    out = out.loc[dates].reindex(risky.index).ffill()
    out["__rebalance__"] = out.index.isin(dates)
    return out
