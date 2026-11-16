"""Export actual holdings and net performance, including partial months."""
from pathlib import Path
import pandas as pd
from metrics import monthly_returns

def export_results(portfolio_df, metrics, cfg):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    portfolio_df.to_csv(out / "portfolio_daily.csv")
    grouped = portfolio_df.groupby(portfolio_df.index.to_period("M"))
    monthly = grouped["nav"].last().to_frame()
    monthly["monthly_return"] = monthly_returns(portfolio_df)
    monthly.to_csv(out / "portfolio_monthly.csv")
    columns = [c for c in portfolio_df if c.startswith("weight_")]
    grouped[columns].last().to_csv(out / "allocations_monthly.csv")
    pd.DataFrame([metrics]).to_csv(out / "performance_metrics.csv", index=False)
