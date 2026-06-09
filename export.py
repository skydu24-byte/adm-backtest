"""
export.py — Export backtest results to CSV.
"""

import logging
from pathlib import Path
from typing import Dict
import pandas as pd

from config import ADMConfig

logger = logging.getLogger(__name__)


def export_results(portfolio_df: pd.DataFrame, metrics: Dict, cfg: ADMConfig) -> None:
    """Write portfolio holdings and performance data to CSV."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Daily portfolio log ────────────────────────────────────────────────
    export_df = portfolio_df.copy()
    export_df = export_df.round(6)
    path_daily = out / "portfolio_daily.csv"
    export_df.to_csv(path_daily)
    logger.info("Daily portfolio log → %s", path_daily.resolve())

    # ── Monthly summary ────────────────────────────────────────────────────
    monthly = portfolio_df["nav"].resample("ME").last()
    monthly_ret = monthly.pct_change().dropna()
    monthly_df = pd.DataFrame({
        "nav": monthly,
        "monthly_return": monthly_ret,
    })
    path_monthly = out / "portfolio_monthly.csv"
    monthly_df.to_csv(path_monthly)
    logger.info("Monthly summary → %s", path_monthly.resolve())

    # ── Allocation table ───────────────────────────────────────────────────
    # Monthly snapshot of position
    alloc = portfolio_df["position"].resample("ME").last().to_frame()
    path_alloc = out / "allocations_monthly.csv"
    alloc.to_csv(path_alloc)
    logger.info("Monthly allocations → %s", path_alloc.resolve())

    # ── Metrics summary ────────────────────────────────────────────────────
    metrics_df = pd.DataFrame([metrics])
    path_metrics = out / "performance_metrics.csv"
    metrics_df.to_csv(path_metrics, index=False)
    logger.info("Performance metrics → %s", path_metrics.resolve())
