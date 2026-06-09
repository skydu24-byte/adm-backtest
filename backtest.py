"""
backtest.py — Orchestration engine.

Coordinates data loading → signal computation → portfolio simulation
→ metrics computation → plot generation.
"""

import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from config import ADMConfig
from data_loader import load_data
from signals import compute_signals
from portfolio import simulate as simulate_portfolio
from metrics import compute_metrics
from plots import generate_plots
from export import export_results

logger = logging.getLogger(__name__)


def run(cfg: ADMConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full ADM backtest pipeline.

    Parameters
    ----------
    cfg : ADMConfig
        Fully populated configuration.

    Returns
    -------
    (portfolio_df, metrics_dict)
        portfolio_df — daily NAV, position, returns, benchmark.
        metrics_dict — annualised performance metrics.
    """
    logger.info("Loading data for tickers...")
    prices = load_data(cfg)
    if cfg.verbose:
        for t, s in prices.items():
            logger.info("  %s: %6d rows  [%s → %s]", t, len(s),
                        s.first_valid_index().date() if s.first_valid_index() else "?",
                        s.last_valid_index().date() if s.last_valid_index() else "?")

    logger.info("Computing ADM signals...")
    signals = compute_signals(prices, cfg)

    logger.info("Running portfolio simulation...")
    portfolio_df = simulate_portfolio(prices, signals, cfg)

    logger.info("Computing performance metrics...")
    metrics = compute_metrics(portfolio_df, cfg)

    logger.info("Generating plots...")
    generate_plots(portfolio_df, metrics, cfg)

    logger.info("Exporting results...")
    export_results(portfolio_df, metrics, cfg)

    if cfg.verbose:
        _print_summary(metrics)

    return portfolio_df, metrics


def _print_summary(metrics: Dict) -> None:
    """Pretty-print a summary table to stdout."""
    print("\n" + "=" * 60)
    print("  ADM BACKTEST — PERFORMANCE SUMMARY")
    print("=" * 60)
    for key, val in metrics.items():
        if isinstance(val, float):
            if "return" in key.lower() or "cagr" in key.lower():
                print(f"  {key:30s}: {val:>10.2%}")
            elif "ratio" in key.lower():
                print(f"  {key:30s}: {val:>10.3f}")
            elif "drawdown" in key.lower():
                print(f"  {key:30s}: {val:>10.2%}")
            elif "volatility" in key.lower():
                print(f"  {key:30s}: {val:>10.2%}")
            else:
                print(f"  {key:30s}: {val:>10.4f}")
        else:
            print(f"  {key:30s}: {val}")
    print("=" * 60)
