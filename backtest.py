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
from signals import compute_signals as adm_signals
from strategies import (
    compute_signals_enhanced_dm,
    compute_signals_managed_futures_dm,
    compute_signals_vigilant_dm,
)
from portfolio import simulate as simulate_portfolio
from metrics import compute_metrics
from plots import generate_plots
from export import export_results

logger = logging.getLogger(__name__)


# ── Signal dispatch ──────────────────────────────────────────────────────────

SIGNAL_REGISTRY = {
    "adm": adm_signals,
    "enhanced_dm": compute_signals_enhanced_dm,
    "managed_futures_dm": compute_signals_managed_futures_dm,
    "vigilant_dm": compute_signals_vigilant_dm,
}


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(cfg: ADMConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full backtest pipeline.

    Parameters
    ----------
    cfg : ADMConfig
        Fully populated configuration.

    Returns
    -------
    (portfolio_df, metrics_dict)
    """
    logger.info("Loading data for tickers...")
    prices = load_data(cfg)
    if cfg.verbose:
        for t, s in sorted(prices.items()):
            first = s.first_valid_index()
            last = s.last_valid_index()
            logger.info("  %s: %6d rows  [%s → %s]",
                        t, len(s),
                        first.date() if first else "?",
                        last.date() if last else "?")

    # ── Compute signals ─────────────────────────────────────────────────────
    signal_fn = SIGNAL_REGISTRY.get(cfg.strategy)
    if signal_fn is None:
        raise ValueError(
            f"Unknown strategy '{cfg.strategy}'. "
            f"Available: {list(SIGNAL_REGISTRY.keys())}"
        )

    logger.info("Computing signals (%s)...", cfg.strategy)
    try:
        signals = signal_fn(prices, cfg)
    except Exception:
        logger.exception("Signal computation failed for strategy '%s'.", cfg.strategy)
        raise

    logger.info("  Signal shape: %s", signals.shape)

    # ── Run simulation ──────────────────────────────────────────────────────
    logger.info("Running portfolio simulation...")
    portfolio_df = simulate_portfolio(prices, signals, cfg)

    # ── Metrics ─────────────────────────────────────────────────────────────
    logger.info("Computing performance metrics...")
    metrics = compute_metrics(portfolio_df, cfg)

    # ── Plots ───────────────────────────────────────────────────────────────
    logger.info("Generating plots...")
    generate_plots(portfolio_df, metrics, cfg, signals=signals)

    # ── Export ──────────────────────────────────────────────────────────────
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
