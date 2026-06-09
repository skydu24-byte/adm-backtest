"""
Example configuration file for ADM backtester.

Usage:
    python main.py --config examples/my_config.py
"""

from config import ADMConfig

# ── Primary universe (US + International equities, TLT for defence) ─────────
cfg = ADMConfig(
    risky_assets=["SPY", "SCZ"],
    defensive_asset="TLT",
    best_of_defensives=[],          # single defensive asset
    benchmark="SPY",

    momentum_windows_months=[1, 3, 6],

    start_date="2000-01-01",
    end_date=None,                  # up to today

    transaction_cost=0.001,
    slippage=0.0005,
    tax_rate=0.0,

    initial_capital=100_000.0,
    verbose=True,
    output_dir="output",
)
