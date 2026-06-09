"""
config.py — Simulation parameters for the Accelerating Dual Momentum strategy.

All settings are grouped into a single dataclass with sensible defaults,
validated on construction.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ADMConfig:
    # ── Universe ──────────────────────────────────────────────────────────────
    risky_assets: List[str] = field(
        default_factory=lambda: ["SPY", "SCZ"]
    )
    """Tickers of the risky assets (at least 2 for ADM comparison)."""

    defensive_asset: str = "TLT"
    """Single defensive asset ticker, e.g. TLT, BND, SHY."""

    best_of_defensives: List[str] = field(default_factory=list)
    """
    If non-empty, the defensive choice is the one with the best 1-month return
    among these tickers (overrides *defensive_asset*).  Typical use:
    ["TLT", "TIP"].
    """

    benchmark: str = "SPY"
    """Benchmark ticker for comparison charts and metrics."""

    cash_asset: str = "BIL"
    """
    Fallback cash-like asset used when no defensive ticker is available and
    the ADM score is negative.
    """

    # ── Momentum windows (in months) ─────────────────────────────────────────
    momentum_windows_months: List[int] = field(
        default_factory=lambda: [1, 3, 6]
    )
    """
    Periods (in months) used to compute total returns that feed the score.
    Change to [3, 6, 12] or [3, 12] for variants.
    """

    # ── Rebalancing ──────────────────────────────────────────────────────────
    rebalance_freq: str = "ME"
    """Pandas offset string — 'M' for month-end, 'W' for weekly, etc."""

    sleeve_rebalance: bool = False
    """
    If True, split the month into two rebalance dates (1st and 15th)
    to reduce date-bias (half-rebalancing).
    """

    # ── Costs ────────────────────────────────────────────────────────────────
    transaction_cost: float = 0.001
    """One-way trading cost as a fraction (0.001 = 0.1 %)."""

    slippage: float = 0.0005
    """Slippage as a fraction (0.0005 = 0.05 %)."""

    tax_rate: float = 0.0
    """Flat tax rate on realised gains (0.0 = no tax)."""

    # ── Data ──────────────────────────────────────────────────────────────────
    start_date: str = "2000-01-01"
    """Earliest date to load."""

    end_date: Optional[str] = None
    """Latest date to load (None = today)."""

    data_source: str = "yahoo"
    """'yahoo' (live download via yfinance) or 'csv' (local files)."""

    data_dir: str = "data"
    """Directory containing CSV files when data_source='csv'."""

    # ── Misc ──────────────────────────────────────────────────────────────────
    initial_capital: float = 100_000.0
    """Starting portfolio value."""

    verbose: bool = True
    """Print progress messages during backtest."""

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "output"
    """Directory for charts and exported CSV."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.risky_assets) < 2:
            raise ValueError(
                "ADM requires at least 2 risky assets, "
                f"got {len(self.risky_assets)}."
            )
        for w in self.momentum_windows_months:
            if w < 1:
                raise ValueError(
                    f"Momentum window must be >= 1 month, got {w}."
                )
        if self.transaction_cost < 0:
            raise ValueError(
                f"transaction_cost must be >= 0, got {self.transaction_cost}."
            )
        if self.slippage < 0:
            raise ValueError(
                f"slippage must be >= 0, got {self.slippage}."
            )
        if self.initial_capital <= 0:
            raise ValueError(
                f"initial_capital must be > 0, got {self.initial_capital}."
            )
