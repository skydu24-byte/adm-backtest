"""
config.py — Simulation parameters for all portfolio strategies.

All settings are grouped into a single dataclass with sensible defaults,
validated on construction.  The *strategy* field selects which signal
engine to use.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ADMConfig:
    # ── Strategy selector ──────────────────────────────────────────────────
    strategy: str = "adm"
    """
    Which strategy to run.  Options:
      - "adm"               : Accelerating Dual Momentum (original)
      - "enhanced_dm"       : Enhanced Dual Momentum — multi-asset, 12m momentum, top-N
      - "managed_futures_dm": Managed Futures Dual Momentum — KMLM/DBMF replaces TLT
      - "vigilant_dm"       : Vigilant Dual Momentum — rate-aware hedge selection
    """

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

    # ── Enhanced / Managed-Futures DM universe ────────────────────────────
    multi_asset_universe: List[str] = field(
        default_factory=lambda: [
            "SPY", "QQQ", "GLD", "DBC", "VNQ", "KMLM", "SHY", "BIL"
        ]
    )
    """Asset pool for enhanced_dm strategy."""

    managed_futures_risky: List[str] = field(
        default_factory=lambda: ["SPY", "VEA"]
    )
    """Equity assets for managed_futures_dm."""

    managed_futures_defensive: List[str] = field(
        default_factory=lambda: ["KMLM", "DBMF", "FLOT", "BIL"]
    )
    """Defensive assets for managed_futures_dm (replaces TLT)."""

    vigilant_risky: List[str] = field(
        default_factory=lambda: ["SPY", "VEA"]
    )
    """Equity assets for vigilant_dm."""

    vigilant_defensive_rising: List[str] = field(
        default_factory=lambda: ["SHY", "GLD", "BIL", "USFR"]
    )
    """Safe havens when rates are rising."""

    vigilant_defensive_falling: List[str] = field(
        default_factory=lambda: ["TLT", "GLD", "SHY", "IEF"]
    )
    """Safe havens when rates are falling."""

    rate_ticker: str = "^TNX"
    """Ticker for the 10-year Treasury yield (used by vigilant_dm)."""

    benchmark: str = "SPY"
    """Benchmark ticker for comparison charts and metrics."""

    cash_asset: str = "BIL"
    """Fallback cash-like asset."""

    # ── Momentum windows (in months) ─────────────────────────────────────────
    momentum_windows_months: List[int] = field(
        default_factory=lambda: [1, 3, 6]
    )
    """
    Periods (in months) used to compute total returns that feed the score.
    For ADM: [1, 3, 6] by default.
    For other strategies, only the first value is used as the 12m lookback
    equivalent (configurable).
    """

    # ── Enhanced DM specific ────────────────────────────────────────────────
    enhanced_top_n: int = 4
    """Number of top assets to hold in enhanced_dm (default: 4)."""

    momentum_lookback_months: int = 12
    """Primary momentum lookback in months for non-ADM strategies."""

    # ── Rebalancing ──────────────────────────────────────────────────────────
    rebalance_freq: str = "ME"
    """Pandas offset string — 'ME' for month-end, 'W' for weekly, etc."""

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
        valid_strategies = {"adm", "enhanced_dm", "managed_futures_dm", "vigilant_dm"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                f"Choose from {sorted(valid_strategies)}."
            )

        if self.strategy == "adm" and len(self.risky_assets) < 2:
            raise ValueError(
                "ADM requires at least 2 risky assets, "
                f"got {len(self.risky_assets)}."
            )

        for w in self.momentum_windows_months:
            if w < 1:
                raise ValueError(
                    f"Momentum window must be >= 1 month, got {w}."
                )
        if self.momentum_lookback_months < 1:
            raise ValueError(
                f"momentum_lookback_months must be >= 1, "
                f"got {self.momentum_lookback_months}."
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
        if self.enhanced_top_n < 1:
            raise ValueError(
                f"enhanced_top_n must be >= 1, got {self.enhanced_top_n}."
            )

    def get_universe(self) -> List[str]:
        """Return the list of all tickers needed for the selected strategy."""
        base = [self.benchmark, self.cash_asset]

        if self.strategy == "adm":
            return list(set(base + self.risky_assets
                            + [self.defensive_asset]
                            + self.best_of_defensives))

        elif self.strategy == "enhanced_dm":
            return list(set(base + self.multi_asset_universe))

        elif self.strategy == "managed_futures_dm":
            return list(set(base + self.managed_futures_risky
                            + self.managed_futures_defensive))

        elif self.strategy == "vigilant_dm":
            return list(set(base + self.vigilant_risky
                            + self.vigilant_defensive_rising
                            + self.vigilant_defensive_falling
                            + [self.rate_ticker]))

        return base
