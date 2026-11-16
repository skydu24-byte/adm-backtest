"""Validated parameters. All prices must use the same reporting currency."""
from dataclasses import dataclass, field
from typing import List, Optional
import math
import pandas as pd

@dataclass
class ADMConfig:
    strategy: str = "adm"
    risky_assets: List[str] = field(default_factory=lambda: ["SPY", "SCZ"])
    defensive_asset: str = "TLT"
    best_of_defensives: List[str] = field(default_factory=list)
    multi_asset_universe: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "GLD", "DBC", "VNQ", "KMLM", "SHY", "BIL"])
    managed_futures_risky: List[str] = field(default_factory=lambda: ["SPY", "VEA"])
    managed_futures_defensive: List[str] = field(default_factory=lambda: ["KMLM", "DBMF", "FLOT", "BIL"])
    vigilant_risky: List[str] = field(default_factory=lambda: ["SPY", "VEA"])
    vigilant_defensive_rising: List[str] = field(default_factory=lambda: ["SHY", "GLD", "BIL", "USFR"])
    vigilant_defensive_falling: List[str] = field(default_factory=lambda: ["TLT", "GLD", "SHY", "IEF"])
    rate_ticker: str = "^TNX"
    benchmark: str = "SPY"
    cash_asset: str = "BIL"
    momentum_windows_months: List[int] = field(default_factory=lambda: [1, 3, 6])
    enhanced_top_n: int = 4
    momentum_lookback_months: int = 12
    rebalance_freq: str = "ME"
    sleeve_rebalance: bool = False
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    tax_rate: float = 0.0
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    data_source: str = "yahoo"
    data_dir: str = "data"
    initial_capital: float = 100_000.0
    verbose: bool = True
    output_dir: str = "output"
    max_ffill_sessions: int = 5

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.strategy not in {"adm", "enhanced_dm", "managed_futures_dm", "vigilant_dm"}:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        for key in ("risky_assets", "multi_asset_universe", "managed_futures_risky", "managed_futures_defensive", "vigilant_risky", "vigilant_defensive_rising", "vigilant_defensive_falling", "best_of_defensives"):
            value = getattr(self, key)
            if not isinstance(value, list) or any(not isinstance(t, str) or not t.strip() for t in value) or len(value) != len(set(value)):
                raise ValueError(f"{key} must be a list of unique nonempty tickers")
            if key != "best_of_defensives" and not value:
                raise ValueError(f"{key} cannot be empty")
        if self.strategy == "adm" and len(self.risky_assets) < 2:
            raise ValueError("ADM requires at least 2 risky assets")
        for key in ("defensive_asset", "rate_ticker", "benchmark", "cash_asset"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a nonempty ticker")
        if not isinstance(self.momentum_windows_months, list) or not self.momentum_windows_months:
            raise ValueError("momentum_windows_months cannot be empty")
        for value in [*self.momentum_windows_months, self.momentum_lookback_months, self.enhanced_top_n, self.max_ffill_sessions]:
            if type(value) is not int or value < 1:
                raise ValueError("Windows, top N and fill limit must be positive integers")
        for key in ("transaction_cost", "slippage", "tax_rate", "initial_capital"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if self.transaction_cost + self.slippage >= 1:
            raise ValueError("Combined per-side costs must be < 1")
        if self.tax_rate != 0:
            raise ValueError("tax_rate is not implemented; use 0 (pre-tax simulation)")
        if self.sleeve_rebalance:
            raise ValueError("Independent half-portfolio sleeves are not implemented")
        if self.rebalance_freq not in {"M", "ME", "W", "W-FRI"}:
            raise ValueError("rebalance_freq must be ME, W or W-FRI (M is an alias)")
        if self.data_source not in {"yahoo", "csv"}:
            raise ValueError("data_source must be yahoo or csv")
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date) if self.end_date else None
        if pd.isna(start) or (end is not None and (pd.isna(end) or end <= start)):
            raise ValueError("end_date must be after start_date")

    def get_universe(self):
        base = [self.benchmark, self.cash_asset]
        if self.strategy == "adm":
            extra = self.risky_assets + [self.defensive_asset] + self.best_of_defensives
        elif self.strategy == "enhanced_dm":
            extra = self.multi_asset_universe
        elif self.strategy == "managed_futures_dm":
            extra = self.managed_futures_risky + self.managed_futures_defensive
        else:
            extra = self.vigilant_risky + self.vigilant_defensive_rising + self.vigilant_defensive_falling + [self.rate_ticker]
        return sorted(set(base + extra))
