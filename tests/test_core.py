"""
Unit tests for the ADM backtester.

Run with:
    cd /opt/data/adm-backtest && uv run pytest tests/
"""

from datetime import datetime
import pandas as pd
import numpy as np

# Ensure the project root is importable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ADMConfig
from signals import total_return, adm_score, pick_risky, pick_best_defensive


# ── Helpers ──────────────────────────────────────────────────────────────────

def _simple_price(length: int = 100, start_val: float = 100.0) -> pd.Series:
    """A steadily rising price series at 0.1% per day."""
    np.random.seed(42)
    vals = start_val * np.cumprod(1 + np.random.normal(0.0005, 0.008, length))
    idx = pd.date_range("2020-01-01", periods=length, freq="B")
    return pd.Series(vals, index=idx)


def _flat_price(length: int = 100, val: float = 100.0) -> pd.Series:
    """A flat price series (no momentum)."""
    idx = pd.date_range("2020-01-01", periods=length, freq="B")
    return pd.Series(val, index=idx)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestTotalReturn:
    def test_positive_trend(self):
        s = _simple_price(200, 100.0)
        tr = total_return(s, months=3)
        # After 3 months the return should be positive (rising trend)
        valid = tr.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 0

    def test_flat_series(self):
        s = _flat_price(200, 100.0)
        tr = total_return(s, months=3)
        valid = tr.dropna()
        assert len(valid) > 0
        # Flat → return ≈ 0
        assert abs(valid.iloc[-1]) < 1e-10

    def test_insufficient_history(self):
        s = _simple_price(10, 100.0)
        tr = total_return(s, months=6)
        # 10 business days is less than 6 months → should be NaN everywhere
        assert tr.isna().all()


class TestAdmScore:
    def test_higher_for_rising_asset(self):
        rising = _simple_price(300, 100.0)
        falling = _simple_price(300, 100.0)
        # Invert the second series
        falling = falling.iloc[::-1].reset_index(drop=True)
        falling.index = rising.index  # same index

        score_rising = adm_score(rising, [1, 3, 6])
        score_falling = adm_score(falling, [1, 3, 6])

        valid_r = score_rising.dropna()
        valid_f = score_falling.dropna()
        if len(valid_r) > 0 and len(valid_f) > 0:
            assert valid_r.iloc[-1] > valid_f.iloc[-1]

    def test_score_shape(self):
        s = _simple_price(300, 100.0)
        score = adm_score(s, [1, 3, 6])
        assert len(score) == len(s)
        assert score.index.equals(s.index)


class TestPickRisky:
    def test_pick_best_positive(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="ME")
        scores = pd.DataFrame({
            "SPY": [0.02, 0.03, -0.01, 0.05, 0.00],
            "SCZ": [0.01, 0.04, -0.02, 0.03, -0.01],
        }, index=idx)
        picks = pick_risky(scores)
        # Row 0: SPY (0.02 > 0) → pick SPY
        assert picks.iloc[0] == "SPY"
        # Row 1: SCZ (0.04 > 0.03) → pick SCZ
        assert picks.iloc[1] == "SCZ"
        # Row 2: best is SPY but -0.01 <= 0 → None
        assert pd.isna(picks.iloc[2])
        # Row 3: SPY 0.05 > 0 → SPY
        assert picks.iloc[3] == "SPY"
        # Row 4: best is SPY but 0.00 <= 0 → None
        assert pd.isna(picks.iloc[4])

    def test_all_nan(self):
        idx = pd.date_range("2020-01-01", periods=3, freq="ME")
        scores = pd.DataFrame({"SPY": [np.nan, np.nan, np.nan],
                               "SCZ": [np.nan, np.nan, np.nan]}, index=idx)
        picks = pick_risky(scores)
        assert picks.isna().all()


class TestPickBestDefensive:
    def test_pick_highest_1m(self):
        idx = pd.date_range("2020-01-01", periods=60, freq="B")
        prices = pd.DataFrame({
            "TLT": 100 * np.exp(np.cumsum(np.random.randn(60) * 0.005)),
            "TIP": 100 * np.exp(np.cumsum(np.random.randn(60) * 0.003)),
        }, index=idx)
        picks = pick_best_defensive(prices)
        assert len(picks) == len(prices)
        assert picks.isin(["TLT", "TIP"]).all()


class TestConfigValidation:
    def test_requires_at_least_two_risky(self):
        import pytest
        with pytest.raises(ValueError, match="at least 2 risky"):
            ADMConfig(risky_assets=["SPY"])

    def test_negative_cost_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="transaction_cost"):
            ADMConfig(transaction_cost=-0.001)

    def test_zero_capital_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="initial_capital"):
            ADMConfig(initial_capital=0)
