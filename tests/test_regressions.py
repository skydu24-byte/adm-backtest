"""Deterministic accounting, scheduling, metrics and input regressions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from config import ADMConfig
from portfolio import simulate
from metrics import compute_metrics, monthly_returns, _max_drawdown_duration
from signals import adm_score, total_return
from schedule import rebalance_dates
from data_loader import load_data
from backtest import SIGNAL_REGISTRY

def fixture(prices, targets, dates=None, fee=0):
    dates = pd.DatetimeIndex(dates or ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01"])
    cfg = ADMConfig(initial_capital=100, transaction_cost=fee, slippage=0, benchmark="A", cash_asset="B")
    data = {t: pd.Series(v, index=dates, dtype=float) for t, v in prices.items()}
    sig = pd.DataFrame(targets, index=dates)
    return simulate(data, sig, cfg), cfg

def test_next_close_execution_does_not_capture_signal_or_execution_day_jump():
    df, _ = fixture({"A": [100, 200, 400, 800], "B": [100]*4}, {"A": [1]*4, "__rebalance__": [True, False, False, False]})
    assert df.nav.tolist() == pytest.approx([100, 100, 200, 400])

def test_initial_signal_is_actually_invested():
    df, _ = fixture({"A": [100]*4, "B": [100]*4}, {"position": ["A"]*4, "__rebalance__": [True, False, False, False]})
    assert df.weight_A.iloc[1] == pytest.approx(1)
    assert df.trade_count.iloc[-1] == 1

def test_weights_drift_without_unreported_daily_rebalancing():
    df, _ = fixture({"A": [100, 100, 200, 400], "B": [100]*4}, {"A": [.5]*4, "B": [.5]*4, "__rebalance__": [True, False, False, False]})
    assert df.nav.iloc[-1] == pytest.approx(250)
    assert df.weight_A.iloc[-1] == pytest.approx(.8)
    assert df.trade_count.iloc[-1] == 1

def test_same_target_rebalances_drift_when_scheduled():
    df, _ = fixture({"A": [100, 100, 200, 200], "B": [100]*4}, {"A": [.5]*4, "B": [.5]*4, "__rebalance__": [True, False, True, False]})
    assert df.weight_A.iloc[-1] == pytest.approx(.5)
    assert df.trade_count.iloc[-1] == 2

def test_costs_are_per_side_and_cash_is_nonnegative():
    df, cfg = fixture({"A": [100]*4, "B": [100]*4}, {"A": [1, 0, 0, 0], "B": [0, 1, 1, 1], "__rebalance__": [True, True, False, False]}, fee=.01)
    expected = 100 / 1.01 * .99 / 1.01
    assert df.nav.iloc[-1] == pytest.approx(expected)
    assert df.fees.sum() == pytest.approx(100 - expected)
    assert df.weight_CASH.min() >= 0
    assert (1 + df.daily_return).prod() == pytest.approx(expected / cfg.initial_capital)

def test_risk_metrics_include_cost_losses():
    df, cfg = fixture({"A": [100]*4, "B": [100]*4}, {"A": [1, 0, 0, 0], "B": [0, 1, 1, 1], "__rebalance__": [True, True, False, False]}, fee=.01)
    metrics = compute_metrics(df, cfg)
    assert metrics["volatility"] > 0
    assert metrics["sharpe_ratio"] < 0

def test_month_end_weekend_uses_last_observed_session():
    idx = pd.bdate_range("2024-03-01", "2024-04-30")
    assert pd.Timestamp("2024-03-29") in rebalance_dates(idx)

def test_weekly_is_not_monthly():
    idx = pd.bdate_range("2024-01-01", "2024-01-31")
    assert len(rebalance_dates(idx, "W-FRI")) == 5
    assert len(rebalance_dates(idx, "ME")) == 1

def test_monthly_return_uses_actual_last_session():
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    s = pd.Series(np.arange(len(idx)) + 100., index=idx)
    ret = total_return(s, 1)
    assert ret.loc["2024-03-29"] == pytest.approx(s.loc["2024-03-29"] / s.loc["2024-02-29"] - 1)

def test_adm_requires_all_windows():
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    assert adm_score(pd.Series(np.arange(len(idx)) + 100., index=idx), [1, 3, 6]).isna().all()

def test_win_rate_counts_months_not_daily_cumulatives():
    idx = pd.to_datetime(["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01"])
    nav = pd.Series([100., 120., 110., 99.], index=idx)
    df = pd.DataFrame({"nav": nav, "daily_return": nav.pct_change().fillna(0), "bench_return": 0., "trade_count": 0})
    assert monthly_returns(df).tolist() == pytest.approx([.1, -.1])
    assert compute_metrics(df, ADMConfig(initial_capital=100))["win_rate"] == .5

def test_drawdown_duration_uses_calendar_days():
    dd = pd.Series([0., -.1, -.2, 0.], index=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]))
    assert _max_drawdown_duration(dd) == 4

def test_missing_target_stays_in_explicit_cash():
    df, _ = fixture({"A": [100]*4, "B": [np.nan, np.nan, 100, 100]}, {"B": [1]*4, "__rebalance__": [True, False, False, False]})
    assert df.weight_CASH.iloc[1] == pytest.approx(1)
    assert df.unavailable_target_weight.iloc[1] == 1

def test_missing_held_price_is_not_valued_at_zero():
    with pytest.raises(ValueError, match="held asset"):
        fixture({"A": [100, 100, np.nan, 100], "B": [100]*4}, {"A": [1]*4, "__rebalance__": [True, False, False, False]})

@pytest.mark.parametrize("kwargs", [{"tax_rate": .3}, {"sleeve_rebalance": True}, {"rebalance_freq": "nonsense"}, {"momentum_windows_months": []}, {"transaction_cost": float("nan")}])
def test_unsupported_or_invalid_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        ADMConfig(**kwargs)

def test_post_construction_mutation_is_validated():
    cfg = ADMConfig()
    cfg.transaction_cost = -1
    with pytest.raises(ValueError):
        cfg.validate()

def test_no_backward_fill_and_gap_limit(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=10)
    cfg = ADMConfig(start_date="2024-01-01", max_ffill_sessions=2)
    def fake(ticker, cfg):
        values = np.full(10, 100.)
        if ticker == "SCZ":
            values[:3] = np.nan
        if ticker == "TLT":
            values[3:7] = np.nan
        return pd.DataFrame({"Adj Close": values}, index=idx)
    monkeypatch.setattr("data_loader._load_single", fake)
    data = load_data(cfg)
    assert data["SCZ"].iloc[:3].isna().all()
    assert data["TLT"].iloc[4] == 100
    assert pd.isna(data["TLT"].iloc[5])

def test_duplicate_dates_rejected(monkeypatch):
    idx = pd.to_datetime(["2024-01-01", "2024-01-01"])
    monkeypatch.setattr("data_loader._load_single", lambda t, c: pd.DataFrame({"Adj Close": [100, 101]}, index=idx))
    with pytest.raises(ValueError, match="duplicate"):
        load_data(ADMConfig())

@pytest.mark.parametrize("strategy", list(SIGNAL_REGISTRY))
def test_all_strategies_generate_valid_accounting(strategy):
    cfg = ADMConfig(strategy=strategy, start_date="2022-01-01", momentum_lookback_months=3, transaction_cost=.001)
    idx = pd.bdate_range("2022-01-03", "2024-02-29")
    prices = {t: pd.Series(100 * np.exp(np.arange(len(idx)) * (.0001 + k * .00005)), index=idx) for k, t in enumerate(cfg.get_universe())}
    sig = SIGNAL_REGISTRY[strategy](prices, cfg)
    df = simulate(prices, sig, cfg)
    assert np.isfinite(df.nav).all()
    assert df.filter(like="weight_").sum(axis=1).to_numpy() == pytest.approx(np.ones(len(idx)))
    assert (1 + df.daily_return).prod() == pytest.approx(df.nav.iloc[-1] / cfg.initial_capital)

def test_future_prices_do_not_change_past_accounting():
    cfg = ADMConfig(strategy="enhanced_dm", momentum_lookback_months=3)
    idx = pd.bdate_range("2022-01-03", "2024-02-29")
    prices = {t: pd.Series(100 * np.exp(np.arange(len(idx)) * .001), index=idx) for t in cfg.get_universe()}
    a = simulate(prices, SIGNAL_REGISTRY[cfg.strategy](prices, cfg), cfg)
    changed = {t: s.copy() for t, s in prices.items()}
    for t, s in changed.items():
        if t != cfg.benchmark:
            s.loc["2023-09-01":] *= 3
    b = simulate(changed, SIGNAL_REGISTRY[cfg.strategy](changed, cfg), cfg)
    pd.testing.assert_frame_equal(a.loc[:"2023-08-31"], b.loc[:"2023-08-31"])

def test_short_period_plots_and_csv_exports(tmp_path):
    from plots import generate_plots
    from export import export_results
    df, cfg = fixture({"A": [100, 100, 110, 120], "B": [100]*4}, {"A": [1]*4, "__rebalance__": [True, False, False, False]})
    cfg.output_dir = str(tmp_path)
    metrics = compute_metrics(df, cfg)
    generate_plots(df, metrics, cfg)
    export_results(df, metrics, cfg)
    assert len(list(tmp_path.glob("*.png"))) == 5
    assert len(list(tmp_path.glob("*.csv"))) == 4
    assert "weight_CASH" in pd.read_csv(tmp_path / "allocations_monthly.csv")


@pytest.mark.parametrize("strategy", list(SIGNAL_REGISTRY))
def test_full_csv_pipeline(strategy, tmp_path):
    from backtest import run
    cfg = ADMConfig(strategy=strategy, data_source="csv", data_dir=str(tmp_path / "data"), output_dir=str(tmp_path / "out"), start_date="2022-01-01", end_date="2023-06-01", momentum_lookback_months=3, verbose=False)
    Path(cfg.data_dir).mkdir()
    idx = pd.bdate_range("2022-01-03", "2023-05-31")
    for k, ticker in enumerate(cfg.get_universe()):
        column = "Close" if ticker == cfg.rate_ticker else "Adj Close"
        pd.DataFrame({"Date": idx, column: 100 * np.exp(np.arange(len(idx)) * (.0001 + .00001 * k))}).to_csv(Path(cfg.data_dir) / f"{ticker}.csv", index=False)
    df, metrics = run(cfg)
    assert len(df) == len(idx)
    assert len(list(Path(cfg.output_dir).glob("*.png"))) == 5
    assert metrics["unavailable_target_events"] == 0

@pytest.mark.parametrize("bad", [-.1, float("nan"), 1.1])
def test_invalid_target_weights_fail(bad):
    with pytest.raises(ValueError, match="Target weights"):
        fixture({"A": [100]*4, "B": [100]*4}, {"A": [bad]*4, "__rebalance__": [True, False, False, False]})
