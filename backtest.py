"""Validated pipeline: data -> signals -> accounting -> metrics -> outputs."""
from data_loader import load_data
from signals import compute_signals
from strategies import compute_signals_enhanced_dm, compute_signals_managed_futures_dm, compute_signals_vigilant_dm
from portfolio import simulate
from metrics import compute_metrics
from plots import generate_plots
from export import export_results

SIGNAL_REGISTRY = {"adm": compute_signals, "enhanced_dm": compute_signals_enhanced_dm, "managed_futures_dm": compute_signals_managed_futures_dm, "vigilant_dm": compute_signals_vigilant_dm}

def run(cfg):
    cfg.validate()  # CLI/GUI mutate fields after construction.
    prices = load_data(cfg)
    signals = SIGNAL_REGISTRY[cfg.strategy](prices, cfg)
    portfolio = simulate(prices, signals, cfg)
    metrics = compute_metrics(portfolio, cfg)
    generate_plots(portfolio, metrics, cfg, signals=signals)
    export_results(portfolio, metrics, cfg)
    if cfg.verbose:
        _print_summary(metrics)
    return portfolio, metrics

def _print_summary(metrics):
    for key, value in metrics.items():
        print(f"{key:35s}: {value}")
