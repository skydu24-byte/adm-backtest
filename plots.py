"""
plots.py — Visualisation suite for ADM backtest results.

Generates:
  1. Equity curve (portfolio vs benchmark)
  2. Drawdown chart
  3. Monthly return histogram
  4. Annual return heatmap
  5. Allocation timeline
"""

import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from config import ADMConfig

logger = logging.getLogger(__name__)

# Style
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "portfolio": "#2196F3",
    "benchmark": "#9E9E9E",
    "positive": "#4CAF50",
    "negative": "#F44336",
    "drawdown": "#FF5722",
    "cash": "#FFC107",
    "risky": "#2196F3",
}


def generate_plots(portfolio_df: pd.DataFrame, metrics: Dict, cfg: ADMConfig) -> None:
    """Generate and save all plots to *output_dir*."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _equity_curve(portfolio_df, cfg, out / "01_equity_curve.png")
    _drawdown(portfolio_df, cfg, out / "02_drawdown.png")
    _monthly_histogram(portfolio_df, cfg, out / "03_monthly_histogram.png")
    _annual_heatmap(portfolio_df, cfg, out / "04_annual_heatmap.png")
    _allocation(portfolio_df, cfg, out / "05_allocation.png")

    logger.info("Plots saved to %s", out.resolve())
    plt.close("all")


def _equity_curve(df: pd.DataFrame, cfg: ADMConfig, path: Path) -> None:
    """Portfolio NAV vs benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Normalise to 100
    port = df["nav"] / df["nav"].iloc[0] * 100
    bench = (1.0 + df["bench_return"]).cumprod() * 100

    ax.plot(port.index, port, label=f"ADM Portfolio", color=COLORS["portfolio"], linewidth=2)
    ax.plot(bench.index, bench, label=f"Benchmark ({cfg.benchmark})", color=COLORS["benchmark"],
            linewidth=1.5, alpha=0.8)

    ax.set_title("Equity Curve — ADM vs Benchmark", fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of 100")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _drawdown(df: pd.DataFrame, cfg: ADMConfig, path: Path) -> None:
    """Portfolio drawdown over time."""
    nav = df["nav"]
    peak = nav.expanding().max()
    dd = (nav - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.fill_between(dd.index, dd * 100, 0, color=COLORS["drawdown"], alpha=0.5)
    ax.plot(dd.index, dd * 100, color=COLORS["drawdown"], linewidth=1)
    ax.set_title("Portfolio Drawdown", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(True, alpha=0.3)

    # Annotate max drawdown
    min_idx = dd.idxmin()
    min_val = dd.min() * 100
    ax.annotate(f"Max: {min_val:.1f}%",
                xy=(min_idx, min_val),
                xytext=(min_idx, min_val + 5),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _monthly_histogram(df: pd.DataFrame, cfg: ADMConfig, path: Path) -> None:
    """Distribution of monthly returns."""
    # Resample daily returns to actual monthly returns
    monthly = df["nav"].resample("ME").last().pct_change().dropna() * 100

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = max(int(np.sqrt(len(monthly))), 15)
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in monthly]

    ax.hist(monthly, bins=bins, edgecolor="white", linewidth=0.5,
            color=COLORS["positive"], alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    mean_ret = monthly.mean()
    ax.axvline(mean_ret, color=COLORS["portfolio"], linewidth=1.5, linestyle="-",
               label=f"Mean: {mean_ret:.2f}%")

    ax.set_title("Distribution of Monthly Returns", fontsize=13, fontweight="bold")
    ax.set_xlabel("Monthly Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _annual_heatmap(df: pd.DataFrame, cfg: ADMConfig, path: Path) -> None:
    """Heatmap of monthly returns by year (calendar view)."""
    nav = df["nav"]
    monthly = nav.resample("ME").last().pct_change().dropna()

    # Build year × month table
    years = monthly.index.year
    months = monthly.index.month
    pivot = pd.DataFrame({"ret": monthly.values * 100, "year": years, "month": months})
    table = pivot.pivot_table(index="year", columns="month", values="ret", aggfunc="first")
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    vmax = max(abs(table.min().min()), abs(table.max().max()), 5)

    fig, ax = plt.subplots(figsize=(12, max(5, len(table) * 0.4 + 2)))
    cmap = LinearSegmentedColormap.from_list(
        "rd_gn",
        [COLORS["negative"], "#FFFFFF", COLORS["positive"]],
        N=256,
    )

    sns.heatmap(table, annot=True, fmt=".1f", cmap=cmap, center=0,
                vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor="#DDD",
                cbar_kws={"label": "Monthly Return (%)"}, ax=ax)

    ax.set_title("Monthly Returns — Calendar Heatmap (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Year")
    ax.set_xlabel("")

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _allocation(df: pd.DataFrame, cfg: ADMConfig, path: Path) -> None:
    """Stacked area chart showing which asset is held over time."""
    positions = df["position"].dropna().unique()
    if len(positions) == 0:
        return

    # Build weekly allocation matrix
    weekly = df.resample("W").last()
    alloc = pd.DataFrame(0.0, index=weekly.index, columns=positions)
    for pos in positions:
        mask = weekly["position"] == pos
        alloc.loc[mask, pos] = 1.0

    fig, ax = plt.subplots(figsize=(12, 3.5))
    colors_list = sns.color_palette("husl", len(positions))
    alloc.plot.area(ax=ax, color=colors_list, linewidth=0, alpha=0.85)

    ax.set_title("Allocation Over Time", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right", ncol=len(positions), fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
