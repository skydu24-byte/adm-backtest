"""Headless plots of net returns and actual, drifting allocations."""
from pathlib import Path
import calendar
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import monthly_returns

def generate_plots(portfolio_df, metrics, cfg, signals=None):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = portfolio_df
    monthly = monthly_returns(df) * 100
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df.nav / cfg.initial_capital * 100, label=cfg.strategy)
    ax.plot(df.index, df.bench_return.add(1).cumprod() * 100, label=cfg.benchmark)
    ax.set(ylabel="Growth of 100 (same input currency)", title="Net portfolio vs benchmark")
    ax.legend()
    save(fig, "01_equity_curve.png")
    fig, ax = plt.subplots(figsize=(11, 3))
    dd = df.nav / df.nav.cummax().clip(lower=cfg.initial_capital) - 1
    ax.fill_between(df.index, dd.to_numpy() * 100, 0, alpha=.4)
    ax.set(ylabel="Drawdown (%)")
    save(fig, "02_drawdown.png")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(monthly.to_numpy(), bins=max(1, min(30, int(np.sqrt(len(monthly))))))
    ax.set(xlabel="Monthly net return (%)", title="Partial boundary months included")
    save(fig, "03_monthly_histogram.png")
    table = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "return": monthly.values}).pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))
    table.columns = list(calendar.month_abbr)[1:]
    fig, ax = plt.subplots(figsize=(11, max(3, len(table) * .35)))
    sns.heatmap(table, annot=True, fmt=".1f", center=0, cmap="RdYlGn", ax=ax)
    save(fig, "04_annual_heatmap.png")
    fig, ax = plt.subplots(figsize=(11, 3))
    weights = df.filter(like="weight_").copy()
    weights.columns = weights.columns.str.removeprefix("weight_")
    weights.loc[:, weights.max() > 0].plot.area(ax=ax, linewidth=0)
    ax.set(ylim=(0, 1), ylabel="Actual weight")
    save(fig, "05_allocation.png")
