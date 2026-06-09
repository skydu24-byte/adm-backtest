# Accelerating Dual Momentum (ADM) — Backtester

A modular, production-quality Python backtesting engine for the **Accelerating Dual Momentum** strategy, as described by the quantitative finance community.

## Strategy in a nutshell

At each month-end:
1. Compute an **ADM score** for each risky asset = sum of total returns over 1, 3 and 6 months (configurable windows).
2. Pick the risky asset with the **highest positive** score → allocate **100%** to it.
3. If **both scores are ≤ 0**, switch to a **defensive asset** (TLT by default, or the best-performing among a set of defensives over 1 month).
4. Hold until the next rebalance.

## Project structure

```
adm-backtest/
├── config.py          # Dataclass for all simulation parameters
├── data_loader.py     # Price data ingestion (Yahoo Finance / CSV)
├── signals.py         # Momentum computation & ADM scoring logic
├── portfolio.py       # Trade execution & NAV simulation
├── backtest.py        # Pipeline orchestration
├── metrics.py         # Performance metrics (Sharpe, DD, CAGR…)
├── plots.py           # Matplotlib/Seaborn visualisations
├── export.py          # CSV export
├── main.py            # CLI entry point
├── requirements.txt
├── examples/
│   └── default_config.py
├── tests/
│   └── test_core.py
├── data/              # CSV data directory (optional)
└── output/            # Generated charts and CSVs
```

## Installation

```bash
# Clone or copy the project
cd adm-backtest

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> **Optional:** yfinance is required for live Yahoo Finance downloads.

## Usage

### Run with defaults (SPY, SCZ, TTL — 2000 to today)

```bash
uv run python main.py -v
```

### Override parameters on the CLI

```bash
uv run python main.py \
  -p transaction_cost=0.002 \
     momentum_windows_months='[3,6,12]' \
     risky_assets='["SPY","EFA"]' \
     start_date=2010-01-01
```

### Use a custom config file

```bash
uv run python main.py --config examples/default_config.py
```

### More variants

```bash
# US + emerging, best-of defensives
uv run python main.py \
  -p risky_assets='["SPY","EEM"]' \
     defensive_asset=IEF \
     best_of_defensives='["IEF","TIP"]' \
     momentum_windows_months='[3,12]'

# Weekly rebalance (experimental)
uv run python main.py -p rebalance_freq=W

# Half-rebalancing (sleeves on 1st and 15th)
uv run python main.py -p sleeve_rebalance=true
```

## Running tests

```bash
uv run pytest tests/ -v
```

## Output

All generated files land in `output/`:

| File | Description |
|------|-------------|
| `01_equity_curve.png` | Portfolio NAV vs benchmark |
| `02_drawdown.png` | Drawdown over time |
| `03_monthly_histogram.png` | Distribution of monthly returns |
| `04_annual_heatmap.png` | Calendar heatmap of monthly returns (%) |
| `05_allocation.png` | Asset allocation over time |
| `portfolio_daily.csv` | Daily NAV, positions, returns |
| `portfolio_monthly.csv` | Monthly NAV & returns |
| `allocations_monthly.csv` | Monthly position snapshots |
| `performance_metrics.csv` | All calculated metrics |

## Key metrics calculated

- Total return & CAGR
- Annualised volatility
- Sharpe & Sortino ratios
- Calmar ratio
- Max drawdown & duration
- Number of trades
- Average holding period (months)
- Win rate (% of positive months)
- All benchmark equivalents

## Configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `risky_assets` | `["SPY", "SCZ"]` | Universe of risky assets (≥2) |
| `defensive_asset` | `"TLT"` | Default defensive asset |
| `best_of_defensives` | `[]` | If set, picks best of these by 1m return |
| `benchmark` | `"SPY"` | Benchmark for comparison |
| `cash_asset` | `"BIL"` | Fallback cash-like asset |
| `momentum_windows_months` | `[1, 3, 6]` | Lookback windows for ADM score |
| `rebalance_freq` | `"M"` | Pandas offset for rebalancing |
| `sleeve_rebalance` | `False` | Half-rebalance on 1st & 15th |
| `transaction_cost` | `0.001` | One-way trading cost (fraction) |
| `slippage` | `0.0005` | Slippage (fraction) |
| `tax_rate` | `0.0` | Flat tax on realised gains |
| `start_date` | `"2000-01-01"` | Data start |
| `end_date` | `None` | Data end (None = today) |
| `data_source` | `"yahoo"` | `"yahoo"` or `"csv"` |
| `data_dir` | `"data"` | CSV directory |
| `initial_capital` | `100_000` | Starting capital |
| `output_dir` | `"output"` | Output directory |

## Implementation choices

- **Score = sum of returns** (not average) — ranking is identical, sum is simpler.
- **Month-end resampling** for momentum — avoids intra-month timing bias ("close-to-close monthly").
- **Forward-filled signals** — signal computed on rebalance day, held until next rebalance.
- **Cost applied at trade boundary** — detected by position change vs previous day.
- **Matplotlib with `Agg` backend** — headless-friendly, no display required.
- **No external DB** — everything runs on in-memory DataFrames; CSV for persistence.
- **Seaborn styling** ships with the repo — clean charts without manual theming.

## Licence

MIT — use freely, modify, share.
