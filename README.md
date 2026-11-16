# Momentum backtester

Python research tool for ADM, Enhanced DM, Managed Futures DM and Vigilant DM.
Not a broker integration and not investment advice. Historical simulations are
not forecasts. This is a research prototype, not a validated execution platform.

## Install and run

Python 3.10+:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python main.py -v
streamlit run app.py
python -m pytest tests/ -q
```

Examples:

```sh
python main.py --strategy enhanced_dm -p enhanced_top_n=3 momentum_lookback_months=6
python main.py -p rebalance_freq=W-FRI start_date=2010-01-01
python main.py --config examples/default_config.py
```

CLI overrides are revalidated by the pipeline. Unknown fields are still warned
and ignored by the existing CLI. Configuration files are executable Python:
only load trusted files. Yahoo downloads need network access; CSV mode does not.

## Accounting and timing contract

- Benchmark sessions define the common calendar. Reallocation signals are made
  at the last observed session of each month (`ME`, alias `M`) or week (`W-FRI`,
  alias `W`). The final partial period can have a terminal signal, but cannot
  execute without a subsequent session.
- A signal at close t executes at close t+1 AFTER the old holdings earn the
  intervening return. The new holdings start earning returns after that close.
  There is no assumption of fills at the open: open-price data is not loaded.
- Fractional shares, long only, no leverage. Quantities remain unchanged between
  rebalances, so actual weights drift. Target weights are not daily holdings.
- Transaction cost plus slippage is charged per side on the actual bought AND
  sold notional. Full switches therefore involve both sale and purchase costs.
  The post-cost invested amount is solved to avoid negative cash.
- Start in uninvested, non-interest-bearing reporting-currency CASH. BIL is an
  investable fallback asset, not the same thing as uninvested cash.
- No backward filling of prices. Short gaps are forward filled for at most
  `max_ffill_sessions` benchmark sessions (default 5). No unavailable history
  is invented. Missing target prices leave that weight in CASH, reported in
  `unavailable_target_weight`; missing held prices beyond the fill limit raise
  an error rather than silently valuing a holding at zero.
- Required tickers and adjusted prices must exist. Missing downloads fail early.
  An asset is scored only after its full lookback is available.
- `start_date` is inclusive and `end_date` exclusive for CSV and Yahoo. Data
  loading begins at start_date: lookback warmup is INCLUDED in the simulation,
  not fetched invisibly before it. Compare strategies on consistent periods.
- All prices must already share the same currency. There is NO FX conversion,
  inflation adjustment, tax, borrowing, spread model beyond fixed slippage,
  market impact or jurisdiction-specific investment-account treatment.
- Nonzero `tax_rate` and `sleeve_rebalance=True` fail explicitly: these features
  are NOT implemented. No silent tax assumptions or fake half-portfolio sleeves.

## Strategies

- `adm`: sum returns over 1, 3 and 6 calendar months (configurable). Pick the
  highest positive risky score; otherwise use TLT or the configured best-of
  defensives ranked on 1-month return. Best-of defensives may all be negative;
  this intentionally does not impose an extra cash filter.
- `enhanced_dm`: select up to top N positive momentum assets, equally weighted
  at rebalance (default N=4), otherwise BIL.
- `managed_futures_dm`: positive momentum of the first risky asset enables the
  best positive risky candidate; otherwise choose the best positive defensive,
  or BIL. Default risky SPY/VEA, defensive KMLM/DBMF/FLOT/BIL.
- `vigilant_dm`: same risky filter; otherwise choose defensive candidates based
  on the 6-calendar-month change of ^TNX. Insufficient rate history falls back
  to BIL, not to an assumed falling-rate regime.

All non-ADM asset momentum uses `momentum_lookback_months` (default 12), not
252 observations. The rate trend window remains explicitly six months.

## Outputs and metrics

`output/` contains five PNG charts, portfolio_daily.csv, portfolio_monthly.csv,
allocations_monthly.csv (actual asset AND CASH weights), and performance_metrics.csv.
Daily returns and risk metrics use NET NAV changes including costs. Monthly
returns compound daily net returns; partial first/last months are included in
plots and monthly win rate. CAGR uses actual elapsed calendar time. Volatility,
Sharpe and Sortino assume 252 sessions/year and a zero risk-free reference;
Sortino downside deviation uses all observations, with positive returns set to
zero. Undefined ratios are NaN, not a misleading zero. Drawdown duration is in
calendar days. `num_trades` counts rebalance events with at least one material
order, not individual buys/sells; `orders` reports daily order counts. Average
holding duration was removed because a single number was misleading for
multi-asset portfolios.

## Remaining research limitations

Survivorship and choice-of-universe bias, parameter overfitting and changing
Yahoo data are not solved by accounting fixes. Short price-gap filling remains
an explicit approximation. The benchmark calendar cannot model asynchronous
international closes or asset-specific holidays faithfully. No local cache,
locked dependency environment, walk-forward validation, broker reconciliation
or corporate-action audit is provided. UI output files use a shared output
folder; deploy one instance per user/run or add isolated storage before
multi-user hosting. Synthetic regression tests do not validate market data or
claim any investment performance. A trusted CSV dataset is recommended for
reproducible research.
