"""
app.py — Streamlit GUI for the ADM Backtester.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import logging
import sys
import os

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import ADMConfig
from backtest import run as run_backtest

# ── Suppress non-critical logs ───────────────────────────────────────────────
logging.getLogger().setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ADM Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 ADM Backtester — Multi-Strategy")
st.markdown("Configurez les paramètres ci-dessous et lancez un backtest.")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.header("Stratégie")

STRATEGY_LABELS = {
    "adm": "ADM (Accelerating Dual Momentum)",
    "enhanced_dm": "Enhanced Dual Momentum",
    "managed_futures_dm": "Managed Futures Dual Momentum",
    "vigilant_dm": "Vigilant Dual Momentum",
}

strategy_key = st.sidebar.selectbox(
    "Stratégie",
    options=list(STRATEGY_LABELS.keys()),
    format_func=lambda k: STRATEGY_LABELS[k],
    index=0,
)

# ── Period ───────────────────────────────────────────────────────────────────
st.sidebar.header("Période")
start_date = st.sidebar.date_input("Début", datetime(2000, 1, 1))
end_date = st.sidebar.date_input("Fin", datetime.today())

# ── Capital ──────────────────────────────────────────────────────────────────
st.sidebar.header("Capital")
initial_capital = st.sidebar.number_input(
    "Capital initial (€)", min_value=1_000, value=100_000, step=10_000,
    format="%d"
)

# ── Costs ────────────────────────────────────────────────────────────────────
st.sidebar.header("Coûts")
transaction_cost = st.sidebar.slider(
    "Frais de transaction", 0.0, 0.01, 0.001, step=0.0005, format="%.3f"
)
slippage = st.sidebar.slider(
    "Slippage", 0.0, 0.005, 0.0005, step=0.0005, format="%.3f"
)

# ── Strategy-specific parameters ────────────────────────────────────────────
st.sidebar.header("Paramètres Avancés")

if strategy_key == "adm":
    risky = st.sidebar.text_input(
        "Actifs risqués (séparés par des virgules)", "SPY, SCZ"
    )
    defensive = st.sidebar.text_input(
        "Actif défensif", "TLT"
    )
    best_of = st.sidebar.text_input(
        "Best-of défensifs (optionnel, virgules)", ""
    )
    momentum_windows = st.sidebar.text_input(
        "Fenêtres momentum (mois)", "1, 3, 6"
    )

elif strategy_key == "enhanced_dm":
    multi_universe = st.sidebar.text_input(
        "Univers d'actifs (virgules)",
        "SPY, QQQ, GLD, DBC, VNQ, KMLM, SHY, BIL",
    )
    top_n = st.sidebar.slider("Top N actifs à détenir", 1, 6, 4)
    lookback = st.sidebar.slider("Lookback momentum (mois)", 3, 24, 12)

elif strategy_key == "managed_futures_dm":
    mf_risky = st.sidebar.text_input("Actifs risqués (virgules)", "SPY, VEA")
    mf_defensive = st.sidebar.text_input("Actifs défensifs (virgules)", "KMLM, DBMF, FLOT, BIL")
    lookback = st.sidebar.slider("Lookback momentum (mois)", 3, 24, 12)

elif strategy_key == "vigilant_dm":
    v_risky = st.sidebar.text_input("Actifs risqués (virgules)", "SPY, VEA")
    v_def_rising = st.sidebar.text_input("Défensifs taux montants (virgules)", "SHY, GLD, BIL, USFR")
    v_def_falling = st.sidebar.text_input("Défensifs taux baissiers (virgules)", "TLT, GLD, SHY, IEF")
    lookback = st.sidebar.slider("Lookback momentum (mois)", 3, 24, 12)

# ── Benchmark ────────────────────────────────────────────────────────────────
benchmark = st.sidebar.text_input("Benchmark", "SPY")

# ── Run button ───────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Lancer le backtest", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: parse comma-separated string to list
# ══════════════════════════════════════════════════════════════════════════════

def _parse_list(text: str) -> list:
    return [t.strip() for t in text.split(",") if t.strip()]


def _parse_int_list(text: str) -> list:
    return [int(t.strip()) for t in text.split(",") if t.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# BUILD CONFIG & RUN
# ══════════════════════════════════════════════════════════════════════════════


def _plot_equity(df: pd.DataFrame, cfg: ADMConfig):
    """Equity curve: portfolio vs benchmark."""
    port = df["nav"] / df["nav"].iloc[0] * 100
    bench = (1.0 + df["bench_return"]).cumprod() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port.index, y=port.values,
        name="ADM Portfolio", line=dict(color="#2196F3", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bench.index, y=bench.values,
        name=f"Benchmark ({cfg.benchmark})",
        line=dict(color="#9E9E9E", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title="Equity Curve — Portefeuille vs Benchmark",
        yaxis_title="Croissance de 100",
        hovermode="x unified",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_drawdown(df: pd.DataFrame):
    """Drawdown over time."""
    nav = df["nav"]
    peak = nav.expanding().max()
    dd = (nav - peak) / peak * 100

    min_idx = dd.idxmin()
    min_val = dd.min()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", name="Drawdown",
        line=dict(color="#FF5722", width=1.5),
        fillcolor="rgba(255,87,34,0.15)",
    ))
    fig.add_annotation(
        x=min_idx, y=min_val,
        text=f"Max: {min_val:.1f}%",
        showarrow=True, arrowhead=1, ax=40, ay=-40,
    )
    fig.update_layout(
        title="Drawdown",
        yaxis_title="Drawdown (%)",
        hovermode="x unified", height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_monthly_hist(df: pd.DataFrame):
    """Distribution of monthly returns."""
    monthly = df["nav"].resample("ME").last().pct_change().dropna() * 100
    mean_ret = monthly.mean()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=monthly.values,
        nbinsx=max(int(len(monthly) ** 0.5), 15),
        marker_color="#2196F3",
        opacity=0.8,
        name="Mensuel",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
    fig.add_vline(
        x=mean_ret, line_color="#4CAF50", line_width=2,
        annotation_text=f"Moy: {mean_ret:.2f}%",
    )
    fig.update_layout(
        title="Distribution des Rendements Mensuels",
        xaxis_title="Rendement mensuel (%)",
        yaxis_title="Fréquence",
        hovermode="x", height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_heatmap(df: pd.DataFrame):
    """Annual heatmap of monthly returns."""
    nav = df["nav"]
    monthly = nav.resample("ME").last().pct_change().dropna() * 100
    years = monthly.index.year
    months = monthly.index.month
    pivot = pd.DataFrame({"ret": monthly.values, "year": years, "month": months})
    table = pivot.pivot_table(
        index="year", columns="month", values="ret", aggfunc="first"
    )
    table.columns = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
                      "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

    vmax = max(abs(table.min().min()), abs(table.max().max()), 3)

    fig = go.Figure(data=go.Heatmap(
        z=table.values,
        x=table.columns,
        y=table.index.astype(str),
        colorscale="RdYlGn",
        zmid=0,
        zmin=-vmax,
        zmax=vmax,
        text=table.values.round(1).astype(str),
        texttemplate="%{text}",
        hovertemplate="%{y} - %{x}: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Rendements Mensuels — Heatmap Calendaire (%)",
        height=max(350, len(table) * 30 + 100),
        xaxis_side="top",
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_allocation(df: pd.DataFrame):
    """Allocation timeline (single asset) or portfolio composition."""
    if "position" not in df.columns:
        st.info("Données d'allocation non disponibles.")
        return

    positions = df["position"].dropna().unique()
    if len(positions) == 0:
        st.info("Aucune donnée d'allocation.")
        return

    weekly = df.resample("W").last()
    alloc = pd.DataFrame(0.0, index=weekly.index, columns=positions)
    for pos in positions:
        mask = weekly["position"] == pos
        alloc.loc[mask, pos] = 1.0

    # Colors
    colors = px.colors.qualitative.Set2 * (len(positions) // len(px.colors.qualitative.Set2) + 1)

    fig = go.Figure()
    for i, pos in enumerate(alloc.columns):
        fig.add_trace(go.Scatter(
            x=alloc.index, y=alloc[pos],
            name=pos, mode="none",
            stackgroup="one",
            line=dict(width=0.5),
            fillcolor=colors[i % len(colors)],
            hovertemplate=f"%{{x}}<br>{pos}: %{{y:.0%}}<extra></extra>",
        ))

    fig.update_layout(
        title="Allocation au Fil du Temps",
        yaxis_title="Poids",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        hovermode="x unified",
        height=350,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


if run_btn or "results" in st.session_state:

    if run_btn:
        # Build config from sidebar
        cfg = ADMConfig(
            strategy=strategy_key,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat() if end_date else None,
            initial_capital=float(initial_capital),
            transaction_cost=transaction_cost,
            slippage=slippage,
            benchmark=benchmark.strip() or "SPY",
            verbose=False,
            output_dir="output",
        )

        # Strategy-specific fields
        if strategy_key == "adm":
            cfg.risky_assets = _parse_list(risky) if risky.strip() else ["SPY", "SCZ"]
            cfg.defensive_asset = defensive.strip() or "TLT"
            if best_of.strip():
                cfg.best_of_defensives = _parse_list(best_of)
            cfg.momentum_windows_months = _parse_int_list(momentum_windows)

        elif strategy_key == "enhanced_dm":
            if multi_universe.strip():
                cfg.multi_asset_universe = _parse_list(multi_universe)
            cfg.enhanced_top_n = top_n
            cfg.momentum_lookback_months = lookback

        elif strategy_key == "managed_futures_dm":
            if mf_risky.strip():
                cfg.managed_futures_risky = _parse_list(mf_risky)
            if mf_defensive.strip():
                cfg.managed_futures_defensive = _parse_list(mf_defensive)
            cfg.momentum_lookback_months = lookback

        elif strategy_key == "vigilant_dm":
            if v_risky.strip():
                cfg.vigilant_risky = _parse_list(v_risky)
            if v_def_rising.strip():
                cfg.vigilant_defensive_rising = _parse_list(v_def_rising)
            if v_def_falling.strip():
                cfg.vigilant_defensive_falling = _parse_list(v_def_falling)
            cfg.momentum_lookback_months = lookback

        # Run
        with st.spinner("Téléchargement des données et calcul en cours..."):
            try:
                portfolio_df, metrics = run_backtest(cfg)
                st.session_state["results"] = (portfolio_df, metrics, cfg)
            except Exception as e:
                st.error(f"Erreur lors du backtest : {e}")
                st.session_state.pop("results", None)
                st.stop()

    # ── Display results ──────────────────────────────────────────────────────
    if "results" in st.session_state:
        portfolio_df, metrics, cfg = st.session_state["results"]

        # ── Metrics cards ──────────────────────────────────────────────────────
        st.header("Résultats")
        col1, col2, col3, col4, col5 = st.columns(5)

        def _pct(v):
            return f"{v:.2%}" if isinstance(v, float) else str(v)

        def _ratio(v):
            return f"{v:.3f}" if isinstance(v, float) else str(v)

        with col1:
            st.metric("CAGR", _pct(metrics.get("cagr", 0)))
            st.metric("Rendement total", _pct(metrics.get("total_return", 0)))

        with col2:
            st.metric("Volatilité", _pct(metrics.get("volatility", 0)))
            st.metric("Sharpe", _ratio(metrics.get("sharpe_ratio", 0)))

        with col3:
            st.metric("Sortino", _ratio(metrics.get("sortino_ratio", 0)))
            st.metric("Calmar", _ratio(metrics.get("calmar_ratio", 0)))

        with col4:
            st.metric("Max Drawdown", _pct(metrics.get("max_drawdown", 0)))
            st.metric(
                "Durée DD (jours)",
                int(metrics.get("max_drawdown_duration_days", 0)),
            )

        with col5:
            st.metric("Trades", int(metrics.get("num_trades", 0)))
            st.metric("Win Rate", _pct(metrics.get("win_rate", 0)))

        st.markdown("---")

        # ── Benchmarks comparison ─────────────────────────────────────────────
        st.subheader("Comparaison Benchmark")
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            st.metric("Benchmark CAGR", _pct(metrics.get("bench_cagr", 0)))
        with bcol2:
            st.metric("Benchmark Return", _pct(metrics.get("bench_total_return", 0)))
        with bcol3:
            st.metric("Benchmark Vol", _pct(metrics.get("bench_volatility", 0)))
        with bcol4:
            st.metric("Benchmark Max DD", _pct(metrics.get("bench_max_drawdown", 0)))

        st.markdown("---")

        # ── Charts ────────────────────────────────────────────────────────────
        st.header("Graphiques")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Equity Curve", "Drawdown", "Distribution", "Heatmap", "Allocation"]
        )

        with tab1:
            _plot_equity(portfolio_df, cfg)

        with tab2:
            _plot_drawdown(portfolio_df)

        with tab3:
            _plot_monthly_hist(portfolio_df)

        with tab4:
            _plot_heatmap(portfolio_df)

        with tab5:
            _plot_allocation(portfolio_df)

        # ── Data export ──────────────────────────────────────────────────────
        st.markdown("---")
        st.header("Données")

        csv_data = portfolio_df.to_csv().encode("utf-8")
        st.download_button(
            "📥 Télécharger portfolio_daily.csv",
            data=csv_data,
            file_name="portfolio_daily.csv",
            mime="text/csv",
        )

        with st.expander("Afficher les données brutes"):
            st.dataframe(portfolio_df.tail(30), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
