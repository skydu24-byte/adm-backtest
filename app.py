"""Streamlit front end. Run: streamlit run app.py."""
from datetime import date
import calendar
import pandas as pd
import plotly.express as px
import streamlit as st
from config import ADMConfig
from backtest import run
from metrics import monthly_returns

st.set_page_config(page_title="Momentum Backtester", layout="wide")
st.title("Momentum Backtester")
st.caption("Simulation avant impots, sans conversion de devises. Signal a la cloture, execution a la cloture de la seance suivante.")
st.warning("Les performances historiques ne constituent pas une prevision. Les mois partiels sont inclus dans les statistiques mensuelles.")
strategy = st.sidebar.selectbox("Strategie", ["adm", "enhanced_dm", "managed_futures_dm", "vigilant_dm"])
start = st.sidebar.date_input("Debut", date(2000, 1, 1))
end = st.sidebar.date_input("Fin exclusive", date.today())
capital = st.sidebar.number_input("Capital initial (devise des prix, pas de conversion EUR)", min_value=1000.0, value=100000.0)
fee = st.sidebar.number_input("Frais par cote (fraction, 0.001 = 0.1 %)", min_value=0.0, max_value=0.1, value=0.001, format="%.4f")
slip = st.sidebar.number_input("Slippage par cote (fraction)", min_value=0.0, max_value=0.1, value=0.0005, format="%.4f")
benchmark = st.sidebar.text_input("Benchmark", "SPY")
freq = st.sidebar.selectbox("Reallocation", ["ME", "W-FRI"])
lookback = st.sidebar.number_input("Lookback hors ADM (mois)", min_value=1, value=12)
top_n = st.sidebar.number_input("Top N (Enhanced)", min_value=1, value=4)

def ticks(label, default):
    return [t.strip() for t in st.sidebar.text_input(label, default).split(",") if t.strip()]

extra = {}
if strategy == "adm":
    extra["risky_assets"] = ticks("Actifs risques", "SPY,SCZ")
    extra["defensive_asset"] = st.sidebar.text_input("Defensif", "TLT")
    extra["best_of_defensives"] = ticks("Defensifs best-of (optionnel)", "")
    windows_text = st.sidebar.text_input("Fenetres ADM (mois)", "1,3,6")
elif strategy == "enhanced_dm":
    extra["multi_asset_universe"] = ticks("Univers", "SPY,QQQ,GLD,DBC,VNQ,KMLM,SHY,BIL")
elif strategy == "managed_futures_dm":
    extra["managed_futures_risky"] = ticks("Actifs risques", "SPY,VEA")
    extra["managed_futures_defensive"] = ticks("Actifs defensifs", "KMLM,DBMF,FLOT,BIL")
else:
    extra["vigilant_risky"] = ticks("Actifs risques", "SPY,VEA")
    extra["vigilant_defensive_rising"] = ticks("Defensifs taux montants", "SHY,GLD,BIL,USFR")
    extra["vigilant_defensive_falling"] = ticks("Defensifs taux baissiers", "TLT,GLD,SHY,IEF")

if st.sidebar.button("Lancer le backtest", type="primary"):
    st.session_state.pop("results", None)
    try:
        if strategy == "adm":
            extra["momentum_windows_months"] = [int(x.strip()) for x in windows_text.split(",")]
        cfg = ADMConfig(strategy=strategy, start_date=start.isoformat(), end_date=end.isoformat(), initial_capital=capital, transaction_cost=fee, slippage=slip, benchmark=benchmark.strip(), rebalance_freq=freq, momentum_lookback_months=int(lookback), enhanced_top_n=int(top_n), verbose=False, **extra)
        with st.spinner("Chargement et calcul..."):
            df, metrics = run(cfg)
        st.session_state["results"] = (df, metrics, cfg)
    except Exception as exc:
        st.error(str(exc))

if "results" in st.session_state:
    df, metrics, cfg = st.session_state["results"]
    st.caption(f"Resultats memorises : {cfg.strategy}, {cfg.start_date} -> {cfg.end_date}. Relancer pour appliquer les parametres modifies.")
    columns = st.columns(4)
    for col, key in zip(columns, ["cagr", "total_return", "max_drawdown", "win_rate"]):
        col.metric(key, f"{metrics[key]:.2%}")
    st.dataframe(pd.DataFrame([metrics]))
    if metrics["unavailable_target_events"]:
        st.warning("Des cibles n'avaient pas de prix disponible : leur poids est reste en liquidites non remunerees. Voir unavailable_target_weight dans le CSV.")
    curves = pd.DataFrame({"Portefeuille net": df.nav / cfg.initial_capital * 100, cfg.benchmark: df.bench_return.add(1).cumprod() * 100})
    st.plotly_chart(px.line(curves, title="Croissance de 100"), use_container_width=True)
    dd = (df.nav / df.nav.cummax().clip(lower=cfg.initial_capital) - 1) * 100
    st.plotly_chart(px.area(dd, title="Drawdown (%)"), use_container_width=True)
    monthly = monthly_returns(df) * 100
    st.plotly_chart(px.histogram(x=monthly.values, labels={"x": "Rendement mensuel net (%)"}), use_container_width=True)
    table = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "return": monthly.values}).pivot(index="year", columns="month", values="return").reindex(columns=range(1, 13))
    table.columns = list(calendar.month_abbr)[1:]
    st.dataframe(table.style.format("{:.2f}", na_rep=""))
    weights = df.filter(like="weight_").copy()
    weights.columns = weights.columns.str.removeprefix("weight_")
    st.plotly_chart(px.area(weights.loc[:, weights.max() > 0], title="Allocations effectivement detenues"), use_container_width=True)
    st.download_button("Exporter le portefeuille quotidien", df.to_csv().encode(), file_name="portfolio_daily.csv", mime="text/csv")
    st.dataframe(df.tail(30))
