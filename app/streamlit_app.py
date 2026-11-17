"""SoyaAI forecast viewer.

Run from the project folder:
    streamlit run app/streamlit_app.py

The app only READS the files the notebook saved in outputs/forecasts, so it
opens instantly and always shows exactly the numbers reported in the notebook.
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FC = ROOT / "outputs" / "forecasts"
TB = ROOT / "outputs" / "tables"

st.set_page_config(page_title="SoyaAI forecasts", layout="wide")
st.title("SoyaAI: soybean oil imports and Indore oil rate")
st.caption("SARIMAX / ARIMAX forecasts with 95% prediction intervals. "
           "Data: UN Comtrade (HS 1507), SOPA daily rates, Yahoo Finance.")

if not (FC / "model_a_forecast.csv").exists():
    st.error("No forecasts found. Run notebooks/soya_forecasting.ipynb first.")
    st.stop()


def band_chart(hist: pd.DataFrame, fc: pd.DataFrame, ylab: str, scen_cols: list[str]):
    base_h = alt.Chart(hist).mark_line(color="#94a3b8").encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=0)), y=alt.Y("value:Q", title=ylab, scale=alt.Scale(zero=False)))
    band = alt.Chart(fc).mark_area(opacity=0.3, color="#93c5fd").encode(
        x="date:T", y="lower:Q", y2="upper:Q")
    mean = alt.Chart(fc).mark_line(color="#2563eb", strokeWidth=2).encode(x="date:T", y="mean:Q")
    layers = [band, base_h, mean]
    if scen_cols:
        long = fc.melt(id_vars="date", value_vars=scen_cols, var_name="scenario", value_name="v")
        layers.append(alt.Chart(long).mark_line(strokeDash=[5, 3]).encode(
            x="date:T", y="v:Q", color=alt.Color("scenario:N", legend=alt.Legend(orient="bottom"))))
    return alt.layer(*layers).properties(height=380).interactive()


tab_a, tab_b, tab_cv = st.tabs(["Model A: monthly imports", "Model B: daily oil rate",
                                "Backtest accuracy"])

with tab_a:
    fc = pd.read_csv(FC / "model_a_forecast.csv", parse_dates=["date"])
    hist = pd.read_csv(FC / "model_a_history.csv", parse_dates=["date"])
    years = st.slider("Years of history", 1, 15, 5, key="ya")
    hist = hist[hist.date >= hist.date.max() - pd.DateOffset(years=years)]
    scen = [c for c in fc.columns if c.startswith("scenario:")]
    pick = st.multiselect("Scenarios", scen, default=[], key="sa")
    st.altair_chart(band_chart(hist, fc, "thousand tonnes per month", pick), width="stretch")
    show = fc[["date", "mean", "lower", "upper"] + pick].copy()
    show["date"] = show.date.dt.strftime("%b %Y")
    st.dataframe(show.set_index("date").round(0), width="stretch")

with tab_b:
    fc = pd.read_csv(FC / "model_b_forecast.csv", parse_dates=["date"])
    hist = pd.read_csv(FC / "model_b_history.csv", parse_dates=["date"])
    days = st.slider("Business days of history", 20, 500, 120, key="db")
    hist = hist.tail(days)
    scen = [c for c in fc.columns if c.startswith("scenario:")]
    pick = st.multiselect("Scenarios", scen, default=[], key="sb")
    st.altair_chart(band_chart(hist, fc, "Rs per 10 kg", pick), width="stretch")
    show = fc[["date", "mean", "lower", "upper"] + pick].copy()
    show["date"] = show.date.dt.strftime("%d %b %Y")
    st.dataframe(show.set_index("date").round(1), width="stretch")

with tab_cv:
    for f, label in [("cv_summary_model_a.csv", "Model A (horizons in months)"),
                     ("cv_summary_model_b.csv", "Model B (horizons in business days)")]:
        if (TB / f).exists():
            st.subheader(label)
            st.dataframe(pd.read_csv(TB / f).round(3), width="stretch")
    st.caption("MASE < 1 means better than the naive benchmark on the training data. "
               "coverage95 is the share of actual values inside the 95% interval.")
