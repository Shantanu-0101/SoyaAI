"""End-to-end experiments shared by the notebook and the Streamlit app.

Two-stage model comparison (same for Model A and Model B):
  Stage 1  benchmarks + SARIMA/ARIMA candidate orders, no outside variables.
           The order with the lowest average MASE becomes the "base" order.
  Stage 2  the base order + each exogenous set (SARIMAX / ARIMAX).
           This isolates the question "do the outside variables help?",
           because the only difference between the models is X.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as E
from . import features as F
from . import models as M


# ---------------------------------------------------------------------------
# Exogenous inputs on the model scale (logs for prices)
# ---------------------------------------------------------------------------
def exog_raw_a(monthly: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=monthly.index)
    X["oil_rate"] = np.log(monthly.oil_rate)
    X["margin"] = monthly.margin
    X["covid"] = monthly.covid
    X["ukraine"] = monthly.ukraine
    X["duty"] = monthly.duty
    X.index.freq = "MS"
    return X


def exog_raw_b(daily: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=daily.index)
    for c in ["cbot_soyoil", "brent", "usdinr"]:
        X[c] = np.log(daily[c])
    X["duty"] = daily.duty
    X["ukraine"] = daily.ukraine
    X = X.ffill()
    X.index.freq = "B"
    return X


def _name(order, sorder):
    return f"{tuple(order)}{tuple(sorder) if sorder and sorder[3] else ''}"


# ---------------------------------------------------------------------------
# Model A: monthly imports
# ---------------------------------------------------------------------------
def target_a(cfg, monthly):
    """Imports in THOUSAND tonnes (kt). Values of ~100-600 keep the SARIMAX optimiser
    numerically well-behaved; raw tonnes (~10^5) gave scale-dependent AIC rankings."""
    y = (monthly.tonnes.loc[cfg["model_a"]["start"]:cfg["model_a"]["end"]] / 1e3).rename("kt")
    y.index.freq = "MS"
    return y


def cv_a(cfg, monthly, verbose=False):
    ma = cfg["model_a"]
    y = target_a(cfg, monthly)
    tr, m = ma["transform"], ma["seasonal_period"]
    origins = list(range(ma["cv_initial_months"] - 1, len(y) - 1, ma["cv_step"]))
    max_h = max(ma["horizons"])

    stage1 = {"Naive": M.Naive, "SeasonalNaive": lambda: M.SeasonalNaive(m),
              "ETS": lambda: M.ETS(m)}
    for o, so in ma["candidate_orders"]:
        stage1[f"SARIMA{_name(o, so)}"] = (lambda o=o, so=so: M.Sarimax(o, so))
    cv1 = E.rolling_origin(y, stage1, origins, max_h, tr, m, verbose=verbose)
    s1 = E.summarize(cv1, ma["horizons"])
    sarima = s1[s1.model.str.startswith("SARIMA")].groupby("model").MASE.mean()
    best = sarima.idxmin()
    base = next((o, so) for o, so in ma["candidate_orders"] if f"SARIMA{_name(o, so)}" == best)

    Xraw = exog_raw_a(monthly)
    meth = ma.get("exog_forecast", {})
    cvs = [cv1]
    for set_name, lags in ma["exog_sets"].items():
        mk = {f"SARIMAX+{set_name}": (lambda lags=lags: M.Sarimax(base[0], base[1], exog_cols=[
            f"{c}_lag{l}" for c, l in lags.items()]))}
        cvs.append(E.rolling_origin(y, mk, origins, max_h, tr, m, Xraw, lags, "MS", exog_methods=meth))
    # same margin model, but the margin is forecast as "no change" instead of AR(1)
    lags = ma["exog_sets"]["margin"]
    mk = {"SARIMAX+margin (no-change driver)": lambda: M.Sarimax(base[0], base[1], exog_cols=[
        f"{c}_lag{l}" for c, l in lags.items()])}
    cvs.append(E.rolling_origin(y, mk, origins, max_h, tr, m, Xraw, lags, "MS"))
    # oracle: the margin model fed the TRUE future margin (not achievable in practice)
    lags = ma["exog_sets"]["margin"]
    mk = {"SARIMAX+margin (oracle X)": lambda: M.Sarimax(base[0], base[1], exog_cols=[
        f"{c}_lag{l}" for c, l in lags.items()])}
    cvs.append(E.rolling_origin(y, mk, origins, max_h, tr, m, Xraw, lags, "MS", oracle_exog=True))
    cv = pd.concat(cvs, ignore_index=True)
    return {"cv": cv, "summary": E.summarize(cv, ma["horizons"]), "stage1": s1,
            "base_order": base, "origins": origins, "y": y}


# ---------------------------------------------------------------------------
# Model B: daily oil rate
# ---------------------------------------------------------------------------
def target_b(cfg, daily):
    mb = cfg["model_b"]
    y = daily.rate.loc[mb["start"]:mb["end"]].copy()
    y.index.freq = "B"
    return y


def cv_b(cfg, daily, verbose=False):
    mb = cfg["model_b"]
    y = target_b(cfg, daily)
    n, step, k = len(y), mb["cv_step"], mb["cv_origins"]
    max_h = max(mb["horizons"])
    last = n - 1 - max_h
    origins = [last - i * step for i in range(k)][::-1]

    stage1 = {"Naive": M.Naive, "ETS": lambda: M.ETS(None)}
    for o in mb["candidate_orders"]:
        stage1[f"ARIMA{tuple(o)}"] = (lambda o=o: M.Sarimax(o, trend="n"))
    cv1 = E.rolling_origin(y, stage1, origins, max_h, "log", 1, verbose=verbose)
    s1 = E.summarize(cv1, mb["horizons"])
    ar = s1[s1.model.str.startswith("ARIMA")].groupby("model").MASE.mean()
    base = next(o for o in mb["candidate_orders"] if f"ARIMA{tuple(o)}" == ar.idxmin())

    Xraw = exog_raw_b(daily)
    cvs = [cv1]
    for set_name, lags in mb["exog_sets"].items():
        for o in {tuple(base), (0, 1, 0)}:
            mk = {f"ARIMAX{o}+{set_name}": (lambda lags=lags, o=o: M.Sarimax(o, exog_cols=[
                f"{c}_lag{l}" for c, l in lags.items()]))}
            cvs.append(E.rolling_origin(y, mk, origins, max_h, "log", 1, Xraw, lags, "B"))
    cv = pd.concat(cvs, ignore_index=True)
    return {"cv": cv, "summary": E.summarize(cv, mb["horizons"]), "stage1": s1,
            "base_order": tuple(base), "origins": origins, "y": y}


# ---------------------------------------------------------------------------
# Final fit, forecast and scenarios
# ---------------------------------------------------------------------------
def fit_and_forecast(y, order, sorder, lags, Xraw, freq, steps, transform="log",
                     scenarios: dict | None = None, methods: dict | None = None):
    """Fit on all data, forecast `steps` ahead; scenarios = {label: future_adjust dict}."""
    yt = E.FWD[transform](y)
    X_tr = X_fu = None
    cols = [f"{c}_lag{l}" for c, l in (lags or {}).items()]
    if lags:
        X_tr, X_fu = F.exog_for_window(Xraw, lags, y.index, steps, freq, methods=methods)
    model = M.Sarimax(order, sorder, exog_cols=cols).fit(yt, X_tr)
    back = E.BACK[transform]
    fc = model.forecast(steps, X_fu).apply(back)
    scen = {}
    for label, adj in (scenarios or {}).items():
        _, X_s = F.exog_for_window(Xraw, lags, y.index, steps, freq, future_adjust=adj, methods=methods)
        scen[label] = model.forecast(steps, X_s)["mean"].apply(back)
    return model, fc, scen
