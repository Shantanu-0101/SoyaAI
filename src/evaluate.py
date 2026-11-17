"""Rolling-origin (expanding window) evaluation, error metrics and residual diagnostics."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

from .features import exog_for_window

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
FWD = {"log": np.log, "log1p": np.log1p, "none": lambda x: x}
BACK = {"log": np.exp, "log1p": np.expm1, "none": lambda x: x}


# ---------------------------------------------------------------------------
# Metrics (all on the ORIGINAL scale: tonnes or Rs/10 kg)
# ---------------------------------------------------------------------------
def mae(e):
    return float(np.nanmean(np.abs(e)))


def rmse(e):
    return float(np.sqrt(np.nanmean(np.square(e))))


def smape(actual, pred):
    """Symmetric MAPE in %, bounded 0-200, defined when actual is zero (unlike MAPE)."""
    a, f = np.asarray(actual, float), np.asarray(pred, float)
    denom = (np.abs(a) + np.abs(f)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(denom == 0, 0.0, np.abs(a - f) / denom)
    return float(100 * np.nanmean(v))


def mase_scale(y_train: pd.Series, m: int) -> float:
    """In-sample MAE of the (seasonal) naive forecast on the training window."""
    y = y_train.dropna()
    return float(np.mean(np.abs(y.values[m:] - y.values[:-m])))


# ---------------------------------------------------------------------------
# Rolling-origin cross-validation
# ---------------------------------------------------------------------------
def rolling_origin(y: pd.Series, models: dict, origins: list[int], max_h: int, transform: str,
                   mase_m: int, exog_raw: pd.DataFrame | None = None, lags: dict | None = None,
                   freq: str = "MS", oracle_exog: bool = False, verbose: bool = False,
                   n_jobs: int = -1, exog_methods: dict | None = None) -> pd.DataFrame:
    """Expanding-window backtest.

    For each origin position o: train on y[0..o], forecast o+1..o+max_h, compare with
    the actual values. Nothing after the origin is used for fitting, for the exogenous
    variables (see features.exog_for_window) or for the MASE scale.

    models : {name: zero-argument function returning a fresh model}
    oracle_exog : if True, use the TRUE future exogenous values (an upper bound that is
                  not achievable in practice; reported only for comparison).
    n_jobs : origins are independent, so they run in parallel on all CPU cores
             (joblib). Results are identical to a sequential loop.
    """
    from joblib import Parallel, delayed
    jobs = (delayed(_one_origin)(y, models, o, max_h, transform, mase_m, exog_raw, lags,
                                 freq, oracle_exog, verbose, exog_methods) for o in origins)
    # max_nbytes=None: send data to workers directly (no memory-mapped temp files, which
    # can fail to clean up on Windows inside Jupyter)
    parts = Parallel(n_jobs=n_jobs, max_nbytes=None, verbose=10 if verbose else 0)(jobs)
    return pd.DataFrame([r for part in parts for r in part])


def _one_origin(y, models, o, max_h, transform, mase_m, exog_raw, lags, freq, oracle_exog, verbose,
                exog_methods=None):
    warnings.filterwarnings("ignore")
    rows = []
    train = y.iloc[: o + 1]
    steps = min(max_h, len(y) - o - 1)
    if steps < 1:
        return rows
    actual = y.iloc[o + 1: o + 1 + steps]
    yt = FWD[transform](train)
    scale = mase_scale(train, mase_m)
    X_tr = X_fu = None
    if exog_raw is not None and lags:
        X_tr, X_fu = exog_for_window(exog_raw, lags, train.index, steps, freq, methods=exog_methods)
        if oracle_exog:
            full = pd.DataFrame({f"{c}_lag{l}": exog_raw[c].ffill().shift(l) for c, l in lags.items()})
            X_fu = full.reindex(X_fu.index)
    back = BACK[transform]
    for name, make in models.items():
        m = make()
        try:
            m.fit(yt, X_tr if m.uses_exog else None)
            fc = m.forecast(steps, X_fu if m.uses_exog else None)
        except Exception as e:  # keep going, record the failure
            if verbose:
                print(f"{name} failed at origin {train.index[-1].date()}: {e}")
            continue
        for h in range(1, steps + 1):
            rows.append({"model": name, "origin": train.index[-1], "h": h,
                         "date": actual.index[h - 1], "actual": actual.iloc[h - 1],
                         "pred": back(fc["mean"].iloc[h - 1]),
                         "lower": back(fc["lower"].iloc[h - 1]),
                         "upper": back(fc["upper"].iloc[h - 1]), "scale": scale})
    return rows


def summarize(cv: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """MAE, RMSE, MASE, sMAPE and 95% interval coverage at each horizon."""
    out = []
    d = cv.dropna(subset=["actual"])
    for (model, h), g in d[d.h.isin(horizons)].groupby(["model", "h"]):
        e = g.actual - g.pred
        out.append({"model": model, "h": h, "n": len(g), "MAE": mae(e), "RMSE": rmse(e),
                    "MASE": float(np.mean(np.abs(e) / g.scale)), "sMAPE": smape(g.actual, g.pred),
                    "coverage95": float(np.mean((g.actual >= g.lower) & (g.actual <= g.upper)))})
    return pd.DataFrame(out).sort_values(["h", "MASE"]).reset_index(drop=True)


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test on squared errors (H0: equal accuracy). Negative stat = model 1 better.

    Uses the Harvey-Leybourne-Newbold small-sample correction.
    """
    from scipy import stats
    d = np.square(e1) - np.square(e2)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < h + 3:                      # too few paired forecasts for a meaningful test
        return float("nan"), float("nan")
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = dbar / np.sqrt(var) if var > 0 else np.nan
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------
def stationarity_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    """ADF (H0: unit root) and KPSS (H0: stationary) side by side.

    Reading: ADF p < 0.05 AND KPSS p > 0.05  -> stationary.
             ADF p > 0.05 AND KPSS p < 0.05  -> non-stationary, difference it.
    """
    rows = []
    for name, s in series.items():
        s = s.dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf = adfuller(s, autolag="AIC")
            kp = kpss(s, regression="c", nlags="auto")
        verdict = ("stationary" if adf[1] < 0.05 and kp[1] > 0.05 else
                   "non-stationary" if adf[1] > 0.05 and kp[1] < 0.05 else "mixed")
        rows.append({"series": name, "n": len(s), "ADF_stat": adf[0], "ADF_p": adf[1],
                     "KPSS_stat": kp[0], "KPSS_p": kp[1], "verdict": verdict})
    return pd.DataFrame(rows).set_index("series")


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------
def residual_diagnostics(res, lags: list[int], burn: int = 0) -> pd.DataFrame:
    """Ljung-Box (autocorrelation), Jarque-Bera (normality), ARCH-LM and
    break-variance (heteroscedasticity) and OLS-CUSUM (structural break)."""
    r = pd.Series(res.resid).iloc[burn:].dropna()
    rows = []
    lb = acorr_ljungbox(r, lags=lags, return_df=True)
    for L, row in lb.iterrows():
        rows.append({"test": f"Ljung-Box lag {L}", "H0": "no autocorrelation",
                     "stat": row.lb_stat, "p_value": row.lb_pvalue})
    from scipy import stats
    jb = stats.jarque_bera(r)
    rows.append({"test": "Jarque-Bera", "H0": "residuals normal", "stat": jb.statistic,
                 "p_value": jb.pvalue})
    arch = het_arch(r, nlags=min(12, len(r) // 10))
    rows.append({"test": "ARCH-LM", "H0": "constant variance (no ARCH)", "stat": arch[0],
                 "p_value": arch[1]})
    try:
        hv = res.test_heteroskedasticity("breakvar")[0]
        rows.append({"test": "Break-variance (1st vs last third)", "H0": "equal variance",
                     "stat": hv[0], "p_value": hv[1]})
    except Exception:
        pass
    cs = breaks_cusumolsresid(r.values - r.values.mean(), ddof=0)
    rows.append({"test": "OLS-CUSUM", "H0": "no structural break", "stat": cs[0], "p_value": cs[1]})
    out = pd.DataFrame(rows)
    out["reject_at_5%"] = out.p_value < 0.05
    return out
