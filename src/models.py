"""Forecasting models with one common interface.

Every model has:
    fit(y, X=None)              y is already transformed (e.g. log tonnes)
    forecast(steps, X_future)   -> DataFrame with columns mean, lower, upper (95%)

Models compared (from simplest to richest):
    Naive          tomorrow = today                       (benchmark for prices)
    SeasonalNaive  next July = last July                  (benchmark for seasonal imports)
    ETS            exponential smoothing, damped trend (+ additive season)
    SARIMA         seasonal ARIMA, no outside variables
    SARIMAX        SARIMA + exogenous variables (ARIMAX when there is no season)
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

Z95 = stats.norm.ppf(0.975)


def _future_index(y: pd.Series, steps: int) -> pd.DatetimeIndex:
    freq = y.index.freq or pd.infer_freq(y.index)
    return pd.date_range(y.index[-1], periods=steps + 1, freq=freq)[1:]


class Naive:
    name = "Naive"
    uses_exog = False

    def fit(self, y, X=None):
        self.y = y.ffill()
        self.sigma = self.y.diff().std()
        return self

    def forecast(self, steps, X_future=None):
        h = np.arange(1, steps + 1)
        mean = np.repeat(self.y.iloc[-1], steps)
        se = self.sigma * np.sqrt(h)
        return pd.DataFrame({"mean": mean, "lower": mean - Z95 * se, "upper": mean + Z95 * se},
                            index=_future_index(self.y, steps))


class SeasonalNaive:
    uses_exog = False

    def __init__(self, m: int = 12):
        self.m = m
        self.name = f"SeasonalNaive(m={m})"

    def fit(self, y, X=None):
        self.y = y.ffill()
        self.sigma = (self.y - self.y.shift(self.m)).std()
        return self

    def forecast(self, steps, X_future=None):
        h = np.arange(1, steps + 1)
        last = self.y.iloc[-self.m:].values
        mean = last[(h - 1) % self.m]
        se = self.sigma * np.sqrt((h - 1) // self.m + 1)
        return pd.DataFrame({"mean": mean, "lower": mean - Z95 * se, "upper": mean + Z95 * se},
                            index=_future_index(self.y, steps))


class ETS:
    uses_exog = False

    def __init__(self, seasonal_periods: int | None = 12, damped: bool = True):
        self.m = seasonal_periods
        self.damped = damped
        self.name = "ETS(A,Ad,A)" if seasonal_periods else "ETS(A,Ad,N)"

    def fit(self, y, X=None):
        self.y = y.ffill()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = ETSModel(self.y, error="add", trend="add", damped_trend=self.damped,
                                seasonal="add" if self.m else None,
                                seasonal_periods=self.m).fit(disp=False, maxiter=500)
        return self

    def forecast(self, steps, X_future=None):
        pr = self.res.get_prediction(start=len(self.y), end=len(self.y) + steps - 1)
        sf = pr.summary_frame(alpha=0.05)
        out = pd.DataFrame({"mean": sf["mean"].values, "lower": sf["pi_lower"].values,
                            "upper": sf["pi_upper"].values}, index=_future_index(self.y, steps))
        return out


class Sarimax:
    """SARIMA when exog_cols is empty, SARIMAX/ARIMAX otherwise.

    NaN values in y are allowed: the Kalman filter skips them in the likelihood.
    """

    def __init__(self, order, seasonal_order=(0, 0, 0, 0), exog_cols=None, trend="n", name=None):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.exog_cols = list(exog_cols or [])
        self.uses_exog = bool(self.exog_cols)
        self.trend = trend
        base = "SARIMAX" if self.uses_exog else "SARIMA"
        if self.seasonal_order[3] == 0:
            base = "ARIMAX" if self.uses_exog else "ARIMA"
        self.name = name or f"{base}{self.order}{self.seasonal_order if self.seasonal_order[3] else ''}"

    def fit(self, y, X=None):
        self.y = y
        exog = X[self.exog_cols] if self.uses_exog else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SARIMAX(y, exog=exog, order=self.order, seasonal_order=self.seasonal_order,
                                 trend=self.trend)
            try:
                self.res = self.model.fit(disp=False, maxiter=200)
            except (np.linalg.LinAlgError, ValueError):
                # rare numerical failure of the default optimiser: retry with Nelder-Mead
                self.res = self.model.fit(disp=False, maxiter=2000, method="nm")
        return self

    def forecast(self, steps, X_future=None):
        exog = X_future[self.exog_cols] if self.uses_exog else None
        fc = self.res.get_forecast(steps=steps, exog=exog)
        sf = fc.summary_frame(alpha=0.05)
        return pd.DataFrame({"mean": sf["mean"].values, "lower": sf["mean_ci_lower"].values,
                             "upper": sf["mean_ci_upper"].values}, index=_future_index(self.y, steps))


# ---------------------------------------------------------------------------
# Order selection
# ---------------------------------------------------------------------------
def auto_arima_order(y, X=None, m=12, d=None, D=None, seasonal=True, seed=42):
    """pmdarima stepwise search (Hyndman-Khandakar algorithm) by AIC."""
    import pmdarima as pm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(y, X=X, seasonal=seasonal, m=m if seasonal else 1, d=d, D=D,
                              max_p=3, max_q=3, max_P=2, max_Q=2, information_criterion="aic",
                              stepwise=True, suppress_warnings=True, error_action="ignore",
                              random_state=seed)
    return model.order, model.seasonal_order, model


def grid_search(y, X=None, d=1, D=0, m=0, p_range=range(0, 3), q_range=range(0, 3),
                P_range=range(0, 2), Q_range=range(0, 2), trend="n"):
    """Exhaustive small grid; returns every fit with its AIC and BIC, best AIC first."""
    rows = []
    seasonal_grid = list(itertools.product(P_range, Q_range)) if m else [(0, 0)]
    for p_, q_ in itertools.product(p_range, q_range):
        for P_, Q_ in seasonal_grid:
            so = (P_, D, Q_, m) if m else (0, 0, 0, 0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = SARIMAX(y, exog=X, order=(p_, d, q_), seasonal_order=so, trend=trend).fit(
                        disp=False, maxiter=200)
                rows.append({"order": (p_, d, q_), "seasonal_order": so, "aic": r.aic, "bic": r.bic,
                             "converged": r.mle_retvals.get("converged", True)})
            except Exception as e:  # pragma: no cover
                rows.append({"order": (p_, d, q_), "seasonal_order": so, "aic": np.nan,
                             "bic": np.nan, "converged": False, "error": str(e)[:60]})
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def parameter_table(res) -> pd.DataFrame:
    """Coefficient, standard error, z, p-value and 95% CI of a fitted SARIMAX."""
    ci = res.conf_int()
    return pd.DataFrame({"coef": res.params, "std_err": res.bse, "z": res.tvalues,
                         "p_value": res.pvalues, "ci_low": ci.iloc[:, 0], "ci_high": ci.iloc[:, 1]})
