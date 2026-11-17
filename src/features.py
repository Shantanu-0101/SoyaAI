"""Exogenous variables: market prices, import-duty level, event dummies, lags.

The key function is `exog_for_window`. It builds the exogenous matrix for one
forecast origin using ONLY information available at that origin:

  * the exogenous series are cut at the origin date;
  * any value needed after the origin is filled with the last known value
    ("no-change" or random-walk forecast of the driver);
  * then each column is lagged by its configured lag.

So a lag of k >= h means the real value is used; a shorter lag means the
model sees a random-walk forecast of the driver, exactly as it would in real
life. This is how we avoid look-ahead leakage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import ccf, grangercausalitytests


# ---------------------------------------------------------------------------
# Market data from Yahoo Finance (optional, cached)
# ---------------------------------------------------------------------------
def load_market(tickers: dict, start: str, end: str, cache: Path, refresh: bool = False):
    """Daily closing prices. Uses a CSV cache so the project runs offline after one download.

    Returns (DataFrame or None, message).
    """
    cache = Path(cache)
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df, f"loaded from cache {cache.name} ({len(df)} rows)"
    try:
        import yfinance as yf
    except ImportError:
        return None, "yfinance not installed: market variables skipped"
    cols = {}
    for name, tk in tickers.items():
        try:
            h = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
        except Exception as e:  # network / rate limit
            return None, f"download failed for {tk}: {e}"
        if h is None or h.empty:
            return None, f"no data returned for {tk} (no internet or rate-limited)"
        close = h["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        cols[name] = close
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)
    return df, f"downloaded {list(tickers.values())} and cached to {cache.name}"


def clean_market(df: pd.DataFrame) -> pd.DataFrame:
    """Remove impossible values (Yahoo occasionally has zeros / tiny prints for futures)."""
    out = df.copy()
    out = out.where(out > 0)
    for c in out.columns:
        s = out[c]
        med = s.rolling(21, center=True, min_periods=5).median()
        bad = (s / med - 1).abs() > 0.25
        out.loc[bad, c] = np.nan
    return out


def import_margin(oil_rate_rs10kg, cbot_cents_lb, usdinr, duty_pct):
    """log(domestic price / landed cost of imported soybean oil).

    landed cost (Rs/kg) = CBOT soybean oil (US cents/lb) / 100 * 2.20462 lb/kg
                          * USD/INR * (1 + effective duty)
    A high margin means importing is profitable, so imports should rise a few
    months later (time for contracting and shipping). Freight, refining and
    the crude/refined price gap are ignored, so only CHANGES in the margin matter.
    """
    landed = cbot_cents_lb / 100 * 2.20462 * usdinr * (1 + duty_pct / 100)
    return np.log(oil_rate_rs10kg / 10) - np.log(landed)


# ---------------------------------------------------------------------------
# Policy and event variables
# ---------------------------------------------------------------------------
def load_duty_events(path) -> pd.DataFrame:
    ev = pd.read_csv(path, parse_dates=["effective_from"])
    return ev.sort_values("effective_from").reset_index(drop=True)


def duty_series(index: pd.DatetimeIndex, events: pd.DataFrame) -> pd.Series:
    """Effective duty (%) on crude soybean oil in force on each date (step function)."""
    s = events.set_index("effective_from").effective_duty_pct
    out = s.reindex(s.index.union(index)).ffill().reindex(index)
    return out.rename("duty")


def window_dummy(index: pd.DatetimeIndex, start: str, end: str, name: str) -> pd.Series:
    return pd.Series(((index >= start) & (index <= end)).astype(float), index=index, name=name)


def daily_to_monthly_share(daily: pd.Series) -> pd.Series:
    """Monthly value of a daily 0/1 dummy or step series = average over calendar days."""
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D")).ffill()
    return full.resample("MS").mean()


# ---------------------------------------------------------------------------
# Leak-free exogenous matrix for one forecast origin
# ---------------------------------------------------------------------------
def ar1_path(known: pd.Series, steps: int) -> np.ndarray:
    """Mean-reverting forecast of a driver from its own AR(1), fitted on known data only:
    x_{T+k} = c + phi * x_{T+k-1}."""
    x = known.dropna()
    y_, x_ = x.values[1:], x.values[:-1]
    phi, c = np.polyfit(x_, y_, 1)
    phi = float(np.clip(phi, -0.99, 0.99))
    out, last = [], x.values[-1]
    for _ in range(steps):
        last = c + phi * last
        out.append(last)
    return np.array(out)


def exog_for_window(raw: pd.DataFrame, lags: dict, train_index: pd.DatetimeIndex,
                    steps: int, freq: str, future_adjust: dict | None = None,
                    methods: dict | None = None):
    """Return (X_train, X_future) using only information up to train_index[-1].

    raw            : exogenous variables on the model frequency (no lag applied yet)
    lags           : {column: lag}
    future_adjust  : {column: function(series_after_origin) -> series}, used for
                     scenarios such as "+10% oil price" or "duty cut".
    methods        : {column: "last" | "ar1"} how a driver is forecast after the origin.
                     "last" = no change (random walk, default; right for prices and policy),
                     "ar1"  = its own AR(1), right for a mean-reverting spread like the margin.
    Gaps INSIDE the known history (e.g. months without SOPA quotes) are linearly
    interpolated; both ends of such a gap are before the origin, so nothing leaks.
    """
    origin = train_index[-1]
    cols = list(lags)
    methods = methods or {}
    known = raw[cols].loc[:origin].interpolate(limit_area="inside").ffill()
    future_index = pd.date_range(origin, periods=steps + 1, freq=freq)[1:]
    full_index = pd.date_range(known.index[0], future_index[-1], freq=freq)
    ext = known.reindex(full_index).ffill()                 # no-change rule after origin
    for c in cols:
        if methods.get(c) == "ar1":
            ext.loc[future_index, c] = ar1_path(known[c], steps)
    if future_adjust:
        for c, f in future_adjust.items():
            if c in ext:
                ext.loc[future_index, c] = f(ext.loc[future_index, c])
    lagged = pd.DataFrame({f"{c}_lag{lags[c]}": ext[c].shift(lags[c]) for c in cols})
    X_train = lagged.reindex(train_index)
    X_future = lagged.loc[future_index]
    # the first `lag` rows have no history; back-fill them with the earliest value
    X_train = X_train.bfill()
    return X_train, X_future


# ---------------------------------------------------------------------------
# Lead/lag screening
# ---------------------------------------------------------------------------
def ccf_table(y: pd.Series, x: pd.Series, max_lag: int) -> pd.Series:
    """corr(y_t, x_{t-k}) for k = 0..max_lag on already-stationary series."""
    d = pd.concat([y, x], axis=1).dropna()
    # statsmodels ccf(a, b)[k] = corr(a_{t+k}, b_t) = corr(y_t, x_{t-k}) when a=y, b=x
    vals = ccf(d.iloc[:, 0], d.iloc[:, 1], adjusted=False, nlags=max_lag + 1)
    return pd.Series(vals[: max_lag + 1], index=pd.RangeIndex(0, max_lag + 1, name="lag"))


def granger_table(y: pd.Series, x: pd.Series, max_lag: int) -> pd.DataFrame:
    """p-values of 'x does not Granger-cause y' (SSR F-test) for lags 1..max_lag."""
    import warnings
    d = pd.concat([y, x], axis=1).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # column 2 (x) is tested as a Granger cause of column 1 (y)
        res = grangercausalitytests(d.values, maxlag=max_lag)
    rows = [{"lag": k, "F": res[k][0]["ssr_ftest"][0], "p_value": res[k][0]["ssr_ftest"][1]}
            for k in range(1, max_lag + 1)]
    return pd.DataFrame(rows).set_index("lag")
