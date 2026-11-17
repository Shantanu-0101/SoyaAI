"""Loading and cleaning of the raw data files.

Two sources:
  * UN Comtrade monthly import files (one Excel file per year, HS 1507 or 1201)
  * SOPA daily refined soybean oil rate, Indore (Rs per 10 kg, quoted as a "low - high" range)

Every cleaning rule is a small, named function so it can be cited in the viva.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read config.yaml. SOYA_QUICK=1 shrinks the backtests for a fast smoke test."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("SOYA_QUICK") == "1":
        cfg["model_a"]["cv_initial_months"] = 178
        cfg["model_b"]["cv_origins"] = 6
    return cfg


def p(rel: str) -> Path:
    """Resolve a path from config.yaml relative to the project root."""
    return ROOT / rel


# ---------------------------------------------------------------------------
# UN Comtrade
# ---------------------------------------------------------------------------
KEEP_COLS = ["freqCode", "period", "refYear", "refMonth", "reporterDesc", "flowDesc",
             "partnerDesc", "partner2Desc", "customsDesc", "motDesc", "cmdCode",
             "classificationCode", "netWgt", "qty", "primaryValue",
             "isNetWgtEstimated", "legacyEstimationFlag", "isAggregate"]


def load_comtrade(folder: str | os.PathLike) -> pd.DataFrame:
    """Concatenate every Comtrade Excel file in a folder and keep the useful columns."""
    files = sorted(glob.glob(str(Path(folder) / "*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No Comtrade .xlsx files in {folder}")
    frames = []
    for f in files:
        df = pd.read_excel(f)
        df["source_file"] = Path(f).name
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return raw[KEEP_COLS + ["source_file"]]


def check_comtrade(raw: pd.DataFrame) -> dict:
    """Facts we must confirm before using the data (reported in notebook Step 0)."""
    return {
        "rows": len(raw),
        "files": raw.source_file.nunique(),
        "hs_codes": sorted(raw.cmdCode.unique().tolist()),
        "frequency": sorted(raw.freqCode.unique().tolist()),
        "reporters": sorted(raw.reporterDesc.unique().tolist()),
        "flows": sorted(raw.flowDesc.unique().tolist()),
        "partners": sorted(raw.partnerDesc.unique().tolist()),
        "partner2": sorted(raw.partner2Desc.unique().tolist()),
        "customs": sorted(raw.customsDesc.unique().tolist()),
        "transport": sorted(raw.motDesc.unique().tolist()),
        "duplicate_periods": int(raw.period.duplicated().sum()),
        "first_period": int(raw.period.min()),
        "last_period": int(raw.period.max()),
        "estimated_weight_rows": int(raw.isNetWgtEstimated.sum()),
    }


def clean_comtrade(raw: pd.DataFrame, start: str, end: str, impute_window: int = 2) -> pd.DataFrame:
    """Return one row per month with tonnes, value and unit price.

    Rules (each one is printed in the notebook):
      1. keep only Import / India / World partner / total customs / total transport rows,
         so aggregate and breakdown rows can never be double-counted;
      2. months with no row in the file are true zero-trade months (Comtrade omits zeros);
      3. a month with a positive USD value but a zero or blank weight is a *missing weight*,
         not zero trade: tonnes = value / average unit price of the neighbouring months;
      4. months after `end` are not published yet and are dropped, not set to zero.
    """
    df = raw[(raw.flowDesc == "Import") & (raw.reporterDesc == "India")
             & (raw.partnerDesc == "World") & (raw.partner2Desc == "World")
             & (raw.customsDesc == "TOTAL CPC") & (raw.motDesc == "TOTAL MOT")].copy()
    if df.period.duplicated().any():
        raise ValueError("More than one World/total row for a month: would double count")

    df["month"] = pd.to_datetime(df.period.astype(str), format="%Y%m")
    df = df.set_index("month").sort_index()
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    out = pd.DataFrame(index=idx)
    out.index.name = "month"
    out["value_usd"] = df.primaryValue.reindex(idx)
    out["tonnes"] = df.netWgt.reindex(idx) / 1000.0
    out["weight_estimated_by_comtrade"] = df.isNetWgtEstimated.reindex(idx).fillna(False).astype(bool)

    # Rule 2: absent month = zero trade
    out["zero_filled"] = out.value_usd.isna()
    out.loc[out.zero_filled, ["value_usd", "tonnes"]] = 0.0

    # Rule 3: missing weight with positive value -> impute
    missing_w = (out.value_usd > 0) & (out.tonnes.isna() | (out.tonnes <= 0))
    out["weight_imputed"] = missing_w
    unit = (out.value_usd / (out.tonnes * 1000)).where(~missing_w & (out.tonnes > 0))
    for m in out.index[missing_w]:
        lo, hi = m - pd.DateOffset(months=impute_window), m + pd.DateOffset(months=impute_window)
        neighbours = unit.loc[lo:hi].drop(m, errors="ignore").dropna()
        out.loc[m, "tonnes"] = out.loc[m, "value_usd"] / neighbours.mean() / 1000.0

    out["usd_per_kg"] = out.value_usd / (out.tonnes * 1000)
    # flag (do not change) unit prices far from their neighbours: a sign the weight is off
    roll = out.usd_per_kg.rolling(5, center=True, min_periods=3).median()
    out["unit_price_outlier"] = ((out.usd_per_kg / roll - 1).abs() > 0.2) & (out.tonnes > 0)
    return out


# ---------------------------------------------------------------------------
# SOPA daily rates
# ---------------------------------------------------------------------------
def load_sopa(path: str | os.PathLike) -> pd.DataFrame:
    """Parse the SOPA export: date text, 'low - high' rate text, station."""
    raw = pd.read_excel(path, dtype=str)
    rate_col = [c for c in raw.columns if "oil" in c.lower()][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"].str.strip(), format="%d %b %Y"),
        "rate_text": raw[rate_col].str.strip(),
        "station": raw["Station"].fillna("(blank)").str.strip(),
    })
    parts = df.rate_text.str.extract(r"^\s*([\d.,]+)\s*-\s*([\d.,]+)\s*$")
    a = pd.to_numeric(parts[0].str.replace(",", ""), errors="coerce")
    b = pd.to_numeric(parts[1].str.replace(",", ""), errors="coerce")
    df["reversed"] = a > b                      # older rows are written "high - low"
    df["low"] = np.minimum(a, b)
    df["high"] = np.maximum(a, b)
    df["spread"] = df.high - df.low
    df["mid"] = (df.low + df.high) / 2
    return df.sort_values("date").reset_index(drop=True)


def clean_sopa(df: pd.DataFrame, max_spread: float = 50, outlier_threshold: float = 0.08) -> pd.DataFrame:
    """Fix typo-like quotes with two transparent rules.

    Rule A (wide range): if high - low > max_spread on an ISOLATED day (the day before and
          the day after have normal ranges), one end is probably a typo. Use the end that is
          closer to the 11-day rolling median of normal quotes. A run of wide ranges
          (June 2013) is a genuine market feature, so its midpoint is kept.
    Rule B (isolated spike): if a quote is more than `outlier_threshold` away from the
          11-day rolling median AND the next quote returns to normal, set it to missing.
          A missing value is later handled by the Kalman filter, so no number is invented.
    """
    out = df.copy().set_index("date")
    normal_mid = out.mid.where(out.spread <= max_spread)
    ref = normal_mid.rolling(11, center=True, min_periods=3).median()

    is_wide = out.spread > max_spread
    wide = is_wide & ~is_wide.shift(1, fill_value=False) & ~is_wide.shift(-1, fill_value=False)
    pick_low = (out.low - ref).abs() <= (out.high - ref).abs()
    out["rate"] = out.mid
    out.loc[wide & pick_low, "rate"] = out.low
    out.loc[wide & ~pick_low, "rate"] = out.high
    out["fix_wide_range"] = wide

    med = out.rate.rolling(11, center=True, min_periods=5).median()
    dev = (out.rate / med - 1).abs()
    nxt_dev = (out.rate.shift(-1) / med - 1).abs()
    spike = (dev > outlier_threshold) & (nxt_dev < outlier_threshold / 2)
    out["fix_spike_removed"] = spike
    out.loc[spike, "rate"] = np.nan
    return out


def to_business_days(rate: pd.Series, start: str, end: str, ffill_limit: int = 3) -> pd.DataFrame:
    """Put the daily rate on a Mon-Fri business-day calendar.

    * A business day with a quote keeps it.
    * A business day without a quote (holiday) takes the last earlier quote
      (a Saturday quote counts) if that quote is at most `ffill_limit` business days old.
    * Longer gaps stay missing (NaN). statsmodels' state-space models skip missing
      observations in the likelihood, so we never manufacture a week of fake prices.
    """
    obs = rate.dropna()
    bdays = pd.bdate_range(start, end)
    out = pd.DataFrame(index=bdays)
    out.index.name = "date"
    out["observed"] = obs.reindex(bdays)
    last_date = pd.Series(obs.index, index=obs.index).reindex(
        obs.index.union(bdays)).ffill().reindex(bdays)
    last_val = obs.reindex(obs.index.union(bdays)).ffill().reindex(bdays)
    age = np.busday_count(last_date.values.astype("datetime64[D]"),
                          bdays.values.astype("datetime64[D]"))
    fill_ok = out.observed.isna() & (age <= ffill_limit) & last_val.notna()
    out["rate"] = out.observed.where(~fill_ok, last_val)
    out["filled"] = fill_ok
    return out


def monthly_mean(daily: pd.Series) -> pd.Series:
    """Average of the days that actually have a value, labelled by month start."""
    return daily.dropna().resample("MS").mean()
