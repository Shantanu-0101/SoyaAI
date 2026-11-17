"""Builds the processed datasets used by both models. Run with:  python -m src.pipeline"""
from __future__ import annotations

import pandas as pd

from . import data as D
from . import features as F


def build_all(cfg: dict, refresh_market: bool = False, verbose: bool = True) -> dict:
    out = {}
    # ---------------- imports (both HS codes, the target is chosen in config) -----
    ma = cfg["model_a"]
    for hs, folder in cfg["paths"]["raw_comtrade"].items():
        raw = D.load_comtrade(D.p(folder))
        out[f"comtrade_raw_{hs}"] = raw
        out[f"imports_{hs}"] = D.clean_comtrade(raw, "2011-01", ma["end"], ma["impute_window"])

    # ---------------- SOPA daily rate -------------------------------------------
    mb = cfg["model_b"]
    sopa_raw = D.load_sopa(D.p(cfg["paths"]["raw_sopa"]))
    sopa = D.clean_sopa(sopa_raw, mb["max_spread"], mb["outlier_threshold"])
    out["sopa_raw"], out["sopa_clean"] = sopa_raw, sopa
    full_b = D.to_business_days(sopa.rate, "2011-01-03", mb["end"], mb["ffill_limit"])
    out["sopa_bday_full"] = full_b

    # ---------------- policy / events ------------------------------------------
    events = F.load_duty_events(D.p(cfg["paths"]["duty_events"]))
    out["duty_events"] = events
    bidx = full_b.index
    duty_d = F.duty_series(bidx, events)
    ev = cfg["events"]
    covid_d = F.window_dummy(bidx, *ev["covid_lockdown"], "covid")
    war_d = F.window_dummy(bidx, *ev["ukraine_shock"], "ukraine")

    # ---------------- market data ------------------------------------------------
    mkt = None
    if cfg["market"]["enabled"]:
        mkt, msg = F.load_market(cfg["market"]["tickers"], cfg["market"]["start"],
                                 cfg["market"]["end"], D.p(cfg["paths"]["market_cache"]),
                                 refresh=refresh_market)
        out["market_msg"] = msg
        if verbose:
            print("market data:", msg)
        if mkt is not None:
            mkt = F.clean_market(mkt)
    out["market_daily"] = mkt

    # ---------------- daily modelling frame (Model B) ---------------------------
    daily = pd.DataFrame({"rate": full_b.rate, "filled": full_b.filled, "duty": duty_d,
                          "covid": covid_d, "ukraine": war_d})
    if mkt is not None:
        m_b = mkt.reindex(mkt.index.union(bidx)).ffill(limit=5).reindex(bidx)
        daily = daily.join(m_b)
    out["daily"] = daily

    # ---------------- monthly modelling frame (Model A) -------------------------
    target = out[f"imports_{ma['target_hs']}"]
    months = target.index
    monthly = pd.DataFrame(index=months)
    monthly["tonnes"] = target.tonnes
    monthly["usd_per_kg"] = target.usd_per_kg
    monthly["oil_rate"] = D.monthly_mean(sopa.rate).reindex(months)
    monthly["duty"] = F.daily_to_monthly_share(F.duty_series(
        pd.date_range("2011-01-01", months[-1] + pd.offsets.MonthEnd(0), freq="D"), events)).reindex(months)
    monthly["covid"] = F.daily_to_monthly_share(F.window_dummy(
        pd.date_range("2011-01-01", months[-1] + pd.offsets.MonthEnd(0)), *ev["covid_lockdown"], "c")).reindex(months)
    monthly["ukraine"] = F.daily_to_monthly_share(F.window_dummy(
        pd.date_range("2011-01-01", months[-1] + pd.offsets.MonthEnd(0)), *ev["ukraine_shock"], "u")).reindex(months)
    if mkt is not None:
        mm = mkt.resample("MS").mean().reindex(months)
        monthly = monthly.join(mm)
        monthly["margin"] = F.import_margin(monthly.oil_rate, monthly.cbot_soyoil,
                                            monthly.usdinr, monthly.duty)
    monthly.index.freq = "MS"
    out["monthly"] = monthly

    # ---------------- save -------------------------------------------------------
    pdir = D.p(cfg["paths"]["processed"])
    pdir.mkdir(parents=True, exist_ok=True)
    for hs in cfg["paths"]["raw_comtrade"]:
        out[f"imports_{hs}"].to_csv(pdir / f"imports_hs{hs}_monthly.csv")
    sopa.to_csv(pdir / "sopa_daily_clean.csv")
    daily.to_csv(pdir / "model_b_daily.csv")
    monthly.to_csv(pdir / "model_a_monthly.csv")
    return out


if __name__ == "__main__":
    cfg = D.load_config()
    res = build_all(cfg)
    print(res["monthly"].tail())
    print(res["daily"].tail())
