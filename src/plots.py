"""Plot helpers. Every function saves a PNG into outputs/figures and returns the figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
COLORS = {"actual": "#1f2937", "fc": "#2563eb", "band": "#93c5fd", "event": "#dc2626",
          "alt": "#ea580c", "grey": "#9ca3af"}

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    return fig


def series_with_events(s: pd.Series, events: pd.DataFrame | None, title: str, ylabel: str,
                       name: str, extra: pd.Series | None = None, extra_label: str = ""):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(s.index, s.values, color=COLORS["actual"], lw=1)
    if extra is not None:
        ax.plot(extra.index, extra.values, color=COLORS["alt"], lw=1.6, label=extra_label)
        ax.legend(loc="upper left")
    if events is not None:
        lo, hi = s.index.min(), s.index.max()
        for _, ev in events.iterrows():
            if lo <= ev.effective_from <= hi:
                ax.axvline(ev.effective_from, color=COLORS["event"], lw=0.8, ls="--", alpha=0.6)
                ax.text(ev.effective_from, ax.get_ylim()[1], f" {ev.effective_duty_pct:g}%",
                        color=COLORS["event"], fontsize=7, va="top", rotation=90)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return save(fig, name)


def month_profile(s: pd.Series, title: str, ylabel: str, name: str):
    """Box plot by calendar month: the visual check for seasonality."""
    df = pd.DataFrame({"v": s.values, "m": s.index.month})
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.boxplot([df.v[df.m == k] for k in range(1, 13)], tick_labels=list("JFMAMJJASOND"))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return save(fig, name)


def acf_pacf(s: pd.Series, lags: int, title: str, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    plot_acf(s.dropna(), lags=lags, ax=axes[0], title=f"ACF: {title}")
    plot_pacf(s.dropna(), lags=lags, ax=axes[1], title=f"PACF: {title}", method="ywm")
    return save(fig, name)


def stl_plot(stl_res, title: str, name: str):
    fig = stl_res.plot()
    fig.set_size_inches(10, 6)
    fig.suptitle(title, y=1.01)
    return save(fig, name)


def ccf_bar(c: pd.Series, n: int, title: str, name: str):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(c.index, c.values, color=COLORS["fc"])
    band = 1.96 / np.sqrt(n)
    ax.axhline(band, color=COLORS["event"], ls="--", lw=0.8)
    ax.axhline(-band, color=COLORS["event"], ls="--", lw=0.8)
    ax.set_xlabel("lag k (driver leads target by k periods)")
    ax.set_title(title)
    return save(fig, name)


def forecast_plot(history: pd.Series, fc: pd.DataFrame, title: str, ylabel: str, name: str,
                  tail: int = 60, scenarios: dict | None = None):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    h = history.iloc[-tail:]
    ax.plot(h.index, h.values, color=COLORS["actual"], lw=1.2, label="actual")
    ax.fill_between(fc.index, fc["lower"], fc["upper"], color=COLORS["band"], alpha=0.5,
                    label="95% interval")
    ax.plot(fc.index, fc["mean"], color=COLORS["fc"], lw=1.6, label="forecast")
    if scenarios:
        styles = ["--", ":", "-."]
        for i, (lab, s) in enumerate(scenarios.items()):
            ax.plot(s.index, s.values, ls=styles[i % 3], lw=1.3, label=lab)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8)
    return save(fig, name)


def residual_plot(res, title: str, name: str, burn: int = 0, lags: int = 24):
    r = pd.Series(res.resid).iloc[burn:].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    axes[0, 0].plot(r.index, r.values, color=COLORS["actual"], lw=0.8)
    axes[0, 0].axhline(0, color=COLORS["grey"])
    axes[0, 0].set_title("Residuals over time")
    axes[0, 1].hist(r, bins=30, density=True, color=COLORS["band"], edgecolor="white")
    x = np.linspace(r.min(), r.max(), 200)
    axes[0, 1].plot(x, np.exp(-(x - r.mean()) ** 2 / (2 * r.var())) / np.sqrt(2 * np.pi * r.var()),
                    color=COLORS["event"])
    axes[0, 1].set_title("Histogram vs normal curve")
    from scipy import stats
    stats.probplot(r, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot")
    plot_acf(r, lags=lags, ax=axes[1, 1], title="ACF of residuals")
    fig.suptitle(title)
    fig.tight_layout()
    return save(fig, name)


def cv_error_by_h(summary: pd.DataFrame, metric: str, title: str, name: str):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for model, g in summary.groupby("model"):
        ax.plot(g.h, g[metric], marker="o", label=model)
    ax.set_xlabel("forecast horizon")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    return save(fig, name)
