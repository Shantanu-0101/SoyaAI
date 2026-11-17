"""Light-theme summary tables of the final models' backtest accuracy.

Reads outputs/tables/cv_summary_model_{a,b}.csv (written by the notebook) and saves
outputs/figures/metrics_model_{a,b}_light.png.

Run from the project folder:
    C:\\Users\\DELL\\venvs\\soyaai\\Scripts\\python scripts\\metric_tables.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
TEXT, RULE = "#111827", "#d1d5db"

SPECS = [
    ("metrics_model_a_light", "cv_summary_model_a.csv", "SARIMAX+margin",
     "Model A: monthly soybean oil imports (SARIMAX + import margin)",
     "MAE", lambda h: f"{h} month" + ("s" if h > 1 else ""), lambda v: f"{v:.0f} kt", lambda v: f"{100 - v:.0f}%"),
    ("metrics_model_b_light", "cv_summary_model_b.csv", "ARIMAX(0, 1, 0)+cbot_brent",
     "Model B: daily oil rate (ARIMAX + CBOT and Brent, lag 1)",
     "MAE\n(Rs per 10 kg)", lambda h: f"{h} day" + ("s" if h > 1 else ""), lambda v: f"{v:.1f}", lambda v: f"{100 - v:.1f}%"),
]

for name, csv, model, title, mae_head, hlab, mae_fmt, acc_fmt in SPECS:
    s = pd.read_csv(TAB / csv)
    s = s[s.model == model].sort_values("h")
    rows = [[hlab(r.h), acc_fmt(r.sMAPE), mae_fmt(r.MAE), f"{r.MASE:.2f}", f"{100 * r.coverage95:.0f}%"]
            for r in s.itertuples()]
    head = ["Horizon", "Accuracy\n(100 − sMAPE)", mae_head, "MASE", "95% interval\ncoverage"]
    fig, ax = plt.subplots(figsize=(9.2, 1.35 + 0.52 * len(rows)), dpi=200)
    fig.patch.set_facecolor("white"); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=head, cellLoc="left", colLoc="left",
                   colWidths=[0.18, 0.22, 0.2, 0.15, 0.22], bbox=[0, 0, 1, 0.86])
    tbl.auto_set_font_size(False)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(RULE); cell.visible_edges = "B"; cell.set_linewidth(0.8); cell.PAD = 0.04
        cell.set_facecolor("white")
        t = cell.get_text(); t.set_color(TEXT)
        t.set_fontsize(11 if r == 0 else 12)
        if r == 0 or (r == 1 and c == 1):
            t.set_fontweight("bold")
    ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", color=TEXT, pad=10)
    fig.savefig(FIG / f"{name}.png", facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", FIG / f"{name}.png")
