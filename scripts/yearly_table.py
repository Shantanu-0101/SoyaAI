"""Year-wise table: imports (HS 1507), Indore oil price, and the same in quintals.

Reads data/processed (written by the notebook / src.pipeline) and saves
outputs/figures/yearly_import_and_price.png and outputs/tables/yearly_import_and_price.csv.

Run from the project folder:
    C:\\Users\\DELL\\venvs\\soyaai\\Scripts\\python scripts\\yearly_table.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
imp = pd.read_csv(ROOT / "data/processed/imports_hs1507_monthly.csv", index_col=0, parse_dates=True)
sopa = pd.read_csv(ROOT / "data/processed/sopa_daily_clean.csv", index_col=0, parse_dates=True)
rate = sopa["rate"].dropna()

years = range(2011, 2027)
rows, csv = [], []
for y in years:
    t = imp.loc[str(y), "tonnes"].sum() / 1e6                       # million tonnes
    pr = rate.loc[str(y)].mean()                                   # Rs per 10 kg
    n_days = rate.loc[str(y)].size
    if y == 2026:   # partial year: compare with the same months of 2025
        t_prev = imp.loc["2025-01":"2025-08", "tonnes"].sum() / 1e6
        p_prev = rate.loc["2025-01":"2025-10"].mean()
    elif y > 2011:
        t_prev = imp.loc[str(y - 1), "tonnes"].sum() / 1e6
        p_prev = rate.loc[str(y - 1)].mean()
    else:
        t_prev = p_prev = None
    tc = None if t_prev is None else 100 * (t / t_prev - 1)
    pc = None if p_prev is None else 100 * (pr / p_prev - 1)
    csv.append({"year": y, "import_million_tonnes": round(t, 3), "import_change_pct": None if tc is None else round(tc, 1),
                "price_rs_per_10kg": round(pr, 1), "price_change_pct": None if pc is None else round(pc, 1),
                "price_days_quoted": n_days})
    fmt = lambda v: "" if v is None else ("  (0%)" if round(v) == 0 else f"  ({v:+.0f}%)")
    label = "2026*" if y == 2026 else (f"{y}†" if y in (2015, 2016) else str(y))
    # 1 million tonnes = 1,000,000,000 kg = 1 crore quintals (1 quintal = 100 kg);
    # Rs per 10 kg x 10 = Rs per quintal
    csv[-1].update({"import_crore_quintals": round(t, 3), "price_rs_per_quintal": round(pr * 10, 0),
                    "import_kg": round(t * 1e9)})
    rows.append([label, f"{t:.2f}{fmt(tc)}", f"{pr:,.0f}{fmt(pc)}",
                 f"{t:.2f} crore qtl  @  Rs {pr * 10:,.0f}/qtl"])

pd.DataFrame(csv).to_csv(ROOT / "outputs/tables/yearly_import_and_price.csv", index=False)
print(pd.DataFrame(csv).to_string(index=False))

TEXT, MUTED, RULE, UP, DOWN = "#111827", "#6b7280", "#d1d5db", "#15803d", "#b91c1c"
head = ["Year", "Import\n(million tonnes, HS 1507)", "Price\n(Rs per 10 kg, Indore)",
        "In quintals\n(import  @  price per quintal)"]
fig, ax = plt.subplots(figsize=(11.2, 0.42 * len(rows) + 1.9), dpi=200)
fig.patch.set_facecolor("white"); ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=head, cellLoc="left", colLoc="left",
               colWidths=[0.1, 0.26, 0.24, 0.4], bbox=[0, 0.15, 1, 0.8])
tbl.auto_set_font_size(False)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(RULE); cell.visible_edges = "B"; cell.set_linewidth(0.8); cell.PAD = 0.05
    cell.set_facecolor("white")
    t = cell.get_text(); t.set_color(TEXT)
    if r == 0:
        t.set_fontsize(10.5); t.set_fontweight("bold")
    else:
        t.set_fontsize(11)
ax.set_title("Soybean oil: India's imports and Indore price, by year",
             loc="left", fontsize=12.5, fontweight="bold", color=TEXT, pad=8)
note = ("(+x%) = change from the previous year.  Price = yearly average of the daily refined-oil rate.\n"
        "* 2026 = Jan–Aug imports and Jan–Oct price, compared with the same months of 2025.\n"
        "† 2015–16 price averaged over fewer days (SOPA data gaps).\n"
        "Import: UN Comtrade HS 1507 (Jul 2022 and Jul 2025 weights imputed from value).\n"
        "Units: 1 quintal (qtl) = 100 kg;  1 million tonnes = 100 crore kg = 1 crore quintals;  "
        "Rs per quintal = Rs per 10 kg x 10.")
fig.text(0.0, 0.0, note, fontsize=7.6, color=MUTED, ha="left", va="bottom", transform=ax.transAxes)
out = ROOT / "outputs/figures/yearly_import_and_price.png"
fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.25)
print("saved", out)
