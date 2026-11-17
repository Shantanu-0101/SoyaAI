# SoyaAI: forecasting India's soybean oil imports and the Indore soybean oil rate

ARIMAX / SARIMAX time-series forecasting with exogenous drivers.

* **Model A (monthly):** India's soybean oil imports (UN Comtrade HS 1507, tonnes), SARIMAX with s = 12, 12-month forecast.
* **Model B (daily):** SOPA refined soybean oil rate, Indore (Rs per 10 kg), ARIMAX, 30-trading-day forecast.
* **Link:** the monthly-averaged, lagged oil rate (on its own, and inside an *import margin*) is an exogenous input to Model A. The backtest tests whether it helps.

The full reasoning for every step is in `notebooks/soya_forecasting.ipynb`.

## Folder structure

```
SoyaAI/
├── config.yaml                 every modelling choice (target, lags, horizons, dates)
├── requirements.txt            pinned package versions
├── data/
│   ├── raw/
│   │   ├── comtrade_1507/      UN Comtrade monthly files, HS 1507 (soybean oil)
│   │   ├── comtrade_1201/      UN Comtrade monthly files, HS 1201 (soybean seed)
│   │   ├── sopa/               SOPA daily rate export
│   │   └── external/           duty-event table (with sources) + cached Yahoo Finance prices
│   └── processed/              cleaned monthly / daily datasets (written by the pipeline)
├── src/
│   ├── data.py                 loading + cleaning rules
│   ├── features.py             duty, dummies, market data, lags, leak-free exogenous matrix
│   ├── models.py               Naive, SeasonalNaive, ETS, SARIMA/SARIMAX, auto_arima, AIC/BIC grid
│   ├── evaluate.py             rolling-origin CV, MAE/RMSE/MASE/sMAPE, Diebold-Mariano, diagnostics
│   ├── plots.py                figures
│   ├── pipeline.py             builds data/processed from data/raw
│   └── experiments.py          two-stage model comparison, final fit, scenarios
├── notebooks/soya_forecasting.ipynb   end-to-end notebook with "Why?" explanations
├── scripts/
│   ├── metric_tables.py        accuracy tables of the final models (light PNG, for slides)
│   └── yearly_table.py         year-wise import / price / quintal table (PNG + CSV)
├── outputs/
│   ├── figures/                all plots (PNG)
│   ├── tables/                 backtest results, parameter tables, diagnostics (CSV)
│   └── forecasts/              final forecasts + scenarios (CSV, read by the app)
└── app/streamlit_app.py        forecast viewer with interval bands
```

## Setup (Windows, run once)

The virtual environment is kept **outside OneDrive**, so OneDrive does not try to sync thousands of package files.

```powershell
py -3.11 -m venv C:\Users\DELL\venvs\soyaai
C:\Users\DELL\venvs\soyaai\Scripts\python -m pip install -r requirements.txt
C:\Users\DELL\venvs\soyaai\Scripts\python -m ipykernel install --user --name soyaai --display-name "Python (soyaai)"
```

## Run

```powershell
# 1. open the notebook and choose the "Python (soyaai)" kernel, then Run All (~6 minutes)
C:\Users\DELL\venvs\soyaai\Scripts\jupyter-lab.exe notebooks\soya_forecasting.ipynb

# or run it headless and save the outputs inside the notebook
C:\Users\DELL\venvs\soyaai\Scripts\python -m nbconvert --to notebook --execute --inplace notebooks\soya_forecasting.ipynb --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=soyaai

# 2. the Streamlit app (after the notebook has run)
C:\Users\DELL\venvs\soyaai\Scripts\streamlit run app\streamlit_app.py

# 3. presentation tables (after the notebook has run)
C:\Users\DELL\venvs\soyaai\Scripts\python scripts\metric_tables.py
C:\Users\DELL\venvs\soyaai\Scripts\python scripts\yearly_table.py
```

**Google Colab:** upload the whole `SoyaAI` folder to Google Drive, open the notebook in Colab, and edit the path in the first code cell if needed. That cell installs the requirements.

**Offline:** Yahoo Finance prices are downloaded once and cached in `data/raw/external/market_daily.csv`. Delete that file to refresh it. If there is no internet and no cache, the pipeline reports it and the market variables are skipped.

## Data sources

| data | source | notes |
|---|---|---|
| Soybean oil imports | UN Comtrade, India, Import, Partner World, HS 1507, monthly 2011-01 to 2026-08 | Jul 2022 and Jul 2025 weights imputed from unit value |
| Soybean seed imports | UN Comtrade, HS 1201 | 39 zero months; kept as an optional target |
| Refined soybean oil rate | SOPA daily rates, Indore | unit Rs per 10 kg, verified against SOPA's published quotes |
| USD/INR, Brent, CBOT soybean oil | Yahoo Finance (`INR=X`, `BZ=F`, `ZL=F`) | cached CSV |
| Import duty on crude soybean oil | Government notifications and news (each row has its source) | `data/raw/external/duty_events_crude_soyoil.csv` |

## Changing the target

In `config.yaml`, set `model_a.target_hs: "1201"` and `model_a.transform: log1p` to model soybean seed instead. For that series, sMAPE and MASE stay defined at zero months (plain MAPE would not).
