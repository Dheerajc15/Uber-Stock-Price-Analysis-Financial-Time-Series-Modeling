# 📈 Uber Technologies (UBER) — Stock Volatility & Risk Analysis

> **A production-style Python analytics project** demonstrating end-to-end quantitative finance skills:
> data engineering → feature construction → statistical diagnostics → GARCH(1,1) volatility modelling → out-of-sample forecasting.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Objective

Uber Technologies (NYSE: UBER) went public in May 2019 at \$45 and has since navigated a global pandemic, a ride-share collapse, and a macro rate-shock cycle. This project uses the full post-IPO price history (May 2019 – February 2025) to answer one central question:

> **Can we model and forecast how *risky* Uber stock is likely to be on any given future trading day — and how did that risk evolve over time?**

This matters to:
- **Options traders** pricing contracts on UBER's future volatility
- **Portfolio managers** sizing positions dynamically
- **Risk managers** computing time-varying Value-at-Risk for regulatory capital

The project is structured as a **modular, reusable Python package** (`src/uber_stock/`) with a single-command pipeline entry point, reproducible outputs, and a fully annotated analysis notebook — designed for professional portfolio presentation.

---

## 🛠️ Implementations Carried Out

### 1. Data Engineering & Cleaning (`data_loader.py`)
- Loaded raw Yahoo Finance CSV with **DD-MM-YYYY date format** — correctly parsed using `format="%d-%m-%Y"` to prevent pandas from silently scrambling dates and producing corrupted returns
- Enforced strict data quality: column standardisation, coercion to numeric, deduplication, positive-price/volume filter
- Dropped pre-IPO rows (guard against phantom rows before 2019-05-10 that would generate a +∞ return on day 1)
- Applied calendar-gap detection to warn of missing data without forward-filling (preserving statistical integrity)

### 2. Feature Engineering (`features.py`)
- **Log-returns** `ln(P_t / P_{t-1})` — decimal and ×100 (pct) forms
- **Annualised rolling volatility** at 21-day (1-month) and 63-day (1-quarter) windows with explicit `min_periods` to document NaN behaviour
- **Cumulative return**, rolling peak, and **drawdown series**
- **Lagged return features** (lag-1, lag-2, lag-5) for regression modelling
- **Volume log-change** as a liquidity proxy
- Calendar features: year, month, quarter, weekday

### 3. Statistical Diagnostics (`diagnostics.py`)
Four formal hypothesis tests run on the full return series:

| Test | H₀ | Expected Result |
|---|---|---|
| **Jarque-Bera** | Returns are Normally distributed | Reject — fat tails confirmed |
| **Augmented Dickey-Fuller** | Unit root present (non-stationary) | Reject — returns are stationary |
| **Ljung-Box** (lags 10, 20) | No autocorrelation in returns | Mostly fail to reject |
| **ARCH LM** | Constant conditional variance | Strongly reject — ARCH effects present |

### 4. Risk Metrics (`diagnostics.py`)
- **CAGR** computed via exact calendar days (not assumed 252 trading days/year)
- **Annualised volatility** = `std(log_return) × √252`
- **Sharpe Ratio** with configurable risk-free rate
- **Value at Risk (VaR 95%)** — 5th percentile of daily simple returns (decimal form)
- **Conditional VaR (CVaR 95%)** — mean of all days worse than the VaR threshold
- **Maximum Drawdown** from peak cumulative return

### 5. GARCH(1,1) Volatility Model (`models.py`)
- Fit a **GARCH(1,1)-AR(1) model with Student-t innovations** using the `arch` library
- Input: log-returns scaled to percentage form (`×100`) for numerical stability
- **COVID-crash winsorisation** at ±4σ before training — prevents the 4–6 extreme March 2020 daily returns from driving the optimizer to an IGARCH boundary (α+β = 1) and infinite-variance Student-t (ν < 3)
- **80/20 chronological train/test split** — no data leakage
- **Rolling 1-step-ahead OOS forecast** on the held-out 20% test set
- **5-day-ahead forward forecast** from the most recent market state
- Full parameter extraction: ω, α, β, ν, persistence, unconditional long-run volatility

### 6. Visualisations (`plots.py`)
- Adjusted close price series with key event annotations
- Daily log-return histogram vs. Normal distribution overlay
- Q-Q plot vs. Normal quantiles
- 21-day and 63-day rolling annualised volatility
- Drawdown curve (fill-between)
- GARCH in-sample conditional volatility vs. realised absolute returns
- OOS predicted vs. realised volatility (test period)
- 5-day-ahead forward volatility forecast with uncertainty shading

### 7. End-to-End Pipeline (`pipeline.py` + `scripts/run_pipeline.py`)
- Single-command execution: `python scripts/run_pipeline.py`
- Automatically creates all output directories
- Persists: 5 figures, 3 CSV tables, 2 JSON/text metric files
- Returns a summary row CSV for README badge-style reporting

---

## 📊 Results Obtained

All numbers below are from the validated pipeline run on **1,444 trading days (2019-05-10 → 2025-02-05)**.

### Risk & Return Summary

| Metric | Value | Interpretation |
|---|---|---|
| **CAGR (Annualised Return)** | **+7.9%** | Moderate growth over 5.75 years from IPO price ~\$41.57 |
| **Annualised Volatility** | **52.7%** | High — consistent with a high-beta mobility/tech stock |
| **Sharpe Ratio** | **0.41** | Below S&P 500 (~0.60 same period) — UBER compensated investors less per unit of risk |
| **VaR 95% (Daily)** | **−4.6%** | On 5% of trading days, losses exceeded 4.6% in a single session |
| **CVaR 95% (Daily)** | **−7.0%** | When bad days happen, the average loss is 7.0% |
| **Maximum Drawdown** | **−68.1%** | Peak-to-trough during the COVID crash (Feb–Mar 2020): ~\$47 → ~\$14 |

### GARCH(1,1)-AR(1) Model Results

```
                    AR - GARCH Model Results
===========================================================================
Dep. Variable:        log_return_pct   Observations:    1,153 (train)
Vol Model:            GARCH(1,1)       Distribution:    Student's t
===========================================================================
             coef       std err     t        P>|t|    95% CI
---------------------------------------------------------------------------
omega (ω)   0.4900      0.300      1.633     0.102    [-0.098, 1.078]
alpha (α)   0.0993      0.036      2.787     0.005    [ 0.029, 0.169]
beta (β)    0.8559      0.058     14.646    <0.001    [ 0.741, 0.970]
nu (ν)      7.988       1.688      4.731    <0.001    [ 4.679, 11.30]
===========================================================================
AIC: 5,865.72   |   BIC: 5,896.02   |   Log-Likelihood: -2,926.86
```

| GARCH Parameter | Value | Interpretation |
|---|---|---|
| **α (ARCH)** | 0.0993 | ~10% of yesterday's shock feeds into today's variance |
| **β (GARCH)** | 0.8559 | 85.6% of yesterday's variance persists — high volatility clustering |
| **α + β (Persistence)** | **0.9552** | Covariance-stationary; vol shocks decay with half-life ~15 trading days |
| **ν (Student-t d.o.f.)** | **7.99** | Moderate fat tails (Normal = ∞); valid for equity returns post-COVID |
| **Unconditional Vol (Ann.)** | **52.5%** | Long-run average UBER volatility the model converges to |
| **Last Cond. Vol (Ann.)** | **37.2%** | Most recent market state — below long-run average (calm period) |

### Volatility Regimes (21-Day Rolling, Annualised)

| Year | Avg Volatility | Key Driver |
|---|---|---|
| 2019 | 46% | IPO uncertainty, market finding fair value |
| 2020 | 73% | COVID crash — single-day move of −23% on 18 Mar 2020 |
| 2021 | 46% | Recovery + SPAC/tech bubble optimism |
| 2022 | 64% | Federal Reserve rate hike cycle — growth stock de-rating |
| 2023 | 36% | Profitability confirmed — volatility normalised |
| 2024 | 40% | Stable operations, macro uncertainty |
| 2025 | 40% | Continued normalisation |

---

## 🔑 Key Takeaways

### 1. Log-Returns Are Essential for Rigorous Financial Analysis
Raw price levels are non-stationary (they drift and have non-constant variance). Log-returns are approximately stationary, additive over time, and symmetric — the correct input for ADF tests, GARCH models, and risk metrics. Using percentage-scaled log-returns (×100) for GARCH specifically ensures numerical stability during maximum-likelihood optimisation.

### 2. Volatility Is Not Constant — and It Clusters
The ARCH LM test (p ≈ 2.8 × 10⁻⁴⁶) provides overwhelming statistical evidence that UBER's daily return variance is time-varying. The GARCH persistence parameter (α+β = 0.955) quantifies that once volatility spikes — as in the COVID crash — it persists for approximately 15 trading days before decaying back toward its long-run average of 52.5%. A constant-variance model would be fundamentally wrong for risk forecasting.

### 3. Extreme Events Demand Fat-Tailed Distributions
The excess kurtosis of 10.17 and Student-t degrees of freedom ν = 8.0 confirm that UBER returns have significantly heavier tails than a Normal distribution. A Normal GARCH model would underestimate the probability of large daily moves by several orders of magnitude — directly impacting options pricing, VaR calculations, and margin requirements.

### 4. The Market Is Efficient for Return Prediction, Not Volatility Prediction
The AR(1) mean model coefficient (0.0138, p = 0.652) confirms that past returns have **no statistically significant predictive power** for the next day's return — consistent with the Efficient Market Hypothesis for a liquid NYSE stock. However, past *variance* (β = 0.856, p < 0.001) is a strong predictor of future variance. You cannot predict *where* a stock will go, but you can predict *how wildly* it will move.

### 5. Data Quality Is the Foundation of Every Model
The single most impactful fix in this project was correctly parsing the CSV date format (`DD-MM-YYYY` → `"%d-%m-%Y"`). Using `format="mixed"` caused pandas to misread ambiguous dates, scrambling the chronological order of prices. This produced corrupted log-returns with `std = 0.132` instead of `0.033`, making annualised volatility appear as **212%** instead of the correct **52.7%**, and GARCH parameters collapse to the degenerate IGARCH boundary. A single incorrect assumption at ingestion invalidated every downstream output.

---

## 🗂️ Project Structure

```text
uber-stock-prices/
│
├── data/
│   ├── raw/
│   │   └── uber_stock_data.csv          # Yahoo Finance OHLCV (DD-MM-YYYY format)
│   └── processed/                       # Reserved for cleaned/feature data exports
│
├── notebooks/
│   └── report.ipynb                     # 14-section annotated analysis notebook
│
├── outputs/
│   ├── figures/
│   │   ├── price_series.png
│   │   ├── return_distribution.png
│   │   ├── return_qq.png
│   │   ├── rolling_volatility.png
│   │   ├── drawdown.png
│   │   ├── garch_insample_fit.png
│   │   ├── garch_oos_forecast.png
│   │   └── garch_forward_forecast.png
│   ├── metrics/
│   │   ├── metrics_summary_table.csv    # One-row KPI summary (all metrics)
│   │   ├── summary_metrics.json         # Full nested metrics bundle
│   │   ├── garch_model_summary.txt      # statsmodels-style GARCH output table
│   │   └── return_model_summary.txt     # OLS regression summary
│   └── tables/
│       ├── uber_finance_features.csv    # Full feature DataFrame
│       ├── return_model_predictions.csv # OLS OOS predictions
│       ├── garch_oos_volatility.csv     # GARCH OOS predicted vs realised vol
│       └── garch_volatility_forecast.csv # 5-day-ahead forward forecast
│
├── scripts/
│   └── run_pipeline.py                  # Single-command entry point
│
├── src/
│   └── uber_stock/
│       ├── __init__.py
│       ├── config.py                    # Path constants + ensure_dirs()
│       ├── data_loader.py               # CSV ingestion, date parsing, cleaning
│       ├── features.py                  # Log-returns, rolling vol, drawdown, lags
│       ├── diagnostics.py               # JB, ADF, Ljung-Box, ARCH LM, risk metrics
│       ├── models.py                    # GARCH(1,1)-AR(1) + OLS return regression
│       ├── plots.py                     # All matplotlib/seaborn visualisations
│       └── pipeline.py                  # Orchestration: load → features → model → save
│
├── pyproject.toml                       # Package metadata + dependencies
├── requirements.txt                     # Pinned full environment
├── .gitignore
└── README.md
```

---

## ⚙️ How to Run

### Prerequisites
- Python 3.10+
- `arch` library: `pip install arch`

### Installation
```bash
git clone https://github.com/Dheerajc15/uber-stock-prices.git
cd uber-stock-prices
pip install -e .
# or for exact environment reproduction:
pip install -r requirements.txt
```

### Run the Full Pipeline
```bash
python scripts/run_pipeline.py
```
This executes all steps end-to-end: data loading → feature engineering → diagnostics → GARCH modelling → visualisations → metrics export.

### Open the Analysis Notebook
```bash
jupyter lab notebooks/report.ipynb
```
The notebook requires the pipeline to be run first (to generate `outputs/`), or run `Cell 1` which calls `run_full_analysis()` automatically.

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data** | pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Statistics** | SciPy, statsmodels |
| **Volatility Modelling** | `arch` (GARCH/EGARCH/GJR-GARCH) |
| **ML Baseline** | scikit-learn (OLS via statsmodels) |
| **Notebook** | JupyterLab |
| **Packaging** | setuptools (`pyproject.toml`) |
| **Version Control** | Git / GitHub |

---


---

*Built with Python · Data sourced from Yahoo Finance · Analysis period: May 2019 – February 2025*
