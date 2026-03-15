"""
pipeline.py — Orchestrates the full Uber stock analysis.

Return signature of run_full_analysis()
----------------------------------------
df              : pd.DataFrame   — feature-engineered price data (DatetimeIndex)
metrics_bundle  : dict           — diagnostics, risk_metrics, garch_metrics
garch_result    : arch result    — fitted GARCH(1,1) object (training set only)
"""
from __future__ import annotations

import json
from datetime import datetime

import matplotlib.dates  as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uber_stock.config import (
    OUTPUT_FIGURES,
    OUTPUT_METRICS,
    OUTPUT_TABLES,
    ensure_dirs,
)
from uber_stock.data_loader  import load_and_clean_stock_data
from uber_stock.diagnostics  import compute_risk_metrics, run_return_diagnostics
from uber_stock.features     import add_finance_features
from uber_stock.models       import run_garch_model
from uber_stock.plots        import (
    plot_drawdown,
    plot_price_series,
    plot_return_distribution,
    plot_return_qq,
    plot_rolling_volatility,
)


# ── ACF / PACF helper ─────────────────────────────────────────────────────

def _save_acf_pacf(series: pd.Series, title_suffix: str, save_path) -> None:
    """Save a side-by-side ACF / PACF figure for *series*."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf( series.dropna(), lags=40, ax=axes[0], title=f"ACF  —  {title_suffix}")
    plot_pacf(series.dropna(), lags=40, ax=axes[1], title=f"PACF —  {title_suffix}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── GARCH conditional-volatility plot ────────────────────────────────────

def _save_garch_vol_plot(
    df: pd.DataFrame,
    garch_result,
    train_n: int,
    oos_df: pd.DataFrame,
    save_path,
) -> None:
    """
    Two-panel figure:
      Top    : full log-return series with ±1σ in-sample GARCH bands
      Bottom : in-sample annualised conditional vol (train) +
               OOS 1-step-ahead forecast (test) +
               |return| realised proxy
    """
    cond_vol_pct = garch_result.conditional_volatility      # pct/day
    cond_vol_ann = (cond_vol_pct / 100.0) * np.sqrt(252)   # decimal ann.
    returns_dec  = df["log_return"].dropna()
    train_index  = returns_dec.index[:train_n]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Top panel — returns + GARCH ±1σ bands ────────────────────────────────
    ax1.plot(returns_dec.index, returns_dec.values,
             color="steelblue", linewidth=0.7, label="Log-return (decimal)")
    cond_vol_daily = cond_vol_pct.values / 100.0            # pct → decimal/day
    ax1.fill_between(
        train_index,
        -cond_vol_daily,
        +cond_vol_daily,
        alpha=0.35, color="orange", label="±1σ GARCH band (train)",
    )
    ax1.set_title("Uber Daily Log-Returns with GARCH(1,1) Volatility Bands")
    ax1.set_ylabel("Log-Return (decimal)")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Bottom panel — annualised conditional vol ────────────────────────────
    ax2.plot(train_index,     cond_vol_ann.values,
             color="darkorange", linewidth=1.0,
             label="In-sample cond. vol, ann. (train)")
    ax2.plot(oos_df.index,    oos_df["predicted_vol_ann"].values,
             color="firebrick", linewidth=1.0, linestyle="--",
             label="OOS 1-step-ahead forecast, ann. (test)")
    ax2.plot(oos_df.index,    oos_df["realized_vol_proxy"].values,
             color="grey", linewidth=0.8, alpha=0.6,
             label="|return| realised proxy (test)")
    ax2.set_title("GARCH(1,1) Annualised Conditional Volatility — In-Sample vs OOS")
    ax2.set_ylabel("Annualised Volatility (decimal)")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Main pipeline ─────────────────────────────────────────────────────────

def run_full_analysis():
    """
    Run the complete Uber stock analysis pipeline.

    Returns
    -------
    df              : pd.DataFrame  — full feature DataFrame (DatetimeIndex)
    metrics_bundle  : dict          — all metrics (diagnostics, risk, GARCH)
    garch_result    : arch result   — GARCH(1,1) fitted on training set
    """
    # 0. Create output directories ─────────────────────────────────────────
    ensure_dirs()

    # 1. Load data and engineer features ───────────────────────────────────
    df = load_and_clean_stock_data()
    df = add_finance_features(df)

    print(f"  Data loaded: {len(df):,} actual trading days "
          f"({df.index[0].date()} → {df.index[-1].date()})")

    # 2. Statistical diagnostics and risk metrics ───────────────────────────
    diag = run_return_diagnostics(df)
    risk = compute_risk_metrics(df)

    # 3. GARCH(1,1)-AR(1) volatility model ─────────────────────────────────
    garch_result, garch_metrics, oos_df, forecast_df = run_garch_model(df)

    # 4. Standard price / return plots ─────────────────────────────────────
    plot_price_series(       df, OUTPUT_FIGURES / "price_series.png")
    plot_return_distribution(df, OUTPUT_FIGURES / "return_distribution.png")
    plot_return_qq(          df, OUTPUT_FIGURES / "return_qq.png")
    plot_rolling_volatility( df, OUTPUT_FIGURES / "rolling_volatility.png")
    plot_drawdown(           df, OUTPUT_FIGURES / "drawdown.png")

    # 5. ACF / PACF of raw log-returns ─────────────────────────────────────
    _save_acf_pacf(
        df["log_return"],
        title_suffix="Log-Returns (decimal)",
        save_path=OUTPUT_FIGURES / "acf_pacf_returns.png",
    )

    # 6. ACF / PACF of GARCH standardised residuals ────────────────────────
    std_resid = pd.Series(
        garch_result.std_resid,
        index=df["log_return"].dropna().index[: len(garch_result.std_resid)],
    )
    _save_acf_pacf(
        std_resid,
        title_suffix="GARCH Standardised Residuals",
        save_path=OUTPUT_FIGURES / "acf_pacf_residuals.png",
    )

    # 7. GARCH conditional-volatility plot ─────────────────────────────────
    _save_garch_vol_plot(
        df,
        garch_result,
        train_n  = garch_metrics["n_train"],
        oos_df   = oos_df,
        save_path= OUTPUT_FIGURES / "garch_conditional_volatility.png",
    )

    # 8. Tabular outputs ───────────────────────────────────────────────────
    df.to_csv(       OUTPUT_TABLES / "uber_finance_features.csv")
    oos_df.to_csv(   OUTPUT_TABLES / "garch_oos_volatility.csv")
    forecast_df.to_csv(OUTPUT_TABLES / "garch_volatility_forecast.csv")

    # 9. Metrics bundle as JSON ────────────────────────────────────────────
    metrics_bundle = {
        "diagnostics":   diag,
        "risk_metrics":  risk,
        "garch_metrics": garch_metrics,
    }

    def _to_json_safe(obj):
        """Recursively convert numpy scalars and NaN for JSON serialisation."""
        if isinstance(obj, dict):
            return {k: _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None          # JSON null rather than invalid NaN token
        return obj

    with open(OUTPUT_METRICS / "summary_metrics.json", "w") as fh:
        json.dump(_to_json_safe(metrics_bundle), fh, indent=2)

    # 10. One-row summary CSV ──────────────────────────────────────────────
    # Includes all key metrics in a single row for easy README / dashboard use.
    gm = garch_metrics
    summary_row = {
        "run_timestamp":         datetime.now().isoformat(timespec="seconds"),
        # Data
        "n_trading_days":        len(df),
        "date_start":            str(df.index[0].date()),
        "date_end":              str(df.index[-1].date()),
        # Risk metrics (all in decimal form)
        "annualized_return":     risk["annualized_return"],
        "annualized_volatility": risk["annualized_volatility"],
        "sharpe_ratio":          risk["sharpe_ratio"],
        "VaR_95_daily":          risk["VaR_95_daily"],
        "CVaR_95_daily":         risk["CVaR_95_daily"],
        "max_drawdown":          risk["max_drawdown"],
        # Statistical tests
        "jb_pvalue":             diag["jarque_bera"]["p_value"],
        "jb_skewness":           diag["jarque_bera"]["skewness"],
        "jb_kurtosis":           diag["jarque_bera"]["kurtosis"],
        "adf_pvalue":            diag["adf"]["p_value"],
        "arch_pvalue":           diag["arch_test"]["p_value"],
        # GARCH parameters
        "garch_omega":           gm["omega"],
        "garch_alpha":           gm["alpha_1"],
        "garch_beta":            gm["beta_1"],
        "garch_persistence":     gm["persistence"],
        "garch_nu":              gm["nu"],
        "garch_uncond_vol_ann":  gm["unconditional_vol_ann"],
        "garch_last_vol_ann":    gm["last_cond_vol_ann"],
        "garch_aic":             gm["aic"],
        "garch_bic":             gm["bic"]
    }
    pd.DataFrame([summary_row]).to_csv(
        OUTPUT_METRICS / "metrics_summary_table.csv", index=False
    )

    return df, metrics_bundle, garch_result