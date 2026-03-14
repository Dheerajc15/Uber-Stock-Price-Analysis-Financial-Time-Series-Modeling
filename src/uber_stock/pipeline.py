"""
pipeline.py — Orchestrates the full Uber stock analysis.

Return signature of run_full_analysis()
----------------------------------------
df              : pd.DataFrame   — feature-engineered price data
metrics_bundle  : dict           — diagnostics, risk, GARCH metrics
garch_result    : arch result    — fitted GARCH(1,1) object (training set)
"""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from uber_stock.config import (
    OUTPUT_FIGURES,
    OUTPUT_METRICS,
    OUTPUT_TABLES,
    ensure_dirs,
)
from uber_stock.data_loader import load_and_clean_stock_data
from uber_stock.diagnostics import compute_risk_metrics, run_return_diagnostics
from uber_stock.features import add_finance_features
from uber_stock.models import run_garch_model
from uber_stock.plots import (
    plot_drawdown,
    plot_price_series,
    plot_return_distribution,
    plot_return_qq,
    plot_rolling_volatility,
)


# ── ACF/PACF helper ───────────────────────────────────────────────────────

def _save_acf_pacf(series: pd.Series, title_suffix: str, save_path) -> None:
    """Save a side-by-side ACF / PACF figure for *series*."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(),  lags=40, ax=axes[0], title=f"ACF — {title_suffix}")
    plot_pacf(series.dropna(), lags=40, ax=axes[1], title=f"PACF — {title_suffix}")
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
      Top    : log-returns with ±1σ in-sample GARCH bands
      Bottom : annualized conditional volatility (in-sample) +
               OOS 1-step-ahead predicted vol (test set)
    """
    # In-sample conditional vol in decimal annualized form
    cond_vol_pct   = garch_result.conditional_volatility          # pct/day
    cond_vol_ann   = (cond_vol_pct / 100.0) * np.sqrt(252)        # decimal ann.
    returns_dec    = df["log_return"].dropna()

    # Align index: cond_vol is on train set only
    train_index = returns_dec.index[:train_n]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # ── Top panel: returns + bands ─────────────────────────────────────────
    ax1.plot(returns_dec.index, returns_dec.values,
             color="steelblue", linewidth=0.7, label="Log-return (decimal)")
    # Only draw bands over the train period
    cond_vol_daily = cond_vol_pct.values / 100.0
    ax1.fill_between(
        train_index,
        -cond_vol_daily,
        +cond_vol_daily,
        alpha=0.35, color="orange", label="±1σ GARCH band (train)"
    )
    ax1.set_title("Uber Daily Log-Returns with GARCH(1,1) Volatility Bands")
    ax1.set_ylabel("Log-Return (decimal)")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Bottom panel: in-sample vol + OOS vol ─────────────────────────────
    ax2.plot(train_index, cond_vol_ann.values,
             color="darkorange", linewidth=1.0, label="In-sample cond. vol (ann.)")
    ax2.plot(oos_df.index, oos_df["predicted_vol_ann"].values,
             color="firebrick", linewidth=1.0, linestyle="--",
             label="OOS 1-step-ahead forecast (ann.)")
    ax2.plot(oos_df.index, oos_df["realized_vol_proxy"].values,
             color="grey", linewidth=0.8, alpha=0.6,
             label="|return| realized proxy")
    ax2.set_title("GARCH(1,1) Annualized Conditional Volatility — In-Sample vs OOS")
    ax2.set_ylabel("Annualized Volatility (decimal)")
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
    df              : pd.DataFrame  — full feature DataFrame (Date as index)
    metrics_bundle  : dict          — all metrics (diagnostics, risk, GARCH)
    garch_result    : arch result   — fitted GARCH(1,1) object (train set)
    """
    # ── 0. Ensure output directories exist ────────────────────────────────
    ensure_dirs()

    # ── 1. Load and feature-engineer data ─────────────────────────────────
    df = load_and_clean_stock_data()
    df = add_finance_features(df)

    # ── 2. Diagnostics & risk metrics ─────────────────────────────────────
    diag = run_return_diagnostics(df)
    risk = compute_risk_metrics(df)

    # ── 3. GARCH(1,1) with train/test split ────────────────────────────────
    garch_result, garch_metrics, oos_df, forecast_df = run_garch_model(df)

    # ── 4. Standard price/return plots ────────────────────────────────────
    plot_price_series(df,            OUTPUT_FIGURES / "price_series.png")
    plot_return_distribution(df,     OUTPUT_FIGURES / "return_distribution.png")
    plot_return_qq(df,               OUTPUT_FIGURES / "return_qq.png")
    plot_rolling_volatility(df,      OUTPUT_FIGURES / "rolling_volatility.png")
    plot_drawdown(df,                OUTPUT_FIGURES / "drawdown.png")

    # ── 5. ACF/PACF of raw returns (pre-GARCH inspection) ─────────────────
    _save_acf_pacf(
        df["log_return"],
        title_suffix="Log-Returns (decimal)",
        save_path=OUTPUT_FIGURES / "acf_pacf_returns.png",
    )

    # ── 6. ACF/PACF of GARCH standardized residuals (model validation) ─────
    std_resid = pd.Series(
        garch_result.std_resid,
        index=df["log_return"].dropna().index[: len(garch_result.std_resid)],
    )
    _save_acf_pacf(
        std_resid,
        title_suffix="GARCH Standardized Residuals",
        save_path=OUTPUT_FIGURES / "acf_pacf_residuals.png",
    )

    # ── 7. GARCH conditional-volatility plot ──────────────────────────────
    _save_garch_vol_plot(
        df,
        garch_result,
        train_n=garch_metrics["n_train"],
        oos_df=oos_df,
        save_path=OUTPUT_FIGURES / "garch_conditional_volatility.png",
    )

    # ── 8. Save tabular outputs ────────────────────────────────────────────
    df.to_csv(OUTPUT_TABLES / "uber_finance_features.csv")
    oos_df.to_csv(OUTPUT_TABLES / "garch_oos_volatility.csv")
    forecast_df.to_csv(OUTPUT_TABLES / "garch_volatility_forecast.csv")

    # ── 9. Save metrics bundle as JSON ───────────────────────────────────
    metrics_bundle = {
        "diagnostics":  diag,
        "risk_metrics": risk,
        "garch_metrics": garch_metrics,
    }

    # Convert any numpy floats → Python float for JSON serialisation
    def _to_json_safe(obj):
        if isinstance(obj, dict):
            return {k: _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    with open(OUTPUT_METRICS / "summary_metrics.json", "w") as f:
        json.dump(_to_json_safe(metrics_bundle), f, indent=2)

    # ── 11. One-row summary CSV ────────────────────────────────────────────
    summary_row = {
        "run_timestamp":   datetime.now().isoformat(timespec="seconds"),
        **{k: v for k, v in risk.items()},
        "garch_alpha":       garch_metrics["alpha_1"],
        "garch_beta":        garch_metrics["beta_1"],
        "garch_persistence": garch_metrics["persistence"],
        "garch_uncond_vol":  garch_metrics["unconditional_vol_ann"],
        "garch_oos_mae":     garch_metrics["oos_mae_vol"],
        "garch_oos_corr":    garch_metrics["oos_corr_vol"],
        "jb_pvalue":         diag["jarque_bera"]["p_value"],
        "adf_pvalue":        diag["adf"]["p_value"],
    }
    pd.DataFrame([summary_row]).to_csv(
        OUTPUT_METRICS / "metrics_summary_table.csv", index=False
    )

    return df, metrics_bundle, garch_result