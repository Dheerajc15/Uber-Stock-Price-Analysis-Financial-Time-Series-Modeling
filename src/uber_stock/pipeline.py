import json
from datetime import datetime
import pandas as pd

from uber_stock.config import OUTPUT_FIGURES, OUTPUT_METRICS, OUTPUT_TABLES
from uber_stock.data_loader import load_and_clean_stock_data
from uber_stock.features import add_finance_features
from uber_stock.diagnostics import run_return_diagnostics, compute_risk_metrics
from uber_stock.models import run_return_regression
from uber_stock.plots import (
    plot_price_series,
    plot_return_distribution,
    plot_return_qq,
    plot_rolling_volatility,
    plot_drawdown,
)


def run_full_analysis():
    df = load_and_clean_stock_data()
    df = add_finance_features(df)

    diag = run_return_diagnostics(df)
    risk = compute_risk_metrics(df)

    # Model
    model, model_metrics, pred_df = run_return_regression(df)

    # visuals
    plot_price_series(df, OUTPUT_FIGURES / "price_series.png")
    plot_return_distribution(df, OUTPUT_FIGURES / "return_distribution.png")
    plot_return_qq(df, OUTPUT_FIGURES / "return_qq.png")
    plot_rolling_volatility(df, OUTPUT_FIGURES / "rolling_volatility.png")
    plot_drawdown(df, OUTPUT_FIGURES / "drawdown.png")

    # Save tables
    df.to_csv(OUTPUT_TABLES / "uber_finance_features.csv", index=False)
    pred_df.to_csv(OUTPUT_TABLES / "return_model_predictions.csv", index=False)

    # Save metrics (JSON)
    metrics_bundle = {
        "diagnostics": diag,
        "risk_metrics": risk,
        "model_metrics": model_metrics,
    }
    with open(OUTPUT_METRICS / "summary_metrics.json", "w") as f:
        json.dump(metrics_bundle, f, indent=2)

    # Save a one-row summary CSV (nice for README)
    summary_row = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        **metrics_bundle["risk_metrics"],
        **metrics_bundle["model_metrics"],
        "jb_pvalue": metrics_bundle["diagnostics"]["jarque_bera"]["p_value"],
        "adf_pvalue": metrics_bundle["diagnostics"]["adf"]["p_value"],
    }
    pd.DataFrame([summary_row]).to_csv(
        OUTPUT_METRICS / "metrics_summary_table.csv", index=False
    )

    # Save model summary text
    model_summary_text = model.summary().as_text()
    with open(OUTPUT_METRICS / "return_model_summary.txt", "w") as f:
        f.write(model_summary_text)

    return df, metrics_bundle, model_summary_text