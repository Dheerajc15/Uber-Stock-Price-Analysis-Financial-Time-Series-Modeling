"""
run_pipeline.py — Entry point for the Uber stock analysis pipeline.

Usage (from the project root):
    python scripts/run_pipeline.py

Outputs are written to:
    outputs/figures/   — all PNG plots
    outputs/tables/    — CSV files (features, GARCH OOS, forecast)
    outputs/metrics/   — JSON metrics bundle, summary CSV
"""

import sys
import os

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uber_stock.pipeline import run_full_analysis


if __name__ == "__main__":
    # run_full_analysis now returns 3 values:
    #   df             — feature DataFrame (Date as index)
    #   metrics        — dict with all metric sub-dicts
    #   garch_result   — fitted GARCH arch ModelResult object
    df, metrics, garch_result = run_full_analysis()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully ✅")
    print("=" * 60)

    print(f"\nData: {len(df)} trading days  |  "
          f"{df.index.min().date()} → {df.index.max().date()}")

    print("\n── Risk Metrics ──────────────────────────────────────────")
    for k, v in metrics["risk_metrics"].items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.4f}  ({v:.2%})")
        else:
            print(f"  {k:<30}: {v}")

    print("\n── GARCH(1,1) Parameters ─────────────────────────────────")
    gm = metrics["garch_metrics"]
    print(f"  omega        : {gm['omega']:.8f}")
    print(f"  alpha[1]     : {gm['alpha_1']:.6f}")
    print(f"  beta[1]      : {gm['beta_1']:.6f}")
    print(f"  persistence  : {gm['persistence']:.6f}  "
          f"{'✅ stationary' if gm['persistence'] < 1.0 else '⚠️  IGARCH'}")
    print(f"  uncond. vol  : {gm['unconditional_vol_ann']:.2%}  (annualized)")
    print(f"  OOS MAE vol  : {gm['oos_mae_vol']:.6f}")
    print(f"  OOS corr vol : {gm['oos_corr_vol']:.4f}")

    print(f"\n  Train size   : {gm['n_train']} days")
    print(f"  Test  size   : {gm['n_test']} days")

    print("\n── Outputs ───────────────────────────────────────────────")
    print("  outputs/figures/   → all PNG plots (including GARCH + OOS)")
    print("  outputs/tables/    → uber_finance_features.csv,")
    print("                       return_model_predictions.csv,")
    print("                       garch_oos_volatility.csv,")
    print("                       garch_volatility_forecast.csv")
    print("  outputs/metrics/   → summary_metrics.json,")
    print("                       metrics_summary_table.csv,")
