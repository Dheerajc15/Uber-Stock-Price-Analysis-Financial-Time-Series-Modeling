from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):  
        return float(mean_squared_error(y_true, y_pred) ** 0.5)


# ── GARCH model ────────────────────────────────────────────────────────────

def run_garch_model(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    forecast_horizon: int = 5,
) -> tuple[object, dict, pd.DataFrame, pd.DataFrame]:
    """
    Fit a GARCH(1,1)-AR(1) model with Student-t innovations.

    Parameters
    ----------
    df               : Feature DataFrame produced by add_finance_features().
                       Must contain the 'log_return_pct' column.
    train_ratio      : Fraction of observations used for training (default 0.80).
    forecast_horizon : Number of future trading days to forecast (default 5).

    Returns
    -------
    garch_result  : arch ModelResult object fitted on the training set.
    garch_metrics : dict of parameters, fit statistics, and OOS metrics.
    oos_df        : DataFrame with OOS predicted_vol_ann and realized_vol_proxy.
    forecast_df   : DataFrame with annualized_vol_forecast for future dates.
    """
    # ── Input validation ──────────────────────────────────────────────────
    if "log_return_pct" not in df.columns:
        raise ValueError(
            "'log_return_pct' column not found. "
            "Call add_finance_features() before run_garch_model(). "
            "This column should equal log_return × 100."
        )

    returns_pct = df["log_return_pct"].dropna()

    # Scale sanity check — std of pct returns must be in range 0.5–20.
    # Values outside this range indicate a unit error (decimal vs pct).
    ret_std = float(returns_pct.std())
    if ret_std < 0.1:
        raise ValueError(
            f"'log_return_pct' has std = {ret_std:.6f}. "
            "This looks like decimal returns (expected std ~0.5–5.0 for pct). "
            "Ensure features.py sets log_return_pct = log_return × 100."
        )
    if ret_std > 20.0:
        raise ValueError(
            f"'log_return_pct' has std = {ret_std:.4f}. "
            "Unusually large — check for outliers or incorrect units."
        )
    if len(returns_pct) < 50:
        raise ValueError(
            f"Only {len(returns_pct)} non-NaN return observations — need ≥ 50."
        )

    # ── Chronological train / test split ──────────────────────────────────
    split_idx = int(len(returns_pct) * train_ratio)
    train_ret = returns_pct.iloc[:split_idx]
    test_ret  = returns_pct.iloc[split_idx:]

    print(f"  GARCH split  →  train: {len(train_ret)} days  |  test: {len(test_ret)} days")
    print(f"  Train period :  {train_ret.index[0].date()}  →  {train_ret.index[-1].date()}")
    print(f"  Test  period :  {test_ret.index[0].date()}   →  {test_ret.index[-1].date()}")
    print(f"  Return std (pct/day): {ret_std:.4f}  ← should be 0.5–5.0")

    # ── Fit GARCH(1,1)-AR(1) on training set ──────────────────────────────
    am_train = arch_model(
        train_ret,
        mean="AR",    lags=1,     # AR(1) mean: captures mild return autocorrelation
        vol="Garch",  p=1, q=1,   # GARCH(1,1): one lag each of ε² and σ²
        dist="t",                  # Student-t: accommodates fat tails
        rescale=False,             # Do NOT let arch rescale — we've already scaled
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        garch_result = am_train.fit(
            disp="off",
            options={"maxiter": 500},
        )

    # ── Extract parameters ─────────────────────────────────────────────────
    params  = garch_result.params
    omega   = float(params.get("omega",    params.get("Const", 0.0)))
    alpha_1 = float(params.get("alpha[1]", 0.0))
    beta_1  = float(params.get("beta[1]",  0.0))
    nu      = float(params.get("nu", float("nan")))
    persistence = alpha_1 + beta_1

    # ── Convergence warnings ───────────────────────────────────────────────
    if persistence >= 1.0:
        warnings.warn(
            f"GARCH persistence α+β = {persistence:.6f} ≥ 1.0 (IGARCH). "
            "Possible causes: (1) model fed decimal returns instead of pct "
            "returns, (2) structural break in the data. "
            "Check that log_return_pct = log_return × 100.",
            UserWarning, stacklevel=2,
        )

    if not np.isnan(nu) and nu < 3.0:
        warnings.warn(
            f"Student-t ν = {nu:.4f} < 3 (implies infinite variance). "
            "This usually signals a scale error in the GARCH input. "
            "Verify log_return_pct units.",
            UserWarning, stacklevel=2,
        )

    # ── Unit conversions: pct/day  →  decimal annualised ──────────────────
    #
    # arch.conditional_volatility is in pct/day (same units as the input).
    # Step 1: divide by 100 to convert pct → decimal
    # Step 2: multiply by √252 to annualise

    cond_vol_pct      = garch_result.conditional_volatility   # Series, pct/day
    last_cond_vol_ann = float(
        (cond_vol_pct.iloc[-1] / 100.0) * np.sqrt(252)
    )

    # Unconditional (long-run) variance in pct²/day
    if persistence < 1.0:
        uncond_var_pct2  = omega / (1.0 - persistence)
        uncond_vol_daily = float(np.sqrt(uncond_var_pct2)) / 100.0   # pct → decimal
        uncond_vol_ann   = uncond_vol_daily * np.sqrt(252)
    else:
        uncond_vol_ann = float("nan")

    # ── Out-of-sample rolling 1-step-ahead forecast ────────────────────────
    sigma2_prev = float(cond_vol_pct.iloc[-1] ** 2)     # pct²/day
    resid_prev  = float(garch_result.resid.iloc[-1])    # pct/day

    oos_predicted_vol_ann: list[float] = []
    realized_vol_proxy:    list[float] = []

    for ret_val_pct in test_ret.values:
        # One-step-ahead variance forecast (pct²/day)
        sigma2_forecast  = omega + alpha_1 * (resid_prev ** 2) + beta_1 * sigma2_prev
        # Convert to decimal annualised
        vol_ann_forecast = (np.sqrt(sigma2_forecast) / 100.0) * np.sqrt(252)
        oos_predicted_vol_ann.append(float(vol_ann_forecast))
        # Realized proxy: |pct return| → decimal
        realized_vol_proxy.append(abs(float(ret_val_pct)) / 100.0)
        # Roll forward
        resid_prev  = ret_val_pct
        sigma2_prev = sigma2_forecast

    oos_df = pd.DataFrame(
        {
            "predicted_vol_ann":  oos_predicted_vol_ann,
            "realized_vol_proxy": realized_vol_proxy,
        },
        index=test_ret.index,
    )

    # ── h-step-ahead forecast on full dataset ─────────────────────────────
    # Re-fit on ALL available data so the forecast uses the most recent state.
    am_full = arch_model(
        returns_pct,
        mean="AR", lags=1,
        vol="Garch", p=1, q=1,
        dist="t",
        rescale=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        garch_full = am_full.fit(disp="off", options={"maxiter": 500})

    # arch.forecast(horizon=h) returns variance for each of the next h steps.
    forecast_obj = garch_full.forecast(horizon=forecast_horizon, reindex=False)
    forecast_var_pct2 = forecast_obj.variance.iloc[-1].values  # pct²/day, shape (h,)
    forecast_vol_ann  = (np.sqrt(forecast_var_pct2) / 100.0) * np.sqrt(252)

    last_date    = returns_pct.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
    )
    forecast_df = pd.DataFrame(
        {"annualized_vol_forecast": forecast_vol_ann.tolist()},
        index=future_dates,
    )
    forecast_df.index.name = "Date"

    # ── OOS evaluation metrics ─────────────────────────────────────────────
    oos_mae  = float(
        mean_absolute_error(oos_df["realized_vol_proxy"], oos_df["predicted_vol_ann"])
    )
    oos_corr = float(
        oos_df["predicted_vol_ann"].corr(oos_df["realized_vol_proxy"])
    )

    # ── Save training-set model summary text ──────────────────────────────
    from uber_stock.config import OUTPUT_METRICS
    with open(OUTPUT_METRICS / "garch_model_summary.txt", "w") as fh:
        fh.write(garch_result.summary().as_text())

    # ── Bundle metrics ─────────────────────────────────────────────────────
    garch_metrics: dict = {
        "omega":                 omega,
        "alpha_1":               alpha_1,
        "beta_1":                beta_1,
        "persistence":           persistence,
        "nu":                    float(nu),
        "unconditional_vol_ann": uncond_vol_ann,
        "last_cond_vol_ann":     last_cond_vol_ann,
        "aic":                   float(garch_result.aic),
        "bic":                   float(garch_result.bic),
        "log_likelihood":        float(garch_result.loglikelihood),
        "n_train":               int(len(train_ret)),
        "n_test":                int(len(test_ret)),
        "oos_mae_vol":           oos_mae,
        "oos_corr_vol":          oos_corr,
    }

    return garch_result, garch_metrics, oos_df, forecast_df
