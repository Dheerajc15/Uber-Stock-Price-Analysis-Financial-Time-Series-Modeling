from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from sklearn.metrics import mean_absolute_error

# ── GARCH(1,1) ─────────────────────────────────────────────────────────────

def run_garch_model(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    forecast_horizon: int = 5,
) -> tuple[object, dict, pd.DataFrame, pd.DataFrame]:
    """
    Fit a GARCH(1,1)-AR(1) model with Student-t innovations on the training
    set, then evaluate out-of-sample using rolling 1-step-ahead forecasts on
    the test set.

    Parameters
    ----------
    df             : DataFrame that includes a `log_return_pct` column
                     (log-returns × 100, i.e. percentage returns).
    train_ratio    : Fraction of data used for in-sample fitting (default 0.80).
    forecast_horizon : Number of future trading days to forecast (default 5).

    Returns
    -------
    garch_result   : Fitted arch ModelResult object.
    garch_metrics  : Dict of parameters and performance metrics.
    oos_df         : DataFrame of out-of-sample 1-step-ahead forecasts
                     (columns: predicted_vol_ann, realized_vol_proxy).
    forecast_df    : DataFrame of h-step-ahead forecasts beyond the sample.
    """
    if "log_return_pct" not in df.columns:
        raise ValueError(
            "Column 'log_return_pct' not found. "
            "Did you call add_finance_features()? "
            "This column should be log_return × 100."
        )

    # ── Drop NaN rows at the start (first row has no return) ──────────────
    returns_pct = df["log_return_pct"].dropna()

    if len(returns_pct) < 50:
        raise ValueError(
            f"Only {len(returns_pct)} non-NaN return observations. "
            "Need at least 50 for a reliable GARCH fit."
        )

    # ── Train / test chronological split ──────────────────────────────────
    split_idx  = int(len(returns_pct) * train_ratio)
    train_ret  = returns_pct.iloc[:split_idx]
    test_ret   = returns_pct.iloc[split_idx:]

    print(f"GARCH split → train: {len(train_ret)} days | test: {len(test_ret)} days")
    print(f"  Train period: {train_ret.index[0].date()} → {train_ret.index[-1].date()}")
    print(f"  Test  period: {test_ret.index[0].date()}  → {test_ret.index[-1].date()}")

    # ── Fit GARCH(1,1) with AR(1) mean on TRAINING set only ───────────────
    # mean="AR"  → r_t = c + φ·r_{t-1} + ε_t  (absorbs any autocorrelation)
    # vol="Garch" → σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    # dist="t"   → Student-t innovations (handles fat tails)
    am = arch_model(
        train_ret,
        mean="AR", lags=1,
        vol="Garch", p=1, q=1,
        dist="t",
        rescale=False,         
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        garch_result = am.fit(
            disp="off",
            options={"maxiter": 500},
        )

    # ── Extract parameters ─────────────────────────────────────────────────
    params = garch_result.params
    omega   = float(params.get("omega",   params.get("Const",   0.0)))
    alpha_1 = float(params.get("alpha[1]", 0.0))
    beta_1  = float(params.get("beta[1]",  0.0))
    persistence = alpha_1 + beta_1

    # Unconditional variance (percentage²) → annualized vol
    if persistence < 1.0:
        uncond_var_pct2   = omega / (1.0 - persistence)
        uncond_vol_daily  = np.sqrt(uncond_var_pct2) / 100.0  
        uncond_vol_ann    = uncond_vol_daily * np.sqrt(252)
    else:
        uncond_vol_ann = float("nan")

    # ── In-sample conditional volatility (training set) ───────────────────
    cond_var_train_pct2 = garch_result.conditional_volatility ** 2  
    last_cond_vol_ann   = float(
        np.sqrt(cond_var_train_pct2.iloc[-1]) / 100.0 * np.sqrt(252)
    )

    # ── Out-of-sample: rolling 1-step-ahead forecast on test set ──────────
    # We refit the model on an expanding window: train + first k test points,
    # then forecast 1 step ahead for point k+1.
    # For speed we use the "fixed" approach: freeze parameters from the full
    # training fit and run the variance recursion forward on the test set.
    # This is the standard "Walk-Forward without re-estimation" approach.

    # Initialise recursion from the last training variance
    sigma2_prev = float(cond_var_train_pct2.iloc[-1])
    resid_prev  = float(garch_result.resid.iloc[-1])

    oos_predicted_vol_ann = []
    realized_vol_proxy    = []   # |r_t| as a proxy for daily realised vol

    for ret_val in test_ret.values:
        # 1-step-ahead forecast of variance (still in pct²)
        sigma2_forecast = omega + alpha_1 * (resid_prev ** 2) + beta_1 * sigma2_prev

        # Convert pct² → decimal → annualized
        daily_vol_decimal = np.sqrt(sigma2_forecast) / 100.0
        ann_vol_forecast   = daily_vol_decimal * np.sqrt(252)
        oos_predicted_vol_ann.append(ann_vol_forecast)

        # Realized vol proxy: absolute return (decimal)
        realized_vol_proxy.append(abs(ret_val) / 100.0)

        # Update recursion: residual = return - AR(1) fitted mean
        # (approximate: use the return itself as the residual for simplicity)
        resid_prev  = ret_val
        sigma2_prev = sigma2_forecast

    oos_df = pd.DataFrame(
        {
            "predicted_vol_ann":  oos_predicted_vol_ann,
            "realized_vol_proxy": realized_vol_proxy,
        },
        index=test_ret.index,
    )

    # ── h-step-ahead forecast beyond the full sample ──────────────────────
    # Fit on the FULL dataset to use all available information for the forecast
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

    forecast_obj   = garch_full.forecast(horizon=forecast_horizon, reindex=False)
    forecast_var   = forecast_obj.variance.iloc[-1].values        
    forecast_daily = np.sqrt(forecast_var) / 100.0               
    forecast_ann   = forecast_daily * np.sqrt(252)              

    # Build date index for the forecast
    last_date = returns_pct.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
    )
    forecast_df = pd.DataFrame(
        {"annualized_vol_forecast": forecast_ann},
        index=future_dates,
    )
    forecast_df.index.name = "Date"

    # ── OOS evaluation metrics ─────────────────────────────────────────────
    oos_mae = float(mean_absolute_error(
        oos_df["realized_vol_proxy"], oos_df["predicted_vol_ann"]
    ))
    # Correlation between predicted and realized (directional accuracy)
    oos_corr = float(
        oos_df["predicted_vol_ann"].corr(oos_df["realized_vol_proxy"])
    )

    garch_metrics = {
        # Parameters (from training-set fit)
        "omega":              omega,
        "alpha_1":            alpha_1,
        "beta_1":             beta_1,
        "persistence":        persistence,
        "unconditional_vol_ann":  uncond_vol_ann,
        "last_cond_vol_ann":      last_cond_vol_ann,
        # Model fit (AIC/BIC from training-set fit)
        "aic":             float(garch_result.aic),
        "bic":             float(garch_result.bic),
        "log_likelihood":  float(garch_result.loglikelihood),
        # Train / test split info
        "n_train": int(len(train_ret)),
        "n_test":  int(len(test_ret)),
        # OOS performance
        "oos_mae_vol":  oos_mae,
        "oos_corr_vol": oos_corr,
    }

    return garch_result, garch_metrics, oos_df, forecast_df