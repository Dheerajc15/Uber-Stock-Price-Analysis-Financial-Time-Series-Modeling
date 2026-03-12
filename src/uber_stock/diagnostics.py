import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def run_return_diagnostics(df: pd.DataFrame) -> dict:
    r = df["log_return"].dropna()

    results = {}

    # Normality
    jb_stat, jb_p = jarque_bera(r)
    results["jarque_bera"] = {"stat": float(jb_stat), "p_value": float(jb_p)}

    # Stationarity
    adf_stat, adf_p, *_ = adfuller(r)
    results["adf"] = {"stat": float(adf_stat), "p_value": float(adf_p)}

    # Autocorrelation in returns
    lb = acorr_ljungbox(r, lags=[10, 20], return_df=True)

    # Convert integer index to string keys for JSON compatibility
    results["ljung_box"] = {str(k): v for k, v in lb.to_dict(orient="index").items()}

    # FIX B14: Unpack by index to be resilient to API changes
    arch_results = het_arch(r)
    arch_stat, arch_p = arch_results[0], arch_results[1]
    results["arch_test"] = {"stat": float(arch_stat), "p_value": float(arch_p)}

    return results


def compute_risk_metrics(df: pd.DataFrame, risk_free_rate_annual: float = 0.0) -> dict:
    # Guard against missing required columns
    required_cols = {"simple_return", "log_return", "drawdown"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_risk_metrics requires columns: {required_cols}. "
            f"Missing: {missing}. Did you call add_finance_features() first?"
        )

    r_simple = df["simple_return"].dropna()
    r_log    = df["log_return"].dropna()

    ann_ret = (1 + r_simple.mean()) ** 252 - 1

    # Use log-returns for annualized volatility 
    ann_vol = r_log.std() * np.sqrt(252)

    rf_daily    = (1 + risk_free_rate_annual) ** (1 / 252) - 1
    excess_daily = r_simple - rf_daily

    # Use excess_daily.std() in the Sharpe denominator
    sharpe = (
        (excess_daily.mean() / excess_daily.std()) * np.sqrt(252)
        if excess_daily.std() > 0
        else np.nan
    )

    var_95  = float(np.quantile(r_simple, 0.05))
    cvar_95 = float(r_simple[r_simple <= var_95].mean())
    max_dd  = float(df["drawdown"].min())

    return {
        "annualized_return":    float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe_ratio":         float(sharpe),
        "VaR_95_daily":         var_95,
        "CVaR_95_daily":        cvar_95,
        "max_drawdown":         max_dd,
    }