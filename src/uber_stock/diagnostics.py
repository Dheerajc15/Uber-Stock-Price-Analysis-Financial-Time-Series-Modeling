import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller


def run_return_diagnostics(df: pd.DataFrame) -> dict:
    """
    Statistical tests on the log-return series.
    Returns are passed in DECIMAL form (log_return column).
    """
    r = df["log_return"].dropna()

    results = {}

    # ── Normality ──────────────────────────────────────────────────────────
    jb_stat, jb_p = jarque_bera(r)
    results["jarque_bera"] = {"stat": float(jb_stat), "p_value": float(jb_p)}

    # ── Stationarity ───────────────────────────────────────────────────────
    adf_stat, adf_p, *_ = adfuller(r)
    results["adf"] = {"stat": float(adf_stat), "p_value": float(adf_p)}

    # ── Autocorrelation in returns ─────────────────────────────────────────
    lb = acorr_ljungbox(r, lags=[10, 20], return_df=True)
    # Convert integer index to string keys for JSON compatibility
    results["ljung_box"] = {
        str(k): v for k, v in lb.to_dict(orient="index").items()
    }

    # ── ARCH / volatility-clustering test ─────────────────────────────────
    arch_results = het_arch(r)
    results["arch_test"] = {
        "stat":    float(arch_results[0]),
        "p_value": float(arch_results[1]),
    }

    return results


def compute_risk_metrics(
    df: pd.DataFrame,
    risk_free_rate_annual: float = 0.0,
) -> dict:
    """
    Compute standard risk/performance metrics from the feature DataFrame.

    Annualized return
    -----------------
    Uses the geometric CAGR formula:

        CAGR = (P_final / P_initial) ^ (252 / N_trading_days) - 1

    This is correct and standard in finance. The previous formula —
    compounding the arithmetic mean of daily returns — produces
    misleadingly large numbers (e.g. 400%+) because it amplifies the
    compounding effect of day-to-day variance.

    Annualized volatility
    ---------------------
    Computed on DECIMAL log-returns (log_return column).
    log_return values are in the range ±0.15 for typical trading days,
    so std() * sqrt(252) gives a realistic annualized figure (~30–60%
    for a high-beta stock like UBER).
    """
    required_cols = {"simple_return", "log_return", "drawdown", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_risk_metrics requires columns: {required_cols}. "
            f"Missing: {missing}. Did you call add_finance_features() first?"
        )

    r_simple = df["simple_return"].dropna()
    r_log    = df["log_return"].dropna()

    # ── Annualized return — CAGR (geometric) ──────────────────────────────
    # Number of trading days in the full series (not just non-NaN returns)
    n_trading_days = len(df)
    years = n_trading_days / 252.0

    # Use the price column from the full DataFrame (before dropna on returns)
    p_initial = float(df["price"].dropna().iloc[0])
    p_final   = float(df["price"].dropna().iloc[-1])

    if years > 0 and p_initial > 0:
        ann_ret = (p_final / p_initial) ** (1.0 / years) - 1.0
    else:
        ann_ret = float("nan")

    # ── Annualized volatility (decimal log-returns) ────────────────────────
    ann_vol = float(r_log.std() * np.sqrt(252))

    # ── Sharpe ratio ───────────────────────────────────────────────────────
    rf_daily     = (1 + risk_free_rate_annual) ** (1.0 / 252) - 1
    excess_daily = r_simple - rf_daily
    sharpe = (
        float((excess_daily.mean() / excess_daily.std()) * np.sqrt(252))
        if excess_daily.std() > 0
        else float("nan")
    )

    # ── Tail risk ──────────────────────────────────────────────────────────
    var_95  = float(np.quantile(r_simple, 0.05))
    cvar_95 = float(r_simple[r_simple <= var_95].mean())
    max_dd  = float(df["drawdown"].min())

    return {
        "annualized_return":     ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio":          sharpe,
        "VaR_95_daily":          var_95,
        "CVaR_95_daily":         cvar_95,
        "max_drawdown":          max_dd,
    }