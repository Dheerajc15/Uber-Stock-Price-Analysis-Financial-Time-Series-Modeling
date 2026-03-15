import warnings

import numpy as np
import pandas as pd


def add_finance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all financial features used downstream by diagnostics, models,
    and plots.

    Column glossary
    ---------------
    log_return       : ln(P_t / P_{t-1})  — decimal form, e.g. 0.023 for +2.3%
    log_return_pct   : log_return × 100   — percentage form, e.g. 2.3
                       *** This is the GARCH input — percent scale is required
                           for numerical stability of the optimizer. ***
    simple_return    : (P_t - P_{t-1}) / P_{t-1}  — decimal pct-change
    log_volume       : ln(Volume)
    volume_log_change: day-on-day difference of log_volume
    vol_21d / vol_63d: rolling annualized volatility from decimal log-returns
    cum_return       : cumulative product of (1 + simple_return)
    drawdown         : cum_return / rolling_peak − 1  (always ≤ 0)
    ret_lag1/2/5     : lagged log_return 
    """
    df = df.copy()

    # ── Guard: prices must be positive before log transform ───────────────
    if not (df["price"] > 0).all():
        n_bad = (df["price"] <= 0).sum()
        raise ValueError(
            f"{n_bad} non-positive price value(s) found. "
            "data_loader.py should have filtered these out already."
        )

    # ── Returns — decimal form ───────────────────────────��─────────────────
    df["simple_return"] = df["price"].pct_change()
    df["log_return"]    = np.log(df["price"] / df["price"].shift(1))

    # ── Returns — percentage form (GARCH input) ────────────────────────────
    # Multiplying by 100 keeps variance in a human-readable range (~0.5–25 pct²)
    # instead of the tiny decimal range (~0.00005–0.0025), which prevents the
    # GARCH optimizer from converging to degenerate IGARCH solutions.
    df["log_return_pct"] = df["log_return"] * 100.0

    # ── Volume transforms ──────────────────────────────────────────────────
    # Guard against zero/NaN volume before log
    vol_safe = df["Volume"].clip(lower=1.0)
    df["log_volume"]        = np.log(vol_safe)
    df["volume_log_change"] = df["log_volume"].diff()

    # ── Time features ──────────────────────────────────────────────────────
    df["weekday"] = df.index.day_name()
    df["year"]    = df.index.year
    df["month"]   = df.index.month
    df["quarter"] = df.index.quarter

    # ── Rolling volatility (annualized) ───────────────────────────────────
    trading_days = 252
    df["vol_21d"] = (
        df["log_return"].rolling(21, min_periods=21).std() * np.sqrt(trading_days)
    )
    df["vol_63d"] = (
        df["log_return"].rolling(63, min_periods=63).std() * np.sqrt(trading_days)
    )

    # ── Warn about unexpected mid-series NaN returns ───────────────────────
    mid_nans = df["simple_return"].iloc[1:].isna().sum()
    if mid_nans > 0:
        warnings.warn(
            f"{mid_nans} NaN value(s) found in simple_return after row 0. "
            "This may indicate gaps in the trading-day data. "
            "Filling with 0 for the cumulative-return calculation only.",
            UserWarning,
            stacklevel=2,
        )

    # ── Cumulative return and drawdown ─────────────────────────────────────
    df["cum_return"]   = (1 + df["simple_return"].fillna(0)).cumprod()
    df["rolling_peak"] = df["cum_return"].cummax()
    df["drawdown"]     = df["cum_return"] / df["rolling_peak"] - 1

    # ── Lagged features  ───────────────────────────
    df["ret_lag1"] = df["log_return"].shift(1)
    df["ret_lag2"] = df["log_return"].shift(2)
    df["ret_lag5"] = df["log_return"].shift(5)

    return df