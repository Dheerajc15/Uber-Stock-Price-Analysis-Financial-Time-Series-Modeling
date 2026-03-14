import warnings

import numpy as np
import pandas as pd


def add_finance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Guard ──────────────────────────────────────────────────────────────
    assert (df["price"] > 0).all(), (
        "All price values must be positive before log transform."
    )

    # ── Returns (DECIMAL form) ─────────────────────────────────────────────
    df["simple_return"] = df["price"].pct_change()
    df["log_return"]    = np.log(df["price"] / df["price"].shift(1))

    # ── Returns (PERCENTAGE form) ─────────────────────────
    df["log_return_pct"] = df["log_return"] * 100.0

    # ── Volume transforms ──────────────────────────────────────────────────
    df["log_volume"]        = np.log(df["Volume"])
    df["volume_log_change"] = df["log_volume"].diff()

    # ── Time features ──────────────────────────────────────────────────────
    df["weekday"] = df.index.day_name()
    df["year"]    = df.index.year
    df["month"]   = df.index.month
    df["quarter"] = df.index.quarter

    # ── Rolling annualized volatility (decimal log-returns × √252) ────────
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
            f"{mid_nans} NaN values found in simple_return (mid-series). "
            "Gaps may indicate missing trading days. Filling with 0 for cumulative product.",
            UserWarning, stacklevel=2,
        )

    # ── Cumulative return & drawdown ───────────────────────────────────────
    df["cum_return"]   = (1 + df["simple_return"].fillna(0)).cumprod()
    df["rolling_peak"] = df["cum_return"].cummax()
    df["drawdown"]     = df["cum_return"] / df["rolling_peak"] - 1

    return df