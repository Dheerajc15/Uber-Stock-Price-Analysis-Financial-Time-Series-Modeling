import numpy as np
import pandas as pd


def add_finance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Explicit guard before log transform
    assert (df["price"] > 0).all(), "All price values must be positive before log transform."

    # Returns
    df["simple_return"] = df["price"].pct_change()
    df["log_return"]    = np.log(df["price"] / df["price"].shift(1))  # Explicit ratio form

    # Volume transforms
    df["log_volume"]       = np.log(df["Volume"])
    df["volume_log_change"] = df["log_volume"].diff()

    # Time features
    df["weekday"] = df["Date"].dt.day_name()
    df["year"]    = df["Date"].dt.year
    df["month"]   = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter

    # Use min_periods to document NaN window behavior explicitly
    trading_days = 252
    df["vol_21d"] = df["log_return"].rolling(21, min_periods=21).std() * np.sqrt(trading_days)
    df["vol_63d"] = df["log_return"].rolling(63, min_periods=63).std() * np.sqrt(trading_days)

    # Warn if unexpected mid-series NaN returns exist before filling
    mid_nans = df["simple_return"].iloc[1:].isna().sum()
    if mid_nans > 0:
        import warnings
        warnings.warn(
            f"{mid_nans} NaN values found in simple_return (mid-series). "
            "Gaps may indicate missing trading days. Filling with 0 for cumulative product.",
            UserWarning,
            stacklevel=2,
        )

    df["cum_return"]   = (1 + df["simple_return"].fillna(0)).cumprod()
    df["rolling_peak"] = df["cum_return"].cummax()
    df["drawdown"]     = df["cum_return"] / df["rolling_peak"] - 1

    # Lagged features for regression
    df["ret_lag1"] = df["log_return"].shift(1)
    df["ret_lag2"] = df["log_return"].shift(2)
    df["ret_lag5"] = df["log_return"].shift(5)

    return df