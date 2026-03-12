import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5


def run_return_regression(df: pd.DataFrame):
    model_df = df.copy()

    # Predict next day's return (shift target backward by 1)
    model_df["target_next_ret"] = model_df["log_return"].shift(-1)

    cols = ["ret_lag1", "ret_lag2", "ret_lag5", "volume_log_change", "target_next_ret"]
    model_df = model_df[cols].dropna()

    # Guard for degenerate splits
    if len(model_df) < 10:
        raise ValueError(
            f"Dataset too small for regression after dropna: {len(model_df)} rows. "
            "Need at least 10 rows."
        )

    # Time-based split (better than random split for time series)
    split_idx = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_idx]
    test  = model_df.iloc[split_idx:]

    if len(test) < 2:
        raise ValueError(
            f"Test set has only {len(test)} row(s); need at least 2 for r2_score. "
            "Consider using a larger dataset or adjusting the split ratio."
        )

    feature_cols = ["ret_lag1", "ret_lag2", "ret_lag5", "volume_log_change"]

    # Force constant addition with has_constant="add"
    X_train = sm.add_constant(train[feature_cols], has_constant="add")
    y_train = train["target_next_ret"]

    X_test = sm.add_constant(test[feature_cols], has_constant="add")
    y_test = test["target_next_ret"]

    model  = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)

    metrics = {
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "r2":   float(r2_score(y_test, y_pred)),
    }

    pred_out = test.copy()
    pred_out["predicted"] = y_pred 

    return model, metrics, pred_out