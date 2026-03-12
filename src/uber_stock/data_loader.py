from __future__ import annotations  

import pandas as pd


def load_and_clean_stock_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        from uber_stock.config import DATA_RAW
        csv_path = DATA_RAW / "uber_stock_data.csv"

    df = pd.read_csv(csv_path)

    # Standardize column names (strip extra spaces)
    df.columns = [c.strip() for c in df.columns]

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Deduplicate on Date to ensure one row per trading day
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Ensure numeric columns
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        required_cols.append("Adj Close")
    df = df.dropna(subset=required_cols)

    if "Adj Close" in df.columns:
        df["price"] = df["Adj Close"]
    else:
        df["price"] = df["Close"]

    # Keep only positive prices/volume
    df = df[(df["price"] > 0) & (df["Volume"] > 0)].copy()

    return df