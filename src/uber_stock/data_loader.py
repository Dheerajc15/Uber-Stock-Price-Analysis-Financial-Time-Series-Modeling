from __future__ import annotations

import warnings

import pandas as pd


def load_and_clean_stock_data(csv_path: str | None = None) -> pd.DataFrame:
    if csv_path is None:
        from uber_stock.config import DATA_RAW
        csv_path = DATA_RAW / "uber_stock_data.csv"

    df = pd.read_csv(csv_path)

    # ── Standardize column names ───────────────────────────────────────────
    df.columns = [c.strip() for c in df.columns]

    # ── Parse dates ────────────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(
        df["Date"], format="mixed", dayfirst=False, errors="coerce"
    )
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        warnings.warn(
            f"{invalid_dates} rows had unparseable dates and were dropped.",
            UserWarning,
            stacklevel=2,
        )
    df = df.dropna(subset=["Date"]).copy()

    # ── Deduplicate and sort ───────────────────────────────────────────────
    df = (
        df.drop_duplicates(subset=["Date"])
          .sort_values("Date")
          .reset_index(drop=True)
    )

    # ── Ensure numeric columns ─────────────────────────────────────────────
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop rows where any required column is NaN ─────────────────────────
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        required_cols.append("Adj Close")
    df = df.dropna(subset=required_cols)

    # ── Set price column ───────────────────────────────────────────────────
    if "Adj Close" in df.columns:
        df["price"] = df["Adj Close"]
    else:
        df["price"] = df["Close"]

    # ── Keep only positive prices and positive volume ──────────────────────
    df = df[(df["price"] > 0) & (df["Volume"] > 0)].copy()

    # ── Set DatetimeIndex from actual trading days only ────────────────────
    df = df.set_index("Date")
    df.index.name = "Date"

    # ── Warn about calendar gaps longer than 5 days ────────────────────────
    # (normal weekends = 3 days; long-weekend holidays = 4 days)
    if len(df) > 1:
        day_gaps = df.index.to_series().diff().dt.days.dropna()
        large_gaps = day_gaps[day_gaps > 5]
        if not large_gaps.empty:
            worst_gap  = int(large_gaps.max())
            worst_date = large_gaps.idxmax().date()
            warnings.warn(
                f"{len(large_gaps)} gap(s) longer than 5 calendar days detected. "
                f"Largest: {worst_gap} days ending {worst_date}. "
                "This may indicate missing data in the CSV. "
                "No forward-fill is applied — only actual trading days are used.",
                UserWarning,
                stacklevel=2,
            )

    return df