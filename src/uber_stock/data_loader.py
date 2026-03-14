from __future__ import annotations

import warnings
import pandas as pd


def load_and_clean_stock_data(csv_path: str | None = None) -> pd.DataFrame:
    """
    Load, clean, and return UBER stock price data.

    Key design decisions
    --------------------
    * Forward-fill (ffill) is used for missing prices — this mimics the
      real-world convention where a stock's last known price carries forward
      on non-trading days.  Backward-fill (bfill) would introduce look-ahead
      bias by using a *future* price to fill a *past* gap.
    * Gaps longer than MAX_GAP_DAYS are NOT filled; instead a warning is raised
      so the user can investigate data quality.
    * Returns are kept in USD.
    """
    MAX_GAP_DAYS = 5  # More than a long weekend 

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
            UserWarning, stacklevel=2,
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

    # ── Drop rows where required columns are all NaN ───────────────────────
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if "Adj Close" in df.columns:
        required_cols.append("Adj Close")
    df = df.dropna(subset=required_cols)

    # ── Set price column ───────────────────────────────────────────────────
    if "Adj Close" in df.columns:
        df["price"] = df["Adj Close"]
    else:
        df["price"] = df["Close"]

    # ── Keep only positive prices/volume ──────────────────────────────────
    df = df[(df["price"] > 0) & (df["Volume"] > 0)].copy()

    # ── Reindex to full calendar range & forward-fill gaps ────────────────
    # Build a complete business-day calendar between the first and last date
    full_bday_index = pd.bdate_range(df["Date"].min(), df["Date"].max())
    df = df.set_index("Date").reindex(full_bday_index)

    # Warn about large gaps BEFORE filling
    gap_mask = df["price"].isna()
    if gap_mask.any():
        # Find consecutive gap lengths
        gap_lengths = (
            gap_mask.astype(int)
            .groupby((gap_mask != gap_mask.shift()).cumsum())
            .sum()
        )
        max_gap = int(gap_lengths.max())
        n_missing = int(gap_mask.sum())
        if max_gap > MAX_GAP_DAYS:
            warnings.warn(
                f"Largest consecutive missing-data gap is {max_gap} business days "
                f"({n_missing} total missing rows). "
                "This may indicate data quality issues beyond normal holidays. "
                "Forward-fill is applied for all gaps.",
                UserWarning, stacklevel=2,
            )

    # Forward-fill: last known price carries forward (no look-ahead bias)
    fill_cols = ["Open", "High", "Low", "Close", "Volume"] + (
        ["Adj Close"] if "Adj Close" in df.columns else []
    ) + ["price"]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Drop any rows still missing after ffill (beginning of series with no prior data)
    df = df.dropna(subset=["price"]).copy()
    df.index.name = "Date"

    return df