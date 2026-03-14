from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot


def plot_price_series(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["price"], color="steelblue", linewidth=1.0)
    ax.set_title("Uber Adjusted Close Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_return_distribution(df: pd.DataFrame, save_path: Path) -> None:
    r = df["log_return"].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(r, kde=True, bins=60, ax=ax)
    ax.set_title("Distribution of Uber Daily Log-Returns (decimal)")
    ax.set_xlabel("Log-Return")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_return_qq(df: pd.DataFrame, save_path: Path) -> None:
    r = df["log_return"].dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    qqplot(r, line="s", ax=ax)
    ax.set_title("Q-Q Plot of Uber Daily Log-Returns")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_rolling_volatility(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["vol_21d"], label="21D Annualized Volatility")
    ax.plot(df.index, df["vol_63d"], label="63D Annualized Volatility")
    ax.set_title("Rolling Volatility (Annualized, from decimal log-returns)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (decimal)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_drawdown(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(
        df.index, df["drawdown"], 0,
        alpha=0.4, color="red", label="Drawdown"
    )
    ax.plot(df.index, df["drawdown"], color="darkred", linewidth=0.8)
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (fraction)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()