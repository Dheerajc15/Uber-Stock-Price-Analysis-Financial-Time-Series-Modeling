from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot


def plot_price_series(df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["price"])
    plt.title("Uber Adjusted Close Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_return_distribution(df: pd.DataFrame, save_path: Path) -> None:
    r = df["log_return"].dropna()
    plt.figure(figsize=(10, 5))
    sns.histplot(r, kde=True, bins=50)
    plt.title("Distribution of Uber Daily Log Returns")
    plt.xlabel("Log Return")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_return_qq(df: pd.DataFrame, save_path: Path) -> None:
    r = df["log_return"].dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    qqplot(r, line="s", ax=ax)
    ax.set_title("Q-Q Plot of Uber Daily Log Returns")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_rolling_volatility(df: pd.DataFrame, save_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["vol_21d"], label="21D Annualized Volatility")
    plt.plot(df["Date"], df["vol_63d"], label="63D Annualized Volatility")
    plt.title("Rolling Volatility (Annualized)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_drawdown(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(df["Date"], df["drawdown"], 0, alpha=0.4, color="red", label="Drawdown")
    ax.plot(df["Date"], df["drawdown"], color="darkred", linewidth=0.8)
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()