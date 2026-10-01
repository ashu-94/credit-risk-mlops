"""
Exploratory Data Analysis for the credit-risk dataset.

Produces a set of plots + a text summary in reports/ that you can screenshot
into your README or talk through in an interview. This is the "data science
craft" layer: understand the data before modelling it.

Usage:  python -m src.eda
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

from src import config


def load():
    if not config.DATA_PATH.exists():
        raise FileNotFoundError("Run `make data` first to generate the dataset.")
    return pd.read_csv(config.DATA_PATH)


def summary_stats(df: pd.DataFrame) -> dict:
    target = df[config.TARGET]
    return {
        "n_rows": int(len(df)),
        "n_features": len(config.FEATURES),
        "default_rate": float(target.mean()),
        "class_balance": {"repaid": int((target == 0).sum()), "defaulted": int((target == 1).sum())},
        "missing_values": int(df.isna().sum().sum()),
    }


def plot_target_balance(df, out):
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Repaid (0)", "Defaulted (1)"], counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Target class balance")
    ax.set_ylabel("count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_numeric_distributions(df, out):
    cols = config.NUMERIC_FEATURES
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(df[col], bins=30, color="#264653", alpha=0.8)
        ax.set_title(col, fontsize=9)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Numeric feature distributions")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def plot_default_rate_by_feature(df, out):
    """Default rate across binned/categorical features — the interview gold."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # credit score bins
    df["_cs_bin"] = pd.cut(df["credit_score"], bins=6)
    r1 = df.groupby("_cs_bin", observed=True)[config.TARGET].mean()
    axes[0].bar(range(len(r1)), r1.values, color="#e9c46a")
    axes[0].set_xticks(range(len(r1)))
    axes[0].set_xticklabels([str(i) for i in r1.index], rotation=45, ha="right", fontsize=7)
    axes[0].set_title("Default rate by credit score")

    # dti bins
    df["_dti_bin"] = pd.qcut(df["debt_to_income"], q=5, duplicates="drop")
    r2 = df.groupby("_dti_bin", observed=True)[config.TARGET].mean()
    axes[1].bar(range(len(r2)), r2.values, color="#f4a261")
    axes[1].set_xticks(range(len(r2)))
    axes[1].set_xticklabels([str(i) for i in r2.index], rotation=45, ha="right", fontsize=7)
    axes[1].set_title("Default rate by debt-to-income")

    # loan purpose
    r3 = df.groupby("loan_purpose", observed=True)[config.TARGET].mean().sort_values()
    axes[2].bar(range(len(r3)), r3.values, color="#e76f51")
    axes[2].set_xticks(range(len(r3)))
    axes[2].set_xticklabels(r3.index, rotation=45, ha="right", fontsize=8)
    axes[2].set_title("Default rate by loan purpose")

    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    df.drop(columns=["_cs_bin", "_dti_bin"], inplace=True)


def plot_correlations(df, out):
    num = df[config.NUMERIC_FEATURES + [config.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(num, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num))); ax.set_xticklabels(num.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(num))); ax.set_yticklabels(num.columns, fontsize=8)
    fig.colorbar(im); ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    df = load()
    stats = summary_stats(df)
    (config.REPORT_DIR / "eda_summary.json").write_text(json.dumps(stats, indent=2))

    plot_target_balance(df, config.REPORT_DIR / "eda_target_balance.png")
    plot_numeric_distributions(df, config.REPORT_DIR / "eda_distributions.png")
    plot_default_rate_by_feature(df, config.REPORT_DIR / "eda_default_rates.png")
    plot_correlations(df, config.REPORT_DIR / "eda_correlations.png")

    print("EDA complete. Summary:")
    print(json.dumps(stats, indent=2))
    print("Plots written to reports/")


if __name__ == "__main__":
    main()
