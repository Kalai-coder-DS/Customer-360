"""Static EDA charts (Matplotlib) using the shared theme.

Each function returns a Matplotlib Figure. Charts follow a few rules so they
communicate rather than decorate:

* one message per chart, stated in the title
* values printed directly on bars, so nobody has to read the axis
* colour only where it means something (retained = blue, churned = red,
  always with a legend or label)
* light grid, no 3-D, no unnecessary colour

The interactive Plotly versions for the dashboard come in Phase 7.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config import theme  # noqa: E402
from src.analysis import eda  # noqa: E402
from src.data.cleaning import CHURN_FLAG  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 24,
    "axes.labelcolor": theme.TEXT_MUTED,
    "axes.edgecolor": theme.BASELINE,
    "axes.facecolor": theme.SURFACE,
    "figure.facecolor": theme.SURFACE,
    "text.color": theme.TEXT,
    "xtick.color": theme.TEXT_SECONDARY,
    "ytick.color": theme.TEXT_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,   # gridlines behind bars
    "grid.color": theme.GRIDLINE,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def _subtitle(ax, text: str) -> None:
    """Small grey explanatory line under the title."""
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=9, color=theme.TEXT_MUTED,
            va="bottom")


def _percent_axis(ax, axis: str = "x") -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def plot_class_balance(df: pd.DataFrame):
    """How many customers stayed vs left."""
    table = eda.class_balance(df)
    labels = {"No": "Retained", "Yes": "Churned"}
    colours = [theme.STATUS["retained"], theme.STATUS["churned"]]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    names = [labels[i] for i in table.index]
    bars = ax.barh(names, table["customers"], color=colours, height=0.55)
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {int(row['customers']):,}  ({row['share']:.1%})", va="center",
                color=theme.TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, table["customers"].max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Customers")
    churn_share = table["share"].get("Yes", 0.0)
    ax.set_title(f"{churn_share:.1%} of customers churned")
    _subtitle(ax, "Class balance of the target variable 'Churn'")
    return fig


def plot_churn_rate_by(df: pd.DataFrame, column: str, title: str | None = None,
                       numeric_order: bool = False):
    """Churn rate for each value of a column, with the overall average marked.

    Groups are sorted by churn rate, or by their own value when ``numeric_order``
    is True (e.g. number of services 0, 1, 2 ...).
    """
    table = eda.churn_rate_by(df, column)
    if numeric_order:
        table = table.sort_index(key=lambda idx: idx.astype(float))
    overall = eda.overall_churn_rate(df)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(table) + 1.6))
    bars = ax.barh(table.index.astype(str), table["churn_rate"], color=theme.STATUS["churned"],
                   height=0.6)
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {row['churn_rate']:.1%}  (n={int(row['customers']):,})", va="center",
                fontsize=9, color=theme.TEXT)
    ax.axvline(overall, color=theme.TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlim(0, max(table["churn_rate"].max(), overall) * 1.45)
    _percent_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Churn rate")
    ax.set_title(title or f"Churn rate by {column}")
    _subtitle(ax, f"Share of customers in each group who left · n = customers in group · "
                  f"dashed line = average ({overall:.1%})")
    return fig


def plot_tenure_band_churn(df: pd.DataFrame):
    """Churn rate across tenure bands (natural order, so a column chart)."""
    table = eda.churn_rate_by(eda.add_tenure_band(df), eda.TENURE_BAND)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(table.index.astype(str), table["churn_rate"], color=theme.STATUS["churned"],
                  width=0.6)
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{row['churn_rate']:.1%}\n(n={int(row['customers']):,})", ha="center",
                va="bottom", fontsize=9, color=theme.TEXT)
    ax.set_ylim(0, table["churn_rate"].max() * 1.35)
    _percent_axis(ax, "y")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Churn rate")
    ax.set_xlabel("Months with the company")
    ax.set_title("Churn rate by how long customers have stayed")
    _subtitle(ax, "Customers grouped into tenure bands")
    return fig


def plot_numeric_by_churn(df: pd.DataFrame, column: str, unit: str = ""):
    """Distribution of a number for retained vs churned customers."""
    data = df[[column, CHURN_FLAG]].dropna()
    values = data[column].astype(float)
    bins = np.histogram_bin_edges(values, bins=30)
    fig, ax = plt.subplots(figsize=(8, 4))
    for flag, name in ((0, "retained"), (1, "churned")):
        subset = values[data[CHURN_FLAG] == flag]
        if subset.empty:
            continue
        ax.hist(subset, bins=bins, density=True, histtype="step", linewidth=2,
                color=theme.STATUS[name], label=f"{name.capitalize()} (median "
                f"{subset.median():,.1f}{unit})")
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"{column}{f' ({unit.strip()})' if unit else ''}")
    ax.set_ylabel("Share of customers")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title(f"{column}: churned vs retained customers")
    _subtitle(ax, "Each line shows how the group is spread; heights are scaled so groups "
                  "of different sizes compare fairly")
    return fig


def plot_churn_drivers(association: pd.DataFrame):
    """Strength of association with churn for every text column (Cramér's V)."""
    table = association.sort_values("cramers_v")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(table) + 1.6))
    bars = ax.barh(table.index, table["cramers_v"], color=theme.PRIMARY, height=0.6)
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {row['cramers_v']:.2f}  {row['strength']}", va="center", fontsize=9,
                color=theme.TEXT)
    ax.set_xlim(0, max(table["cramers_v"].max() * 1.35, 0.1))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cramér's V (0 = no link, 1 = perfect link)")
    ax.set_title("Which customer characteristics go together with churn?")
    _subtitle(ax, "Association strength, not cause. Higher means churn differs more "
                  "between the column's groups")
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame):
    """Spearman correlation between number columns and churn."""
    cmap = LinearSegmentedColormap.from_list("c360_diverging", theme.DIVERGING)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    image = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    labels = [c.replace("churn_flag", "Churn") for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10,
                    color=theme.PLUM_BG if value > 0.6 else theme.TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    ax.set_title("How the numbers move together")
    _subtitle(ax, "Spearman correlation: +1 rise together, -1 opposite, 0 unrelated")
    return fig


def plot_missing_values(missing: dict[str, int], n_rows: int):
    """Missing values per column in the raw data, before cleaning."""
    fig, ax = plt.subplots(figsize=(8, 2.6 if missing else 2))
    if not missing:
        ax.axis("off")
        ax.text(0.5, 0.5, "No missing values found in the raw data", ha="center",
                va="center", fontsize=12, color=theme.TEXT_SECONDARY)
        ax.set_title("Missing values before cleaning")
        return fig
    items = sorted(missing.items(), key=lambda kv: kv[1])
    names, counts = zip(*items)
    bars = ax.barh(names, counts, color=theme.PRIMARY, height=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {count:,} of {n_rows:,} rows ({count / n_rows:.2%})", va="center",
                fontsize=9)
    ax.set_xlim(0, max(counts) * 2.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Missing values")
    ax.set_title("Missing values before cleaning")
    _subtitle(ax, "Blank text counts as missing")
    return fig


def save_figure(fig, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
