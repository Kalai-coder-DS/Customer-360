"""Charts for Phase 3 customer segmentation (Matplotlib, shared theme).

Churn-rate-by-segment charts reuse ``eda_charts.plot_churn_rate_by`` so every
churn chart in the project looks and reads the same.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import theme
from src.visualization.eda_charts import _percent_axis, _subtitle  # shared helpers


def plot_k_selection(k_scores: pd.DataFrame, chosen_k: int):
    """Elbow (inertia) and silhouette side by side: two charts, two scales."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    ks = k_scores.index.to_numpy()
    for ax, col, title, note in (
        (left, "inertia", "Elbow method", "Lower = tighter groups; look for the bend"),
        (right, "silhouette", "Silhouette score", "Higher = clearer separation"),
    ):
        ax.plot(ks, k_scores[col], color=theme.PRIMARY, linewidth=2, marker="o",
                markersize=6)
        ax.plot([chosen_k], [k_scores.loc[chosen_k, col]], marker="o", markersize=11,
                markerfacecolor="none", markeredgecolor=theme.MAGENTA,
                markeredgewidth=2)
        ax.set_xticks(ks)
        ax.set_xlabel("Number of segments (K)")
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
        _subtitle(ax, note)
    right.annotate(f"chosen K = {chosen_k}", (chosen_k, k_scores.loc[chosen_k, "silhouette"]),
                   xytext=(8, -16), textcoords="offset points", fontsize=9,
                   color=theme.TEXT_SECONDARY)
    fig.tight_layout()
    return fig


def plot_customers_vs_revenue(named: pd.DataFrame):
    """Each segment's share of customers next to its share of monthly revenue."""
    table = named.sort_values("share_of_revenue")
    y = np.arange(len(table))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(table) + 1.8))
    series = (("share_of_customers", "Share of customers", theme.VIOLET, h / 2),
              ("share_of_revenue", "Share of monthly revenue", theme.MAGENTA, -h / 2))
    for col, label, colour, offset in series:
        bars = ax.barh(y + offset, table[col], height=h - 0.04, color=colour, label=label)
        for bar, value in zip(bars, table[col]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.0%}",
                    va="center", fontsize=8.5, color=theme.TEXT)
    ax.set_yticks(y, table["segment_name"])
    _percent_axis(ax, "x")
    ax.set_xlim(0, table[["share_of_customers", "share_of_revenue"]].to_numpy().max() * 1.25)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Who the segments are: size vs revenue")
    _subtitle(ax, "A segment whose revenue share beats its customer share is worth more "
                  "per customer")
    return fig


def plot_segment_map(df: pd.DataFrame, labels: pd.Series, names: dict[int, str],
                     sample: int = 2500, seed: int = 42):
    """Tenure vs monthly charges, coloured by segment (a random sample of customers)."""
    data = pd.DataFrame({"tenure": df["tenure"].astype(float).values,
                         "monthly": df["MonthlyCharges"].astype(float).values,
                         "cluster": labels.values})
    if len(data) > sample:
        data = data.sample(sample, random_state=seed)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for cluster in sorted(names):
        subset = data[data["cluster"] == cluster]
        ax.scatter(subset["tenure"], subset["monthly"], s=14, alpha=0.7,
                   color=theme.CATEGORICAL[cluster % len(theme.CATEGORICAL)],
                   edgecolors=theme.SURFACE, linewidths=0.4, label=names[cluster])
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly charges (USD)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, markerscale=1.6)
    ax.set_title("Where each segment sits")
    _subtitle(ax, f"Each dot is a customer (random sample of {len(data):,}); the number "
                  "of services also shapes the groups but is not shown on this map")
    return fig


def plot_segment_profile(named: pd.DataFrame):
    """Heatmap of segment averages; colour shows high vs low within each column."""
    columns = {
        "avg_tenure": ("Tenure (mo)", "{:.0f}"),
        "avg_monthly_charges": ("Monthly $", "{:.0f}"),
        "avg_services": ("Services", "{:.1f}"),
        "share_month_to_month": ("Month-to-month", "{:.0%}"),
        "share_fiber": ("Fiber optic", "{:.0%}"),
        "churn_rate": ("Churn rate", "{:.0%}"),
    }
    values = named[list(columns)].astype(float)
    spread = (values.max() - values.min()).replace(0, 1)
    scaled = (values - values.min()) / spread
    cmap = LinearSegmentedColormap.from_list("c360_seq", theme.SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(named) + 1.8))
    ax.imshow(scaled.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), [label for label, _ in columns.values()])
    ax.set_yticks(range(len(named)), named["segment_name"])
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(named)):
        for j, (col, (_, fmt)) in enumerate(columns.items()):
            ax.text(j, i, fmt.format(values.iat[i, j]), ha="center", va="center", fontsize=9.5,
                    color=theme.PLUM_BG if scaled.iat[i, j] > 0.75 else theme.TEXT)
    ax.set_title("Segment profiles at a glance", pad=30)
    ax.text(0, -0.04, "Darker = higher within that column, so compare down each column",
            transform=ax.transAxes, fontsize=9, color=theme.TEXT_MUTED, va="top")
    return fig
