"""Charts for Phase 5 explainable AI (Matplotlib, shared theme).

Colour meaning is fixed across the project: coral = pushes churn risk up,
violet = pushes it down. Every bar is also labelled, so colour is never the
only cue.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from config import theme
from src.explain.explainer import FEATURE_LABELS, readable_value, sigmoid
from src.visualization.eda_charts import _subtitle

UP, DOWN = theme.STATUS["churned"], theme.STATUS["retained"]


def plot_global_importance(table: pd.DataFrame, top: int = 12):
    data = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(data) + 1.8))
    ax.barh(data["label"], data["importance"], xerr=data["spread"], height=0.6,
            color=theme.PRIMARY, ecolor=theme.TEXT_SECONDARY, capsize=3)
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(max(row["importance"] + row["spread"], 0) + 0.002, i,
                f" {row['importance']:.3f}", va="center", fontsize=9)
    ax.axvline(0, color=theme.BASELINE, linewidth=1)
    ax.set_xlim(min(0, data["importance"].min()) - 0.005,
                (data["importance"] + data["spread"]).max() * 1.25 + 0.005)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("Which features the model relies on most")
    _subtitle(ax, "Permutation importance · bigger drop = more reliance · "
                  "whisker = spread over 5 shuffles · not proof of cause")
    return fig


def plot_customer_explanation(contrib: pd.Series, row: pd.Series, probability: float,
                              risk: str, base: float, customer_id: str, top: int = 8):
    """Waterfall-style bars: each factor's push from an average customer."""
    chosen = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)
    chosen = chosen.iloc[::-1]
    labels = [readable_value(f, row[f]) for f in chosen.index]
    colours = [UP if v > 0 else DOWN for v in chosen]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(chosen) + 2))
    bars = ax.barh(labels, chosen.values, color=colours, height=0.6)
    span = max(chosen.abs().max(), 0.1)
    for bar, value in zip(bars, chosen.values):
        word = "raises" if value > 0 else "lowers"
        x = bar.get_width()
        ax.text(x + (0.02 * span if value >= 0 else -0.02 * span),
                bar.get_y() + bar.get_height() / 2, f"{word} {abs(value):.2f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=8.5,
                color=theme.TEXT)
    ax.axvline(0, color=theme.TEXT_SECONDARY, linewidth=1)
    ax.set_xlim(-span * 1.6, span * 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Push on churn risk (log-odds) · coral raises, violet lowers")
    ax.set_title(f"Why customer {customer_id}: {probability:.0%} churn risk ({risk})")
    _subtitle(ax, f"Starting point: an average customer at {sigmoid(base):.0%} · "
                  f"top {len(chosen)} factors shown")
    return fig


def plot_feature_direction(direction: pd.DataFrame, feature: str):
    """Average push on churn risk for each value of one feature."""
    data = direction.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(data) + 1.8))
    colours = [UP if v > 0 else DOWN for v in data["avg_contribution"]]
    bars = ax.barh(data.index, data["avg_contribution"], color=colours, height=0.6)
    span = max(data["avg_contribution"].abs().max(), 0.1)
    for bar, (_, r) in zip(bars, data.iterrows()):
        v = r["avg_contribution"]
        ax.text(v + (0.03 * span if v >= 0 else -0.03 * span),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}  (n={int(r['customers']):,})", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color=theme.TEXT_SECONDARY, linewidth=1)
    ax.set_xlim(-span * 1.7, span * 1.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average push on churn risk (log-odds)")
    ax.set_title(f"How {FEATURE_LABELS.get(feature, feature)} moves the prediction")
    _subtitle(ax, "Coral raises, violet lowers, compared with an average customer")
    return fig
