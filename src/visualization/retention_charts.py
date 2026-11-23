"""Charts for Phase 6 retention intelligence (Matplotlib, shared theme)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import settings, theme
from src.retention.planner import TIER_ORDER
from src.visualization.eda_charts import _percent_axis, _subtitle

TIER_COLOURS = {
    TIER_ORDER[0]: theme.RISK["High"],
    TIER_ORDER[1]: theme.RISK["Medium"],
    TIER_ORDER[2]: theme.PRIMARY,
    TIER_ORDER[3]: theme.TEXT_MUTED,
}


def _money(value: float) -> str:
    return f"${value:,.0f}"


def plot_risk_value_matrix(plan: pd.DataFrame, value_cutoff: float, sample: int = 3000,
                           seed: int = 42):
    """Churn probability vs monthly charges, coloured by priority tier."""
    data = plan.sample(min(sample, len(plan)), random_state=seed)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for tier in reversed(TIER_ORDER):
        part = data[data["priority_tier"] == tier]
        ax.scatter(part["churn_probability"], part["MonthlyCharges"], s=12, alpha=0.75,
                   color=TIER_COLOURS[tier], edgecolors=theme.SURFACE, linewidths=0.3,
                   label=f"{tier} ({int((plan['priority_tier'] == tier).sum()):,})")
    for thr in settings.RISK_THRESHOLDS.values():
        ax.axvline(thr, color=theme.TEXT_SECONDARY, linestyle="--", linewidth=0.9)
    ax.axhline(value_cutoff, color=theme.TEXT_SECONDARY, linestyle=":", linewidth=0.9)
    ax.text(1.0, value_cutoff, f" high value ≥ ${value_cutoff:.0f}", fontsize=8,
            color=theme.TEXT_SECONDARY, va="bottom", ha="right")
    ax.set_xlim(0, 1)
    _percent_axis(ax, "x")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Monthly charges (USD)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, markerscale=1.8)
    ax.set_title("Who to contact first: risk vs value")
    _subtitle(ax, "Dashed lines = risk-level cut-offs · dotted line = high-value cut-off · "
                  "each dot is a customer")
    return fig


def plot_expected_loss_by(summary: pd.DataFrame, label: str):
    data = summary.sort_values("expected_loss_horizon")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(data) + 1.8))
    bars = ax.barh(data.index.astype(str), data["expected_loss_horizon"], color=theme.AMBER,
                   height=0.6)
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {_money(row['expected_loss_horizon'])} ({row['share_of_expected_loss']:.0%})"
                f" · {int(row['customers']):,} customers", va="center", fontsize=8.5)
    ax.set_xlim(0, data["expected_loss_horizon"].max() * 1.7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _money(v)))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Expected revenue at risk over {settings.RETENTION_HORIZON_MONTHS} months")
    ax.set_title(f"Where the revenue at risk sits, by {label}")
    _subtitle(ax, "Sum of churn probability × monthly charges × horizon · an estimate, "
                  "not a forecast")
    return fig


def plot_action_counts(counts: pd.Series):
    data = counts.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1.8))
    bars = ax.barh(data.index, data.values, color=theme.PRIMARY, height=0.6)
    for bar, value in zip(bars, data.values):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {value:,}",
                va="center", fontsize=9)
    ax.set_xlim(0, data.max() * 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Customers recommended this action (first or second choice)")
    ax.set_title("Recommended retention actions")
    _subtitle(ax, "Actions are matched to the factors that raise each customer's risk")
    return fig


def plot_impact_scenarios(scenarios: pd.DataFrame):
    """Net value (revenue protected minus cost) per target group and scenario."""
    targets = list(dict.fromkeys(scenarios.index.get_level_values(0)))
    names = list(settings.SCENARIO_MULTIPLIERS)
    colours = theme.SCENARIO_COLOURS
    y = np.arange(len(targets))
    h = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(targets) + 2.2))
    for i, name in enumerate(names):
        values = [scenarios.loc[(t, name), "net_value"] for t in targets]
        offset = i * h - (len(names) - 1) / 2 * h
        bars = ax.barh(y + offset, values, height=h - 0.03, color=colours[name], label=name)
        for bar, value in zip(bars, values):
            ax.text(value, bar.get_y() + bar.get_height() / 2,
                    f" {_money(value)} " if value >= 0 else f" -{_money(-value)} ",
                    va="center", ha="left" if value >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color=theme.TEXT_SECONDARY, linewidth=1)
    ax.set_yticks(y, targets)
    ax.invert_yaxis()
    span = scenarios["net_value"].abs().max() or 1
    ax.set_xlim(min(0, scenarios["net_value"].min()) - 0.15 * span, span * 1.35)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _money(v)))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_xlabel("Estimated net value (revenue protected − campaign cost)")
    ax.set_title("Estimated campaign impact under different assumptions")
    _subtitle(ax, "ILLUSTRATIVE: based on assumed costs and success rates in the playbook, "
                  "not measured results")
    return fig
