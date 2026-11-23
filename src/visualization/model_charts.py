"""Charts for Phase 4 churn prediction (Matplotlib, shared theme)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import theme
from src.models.train import MODEL_LABELS
from src.visualization.eda_charts import _percent_axis, _subtitle

# Fixed colour per model (colour follows the model, never its rank).
MODEL_COLOURS = theme.MODEL_COLOURS
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1",
                 "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


def plot_model_comparison(test_metrics: pd.DataFrame):
    """Test-set metrics for each model at its tuned threshold."""
    tuned = test_metrics.xs("tuned", level="threshold_type")
    models = list(tuned.index)
    metrics = list(METRIC_LABELS)
    y = np.arange(len(metrics))
    h = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, model in enumerate(models):
        offset = i * h - (len(models) - 1) / 2 * h
        values = tuned.loc[model, metrics].astype(float).to_numpy()
        bars = ax.barh(y + offset, values, height=h - 0.03, color=MODEL_COLOURS[model],
                       label=MODEL_LABELS[model])
        for bar, value in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.2f}",
                    va="center", fontsize=8.5, color=theme.TEXT)
    ax.set_yticks(y, [METRIC_LABELS[m] for m in metrics])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title("How the models compare on unseen customers")
    _subtitle(ax, "Held-out test set (20% of customers) · each model at its tuned threshold "
                  "· higher is better")
    return fig


def plot_cv_scores(cv_scores: pd.DataFrame):
    """Cross-validated ROC-AUC (mean ± spread across the 5 folds)."""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    names = list(cv_scores.index)
    means = cv_scores["roc_auc_mean"].to_numpy()
    stds = cv_scores["roc_auc_std"].to_numpy()
    ypos = np.arange(len(names))
    ax.barh(ypos, means, xerr=stds, height=0.55, capsize=4,
            color=[MODEL_COLOURS[n] for n in names], ecolor=theme.TEXT_SECONDARY)
    for yp, m, s in zip(ypos, means, stds):
        ax.text(m + s + 0.01, yp, f"{m:.3f} ± {s:.3f}", va="center", fontsize=9)
    ax.set_yticks(ypos, [MODEL_LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC-AUC (0.5 = random guessing, 1.0 = perfect)")
    ax.set_title("Cross-validation on the training data")
    _subtitle(ax, "5 folds: train on 4, score on the 5th, repeat · bar = average, "
                  "whisker = spread")
    return fig


def plot_roc_curves(curves: dict, test_metrics: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color=theme.TEXT_MUTED, linewidth=1,
            label="Random guessing (0.50)")
    for name, data in curves.items():
        auc = test_metrics.loc[(name, "tuned"), "roc_auc"]
        ax.plot(data["roc"]["fpr"], data["roc"]["tpr"], color=MODEL_COLOURS[name],
                linewidth=2, label=f"{MODEL_LABELS[name]} ({auc:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False alarms: share of stayers wrongly flagged")
    ax.set_ylabel("Churners caught: share of churners flagged")
    ax.legend(loc="lower right")
    ax.set_title("ROC curve: catching churners vs false alarms")
    _subtitle(ax, "Closer to the top-left corner is better · number = ROC-AUC")
    return fig


def plot_pr_curves(curves: dict, test_metrics: pd.DataFrame, churn_rate: float):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.axhline(churn_rate, linestyle="--", color=theme.TEXT_MUTED, linewidth=1,
               label=f"Random guessing ({churn_rate:.2f})")
    for name, data in curves.items():
        auc = test_metrics.loc[(name, "tuned"), "pr_auc"]
        ax.plot(data["pr"]["recall"], data["pr"]["precision"], color=MODEL_COLOURS[name],
                linewidth=2, label=f"{MODEL_LABELS[name]} ({auc:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall: share of churners caught")
    ax.set_ylabel("Precision: share of flagged who churn")
    ax.legend(loc="upper right")
    ax.set_title("Precision-recall trade-off")
    _subtitle(ax, "Catching more churners usually means more false alarms · number = PR-AUC")
    return fig


def plot_confusion(confusion: pd.DataFrame, model_label: str, threshold: float):
    values = confusion.to_numpy()
    shares = values / values.sum(axis=1, keepdims=True)
    notes = np.array([["Correctly left alone", "False alarm"],
                      ["Missed churner", "Correctly flagged"]])
    good = np.array([[1, 0], [0, 1]])
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    colours = np.where(good == 1, 1.0, 0.25)
    ax.imshow(colours, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "cm", [theme.CARD_RAISED, theme.PRIMARY]), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ink = theme.TEXT
            ax.text(j, i - 0.12, f"{values[i, j]:,}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=ink)
            ax.text(j, i + 0.16, f"{notes[i, j]}\n{shares[i, j]:.0%} of row", ha="center",
                    va="center", fontsize=9, color=ink)
    ax.set_xticks([0, 1], confusion.columns)
    ax.set_yticks([0, 1], confusion.index)
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"What the {model_label} got right and wrong", pad=30)
    ax.text(0, -0.06, f"Test set · decision threshold {threshold:.2f} "
                      "(customers at or above it are flagged)",
            transform=ax.transAxes, fontsize=9, color=theme.TEXT_MUTED, va="top")
    return fig


def plot_calibration(calibration: pd.DataFrame, model_label: str):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color=theme.TEXT_MUTED, linewidth=1,
            label="Perfectly honest probabilities")
    ax.plot(calibration["mean_predicted"], calibration["observed"], marker="o",
            markersize=7, linewidth=2, color=theme.PRIMARY, label=model_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _percent_axis(ax, "x")
    _percent_axis(ax, "y")
    ax.set_xlabel("Predicted churn probability (group average)")
    ax.set_ylabel("Share that actually churned")
    ax.legend(loc="upper left")
    ax.set_title("Can the probabilities be trusted?")
    _subtitle(ax, "Customers grouped into 10 equal bins · points near the line mean "
                  "'70% risk' really is about 70%")
    return fig


def plot_risk_levels(risk_table: pd.DataFrame):
    from config.settings import RISK_THRESHOLDS
    fig, ax = plt.subplots(figsize=(8, 3))
    names = list(risk_table.index)
    bars = ax.barh(names, risk_table["customers"], height=0.55,
                   color=[theme.RISK[n] for n in names])
    for bar, (_, row) in zip(bars, risk_table.iterrows()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f"  {int(row['customers']):,} customers · {row['actual_churn_rate']:.0%} "
                "actually churned", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(risk_table["customers"].max(), 1) * 1.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Customers in the test set")
    ax.set_title("Risk levels vs what actually happened")
    _subtitle(ax, f"Low < {RISK_THRESHOLDS['Medium']:.0%} ≤ Medium < "
                  f"{RISK_THRESHOLDS['High']:.0%} ≤ High predicted churn probability")
    return fig
