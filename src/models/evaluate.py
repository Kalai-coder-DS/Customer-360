"""Model evaluation: metrics, confusion matrix, curves and calibration.

Why so many metrics? With 26.5% churners, a model that says "nobody churns"
scores 73.5% accuracy and is useless. So we report:

* **Precision** - of the customers we flag as churners, how many really leave?
  (low precision = wasted retention offers)
* **Recall** - of the customers who really leave, how many did we flag?
  (low recall = churners we miss entirely)
* **F1** - one number that balances precision and recall
* **ROC-AUC** - how well the model ranks churners above non-churners across all
  thresholds (0.5 = random guessing, 1.0 = perfect)
* **PR-AUC** - like ROC-AUC but focused on the churners; more informative when
  the positive class is the minority (random guessing = the churn rate)
* **Brier score** - how close predicted probabilities are to what happened
  (0 = perfect; lower is better). Checks that "70% risk" really means about 70%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import settings

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """All metrics for one set of predictions."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, proba)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def confusion_table(y_true, proba, threshold: float = 0.5) -> pd.DataFrame:
    """2x2 table: rows = what happened, columns = what the model predicted."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, pred, labels=[0, 1])
    return pd.DataFrame(matrix, index=["Actually stayed", "Actually churned"],
                        columns=["Predicted stay", "Predicted churn"])


def best_f1_threshold(y_true, proba, grid=np.arange(0.10, 0.91, 0.01)) -> float:
    """Decision threshold that maximises F1. Must be run on training-fold
    (out-of-fold) predictions, never on the test set."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def risk_level(proba, thresholds: dict[str, float] = settings.RISK_THRESHOLDS) -> np.ndarray:
    """Turn probabilities into Low / Medium / High risk labels."""
    proba = np.asarray(proba, dtype=float)
    return np.where(proba >= thresholds["High"], "High",
                    np.where(proba >= thresholds["Medium"], "Medium", "Low"))


def risk_level_table(y_true, proba) -> pd.DataFrame:
    """Customers per risk level and how many of them actually churned."""
    data = pd.DataFrame({"risk": risk_level(proba), "churned": np.asarray(y_true)})
    table = data.groupby("risk")["churned"].agg(customers="size", actual_churn_rate="mean")
    table = table.reindex(["Low", "Medium", "High"]).fillna(0)
    table["customers"] = table["customers"].astype(int)
    table["share"] = table["customers"] / max(table["customers"].sum(), 1)
    return table


def curve_data(y_true, proba) -> dict[str, pd.DataFrame]:
    """ROC, precision-recall and calibration curves for charts."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    return {
        "roc": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "pr": pd.DataFrame({"recall": recall, "precision": precision}),
        "calibration": pd.DataFrame({"mean_predicted": mean_pred, "observed": frac_pos}),
    }
