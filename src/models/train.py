"""Train, compare and select churn prediction models.

The process, step by step:

1. **Split**: 80% of customers for training, 20% held back as a final test.
   The split is *stratified*, so both parts keep the same 26.5% churn rate.
2. **Cross-validate** on the training part only: the training data is cut
   into 5 folds; each model trains on 4 and is scored on the 5th, five times.
   This gives a stable estimate (mean ± spread) without touching the test set.
3. **Compare** three models:
   * Baseline: always predicts the average churn rate (the score to beat)
   * Logistic Regression: a simple, transparent linear model
   * Random Forest: many decision trees voting; captures non-linear patterns
4. **Select** the best model by cross-validated ROC-AUC (never by test score,
   which would quietly turn the test set into training data).
5. **Choose a decision threshold** that balances precision and recall (best F1),
   using out-of-fold predictions on the training data.
6. **Test once** on the held-back 20% for an honest final score.
7. **Refit** the chosen model on all customers and score everyone for the
   dashboard. Their scores come from a model that has seen them, so the
   honest performance figures are the test-set ones.

Class imbalance is handled with the threshold (step 5) rather than by
re-weighting classes, so predicted probabilities stay meaningful:
"70% risk" means roughly 7 in 10 similar customers churned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from config import settings
from src.data.schema import ID_COLUMN
from src.models import evaluate
from src.models.features import build_preprocessor, prepare_xy
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_LABELS = {
    "baseline": "Baseline (average rate)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
CV_SCORING = {
    "roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1",
    "recall": "recall",
    # zero_division=0: the baseline never predicts churn, so its precision is 0, not an error
    "precision": make_scorer(precision_score, zero_division=0),
    "accuracy": "accuracy",
}


def make_models() -> dict[str, Pipeline]:
    seed = settings.RANDOM_STATE
    estimators = {
        "baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt",
            n_jobs=-1, random_state=seed),
    }
    return {name: Pipeline([("prep", build_preprocessor()), ("model", est)])
            for name, est in estimators.items()}


@dataclass
class TrainingResult:
    chosen: str
    threshold: float
    cv_scores: pd.DataFrame            # mean and std per model and metric
    test_metrics: pd.DataFrame         # per model, at 0.5 and at its tuned threshold
    thresholds: dict[str, float]
    confusion: pd.DataFrame            # chosen model, tuned threshold, test set
    risk_table: pd.DataFrame           # chosen model, test set
    curves: dict[str, dict[str, pd.DataFrame]]
    overfit_check: dict[str, float]
    final_pipeline: Pipeline
    predictions: pd.DataFrame          # every customer
    split: dict[str, int] = field(default_factory=dict)


def train_and_evaluate(df: pd.DataFrame) -> TrainingResult:
    X, y = prepare_xy(df)
    ids = df[ID_COLUMN].reset_index(drop=True)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)

    # 1. Stratified train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=settings.TEST_SIZE, stratify=y,
        random_state=settings.RANDOM_STATE)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    logger.info("Split: %s training / %s test customers", len(train_idx), len(test_idx))

    cv = StratifiedKFold(n_splits=settings.CV_FOLDS, shuffle=True,
                         random_state=settings.RANDOM_STATE)
    models = make_models()

    # 2. Cross-validation on the training set
    cv_rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=1)
        row = {"model": name}
        for metric in CV_SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        cv_rows.append(row)
        logger.info("CV %-20s ROC-AUC %.3f ± %.3f", name, row["roc_auc_mean"],
                    row["roc_auc_std"])
    cv_scores = pd.DataFrame(cv_rows).set_index("model")

    # 3-4. Choose the best real model by cross-validated ROC-AUC
    candidates = cv_scores.drop(index="baseline")
    chosen = str(candidates["roc_auc_mean"].idxmax())

    # 5. Tuned thresholds from out-of-fold predictions (training data only)
    thresholds = {"baseline": 0.5}
    for name in ("logistic_regression", "random_forest"):
        oof = cross_val_predict(models[name], X_train, y_train, cv=cv,
                                method="predict_proba")[:, 1]
        thresholds[name] = evaluate.best_f1_threshold(y_train, oof)

    # 6. Fit on the full training set and score the untouched test set
    test_rows, curves, test_proba = [], {}, {}
    for name, pipe in models.items():
        fitted = clone(pipe).fit(X_train, y_train)
        proba = fitted.predict_proba(X_test)[:, 1]
        test_proba[name] = proba
        for label, thr in (("default 0.50", 0.5), ("tuned", thresholds[name])):
            metrics = evaluate.classification_metrics(y_test, proba, thr)
            test_rows.append({"model": name, "threshold_type": label, "threshold": thr,
                              **metrics})
        if name != "baseline":
            curves[name] = evaluate.curve_data(y_test, proba)
        if name == chosen:
            train_auc = evaluate.classification_metrics(
                y_train, fitted.predict_proba(X_train)[:, 1])["roc_auc"]
    test_metrics = pd.DataFrame(test_rows).set_index(["model", "threshold_type"])

    threshold = thresholds[chosen]
    confusion = evaluate.confusion_table(y_test, test_proba[chosen], threshold)
    risk_table = evaluate.risk_level_table(y_test, test_proba[chosen])
    test_auc = float(test_metrics.loc[(chosen, "tuned"), "roc_auc"])
    overfit_check = {"train_roc_auc": float(train_auc), "test_roc_auc": test_auc,
                     "gap": float(train_auc - test_auc)}

    # 7. Refit on all customers and score everyone
    final_pipeline = clone(models[chosen]).fit(X, y)
    all_proba = final_pipeline.predict_proba(X)[:, 1]
    in_test = np.zeros(len(X), dtype=bool)
    in_test[test_idx] = True
    predictions = pd.DataFrame({
        ID_COLUMN: ids,
        "churn_probability": np.round(all_proba, 4),
        "risk_level": evaluate.risk_level(all_proba),
        "predicted_churn": (all_proba >= threshold).astype(int),
        "actual_churn": y.values,
        "in_test_set": in_test.astype(int),
    })

    return TrainingResult(
        chosen=chosen, threshold=threshold, cv_scores=cv_scores, test_metrics=test_metrics,
        thresholds=thresholds, confusion=confusion, risk_table=risk_table, curves=curves,
        overfit_check=overfit_check, final_pipeline=final_pipeline, predictions=predictions,
        split={"train": len(train_idx), "test": len(test_idx),
               "train_churn_rate": float(y_train.mean()),
               "test_churn_rate": float(y_test.mean())},
    )
