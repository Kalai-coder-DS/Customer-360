"""Explainable AI: why does the model give a customer their churn score?

The project computes **Shapley values** - the same idea the popular SHAP
library is built on - without adding the SHAP package (which is heavy and
often lags new Python versions). Two exact-or-faithful methods are used:

* **Logistic Regression -> exact linear Shapley values.** For a linear model
  each feature's contribution is simply
      coefficient x (customer's value - average customer's value)
  in log-odds units. This is mathematically identical to SHAP's
  LinearExplainer and is instant for every customer.

* **Random Forest (or any model) -> sampling Shapley values.** Features are
  switched one by one from an "average customer" (a random background
  customer) to this customer's real values, in many random orders. The change
  in the model's output at each switch, averaged, is the feature's
  contribution. More permutations = more precise.

Both work in **log-odds** (the scale where contributions add up exactly):
    base value + sum of contributions = log-odds of this customer's prediction
and the probability is then 1 / (1 + e^-log-odds).

Contributions are reported per *original* feature (e.g. "Contract"), with
one-hot columns added back together, so explanations read naturally.

Important: a contribution shows what the *model* associates with churn,
learned from historical data. It is not proof that changing that feature
would change the customer's behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import settings
from src.models.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES

EPS = 1e-6

FEATURE_LABELS = {
    "tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "SeniorCitizen": "Senior citizen", "services_count": "Number of services",
    "avg_monthly_spend": "Average monthly spend", "auto_payment": "Automatic payment",
    "gender": "Gender", "Partner": "Has partner", "Dependents": "Has dependents",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies", "Contract": "Contract",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
}


# --- Helpers ---------------------------------------------------------------

def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def readable_value(feature: str, value) -> str:
    """'Contract: Month-to-month', 'Tenure: 2 months', 'Monthly charges: $95.10'."""
    label = FEATURE_LABELS.get(feature, feature)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{label}: unknown"
    if feature == "tenure":
        return f"{label}: {float(value):.0f} months"
    if feature in ("MonthlyCharges", "avg_monthly_spend"):
        return f"{label}: ${float(value):,.2f}/month"
    if feature == "TotalCharges":
        return f"{label}: ${float(value):,.2f}"
    if feature == "services_count":
        return f"{label}: {float(value):.0f}"
    if feature in ("SeniorCitizen", "auto_payment"):
        return f"{label}: {'Yes' if float(value) >= 0.5 else 'No'}"
    return f"{label}: {value}"


def strength(contribution: float) -> str:
    size = abs(contribution)
    if size >= 1.0:
        return "strong"
    if size >= 0.4:
        return "moderate"
    if size >= 0.1:
        return "slight"
    return "minimal"


def encoded_to_original(pipeline: Pipeline) -> list[str]:
    """Map every column produced by the preprocessor back to its original feature."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    by_length = sorted(CATEGORICAL_FEATURES, key=len, reverse=True)
    mapping = []
    for name in names:
        if name in NUMERIC_FEATURES:
            mapping.append(name)
            continue
        match = next((f for f in by_length if name.startswith(f + "_")), None)
        if match is None:
            raise ValueError(f"Cannot map encoded column {name!r} to an original feature")
        mapping.append(match)
    return mapping


def is_linear(pipeline: Pipeline) -> bool:
    return isinstance(pipeline.named_steps["model"], LogisticRegression)


# --- Local explanations ----------------------------------------------------

def linear_contributions(pipeline: Pipeline, X: pd.DataFrame,
                         background: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact Shapley values for Logistic Regression (log-odds), per original feature."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    coef = model.coef_.ravel()
    mean_t = prep.transform(background[MODEL_FEATURES]).mean(axis=0)
    encoded = (prep.transform(X[MODEL_FEATURES]) - mean_t) * coef
    base = float(model.intercept_[0] + coef @ mean_t)
    groups = encoded_to_original(pipeline)
    frame = pd.DataFrame(encoded, columns=groups, index=X.index)
    grouped = frame.T.groupby(level=0, sort=False).sum().T
    return grouped[MODEL_FEATURES], base


def sampling_contributions(pipeline: Pipeline, x_row: pd.Series, background: pd.DataFrame,
                           n_permutations: int | None = None,
                           seed: int = settings.RANDOM_STATE) -> tuple[pd.Series, float]:
    """Model-agnostic Shapley values for one customer (log-odds), by permutation sampling.

    For each random order of features, start from a random background customer
    and switch features to this customer's values one at a time, recording how
    the model's log-odds change at each switch.
    """
    n_permutations = n_permutations or settings.SHAPLEY_PERMUTATIONS
    rng = np.random.default_rng(seed)
    features = MODEL_FEATURES
    n_feat = len(features)
    x = x_row[features]
    rows, orders = [], []
    for _ in range(n_permutations):
        order = rng.permutation(n_feat)
        current = background.iloc[rng.integers(len(background))][features].copy()
        rows.append(current.copy())
        for j in order:
            current.iloc[j] = x.iloc[j]
            rows.append(current.copy())
        orders.append(order)
    batch = pd.DataFrame(rows, columns=features)
    for col in NUMERIC_FEATURES:
        batch[col] = batch[col].astype(float)
    out = logit(pipeline.predict_proba(batch)[:, 1]).reshape(n_permutations, n_feat + 1)
    phi = np.zeros(n_feat)
    for p, order in enumerate(orders):
        phi[order] += np.diff(out[p])
    phi /= n_permutations
    base = float(out[:, 0].mean())
    return pd.Series(phi, index=features), base


def explain_customers(pipeline: Pipeline, X: pd.DataFrame, background: pd.DataFrame,
                      n_permutations: int | None = None
                      ) -> tuple[pd.DataFrame, pd.Series, str]:
    """Contributions for every row of X. Returns (contributions, base per row, method)."""
    n_permutations = n_permutations or settings.SHAPLEY_PERMUTATIONS
    if is_linear(pipeline):
        contrib, base = linear_contributions(pipeline, X, background)
        return contrib, pd.Series(base, index=X.index), "exact linear Shapley values"
    results, bases = [], []
    for i, (idx, row) in enumerate(X.iterrows()):
        phi, base = sampling_contributions(pipeline, row, background, n_permutations,
                                           seed=settings.RANDOM_STATE + i)
        results.append(phi.rename(idx))
        bases.append(base)
    contrib = pd.DataFrame(results)
    return contrib, pd.Series(bases, index=X.index), (
        f"sampling Shapley values ({n_permutations} permutations)")


# --- Turning numbers into words --------------------------------------------

def top_factors(contributions: pd.Series, row: pd.Series, n: int = 3
                ) -> tuple[list[dict], list[dict]]:
    """Largest risk-raising and risk-lowering factors (ignoring minimal ones)."""
    def factor(feature):
        value = contributions[feature]
        return {"feature": feature, "label": readable_value(feature, row[feature]),
                "contribution": round(float(value), 4), "strength": strength(value)}

    ordered = contributions.sort_values()
    raising = [factor(f) for f in ordered.index[::-1]
               if contributions[f] > 0 and strength(contributions[f]) != "minimal"][:n]
    lowering = [factor(f) for f in ordered.index
                if contributions[f] < 0 and strength(contributions[f]) != "minimal"][:n]
    return raising, lowering


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def explanation_text(probability: float, risk: str, model_label: str,
                     raising: list[dict], lowering: list[dict]) -> str:
    """Plain-English explanation that only names factors the model actually used."""
    text = (f"This customer has an estimated {probability:.0%} probability of churning "
            f"according to the trained {model_label} model, which places them in the "
            f"{risk} risk level. ")
    if raising:
        text += ("The factors that raise this estimate the most are "
                 + _join([f"{f['label']} ({f['strength']})" for f in raising]) + ". ")
    else:
        text += "No single factor noticeably raises this customer's risk. "
    if lowering:
        text += ("Factors that lower it: "
                 + _join([f"{f['label']} ({f['strength']})" for f in lowering]) + ". ")
    text += ("These are patterns the model learned from past customers; they show "
             "association, not proven cause.")
    return text


def contributions_json(contributions: pd.Series) -> str:
    return json.dumps({k: round(float(v), 4) for k, v in contributions.items()})


# --- Global explanations ---------------------------------------------------

def global_importance(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                      n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance: how much ROC-AUC drops when one feature is shuffled.

    Shuffling breaks the link between that feature and churn while keeping
    everything else, so a big drop means the model relies on the feature.
    Works for any model. Measured on the deployed model, so it describes what
    the model relies on, not proof of cause.
    """
    result = permutation_importance(pipeline, X[MODEL_FEATURES], y, scoring="roc_auc",
                                    n_repeats=n_repeats, random_state=settings.RANDOM_STATE,
                                    n_jobs=1)
    table = pd.DataFrame({"importance": result.importances_mean,
                          "spread": result.importances_std}, index=MODEL_FEATURES)
    table["label"] = [FEATURE_LABELS.get(f, f) for f in table.index]
    return table.sort_values("importance", ascending=False)


def mean_abs_contribution(contributions: pd.DataFrame) -> pd.Series:
    """Average size of each feature's contribution across customers."""
    return contributions.abs().mean().sort_values(ascending=False)


def contribution_direction(contributions: pd.DataFrame, X: pd.DataFrame,
                           feature: str) -> pd.DataFrame:
    """Average contribution for each value (or value band) of one feature."""
    values = X[feature]
    if feature in NUMERIC_FEATURES and values.nunique() > 8:
        values = pd.qcut(values.astype(float), q=5, duplicates="drop").astype(str)
    data = pd.DataFrame({"value": values.astype(str).values,
                         "contribution": contributions[feature].values})
    return (data.groupby("value")["contribution"].agg(["mean", "size"])
            .rename(columns={"mean": "avg_contribution", "size": "customers"})
            .sort_values("avg_contribution", ascending=False))


def logistic_coefficients(pipeline: Pipeline) -> pd.DataFrame | None:
    """Coefficients and odds ratios for Logistic Regression (None for other models).

    Numbers are standardised, so a numeric coefficient is the effect of one
    standard deviation. For categories, compare values within the same feature.
    """
    if not is_linear(pipeline):
        return None
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coef = pipeline.named_steps["model"].coef_.ravel()
    table = pd.DataFrame({"encoded_feature": names, "feature": encoded_to_original(pipeline),
                          "coefficient": coef, "odds_ratio": np.exp(coef)})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index)
