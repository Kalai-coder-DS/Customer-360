"""Feature engineering and preprocessing for churn prediction.

Two kinds of work happen here:

1. **Engineered features** (``add_engineered_features``): new columns
   calculated from a single customer's own row, so they cannot leak
   information between customers:
   * ``services_count``   - how many services the customer uses
   * ``avg_monthly_spend`` - TotalCharges / tenure (MonthlyCharges if tenure is 0)
   * ``auto_payment``     - 1 if paying by automatic bank transfer or card

2. **Preprocessing** (``build_preprocessor``): steps that *learn* from data
   (medians for gaps, scaling, the list of categories). They live inside a
   scikit-learn Pipeline, so during cross-validation and testing they are
   learned from training rows only. This is how the project prevents data
   leakage.

Columns that are never used as inputs: the customer ID (no meaning), the
target itself (``Churn`` / ``churn_flag``) and the Phase 3 segment labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.analysis.eda import services_count
from src.data.cleaning import CHURN_FLAG
from src.data.schema import CATEGORICAL_VALUES, ID_COLUMN, TARGET_COLUMN

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                    "services_count", "avg_monthly_spend", "auto_payment"]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_VALUES if c != TARGET_COLUMN]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_COLUMNS = [ID_COLUMN, TARGET_COLUMN, CHURN_FLAG]

ENGINEERED_DESCRIPTIONS = {
    "services_count": "Number of services the customer subscribes to (0-9).",
    "avg_monthly_spend": "Average amount actually paid per month so far (TotalCharges / tenure).",
    "auto_payment": "1 if the customer pays by automatic bank transfer or credit card.",
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the engineered columns added (row-level only)."""
    out = df.copy()
    tenure = out["tenure"].astype(float)
    out["services_count"] = services_count(out).astype(float)
    out["avg_monthly_spend"] = np.where(
        tenure > 0, out["TotalCharges"].astype(float) / tenure.where(tenure > 0, 1),
        out["MonthlyCharges"].astype(float),
    )
    out["auto_payment"] = out["PaymentMethod"].astype(str).str.contains("automatic").astype(float)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(float)
    return out


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a clean DataFrame into model inputs (X) and target (y = 1 churned)."""
    data = add_engineered_features(df)
    X = data[MODEL_FEATURES].copy()
    y = (data[TARGET_COLUMN] == "Yes").astype(int).rename("churn")
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Learnable preprocessing: fill gaps, scale numbers, one-hot encode categories."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)
