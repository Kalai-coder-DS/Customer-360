"""Shared test fixtures.

The rows below are a tiny HAND-WRITTEN sample that follows the Telco schema.
They are test inputs only, not real customers, and are never used for analysis.
Row 3 has tenure 0 and a blank TotalCharges, mirroring a known quirk of the
real dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import EXPECTED_COLUMNS

_ROWS = [
    ["T-0001", "Female", 0, "Yes", "No", 1, "No", "No phone service", "DSL", "No", "Yes",
     "No", "No", "No", "No", "Month-to-month", "Yes", "Electronic check", 29.85, "29.85", "No"],
    ["T-0002", "Male", 0, "No", "No", 34, "Yes", "No", "DSL", "Yes", "No", "Yes", "No",
     "No", "No", "One year", "No", "Mailed check", 56.95, "1889.5", "No"],
    ["T-0003", "Male", 1, "No", "No", 0, "Yes", "No", "Fiber optic", "No", "No", "No",
     "No", "No", "No", "Two year", "Yes", "Bank transfer (automatic)", 70.70, " ", "No"],
    ["T-0004", "Female", 0, "No", "No", 2, "Yes", "No", "Fiber optic", "No", "No", "No",
     "No", "No", "No", "Month-to-month", "Yes", "Electronic check", 70.70, "151.65", "Yes"],
    ["T-0005", "Female", 1, "Yes", "Yes", 8, "Yes", "Yes", "Fiber optic", "No", "No",
     "Yes", "No", "Yes", "Yes", "Month-to-month", "Yes", "Credit card (automatic)",
     99.65, "820.5", "Yes"],
    ["T-0006", "Male", 0, "No", "Yes", 22, "Yes", "Yes", "No", "No internet service",
     "No internet service", "No internet service", "No internet service",
     "No internet service", "No internet service", "One year", "No", "Mailed check",
     25.10, "540.05", "No"],
]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(_ROWS, columns=EXPECTED_COLUMNS)


def make_random_customers(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Random rows that follow the Telco schema, for TESTS ONLY.

    Values are random draws, so any pattern in them means nothing. They exist
    so statistical and chart code can be exercised on a realistic-sized table.
    """
    import numpy as np

    from src.data.schema import CATEGORICAL_VALUES

    rng = np.random.default_rng(seed)
    data = {"customerID": [f"R-{i:05d}" for i in range(n)]}
    for col, allowed in CATEGORICAL_VALUES.items():
        data[col] = rng.choice(sorted(allowed), size=n)
    data["SeniorCitizen"] = rng.integers(0, 2, size=n)
    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(18, 119, size=n).round(2)
    data["tenure"] = tenure
    data["MonthlyCharges"] = monthly
    data["TotalCharges"] = np.where(tenure == 0, " ", (tenure * monthly).round(2).astype(str))
    return pd.DataFrame(data)[EXPECTED_COLUMNS]


def make_customers_with_signal(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Random customers whose churn follows a KNOWN rule, for TESTS ONLY.

    Month-to-month contracts and short tenure raise the churn chance. Tests use
    this to check that the models really learn a pattern that is present.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    df = make_random_customers(n, seed)
    tenure = df["tenure"].astype(float)
    logit = (-1.5 + 2.0 * (df["Contract"] == "Month-to-month") - 0.04 * tenure
             + 0.6 * (df["InternetService"] == "Fiber optic"))
    prob = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.random(n) < prob, "Yes", "No")
    return df


def build_full_pipeline(tmp_path, mp, n: int = 400, seed: int = 31):
    """Run Phases 1-6 on test data into tmp_path, redirecting every output there.

    ``mp`` is a pytest MonkeyPatch. Returns the database path.
    """
    import sys
    from pathlib import Path

    from config import settings

    for name, value in {
        "DB_PATH": tmp_path / "c360.db",
        "REPORTS_DIR": tmp_path / "reports", "FIGURES_DIR": tmp_path / "reports" / "figures",
        "PROCESSED_DATA_DIR": tmp_path / "processed",
        "CLEAN_DATA_FILE": tmp_path / "processed" / "clean.csv",
        "MODELS_DIR": tmp_path / "models",
        "CHURN_MODEL_FILE": tmp_path / "models" / "churn.joblib",
        "SEGMENT_MODEL_FILE": tmp_path / "models" / "seg.joblib",
        "SLOW_EXPLAIN_MAX_CUSTOMERS": 15, "SHAPLEY_PERMUTATIONS": 15,
    }.items():
        mp.setattr(settings, name, value)
    scripts = str(Path(__file__).resolve().parent.parent / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import ingest_data
    import run_eda
    import run_explain
    import run_modeling
    import run_retention
    import run_segmentation

    csv = tmp_path / "customers.csv"
    make_customers_with_signal(n, seed).to_csv(csv, index=False)
    db = settings.DB_PATH
    assert ingest_data.main(["--csv", str(csv), "--db", str(db)]) == 0
    for script in (run_eda, run_segmentation, run_modeling, run_explain, run_retention):
        assert script.main(["--db", str(db)]) == 0
    return db


@pytest.fixture
def random_df() -> pd.DataFrame:
    return make_random_customers()


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "sample.csv"
    sample_df.to_csv(path, index=False)
    return path
