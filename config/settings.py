"""Central configuration for Customer360 AI.

Every path, file name and tunable constant lives here so the rest of the code
never hard-codes them. Values can be overridden with environment variables or
a local ``.env`` file (see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into os.environ.

    A tiny built-in reader, so the project needs no extra package. Values that
    are already set in the real environment take priority.
    """
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _resolve(path_value: str) -> Path:
    """Turn a relative path from settings into an absolute project path."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# --- Folders ---------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs"

# --- Dataset ---------------------------------------------------------------
DATASET_NAME = "IBM Telco Customer Churn"
RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
# Public copy published by IBM on GitHub (same data as the Kaggle mirror).
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATASET_KAGGLE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
# Published size of the dataset, used as a sanity check after download.
EXPECTED_ROWS = 7043
EXPECTED_COLUMNS = 21

# --- Database --------------------------------------------------------------
DB_PATH = _resolve(os.getenv("C360_DB_PATH", "database/customer360.db"))
RAW_TABLE = "customers_raw"
CLEAN_TABLE = "customers_clean"          # Phase 2
LOADS_TABLE = "data_loads"

# --- Phase 3: segmentation -------------------------------------------------
SEGMENTS_TABLE = "customer_segments"
SEGMENT_MODEL_FILE = MODELS_DIR / "segmentation_kmeans.joblib"
K_SEARCH_RANGE = range(2, 9)      # cluster counts evaluated (elbow + silhouette)
K_CHOICE_RANGE = range(3, 7)      # counts a business can act on; best silhouette wins

# --- Phase 4: churn prediction ---------------------------------------------
CHURN_MODEL_FILE = MODELS_DIR / "churn_model.joblib"
PREDICTIONS_TABLE = "churn_predictions"
TEST_SIZE = 0.20                  # share of customers held back for the final test
CV_FOLDS = 5                      # folds for cross-validation on the training set
# Risk levels from churn probability: below 0.30 Low, 0.30-0.60 Medium, 0.60+ High
RISK_THRESHOLDS = {"Medium": 0.30, "High": 0.60}

# --- Phase 5: explainable AI -----------------------------------------------
EXPLANATIONS_TABLE = "churn_explanations"
EXPLAIN_BACKGROUND_SIZE = 200     # reference customers that define "an average customer"
SHAPLEY_PERMUTATIONS = 100        # sampling effort for non-linear models (Random Forest)
SLOW_EXPLAIN_MAX_CUSTOMERS = 300  # non-linear models: precompute for the riskiest N

# --- Phase 6: retention intelligence ---------------------------------------
RETENTION_TABLE = "retention_actions"
# ASSUMPTIONS for impact estimates - illustrative, replace with real company figures.
RETENTION_HORIZON_MONTHS = 12     # revenue counted for a saved customer
SCENARIO_MULTIPLIERS = {"Conservative": 0.5, "Base": 1.0, "Optimistic": 1.5}
HIGH_VALUE_MONTHLY = None         # None = use the median MonthlyCharges as the cut-off

# --- Phase 8: deployment ---------------------------------------------------
# On first start with no data, the dashboard downloads the dataset and builds
# everything (needed on Streamlit Community Cloud, where data is not in Git).
AUTO_BOOTSTRAP = os.getenv("C360_AUTO_BOOTSTRAP", "1") == "1"
# Public demo: uploads can be scored in memory but are never saved to the database.
DEMO_MODE = os.getenv("C360_DEMO_MODE", "0") == "1"

# --- Phase 2 outputs -------------------------------------------------------
CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "customers_clean.csv"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Logging ---------------------------------------------------------------
LOG_LEVEL = os.getenv("C360_LOG_LEVEL", "INFO").upper()
LOG_FILE = LOG_DIR / "customer360.log"

# --- Reproducibility -------------------------------------------------------
# One seed used by every random step (splits, clustering, models).
RANDOM_STATE = 42


def ensure_directories() -> None:
    """Create the working folders if they do not exist yet."""
    for folder in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR,
                   FIGURES_DIR, LOG_DIR, DB_PATH.parent):
        folder.mkdir(parents=True, exist_ok=True)
