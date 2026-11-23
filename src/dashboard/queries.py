"""Data access and calculations for the dashboard (no Streamlit code here).

Keeping these functions free of Streamlit means they can be unit-tested and
reused. The pages only lay out what these functions return.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from src.data.cleaning import clean_customer_data
from src.data.schema import ID_COLUMN
from src.data.validator import validate_customer_data
from src.database.db import get_connection, list_tables, read_table
from src.models.evaluate import risk_level
from src.models.features import prepare_xy

TABLES = {
    "raw": settings.RAW_TABLE,
    "clean": settings.CLEAN_TABLE,
    "segments": settings.SEGMENTS_TABLE,
    "predictions": settings.PREDICTIONS_TABLE,
    "explanations": settings.EXPLANATIONS_TABLE,
    "retention": settings.RETENTION_TABLE,
    "loads": settings.LOADS_TABLE,
}

PIPELINE_STEPS = [
    ("clean", "python scripts/run_eda.py"),
    ("segments", "python scripts/run_segmentation.py"),
    ("predictions", "python scripts/run_modeling.py"),
    ("explanations", "python scripts/run_explain.py"),
    ("retention", "python scripts/run_retention.py"),
]


# --- Loading ---------------------------------------------------------------

def db_signature(db_path: Path | str | None = None) -> float:
    """Last-modified time of the database; used to refresh cached data."""
    path = Path(db_path) if db_path else settings.DB_PATH
    return path.stat().st_mtime if path.exists() else 0.0


def load_tables(db_path: Path | str | None = None) -> dict[str, pd.DataFrame | None]:
    """Every pipeline table, or None for tables that have not been created yet."""
    path = Path(db_path) if db_path else settings.DB_PATH
    result: dict[str, pd.DataFrame | None] = {key: None for key in TABLES}
    if not path.exists():
        return result
    with closing(get_connection(path)) as conn:
        existing = set(list_tables(conn))
        for key, table in TABLES.items():
            if table in existing:
                result[key] = read_table(table, conn)
    return result


def missing_steps(tables: dict[str, pd.DataFrame | None]) -> list[str]:
    """Commands still needed to complete the pipeline, in order."""
    return [cmd for key, cmd in PIPELINE_STEPS if tables.get(key) is None]


def read_report_csv(relative: str, **kwargs) -> pd.DataFrame | None:
    path = settings.REPORTS_DIR / relative
    return pd.read_csv(path, **kwargs) if path.exists() else None


def read_report_json(relative: str) -> dict | None:
    path = settings.REPORTS_DIR / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def figure_path(name: str) -> Path | None:
    path = settings.FIGURES_DIR / f"{name}.png"
    return path if path.exists() else None


def list_report_files() -> list[Path]:
    """All downloadable report files (Markdown, CSV, JSON)."""
    if not settings.REPORTS_DIR.exists():
        return []
    files = [p for p in settings.REPORTS_DIR.rglob("*")
             if p.is_file() and p.suffix in {".md", ".csv", ".json"}]
    return sorted(files, key=lambda p: (p.parent != settings.REPORTS_DIR, str(p)))


# --- Executive overview ----------------------------------------------------

def executive_kpis(tables: dict[str, pd.DataFrame | None]) -> dict:
    clean = tables["clean"]
    kpis: dict = {}
    if clean is None:
        return kpis
    churned = clean["churn_flag"] == 1
    kpis.update({
        "customers": int(len(clean)),
        "active": int((~churned).sum()),
        "churned": int(churned.sum()),
        "churn_rate": float(churned.mean()),
        "avg_monthly_charges": float(clean["MonthlyCharges"].mean()),
        "avg_tenure": float(clean["tenure"].mean()),
        "monthly_revenue_active": float(clean.loc[~churned, "MonthlyCharges"].sum()),
        "monthly_revenue_lost": float(clean.loc[churned, "MonthlyCharges"].sum()),
    })
    preds = tables["predictions"]
    if preds is not None:
        active_ids = set(clean.loc[~churned, ID_COLUMN])
        active_preds = preds[preds[ID_COLUMN].isin(active_ids)]
        kpis["high_risk_active"] = int((active_preds["risk_level"] == "High").sum())
        kpis["medium_risk_active"] = int((active_preds["risk_level"] == "Medium").sum())
    plan = tables["retention"]
    if plan is not None:
        kpis["revenue_at_risk"] = float(plan["expected_loss_horizon"].sum())
        kpis["act_now"] = int((plan["priority_tier"] == "1 - Act now").sum())
    return kpis


def churn_rate_table(clean: pd.DataFrame, column: str) -> pd.DataFrame:
    """Churn rate and customer count per value of a column (for charts)."""
    grouped = clean.groupby(column, observed=True)["churn_flag"].agg(["mean", "size"])
    grouped.columns = ["churn_rate", "customers"]
    return grouped.reset_index().sort_values("churn_rate", ascending=False)


def tenure_band_table(clean: pd.DataFrame) -> pd.DataFrame:
    from src.analysis.eda import TENURE_BAND, TENURE_LABELS, add_tenure_band
    table = churn_rate_table(add_tenure_band(clean), TENURE_BAND)
    table[TENURE_BAND] = pd.Categorical(table[TENURE_BAND].astype(str), TENURE_LABELS,
                                        ordered=True)
    return table.sort_values(TENURE_BAND)


# --- Customer 360 ----------------------------------------------------------

def customer_ids_by_risk(tables: dict[str, pd.DataFrame | None]) -> list[str]:
    """Customer IDs, riskiest first when predictions exist."""
    preds = tables["predictions"]
    if preds is not None:
        return preds.sort_values("churn_probability", ascending=False)[ID_COLUMN].tolist()
    clean = tables["clean"]
    return clean[ID_COLUMN].tolist() if clean is not None else []


def customer_record(tables: dict[str, pd.DataFrame | None], customer_id: str) -> dict:
    """One customer's row from every table (None where not available)."""
    record = {}
    for key in ("clean", "segments", "predictions", "explanations", "retention"):
        frame = tables.get(key)
        if frame is None:
            record[key] = None
            continue
        match = frame[frame[ID_COLUMN] == customer_id]
        record[key] = match.iloc[0] if len(match) else None
    return record


def parse_contributions(value) -> pd.Series | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return pd.Series(json.loads(value), dtype=float)


def services_used(row: pd.Series) -> list[str]:
    services = []
    if row.get("PhoneService") == "Yes":
        services.append("Phone" + (" (multiple lines)" if row.get("MultipleLines") == "Yes"
                                   else ""))
    if row.get("InternetService") not in (None, "No"):
        services.append(f"Internet ({row['InternetService']})")
    for col, name in (("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
                      ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
                      ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")):
        if row.get(col) == "Yes":
            services.append(name)
    return services


# --- Retention -------------------------------------------------------------

def filter_plan(plan: pd.DataFrame, tiers=None, segments=None, actions=None,
                min_probability: float = 0.0, search: str = "") -> pd.DataFrame:
    data = plan
    if tiers:
        data = data[data["priority_tier"].isin(tiers)]
    if segments and "segment_name" in data:
        data = data[data["segment_name"].isin(segments)]
    if actions:
        data = data[data["action_1"].isin(actions) | data["action_2"].isin(actions)]
    data = data[data["churn_probability"] >= min_probability]
    if search:
        data = data[data[ID_COLUMN].str.contains(search.strip(), case=False, regex=False)]
    return data


def impact_estimate(plan: pd.DataFrame, tiers: list[str], success_multiplier: float = 1.0,
                    horizon_months: int = settings.RETENTION_HORIZON_MONTHS) -> dict:
    """Live what-if version of the Phase 6 impact calculation."""
    group = plan[plan["priority_tier"].isin(tiers)]
    prob = group["churn_probability"].astype(float)
    monthly = group["MonthlyCharges"].astype(float)
    success = np.clip(group["success_rate"].astype(float) * success_multiplier, 0, 1)
    revenue = float((prob * success * monthly * horizon_months).sum())
    cost = float(group["actions_cost"].sum())
    return {
        "customers_contacted": int(len(group)),
        "expected_churners": float(prob.sum()),
        "customers_saved": float((prob * success).sum()),
        "revenue_protected": revenue,
        "campaign_cost": cost,
        "net_value": revenue - cost,
        "roi": (revenue - cost) / cost if cost else float("nan"),
    }


# --- Data management -------------------------------------------------------

def validate_upload(df: pd.DataFrame):
    return validate_customer_data(df)


def score_customers(raw: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, object]:
    """Clean an uploaded table and score it with the saved churn model."""
    clean, log = clean_customer_data(raw)
    X, _ = prepare_xy(clean)
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    scored = clean[[ID_COLUMN, "tenure", "Contract", "MonthlyCharges"]].copy()
    scored["churn_probability"] = np.round(proba, 4)
    scored["risk_level"] = risk_level(proba, bundle.get("risk_thresholds",
                                                        settings.RISK_THRESHOLDS))
    scored["predicted_churn"] = (proba >= bundle["threshold"]).astype(int)
    return scored.sort_values("churn_probability", ascending=False), log


def save_upload(df: pd.DataFrame, table: str, db_path: Path | str | None = None) -> int:
    """Store an uploaded table in SQLite and record it in the audit log."""
    from src.database.db import init_db, record_load, save_dataframe
    report = validate_customer_data(df)
    with closing(get_connection(db_path)) as conn:
        init_db(conn)
        save_dataframe(df, table, conn)
        return record_load(conn, source_file="dashboard upload", table_name=table,
                           n_rows=report.n_rows, n_columns=report.n_columns,
                           is_valid=report.is_valid, n_warnings=len(report.warnings))


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def sqlite_version() -> str:
    return sqlite3.sqlite_version
