"""End-to-end test of the Phase 2 script (ingest -> clean -> EDA -> report)."""

import sys
from contextlib import closing
from pathlib import Path

from config import settings
from src.database.db import get_connection, read_table
from tests.conftest import make_random_customers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ingest_data  # noqa: E402
import run_eda  # noqa: E402


def _redirect_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(settings, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(settings, "CLEAN_DATA_FILE", tmp_path / "processed" / "clean.csv")


def test_run_eda_end_to_end(tmp_path, monkeypatch):
    _redirect_outputs(tmp_path, monkeypatch)
    csv = tmp_path / "customers.csv"
    make_random_customers(300, seed=1).to_csv(csv, index=False)
    db = tmp_path / "c360.db"
    assert ingest_data.main(["--csv", str(csv), "--db", str(db)]) == 0

    assert run_eda.main(["--db", str(db)]) == 0

    with closing(get_connection(db)) as conn:
        clean = read_table(settings.CLEAN_TABLE, conn)
    assert len(clean) == 300 and "churn_flag" in clean.columns
    assert clean["TotalCharges"].notna().all()

    reports = tmp_path / "reports"
    assert (reports / "eda_report.md").read_text(encoding="utf-8").count("![") == 13
    assert len(list((reports / "figures").glob("*.png"))) == 13
    assert (reports / "eda" / "categorical_association.csv").exists()
    assert (tmp_path / "processed" / "clean.csv").exists()


def test_run_eda_without_raw_table_fails_cleanly(tmp_path, monkeypatch):
    _redirect_outputs(tmp_path, monkeypatch)
    assert run_eda.main(["--db", str(tmp_path / "empty.db")]) == 1
