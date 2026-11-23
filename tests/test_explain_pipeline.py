"""End-to-end test of the Phase 5 script (after Phases 1, 2 and 4)."""

import json
import sys
from contextlib import closing
from pathlib import Path

from config import settings
from src.database.db import get_connection, read_table
from tests.conftest import make_customers_with_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ingest_data  # noqa: E402
import run_eda  # noqa: E402
import run_explain  # noqa: E402
import run_modeling  # noqa: E402


def _redirect(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(settings, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(settings, "CLEAN_DATA_FILE", tmp_path / "processed" / "clean.csv")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "CHURN_MODEL_FILE", tmp_path / "models" / "churn.joblib")
    # keep the test quick if Random Forest is chosen (slower, sampling-based explanations)
    monkeypatch.setattr(settings, "SLOW_EXPLAIN_MAX_CUSTOMERS", 15)
    monkeypatch.setattr(settings, "SHAPLEY_PERMUTATIONS", 15)


def test_run_explain_end_to_end(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    csv, db = tmp_path / "c.csv", tmp_path / "c.db"
    make_customers_with_signal(500, seed=8).to_csv(csv, index=False)
    assert ingest_data.main(["--csv", str(csv), "--db", str(db)]) == 0
    assert run_eda.main(["--db", str(db)]) == 0
    assert run_modeling.main(["--db", str(db)]) == 0

    assert run_explain.main(["--db", str(db)]) == 0

    with closing(get_connection(db)) as conn:
        table = read_table(settings.EXPLANATIONS_TABLE, conn)
    assert len(table) > 0
    assert table["explanation"].str.contains("probability of churning").all()
    first = json.loads(table.loc[0, "contributions_json"])
    assert "Contract" in first and "tenure" in first
    report = (tmp_path / "reports" / "explain_report.md").read_text(encoding="utf-8")
    assert "Correlation is not causation" in report and report.count("![") == 8


def test_run_explain_needs_model(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    assert run_explain.main(["--db", str(tmp_path / "x.db")]) == 1
