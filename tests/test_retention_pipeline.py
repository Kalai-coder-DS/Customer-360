"""End-to-end test of the Phase 6 script (runs every earlier phase first)."""

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
import run_retention  # noqa: E402
import run_segmentation  # noqa: E402


def _redirect(tmp_path, monkeypatch):
    for name, value in {
        "REPORTS_DIR": tmp_path / "reports", "FIGURES_DIR": tmp_path / "reports" / "figures",
        "PROCESSED_DATA_DIR": tmp_path / "processed",
        "CLEAN_DATA_FILE": tmp_path / "processed" / "clean.csv",
        "MODELS_DIR": tmp_path / "models",
        "CHURN_MODEL_FILE": tmp_path / "models" / "churn.joblib",
        "SEGMENT_MODEL_FILE": tmp_path / "models" / "seg.joblib",
        "SLOW_EXPLAIN_MAX_CUSTOMERS": 15, "SHAPLEY_PERMUTATIONS": 15,
    }.items():
        monkeypatch.setattr(settings, name, value)


def test_full_pipeline_through_retention(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    csv, db = tmp_path / "c.csv", tmp_path / "c.db"
    make_customers_with_signal(500, seed=13).to_csv(csv, index=False)
    args = ["--db", str(db)]
    assert ingest_data.main(["--csv", str(csv)] + args) == 0
    for script in (run_eda, run_segmentation, run_modeling, run_explain):
        assert script.main(args) == 0

    assert run_retention.main(args) == 0

    with closing(get_connection(db)) as conn:
        plan = read_table(settings.RETENTION_TABLE, conn)
    assert len(plan) == 500
    assert {"priority_tier", "action_1", "recommendation", "segment_name"} <= set(plan.columns)
    out = tmp_path / "reports"
    assert (out / "retention" / "priority_call_list.csv").exists()
    report = (out / "retention_report.md").read_text(encoding="utf-8")
    assert "estimates built on stated assumptions" in report
    assert len(list((out / "figures").glob("retention_*.png"))) == 4


def test_retention_needs_predictions(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    assert run_retention.main(["--db", str(tmp_path / "x.db")]) == 1
