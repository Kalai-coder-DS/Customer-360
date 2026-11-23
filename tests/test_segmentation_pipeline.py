"""End-to-end test of the Phase 3 script."""

import sys
from contextlib import closing
from pathlib import Path

import joblib

from config import settings
from src.database.db import get_connection, read_table
from tests.conftest import make_random_customers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ingest_data  # noqa: E402
import run_eda  # noqa: E402
import run_segmentation  # noqa: E402


def _redirect(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(settings, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(settings, "CLEAN_DATA_FILE", tmp_path / "processed" / "clean.csv")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "SEGMENT_MODEL_FILE", tmp_path / "models" / "seg.joblib")


def test_run_segmentation_end_to_end(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    csv, db = tmp_path / "c.csv", tmp_path / "c.db"
    make_random_customers(400, seed=7).to_csv(csv, index=False)
    assert ingest_data.main(["--csv", str(csv), "--db", str(db)]) == 0
    assert run_eda.main(["--db", str(db)]) == 0

    assert run_segmentation.main(["--db", str(db), "--k", "4"]) == 0

    with closing(get_connection(db)) as conn:
        segments = read_table(settings.SEGMENTS_TABLE, conn)
    assert len(segments) == 400
    assert segments["cluster"].nunique() == 4
    assert segments["segment_name"].nunique() == 4
    model = joblib.load(tmp_path / "models" / "seg.joblib")
    assert set(model) >= {"pipeline", "features", "segment_names"}
    report = (tmp_path / "reports" / "segmentation_report.md").read_text(encoding="utf-8")
    assert "Suggested action" in report and report.count("![") == 6
    assert len(list((tmp_path / "reports" / "figures").glob("segmentation_*.png"))) == 6


def test_run_segmentation_needs_clean_table(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    assert run_segmentation.main(["--db", str(tmp_path / "empty.db")]) == 1
