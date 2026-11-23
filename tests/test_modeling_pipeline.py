"""End-to-end test of the Phase 4 script."""

import sys
from contextlib import closing
from pathlib import Path

import joblib

from config import settings
from src.database.db import get_connection, read_table
from src.models.features import prepare_xy
from tests.conftest import make_customers_with_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ingest_data  # noqa: E402
import run_eda  # noqa: E402
import run_modeling  # noqa: E402


def _redirect(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(settings, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(settings, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(settings, "CLEAN_DATA_FILE", tmp_path / "processed" / "clean.csv")
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(settings, "CHURN_MODEL_FILE", tmp_path / "models" / "churn.joblib")


def test_run_modeling_end_to_end(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    csv, db = tmp_path / "c.csv", tmp_path / "c.db"
    make_customers_with_signal(600, seed=21).to_csv(csv, index=False)
    assert ingest_data.main(["--csv", str(csv), "--db", str(db)]) == 0
    assert run_eda.main(["--db", str(db)]) == 0

    assert run_modeling.main(["--db", str(db)]) == 0

    with closing(get_connection(db)) as conn:
        preds = read_table(settings.PREDICTIONS_TABLE, conn)
        clean = read_table(settings.CLEAN_TABLE, conn)
    assert len(preds) == 600

    bundle = joblib.load(tmp_path / "models" / "churn.joblib")
    assert {"pipeline", "threshold", "risk_thresholds", "model_label"} <= set(bundle)
    X, _ = prepare_xy(clean.head(5))
    proba = bundle["pipeline"].predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()

    report = (tmp_path / "reports" / "model_report.md").read_text(encoding="utf-8")
    assert "ROC-AUC" in report and report.count("![") == 7
    assert len(list((tmp_path / "reports" / "figures").glob("model_*.png"))) == 7


def test_run_modeling_needs_clean_table(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    assert run_modeling.main(["--db", str(tmp_path / "empty.db")]) == 1
