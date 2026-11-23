"""End-to-end test of the Phase 1 ingest script."""

import sys
from contextlib import closing
from pathlib import Path

from config import settings
from src.database.db import get_connection, load_history, read_table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ingest_data  # noqa: E402


def test_ingest_stores_valid_data(tmp_path, sample_csv, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    db = tmp_path / "c360.db"

    assert ingest_data.main(["--csv", str(sample_csv), "--db", str(db)]) == 0

    with closing(get_connection(db)) as conn:
        assert len(read_table(settings.RAW_TABLE, conn)) == 6
    history = load_history(db)
    assert len(history) == 1 and history.iloc[0]["is_valid"] == 1
    assert (tmp_path / "reports" / "phase1_validation_report.json").exists()


def test_ingest_refuses_invalid_data(tmp_path, sample_df, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    bad_csv = tmp_path / "bad.csv"
    sample_df.drop(columns=["Churn"]).to_csv(bad_csv, index=False)
    db = tmp_path / "c360.db"

    assert ingest_data.main(["--csv", str(bad_csv), "--db", str(db)]) == 1
    assert not db.exists() or load_history(db).empty


def test_ingest_reports_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REPORTS_DIR", tmp_path / "reports")
    assert ingest_data.main(["--csv", str(tmp_path / "none.csv"),
                             "--db", str(tmp_path / "x.db")]) == 1


def test_settings_paths_are_inside_project():
    root = settings.PROJECT_ROOT
    for path in (settings.RAW_DATA_FILE, settings.DB_PATH, settings.LOG_FILE):
        assert root in path.parents
