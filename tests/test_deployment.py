"""Deployment readiness: one-call pipeline, first-start detection, required files."""

from pathlib import Path

import pytest

from config import settings
from src.dashboard import queries
from src.pipeline import pipeline_complete, run_pipeline
from tests.conftest import make_customers_with_signal

ROOT = Path(__file__).resolve().parent.parent


def _redirect(tmp_path, monkeypatch):
    for name, value in {
        "DB_PATH": tmp_path / "c360.db",
        "REPORTS_DIR": tmp_path / "reports", "FIGURES_DIR": tmp_path / "reports" / "figures",
        "PROCESSED_DATA_DIR": tmp_path / "processed",
        "CLEAN_DATA_FILE": tmp_path / "processed" / "clean.csv",
        "MODELS_DIR": tmp_path / "models",
        "CHURN_MODEL_FILE": tmp_path / "models" / "churn.joblib",
        "SEGMENT_MODEL_FILE": tmp_path / "models" / "seg.joblib",
        "RAW_DATA_FILE": tmp_path / "raw" / "missing.csv",
        "SLOW_EXPLAIN_MAX_CUSTOMERS": 15, "SHAPLEY_PERMUTATIONS": 15,
    }.items():
        monkeypatch.setattr(settings, name, value)


def test_run_pipeline_builds_everything_the_dashboard_needs(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    csv = tmp_path / "customers.csv"
    make_customers_with_signal(300, seed=17).to_csv(csv, index=False)
    messages = []
    assert queries.load_tables(settings.DB_PATH)["clean"] is None
    assert not pipeline_complete(queries.load_tables(settings.DB_PATH))

    ok = run_pipeline(csv=csv, progress=messages.append)
    assert ok, messages
    assert messages[0].startswith("Phase 1") and messages[-1].startswith("All phases complete")
    assert pipeline_complete(queries.load_tables(settings.DB_PATH))


def test_pipeline_stops_at_first_failure(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    messages = []
    assert run_pipeline(csv=tmp_path / "does_not_exist.csv", progress=messages.append) is False
    assert messages[-1].startswith("Stopped: 'Phase 1")


def test_pipeline_complete_detection():
    full = {k: object() for k in ("clean", "segments", "predictions", "explanations",
                                  "retention")}
    assert pipeline_complete(full)
    assert not pipeline_complete({**full, "explanations": None})


def test_settings_switches_have_safe_defaults():
    assert isinstance(settings.AUTO_BOOTSTRAP, bool)
    assert isinstance(settings.DEMO_MODE, bool)


@pytest.mark.parametrize("path", [
    "requirements.txt", "app/streamlit_app.py", ".streamlit/config.toml",
    ".github/workflows/tests.yml", "docs/deployment.md", ".gitignore",
])
def test_deployment_files_exist(path):
    assert (ROOT / path).exists(), path


def test_secrets_and_data_are_never_committed():
    ignored = (ROOT / ".gitignore").read_text()
    for pattern in (".env", ".streamlit/secrets.toml", "database/*.db", "data/raw/*", "venv/"):
        assert pattern in ignored, pattern


def test_requirements_cover_every_third_party_import():
    reqs = (ROOT / "requirements.txt").read_text().lower()
    for package in ("pandas", "numpy", "scipy", "scikit-learn", "joblib", "matplotlib",
                    "plotly", "streamlit", "pytest"):
        assert package in reqs, package
