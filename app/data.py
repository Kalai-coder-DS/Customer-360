"""Cached data loading for the Streamlit pages.

Data is cached and reloaded automatically when the SQLite file changes
(e.g. after re-running a pipeline script).
"""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from config import settings
from src.dashboard import queries


@st.cache_data(show_spinner="Loading customer data...")
def _load(db_path: str, signature: float):
    return queries.load_tables(db_path)


def tables() -> dict:
    """All pipeline tables; reloaded when the database file changes."""
    return _load(str(settings.DB_PATH), queries.db_signature())


@st.cache_resource(show_spinner="Loading churn model...")
def _bundle(model_path: str, signature: float):
    return joblib.load(model_path)


def churn_bundle():
    path = settings.CHURN_MODEL_FILE
    if not path.exists():
        return None
    return _bundle(str(path), path.stat().st_mtime)


@st.cache_data
def report_csv(path: str, signature: float, **kwargs):
    return pd.read_csv(path, **kwargs) if signature else None


def report(relative: str, **kwargs):
    """A CSV from the reports folder (None if not created yet)."""
    path = settings.REPORTS_DIR / relative
    return report_csv(str(path), path.stat().st_mtime if path.exists() else 0.0, **kwargs)


def demo_mode() -> bool:
    """Public-demo switch: env var C360_DEMO_MODE=1 or the same key in Streamlit secrets."""
    if settings.DEMO_MODE:
        return True
    try:
        return str(st.secrets.get("C360_DEMO_MODE", "0")) == "1"
    except Exception:  # no secrets file: normal local use
        return False


@st.cache_resource
def _build_lock():
    import threading
    return threading.Lock()


def ensure_pipeline() -> None:
    """First start with no data (e.g. a fresh cloud deployment): build everything once.

    A shared lock makes sure only one visitor's session runs the build; others wait.
    """
    from src.pipeline import pipeline_complete, run_pipeline
    if not settings.AUTO_BOOTSTRAP or pipeline_complete(tables()):
        return
    lock = _build_lock()
    with st.status("First start: downloading the dataset and building the models. This "
                   "takes about a minute and only happens once.", expanded=True) as status:
        with lock:
            if pipeline_complete(queries.load_tables()):  # built by another session meanwhile
                status.update(label="Data ready", state="complete", expanded=False)
                return
            ok = run_pipeline(download=True, progress=lambda message: status.write(message))
        if ok:
            status.update(label="Data and models ready", state="complete", expanded=False)
            _load.clear()
        else:
            status.update(label="Automatic setup failed", state="error")
            st.error("The dataset could not be downloaded or processed. Run "
                     "`python scripts/run_all.py --download` in the project folder, or see the "
                     "server log for details.", icon=":material/error:")
