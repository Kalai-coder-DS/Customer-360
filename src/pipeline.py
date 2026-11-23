"""Run the complete Customer360 AI pipeline (Phases 1-6) from Python.

Used by ``scripts/run_all.py`` (command line) and by the dashboard's
first-start bootstrap (cloud deployment, where no data exists yet).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

from config import settings

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _scripts():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import download_data
    import ingest_data
    import run_eda
    import run_explain
    import run_modeling
    import run_retention
    import run_segmentation
    return download_data, [
        ("Phase 1 · Validate and store data", ingest_data),
        ("Phase 2 · Clean data and EDA", run_eda),
        ("Phase 3 · Customer segmentation", run_segmentation),
        ("Phase 4 · Churn prediction", run_modeling),
        ("Phase 5 · Explainable AI", run_explain),
        ("Phase 6 · Retention intelligence", run_retention),
    ]


def run_pipeline(download: bool = False, csv: Path | None = None,
                 progress: Callable[[str], None] = print) -> bool:
    """Run every phase in order. Returns True on success, False at the first failure.

    download: fetch the dataset first if it is not already in data/raw/.
    csv:      use this CSV instead of the default dataset file.
    progress: called with a short message before each step (for UIs and logs).
    """
    download_data, steps = _scripts()
    if download and csv is None and not settings.RAW_DATA_FILE.exists():
        progress("Downloading the dataset")
        if download_data.main([]) != 0:
            return False
    start = time.time()
    for i, (name, module) in enumerate(steps):
        progress(name)
        args = ["--csv", str(csv)] if (csv is not None and i == 0) else []
        if module.main(args) != 0:
            progress(f"Stopped: '{name}' failed.")
            return False
    progress(f"All phases complete in {time.time() - start:.0f}s")
    return True


def pipeline_complete(tables: dict) -> bool:
    """True when every table the dashboard needs exists."""
    return all(tables.get(k) is not None for k in
               ("clean", "segments", "predictions", "explanations", "retention"))
