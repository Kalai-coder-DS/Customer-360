"""Run the complete pipeline (Phases 1-6) in order, stopping at the first failure.

Usage:
    python scripts/run_all.py              # assumes the dataset is in data/raw/
    python scripts/run_all.py --download   # download the dataset first if missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow project imports

from src.pipeline import run_pipeline  # noqa: E402


def _progress(message: str) -> None:
    if message.startswith("Phase") or message.startswith("Downloading"):
        print(f"\n{'=' * 70}\n{message}\n{'=' * 70}")
    else:
        print(f"\n{message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true",
                        help="download the dataset first if it is missing")
    args = parser.parse_args(argv)
    ok = run_pipeline(download=args.download, progress=_progress)
    if ok:
        print("Start the dashboard with:\n  streamlit run app/streamlit_app.py")
        return 0
    print("Fix the error above and run again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
