"""Check that Python and every required library are installed.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import sqlite3
import sys

REQUIRED = {
    "pandas": "pandas", "numpy": "numpy", "scipy": "scipy",
    "scikit-learn": "sklearn", "matplotlib": "matplotlib", "seaborn": "seaborn",
    "plotly": "plotly", "streamlit": "streamlit", "pytest": "pytest",
}


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]}", end="")
    if sys.version_info < (3, 10):
        print("  -> needs 3.10 or newer")
        ok = False
    else:
        print("  OK")
    print(f"SQLite {sqlite3.sqlite_version}  OK (built into Python)")

    for package, module in REQUIRED.items():
        try:
            version = getattr(importlib.import_module(module), "__version__", "installed")
            print(f"{package:<13} {version:<10} OK")
        except ImportError:
            print(f"{package:<13} {'-':<10} MISSING  -> pip install -r requirements.txt")
            ok = False

    print("\nEnvironment ready." if ok else "\nSome items need attention (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
