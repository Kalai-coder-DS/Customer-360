"""Phase 1 pipeline: load the CSV, validate it, and store it in SQLite.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --csv path/to/file.csv --db path/to/file.db

Exit code 0 = data stored, 1 = data could not be loaded or failed validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow project imports

from config import settings  # noqa: E402
from src.data.loader import DataLoadError, load_customer_csv  # noqa: E402
from src.data.validator import validate_customer_data  # noqa: E402
from src.database.db import (  # noqa: E402
    DatabaseError, file_sha256, get_connection, init_db, record_load, save_dataframe,
)
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("ingest_data")


def run(csv_path: Path, db_path: Path) -> int:
    settings.ensure_directories()

    # Step 1 - Load
    try:
        df = load_customer_csv(csv_path)
    except DataLoadError as exc:
        logger.error("%s", exc)
        return 1

    # Step 2 - Validate
    report = validate_customer_data(df)
    print("\n=== Data quality report ===")
    for line in report.summary_lines():
        print("  " + line)

    report_path = settings.REPORTS_DIR / "phase1_validation_report.json"
    report_path.write_text(json.dumps(
        {**report.__dict__, "is_valid": report.is_valid, "churn_rate": report.churn_rate},
        indent=2, default=str,
    ), encoding="utf-8")
    print(f"\n  Full report saved to {report_path}")

    if not report.is_valid:
        logger.error("Validation failed; data was NOT stored. Fix the errors above.")
        return 1

    # Step 3 - Store raw data and audit record
    try:
        with closing(get_connection(db_path)) as conn:
            init_db(conn)
            save_dataframe(df, settings.RAW_TABLE, conn)
            load_id = record_load(
                conn, source_file=str(csv_path), table_name=settings.RAW_TABLE,
                n_rows=report.n_rows, n_columns=report.n_columns,
                is_valid=report.is_valid, n_warnings=len(report.warnings),
                sha256=file_sha256(csv_path),
            )
    except DatabaseError as exc:
        logger.error("%s", exc)
        return 1

    print(f"\n  Stored {report.n_rows:,} rows in table '{settings.RAW_TABLE}' "
          f"of {db_path} (load #{load_id}).")
    if report.warnings:
        print("  Warnings are expected at this stage; Phase 2 cleans them.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=settings.RAW_DATA_FILE)
    parser.add_argument("--db", type=Path, default=settings.DB_PATH)
    args = parser.parse_args(argv)
    return run(args.csv, args.db)


if __name__ == "__main__":
    sys.exit(main())
