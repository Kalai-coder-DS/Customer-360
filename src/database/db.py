"""SQLite storage for Customer360 AI.

SQLite is a database stored in a single file (``database/customer360.db``).
It needs no server, which suits a laptop project.

Tables created in Phase 1:

* ``customers_raw`` - the dataset exactly as loaded from the CSV
* ``data_loads``    - an audit log: when each file was loaded and its quality status
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SAFE_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class DatabaseError(Exception):
    """Raised when a database operation fails."""


def _check_table_name(table: str) -> str:
    """Allow only simple table names, since names cannot be SQL parameters."""
    if not _SAFE_TABLE_NAME.match(table):
        raise DatabaseError(f"Invalid table name: {table!r}")
    return table


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database file."""
    path = Path(db_path) if db_path else settings.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return sqlite3.connect(path)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not open database at {path}: {exc}") from exc


def init_db(conn: sqlite3.Connection) -> None:
    """Create the audit table if it does not exist."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {settings.LOADS_TABLE} (
            load_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            loaded_at    TEXT    NOT NULL,
            source_file  TEXT    NOT NULL,
            file_sha256  TEXT,
            table_name   TEXT    NOT NULL,
            n_rows       INTEGER NOT NULL,
            n_columns    INTEGER NOT NULL,
            is_valid     INTEGER NOT NULL,
            n_warnings   INTEGER NOT NULL
        )
        """
    )
    conn.commit()


def save_dataframe(df: pd.DataFrame, table: str, conn: sqlite3.Connection,
                   if_exists: str = "replace") -> int:
    """Write a DataFrame to a table. Returns the number of rows written."""
    _check_table_name(table)
    try:
        df.to_sql(table, conn, if_exists=if_exists, index=False)
    except (sqlite3.Error, ValueError) as exc:
        raise DatabaseError(f"Could not save table '{table}': {exc}") from exc
    logger.info("Saved %s rows to table '%s'", len(df), table)
    return len(df)


def read_table(table: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Read a whole table into a DataFrame."""
    _check_table_name(table)
    if table not in list_tables(conn):
        raise DatabaseError(f"Table '{table}' does not exist. Run the ingest script first.")
    return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return sorted(r[0] for r in rows)


def file_sha256(path: Path | str) -> str:
    """Fingerprint of a file, so we can tell later exactly which file was loaded."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_load(conn: sqlite3.Connection, *, source_file: str, table_name: str,
                n_rows: int, n_columns: int, is_valid: bool, n_warnings: int,
                sha256: str | None = None) -> int:
    """Add an entry to the audit log and return its load_id."""
    init_db(conn)
    cursor = conn.execute(
        f"""INSERT INTO {settings.LOADS_TABLE}
            (loaded_at, source_file, file_sha256, table_name, n_rows, n_columns,
             is_valid, n_warnings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), source_file, sha256,
         table_name, n_rows, n_columns, int(is_valid), n_warnings),
    )
    conn.commit()
    return int(cursor.lastrowid)


def load_history(db_path: Path | str | None = None) -> pd.DataFrame:
    """Return the audit log, newest first."""
    with closing(get_connection(db_path)) as conn:
        init_db(conn)
        return pd.read_sql_query(
            f"SELECT * FROM {settings.LOADS_TABLE} ORDER BY load_id DESC", conn
        )
