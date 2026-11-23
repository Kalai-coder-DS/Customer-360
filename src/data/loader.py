"""Load customer CSV files into pandas, with clear error messages.

The loader keeps the data exactly as it is in the file (the "raw layer").
Cleaning and type fixes happen in Phase 2, so nothing here changes values.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, Union

import pandas as pd

from config import settings
from src.data.schema import ID_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)

CsvSource = Union[str, Path, IO]


class DataLoadError(Exception):
    """Raised when a CSV file cannot be read as customer data."""


def load_customer_csv(source: CsvSource = settings.RAW_DATA_FILE) -> pd.DataFrame:
    """Read a customer CSV from a path or an open file (e.g. a web upload).

    The customer ID is always read as text so IDs like "0012" keep their zeros.
    Column names are stripped of stray spaces. Values are not modified.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DataLoadError(
                f"Dataset not found at '{path}'. "
                "Run 'python scripts/download_data.py' or place the CSV there manually."
            )
        label = str(path)
    else:
        label = getattr(source, "name", "uploaded file")

    try:
        df = pd.read_csv(source, dtype={ID_COLUMN: str})
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"'{label}' is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"'{label}' is not a readable CSV file: {exc}") from exc

    if df.empty:
        raise DataLoadError(f"'{label}' has column headers but no rows.")

    df.columns = [str(col).strip() for col in df.columns]
    logger.info("Loaded %s rows x %s columns from %s", len(df), df.shape[1], label)
    return df
