"""Turn raw customer data into a clean, analysis-ready table.

Design rule: cleaning only applies fixes that depend on a single row, or that
are certain from business logic. Anything that needs statistics from the
whole dataset (filling gaps with a median, scaling numbers, encoding
categories as numbers) is deliberately left for the Phase 4 model pipeline.
There it is learned from the *training* data only. Learning it here, from all
rows, would let information from the test set leak into the model
("data leakage") and make its scores look better than they really are.

Every change is recorded in a ``CleaningLog`` so nothing happens silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.schema import (
    CATEGORICAL_VALUES,
    EXPECTED_COLUMNS,
    ID_COLUMN,
    NUMERIC_CODES,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHURN_FLAG = "churn_flag"  # 1 = churned, 0 = stayed; numeric copy of the target


@dataclass
class CleaningLog:
    """A record of every change made during cleaning."""

    rows_before: int = 0
    rows_after: int = 0
    dropped_columns: list[str] = field(default_factory=list)
    duplicate_rows_removed: int = 0
    duplicate_ids_removed: int = 0
    blank_text_to_missing: dict[str, int] = field(default_factory=dict)
    total_charges_filled_zero: int = 0
    invalid_values_to_missing: dict[str, int] = field(default_factory=dict)
    rows_missing_target_removed: int = 0
    remaining_missing: dict[str, int] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.steps.append(message)
        logger.info(message)


def _strip_text(df: pd.DataFrame, log: CleaningLog) -> pd.DataFrame:
    """Remove stray spaces around text and turn whitespace-only text into NaN."""
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            text = df[col].astype("string").str.strip()
            keep = text.fillna("") != ""          # False for blank or missing
            blanks = int((text.notna() & ~keep).sum())
            if blanks:
                log.blank_text_to_missing[col] = blanks
            df[col] = text.astype(object).where(keep, np.nan)
    if log.blank_text_to_missing:
        log.add(f"Converted blank text to missing values: {log.blank_text_to_missing}")
    return df


def clean_customer_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """Clean a raw Telco-format DataFrame. The input is not modified."""
    log = CleaningLog(rows_before=len(raw))
    df = raw.copy()

    # Step 1 - keep only the columns the project understands
    log.dropped_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if log.dropped_columns:
        df = df.drop(columns=log.dropped_columns)
        log.add(f"Dropped unexpected columns: {log.dropped_columns}")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot clean data; required columns missing: {missing}")
    df = df[EXPECTED_COLUMNS]

    # Step 2 - tidy text: trim spaces, blanks become missing
    df = _strip_text(df, log)

    # Step 3 - duplicates: identical rows, then repeated customer IDs (keep first)
    before = len(df)
    df = df.drop_duplicates()
    log.duplicate_rows_removed = before - len(df)
    before = len(df)
    df = df[~(df[ID_COLUMN].notna() & df[ID_COLUMN].duplicated(keep="first"))]
    log.duplicate_ids_removed = before - len(df)
    if log.duplicate_rows_removed or log.duplicate_ids_removed:
        log.add(f"Removed {log.duplicate_rows_removed} duplicate rows and "
                f"{log.duplicate_ids_removed} repeated customer IDs (first kept)")

    # Step 4 - numbers: convert text to numeric; impossible values become missing
    for col in NUMERIC_COLUMNS:
        numbers = pd.to_numeric(df[col], errors="coerce")
        newly_missing = int(numbers.isna().sum() - df[col].isna().sum())
        negative = numbers < 0
        invalid_code = (~numbers.isin(NUMERIC_CODES[col]) & numbers.notna()
                        if col in NUMERIC_CODES else pd.Series(False, index=df.index))
        bad = negative | invalid_code
        numbers = numbers.mask(bad)
        if newly_missing + int(bad.sum()):
            log.invalid_values_to_missing[col] = newly_missing + int(bad.sum())
        df[col] = numbers

    # Step 5 - business rule: a customer with 0 months' tenure has not been billed
    # yet, so their TotalCharges is 0. This is certain, not a statistical guess.
    new_customer_gap = df["TotalCharges"].isna() & (df["tenure"] == 0)
    log.total_charges_filled_zero = int(new_customer_gap.sum())
    if log.total_charges_filled_zero:
        df.loc[new_customer_gap, "TotalCharges"] = 0.0
        log.add(f"Set TotalCharges = 0 for {log.total_charges_filled_zero} new customers "
                "(tenure 0, not yet billed)")

    # Step 6 - categories: values outside the allowed list become missing
    for col, allowed in CATEGORICAL_VALUES.items():
        invalid = df[col].notna() & ~df[col].isin(allowed)
        if invalid.any():
            log.invalid_values_to_missing[col] = (
                log.invalid_values_to_missing.get(col, 0) + int(invalid.sum())
            )
            df.loc[invalid, col] = np.nan
    if log.invalid_values_to_missing:
        log.add(f"Set invalid values to missing: {log.invalid_values_to_missing}")

    # Step 7 - target: rows without a churn label cannot be used for learning
    no_target = df[TARGET_COLUMN].isna()
    log.rows_missing_target_removed = int(no_target.sum())
    if log.rows_missing_target_removed:
        df = df[~no_target]
        log.add(f"Removed {log.rows_missing_target_removed} rows with no Churn value")

    # Step 8 - final types and a numeric churn flag for analysis
    df["tenure"] = df["tenure"].astype("Int64")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("Int64")
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    df[CHURN_FLAG] = (df[TARGET_COLUMN] == "Yes").astype(int)
    df = df.reset_index(drop=True)

    log.remaining_missing = {c: int(n) for c, n in df.isna().sum().items() if n}
    if log.remaining_missing:
        log.add(f"Missing values left for the model pipeline to impute (training data "
                f"only): {log.remaining_missing}")
    log.rows_after = len(df)
    log.add(f"Cleaning complete: {log.rows_before:,} rows in, {log.rows_after:,} rows out")
    return df, log
