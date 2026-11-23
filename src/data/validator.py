"""Data quality checks for customer data.

The validator only *reports* problems; it never changes the data. Problems are
split into two levels:

* errors   - the data cannot be used (e.g. a required column is missing)
* warnings - the data can be used but needs cleaning (e.g. blank values)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.schema import (
    CATEGORICAL_VALUES,
    EXPECTED_COLUMNS,
    ID_COLUMN,
    NUMERIC_CODES,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


@dataclass
class ValidationReport:
    """Result of validating one dataset."""

    n_rows: int = 0
    n_columns: int = 0
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)   # NaN + blank text
    blank_text_values: dict[str, int] = field(default_factory=dict)  # subset: "" or " "
    duplicate_rows: int = 0
    duplicate_ids: int = 0
    invalid_categories: dict[str, list[str]] = field(default_factory=dict)
    non_numeric_values: dict[str, int] = field(default_factory=dict)
    negative_values: dict[str, int] = field(default_factory=dict)
    target_distribution: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when there are no blocking errors (warnings are allowed)."""
        return not self.errors

    @property
    def total_missing(self) -> int:
        return sum(self.missing_values.values())

    @property
    def churn_rate(self) -> float | None:
        total = sum(self.target_distribution.values())
        if not total:
            return None
        return self.target_distribution.get("Yes", 0) / total

    def summary_lines(self) -> list[str]:
        """Human-readable summary for the terminal and logs."""
        lines = [
            f"Rows: {self.n_rows:,}   Columns: {self.n_columns}",
            f"Status: {'VALID' if self.is_valid else 'INVALID'}"
            f" ({len(self.errors)} errors, {len(self.warnings)} warnings)",
        ]
        if self.target_distribution:
            dist = ", ".join(f"{k}: {v:,}" for k, v in sorted(self.target_distribution.items()))
            rate = self.churn_rate
            lines.append(f"Target '{TARGET_COLUMN}': {dist}"
                         + (f"  (churn rate {rate:.1%})" if rate is not None else ""))
        lines += [f"ERROR:   {msg}" for msg in self.errors]
        lines += [f"WARNING: {msg}" for msg in self.warnings]
        return lines


def _blank_text_mask(series: pd.Series) -> pd.Series:
    """True where a value is text containing only whitespace (e.g. ' ')."""
    as_text = series.astype("string")
    return as_text.notna() & (as_text.str.strip() == "")


def validate_customer_data(df: pd.DataFrame) -> ValidationReport:
    """Run every quality check and return a report."""
    report = ValidationReport(n_rows=len(df), n_columns=df.shape[1])

    if df.empty:
        report.errors.append("The dataset has no rows.")
        return report

    # 1. Structure: required and unexpected columns
    report.missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    report.unexpected_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if report.missing_columns:
        report.errors.append(f"Missing required columns: {', '.join(report.missing_columns)}")
    if report.unexpected_columns:
        report.warnings.append(
            f"Unexpected columns (will be ignored): {', '.join(report.unexpected_columns)}"
        )

    present = [c for c in EXPECTED_COLUMNS if c in df.columns]

    # 2. Missing values: real NaN plus blank text such as " "
    for col in present:
        blanks = int(_blank_text_mask(df[col]).sum())
        missing = int(df[col].isna().sum()) + blanks
        if blanks:
            report.blank_text_values[col] = blanks
        if missing:
            report.missing_values[col] = missing
            report.warnings.append(
                f"'{col}' has {missing:,} missing value(s)"
                + (f" ({blanks:,} are blank text)" if blanks else "")
            )

    # 3. Duplicates: identical rows and repeated customer IDs
    report.duplicate_rows = int(df.duplicated().sum())
    if report.duplicate_rows:
        report.warnings.append(f"{report.duplicate_rows:,} fully duplicated row(s)")
    if ID_COLUMN in df.columns:
        report.duplicate_ids = int(df[ID_COLUMN].dropna().duplicated().sum())
        if report.duplicate_ids:
            report.warnings.append(f"{report.duplicate_ids:,} repeated '{ID_COLUMN}' value(s)")

    # 4. Categorical columns: values outside the allowed set
    for col, allowed in CATEGORICAL_VALUES.items():
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        bad = sorted(set(values) - set(allowed))
        if bad:
            report.invalid_categories[col] = bad
            shown = ", ".join(repr(v) for v in bad[:5])
            report.warnings.append(f"'{col}' has unexpected value(s): {shown}")

    # 5. Numeric columns: text that is not a number, and negative numbers
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col]
        filled = series[series.notna() & ~_blank_text_mask(series)]
        numbers = pd.to_numeric(filled, errors="coerce")
        non_numeric = int(numbers.isna().sum())
        if non_numeric:
            report.non_numeric_values[col] = non_numeric
            report.warnings.append(f"'{col}' has {non_numeric:,} non-numeric value(s)")
        negatives = int((numbers < 0).sum())
        if negatives:
            report.negative_values[col] = negatives
            report.warnings.append(f"'{col}' has {negatives:,} negative value(s)")
        if col in NUMERIC_CODES:
            bad_codes = sorted(set(numbers.dropna().unique()) - set(NUMERIC_CODES[col]))
            if bad_codes:
                report.invalid_categories[col] = [str(v) for v in bad_codes]
                report.warnings.append(f"'{col}' has unexpected code(s): {bad_codes[:5]}")

    # 6. Target column: class balance (important for churn modelling later)
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].dropna().astype(str).str.strip().value_counts()
        report.target_distribution = {str(k): int(v) for k, v in counts.items()}
        if len(report.target_distribution) < 2:
            report.errors.append(
                f"Target '{TARGET_COLUMN}' needs both 'Yes' and 'No' values to train a model."
            )

    return report
