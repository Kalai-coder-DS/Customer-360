"""Phase 2 pipeline: clean the raw data, analyse it and produce an EDA report.

Usage:
    python scripts/run_eda.py
    python scripts/run_eda.py --db path/to/file.db

Reads   : SQLite table 'customers_raw' (created by scripts/ingest_data.py)
Writes  : SQLite table 'customers_clean', data/processed/customers_clean.csv,
          reports/eda_report.md, reports/eda/*.csv, reports/figures/*.png,
          reports/phase2_cleaning_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow project imports

import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from src.analysis import eda  # noqa: E402
from src.data.cleaning import clean_customer_data  # noqa: E402
from src.data.validator import validate_customer_data  # noqa: E402
from src.database.db import DatabaseError, get_connection, read_table, save_dataframe  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.visualization import eda_charts as charts  # noqa: E402

logger = get_logger("run_eda")


def _md_table(df: pd.DataFrame, float_format: str = "{:,.2f}") -> str:
    """Render a small DataFrame as a Markdown table (no extra packages)."""
    def fmt(value):
        if isinstance(value, float):
            return float_format.format(value)
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    frame = df.reset_index()
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def _pct_table(df: pd.DataFrame, pct_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in pct_cols:
        if col in out:
            out[col] = out[col].map(lambda v: f"{v:.1%}")
    return out


def run(db_path: Path) -> int:
    settings.ensure_directories()
    tables_dir = settings.REPORTS_DIR / "eda"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the raw layer from SQLite
    try:
        with closing(get_connection(db_path)) as conn:
            raw = read_table(settings.RAW_TABLE, conn)
    except DatabaseError as exc:
        logger.error("%s", exc)
        return 1
    raw_missing = validate_customer_data(raw).missing_values

    # 2. Clean and store
    try:
        clean, log = clean_customer_data(raw)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    with closing(get_connection(db_path)) as conn:
        save_dataframe(clean, settings.CLEAN_TABLE, conn)
    clean.to_csv(settings.CLEAN_DATA_FILE, index=False)
    (settings.REPORTS_DIR / "phase2_cleaning_log.json").write_text(
        json.dumps(log.__dict__, indent=2, default=str), encoding="utf-8")

    # 3. Analyse
    overview = eda.overview(clean)
    numeric = eda.numeric_summary(clean)
    categorical = eda.categorical_summary(clean)
    cat_assoc = eda.categorical_association(clean)
    num_assoc = eda.numeric_association(clean)
    corr = eda.correlation_matrix(clean)
    banded = eda.add_tenure_band(clean)
    tenure = eda.churn_rate_by(banded, eda.TENURE_BAND)
    with_services = clean.assign(services_count=eda.services_count(clean))
    services = eda.churn_rate_by(with_services, "services_count").sort_index(
        key=lambda idx: idx.astype(int))
    findings = eda.key_findings(clean)

    for name, table in {"numeric_summary": numeric, "categorical_summary": categorical,
                        "categorical_association": cat_assoc,
                        "numeric_association": num_assoc, "correlation": corr,
                        "churn_by_tenure_band": tenure,
                        "churn_by_services_count": services}.items():
        table.to_csv(tables_dir / f"{name}.csv")

    # 4. Charts
    fig_dir = settings.FIGURES_DIR
    figures = {
        "01_missing_values": charts.plot_missing_values(raw_missing, len(raw)),
        "02_class_balance": charts.plot_class_balance(clean),
        "03_churn_drivers": charts.plot_churn_drivers(cat_assoc),
        "04_tenure_bands": charts.plot_tenure_band_churn(clean),
    }
    for i, feature in enumerate(cat_assoc.head(4).index, start=5):
        figures[f"{i:02d}_churn_by_{feature}"] = charts.plot_churn_rate_by(clean, feature)
    figures["09_churn_by_services_count"] = charts.plot_churn_rate_by(
        with_services, "services_count", "Churn rate by number of services subscribed",
        numeric_order=True)
    figures["10_tenure_distribution"] = charts.plot_numeric_by_churn(clean, "tenure", " months")
    figures["11_monthly_charges_distribution"] = charts.plot_numeric_by_churn(
        clean, "MonthlyCharges", " USD")
    figures["12_total_charges_distribution"] = charts.plot_numeric_by_churn(
        clean, "TotalCharges", " USD")
    figures["13_correlation"] = charts.plot_correlation_heatmap(corr)
    saved = [charts.save_figure(fig, fig_dir, name) for name, fig in figures.items()]

    # 5. Markdown report
    report = settings.REPORTS_DIR / "eda_report.md"
    cat_view = _pct_table(cat_assoc.drop(columns=["p_value"]),
                          ["highest_churn_rate", "lowest_churn_rate"])
    lines = [
        "# Customer360 AI – Exploratory Data Analysis report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} from table `{settings.CLEAN_TABLE}`. "
        "The dataset describes a fictional telecom company (IBM sample data)._",
        "",
        "## Overview",
        f"- Customers: **{overview['customers']:,}**",
        f"- Churned: **{overview['churned']:,}** ({overview['churn_rate']:.1%})",
        f"- Average tenure: **{overview['avg_tenure_months']:.1f} months**",
        f"- Average monthly charges: **${overview['avg_monthly_charges']:.2f}**",
        f"- Monthly charges of churned customers: **${overview['monthly_charges_of_churned']:,.0f}**"
        " (what those customers were paying per month before leaving)",
        "",
        "## Key findings",
        *[f"{i}. {text}" for i, text in enumerate(findings, start=1)],
        "",
        "## Cleaning steps",
        *[f"- {step}" for step in log.steps],
        "",
        "## Number columns",
        _md_table(numeric),
        "",
        "## Churned vs retained (number columns)",
        "Effect size runs from -1 to +1; positive means churned customers have higher values.",
        "",
        _md_table(num_assoc.drop(columns=["p_value"])),
        "",
        "## Association of each characteristic with churn",
        "Cramér's V: under 0.1 negligible, 0.1–0.2 weak, 0.2–0.3 moderate, 0.3+ strong.",
        "",
        _md_table(cat_view, "{:.3f}"),
        "",
        "## Churn by tenure band",
        _md_table(_pct_table(tenure, ["churn_rate", "share_of_customers"])),
        "",
        "## Charts",
        *[f"![{p.stem}](figures/{p.name})" for p in saved],
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    # 6. Terminal summary
    print("\n=== Phase 2 summary ===")
    print(f"  Clean rows: {log.rows_after:,} (from {log.rows_before:,})")
    print(f"  Saved table '{settings.CLEAN_TABLE}' and {settings.CLEAN_DATA_FILE.name}")
    print(f"  Charts: {len(saved)} in {fig_dir}")
    print(f"  Report: {report}\n")
    print("  Key findings:")
    for i, text in enumerate(findings, start=1):
        print(f"   {i}. {text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=settings.DB_PATH)
    args = parser.parse_args(argv)
    return run(args.db)


if __name__ == "__main__":
    sys.exit(main())
