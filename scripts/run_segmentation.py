"""Phase 3 pipeline: RFM-style scoring and K-Means customer segmentation.

Usage:
    python scripts/run_segmentation.py            # K chosen automatically (3-6)
    python scripts/run_segmentation.py --k 4      # force a number of segments

Reads   : SQLite table 'customers_clean' (created by scripts/run_eda.py)
Writes  : SQLite table 'customer_segments', models/segmentation_kmeans.joblib,
          reports/segmentation_report.md, reports/segmentation/*.csv,
          reports/figures/segmentation_*.png
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow project imports

import joblib  # noqa: E402

from config import settings  # noqa: E402
from src.data.cleaning import CHURN_FLAG  # noqa: E402
from src.data.schema import ID_COLUMN  # noqa: E402
from src.database.db import DatabaseError, get_connection, read_table, save_dataframe  # noqa: E402
from src.segmentation import clustering, rfm  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.visualization import eda_charts, segment_charts  # noqa: E402

logger = get_logger("run_segmentation")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eda import _md_table, _pct_table  # noqa: E402  (shared report helpers)


def run(db_path: Path, k: int | None) -> int:
    settings.ensure_directories()
    out_dir = settings.REPORTS_DIR / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load clean data
    try:
        with closing(get_connection(db_path)) as conn:
            df = read_table(settings.CLEAN_TABLE, conn)
    except DatabaseError as exc:
        logger.error("%s Run 'python scripts/run_eda.py' first.", exc)
        return 1
    if k is not None and not 2 <= k < len(df):
        logger.error("--k must be between 2 and the number of customers - 1.")
        return 1

    # 2. RFM-style scores
    scores = rfm.compute_rfm_scores(df)
    scores[CHURN_FLAG] = df[CHURN_FLAG].values
    rfm_table = scores.groupby("rfm_segment").agg(
        customers=(ID_COLUMN, "size"), churn_rate=(CHURN_FLAG, "mean"),
        avg_tenure=("tenure", "mean"), avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_services=("services_count", "mean"),
    ).reindex([s for s in rfm.RFM_SEGMENT_ORDER if s in set(scores["rfm_segment"])])
    rfm_table["share_of_customers"] = rfm_table["customers"] / rfm_table["customers"].sum()
    rfm_table["description"] = [rfm.RFM_SEGMENT_DESCRIPTIONS[s] for s in rfm_table.index]

    # 3. K-Means
    result = clustering.fit_segments(df, k)
    named = clustering.name_segments(clustering.profile_segments(df, result.labels), df)
    names = named["segment_name"].to_dict()
    logger.info("K-Means chose K=%s (silhouette %.3f)", result.k,
                result.k_scores.loc[result.k, "silhouette"])

    # 4. Store: one row per customer
    segments = scores.drop(columns=[CHURN_FLAG]).assign(
        cluster=result.labels.values,
        segment_name=result.labels.map(names).values,
    )
    with closing(get_connection(db_path)) as conn:
        save_dataframe(segments, settings.SEGMENTS_TABLE, conn)
    joblib.dump({"pipeline": result.pipeline, "features": clustering.CLUSTER_FEATURES,
                 "cluster_order": result.pipeline.cluster_order_, "segment_names": names},
                settings.SEGMENT_MODEL_FILE)
    result.k_scores.to_csv(out_dir / "k_selection.csv")
    named.to_csv(out_dir / "cluster_profiles.csv")
    rfm_table.to_csv(out_dir / "rfm_segments.csv")

    # 5. Charts
    fig_dir = settings.FIGURES_DIR
    labelled = df.assign(segment_name=segments["segment_name"].values,
                         rfm_segment=segments["rfm_segment"].values)
    figures = {
        "segmentation_01_k_selection": segment_charts.plot_k_selection(result.k_scores, result.k),
        "segmentation_02_customers_vs_revenue": segment_charts.plot_customers_vs_revenue(named),
        "segmentation_03_segment_map": segment_charts.plot_segment_map(df, result.labels, names),
        "segmentation_04_profiles": segment_charts.plot_segment_profile(named),
        "segmentation_05_churn_by_segment": eda_charts.plot_churn_rate_by(
            labelled, "segment_name", "Churn rate by K-Means segment"),
        "segmentation_06_churn_by_rfm": eda_charts.plot_churn_rate_by(
            labelled, "rfm_segment", "Churn rate by RFM-style segment"),
    }
    saved = [eda_charts.save_figure(fig, fig_dir, name) for name, fig in figures.items()]

    # 6. Report
    overall = float(df[CHURN_FLAG].mean())
    profile_view = _pct_table(
        named[["segment_name", "customers", "share_of_customers", "avg_tenure",
               "avg_monthly_charges", "avg_services", "share_month_to_month", "churn_rate",
               "share_of_revenue"]],
        ["share_of_customers", "share_month_to_month", "churn_rate", "share_of_revenue"])
    lines = [
        "# Customer360 AI – Customer segmentation report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Fictional telecom data (IBM sample); "
        "suggested actions are hypotheses to test, not proven results._",
        "",
        "## Approach",
        "- **RFM-style scores (1–5):** Loyalty = tenure, Engagement = number of services, "
        "Monetary = monthly charges. True RFM needs purchase dates, which this dataset does "
        "not have, so these are labelled substitutes.",
        f"- **K-Means** on tenure, monthly charges and number of services (standardised). "
        f"K = **{result.k}**, chosen by the best silhouette score among 3–6 segments "
        f"(silhouette {result.k_scores.loc[result.k, 'silhouette']:.3f}).",
        "- Churn was **not** used to build segments; it is only used to describe them.",
        f"- Overall churn rate for comparison: **{overall:.1%}**.",
        "",
        "## K-Means segments",
        _md_table(profile_view.set_index("segment_name")),
        "",
        *[line for _, row in named.iterrows() for line in (
            f"### {row['segment_name']}",
            row["description"],
            f"**Suggested action:** {row['suggested_action']}",
            "",
        )],
        "## RFM-style segments",
        _md_table(_pct_table(rfm_table, ["churn_rate", "share_of_customers"])),
        "",
        "## Choosing K",
        _md_table(result.k_scores, "{:,.3f}"),
        "",
        "## Charts",
        *[f"![{p.stem}](figures/{p.name})" for p in saved],
        "",
    ]
    report = settings.REPORTS_DIR / "segmentation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    # 7. Terminal summary
    print("\n=== Phase 3 summary ===")
    print(f"  K-Means segments: {result.k}  (silhouette "
          f"{result.k_scores.loc[result.k, 'silhouette']:.3f}; overall churn {overall:.1%})")
    for _, row in named.iterrows():
        print(f"   - {row['segment_name']:<36} {int(row['customers']):>5,} customers  "
              f"churn {row['churn_rate']:>6.1%}  revenue share {row['share_of_revenue']:>4.0%}")
    print("  RFM-style segments:")
    for name, row in rfm_table.iterrows():
        print(f"   - {name:<36} {int(row['customers']):>5,} customers  "
              f"churn {row['churn_rate']:>6.1%}")
    print(f"\n  Saved table '{settings.SEGMENTS_TABLE}', model {settings.SEGMENT_MODEL_FILE.name}")
    print(f"  Charts: {len(saved)}   Report: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=settings.DB_PATH)
    parser.add_argument("--k", type=int, default=None, help="force a number of segments")
    args = parser.parse_args(argv)
    return run(args.db, args.k)


if __name__ == "__main__":
    sys.exit(main())
