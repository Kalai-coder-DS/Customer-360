"""Write docs/RESULTS.md: the project's key numbers, taken from YOUR pipeline outputs.

Every figure is read from the database and the reports folder, so nothing is
typed in by hand. Use this page for your README, CV and interview answers.

Usage:
    python scripts/run_all.py          # (once) produce the outputs
    python scripts/build_results.py    # write docs/RESULTS.md
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
from src.database.db import DatabaseError, get_connection, read_table  # noqa: E402

DEFAULT_OUT = settings.PROJECT_ROOT / "docs" / "RESULTS.md"


def _csv(relative: str, **kwargs) -> pd.DataFrame | None:
    path = settings.REPORTS_DIR / relative
    return pd.read_csv(path, **kwargs) if path.exists() else None


def _json(relative: str) -> dict | None:
    path = settings.REPORTS_DIR / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build(db_path: Path) -> str:
    with closing(get_connection(db_path)) as conn:
        clean = read_table(settings.CLEAN_TABLE, conn)
    n, churned = len(clean), int(clean["churn_flag"].sum())
    lines = [
        "# Customer360 AI: key results",
        "",
        f"_Generated {datetime.now():%Y-%m-%d} by `scripts/build_results.py` from the pipeline "
        "outputs. Every number below comes from the data; none are typed in by hand. "
        "Dataset: IBM Telco Customer Churn (fictional telecom company)._",
        "",
        "## 1. The customer base",
        f"- **{n:,} customers**, of whom **{churned:,} churned ({churned / n:.1%})**.",
        f"- Average monthly charge **${clean['MonthlyCharges'].mean():.2f}**; average tenure "
        f"**{clean['tenure'].mean():.0f} months**.",
        f"- Monthly revenue of customers who left: "
        f"**${clean.loc[clean['churn_flag'] == 1, 'MonthlyCharges'].sum():,.0f}**.",
        f"- A model that always predicts 'no churn' would be {1 - churned / n:.1%} accurate, "
        "which is why accuracy alone is not used.",
    ]

    assoc = _csv("eda/categorical_association.csv", index_col=0)
    bands = _csv("eda/churn_by_tenure_band.csv", index_col=0)
    if assoc is not None:
        lines += ["", "## 2. What goes together with churn (EDA)"]
        for feature, row in assoc.head(3).iterrows():
            lines.append(f"- **{feature}** (Cramér's V {row['cramers_v']:.2f}, {row['strength']}): "
                         f"'{row['highest_churn_group']}' churn {row['highest_churn_rate']:.1%} vs "
                         f"'{row['lowest_churn_group']}' {row['lowest_churn_rate']:.1%}.")
    if bands is not None and len(bands) >= 2:
        lines.append(f"- Tenure: {bands['churn_rate'].iloc[0]:.1%} churn in {bands.index[0]} vs "
                     f"{bands['churn_rate'].iloc[-1]:.1%} for {bands.index[-1]}.")

    profiles = _csv("segmentation/cluster_profiles.csv")
    k_scores = _csv("segmentation/k_selection.csv", index_col=0)
    rfm = _csv("segmentation/rfm_segments.csv", index_col=0)
    if profiles is not None:
        k = len(profiles)
        sil = f", silhouette {k_scores.loc[k, 'silhouette']:.3f}" if k_scores is not None else ""
        lines += ["", f"## 3. Customer segments (K-Means, K = {k}{sil})", "",
                  "| Segment | Customers | Churn rate | Revenue share |", "|---|---|---|---|"]
        for _, row in profiles.sort_values("churn_rate", ascending=False).iterrows():
            lines.append(f"| {row['segment_name']} | {int(row['customers']):,} | "
                         f"{row['churn_rate']:.1%} | {row['share_of_revenue']:.0%} |")
    if rfm is not None:
        worst = rfm.sort_values("churn_rate", ascending=False).iloc[0]
        lines.append(f"\nHighest-churn RFM-style segment: **{rfm.sort_values('churn_rate').index[-1]}**"
                     f" ({int(worst['customers']):,} customers, {worst['churn_rate']:.1%} churn).")

    summary = _json("modeling/summary.json")
    cv = _csv("modeling/cv_scores.csv", index_col=0)
    if summary is not None:
        m, split = summary["test_metrics"], summary["split"]
        labels = {"baseline": "Baseline", "logistic_regression": "Logistic Regression",
                  "random_forest": "Random Forest"}
        lines += ["", "## 4. Churn prediction",
                  f"- Chosen model: **{labels.get(summary['chosen_model'], summary['chosen_model'])}** "
                  "(best cross-validated ROC-AUC).",
                  f"- Test set ({split['test']:,} unseen customers): **ROC-AUC {m['roc_auc']:.3f}**, "
                  f"PR-AUC {m['pr_auc']:.3f}, recall {m['recall']:.1%}, precision "
                  f"{m['precision']:.1%}, F1 {m['f1']:.3f}, accuracy {m['accuracy']:.1%} "
                  f"(threshold {summary['threshold']:.2f}).",
                  f"- Overfitting check: training ROC-AUC {summary['overfit_check']['train_roc_auc']:.3f}"
                  f" vs test {summary['overfit_check']['test_roc_auc']:.3f}."]
        if cv is not None:
            lines += ["", "| Model | CV ROC-AUC (mean ± std) |", "|---|---|"]
            for name, row in cv.iterrows():
                lines.append(f"| {labels.get(name, name)} | {row['roc_auc_mean']:.3f} ± "
                             f"{row['roc_auc_std']:.3f} |")

    importance = _csv("explain/global_importance.csv", index_col=0)
    if importance is not None:
        lines += ["", "## 5. What the model relies on (permutation importance)"]
        for i, (_, row) in enumerate(importance.head(5).iterrows(), start=1):
            lines.append(f"{i}. **{row['label']}** (ROC-AUC drop {row['importance']:.4f})")
        lines.append("\n_Associations learned from past data, not proven causes._")

    tiers = _csv("retention/tier_summary.csv", index_col=0)
    scenarios = _csv("retention/impact_scenarios.csv")
    if tiers is not None:
        lines += ["", "## 6. Retention plan (estimates under stated assumptions)",
                  f"- Revenue at risk over {settings.RETENTION_HORIZON_MONTHS} months: "
                  f"**${tiers['expected_loss_horizon'].sum():,.0f}**."]
        for tier, row in tiers.iterrows():
            lines.append(f"- {tier}: {int(row['customers']):,} customers")
        if scenarios is not None:
            base = scenarios[(scenarios["scenario"] == "Base")
                             & (scenarios["target"] == "Act now + Act soon")]
            if len(base):
                b = base.iloc[0]
                lines.append(f"- Base case for contacting 'Act now + Act soon': ~"
                             f"{b['customers_saved']:.0f} customers saved, ${b['revenue_protected']:,.0f} "
                             f"protected for ${b['campaign_cost']:,.0f} (illustrative costs and "
                             "success rates).")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=settings.DB_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        text = build(args.db)
    except DatabaseError as exc:
        print(f"{exc}\nRun 'python scripts/run_all.py' first.")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
