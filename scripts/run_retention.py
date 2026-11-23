"""Phase 6 pipeline: prioritise at-risk customers, recommend actions, estimate impact.

Usage:
    python scripts/run_retention.py

Reads   : SQLite 'customers_clean', 'churn_predictions', and (if present)
          'churn_explanations' and 'customer_segments'
Writes  : SQLite table 'retention_actions', reports/retention_report.md,
          reports/retention/*.csv (incl. priority call list), reports/figures/retention_*.png
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow project imports

from config import settings  # noqa: E402
from src.database.db import (  # noqa: E402
    DatabaseError, get_connection, list_tables, read_table, save_dataframe,
)
from src.retention import planner  # noqa: E402
from src.retention.playbook import playbook_table  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.visualization import eda_charts, retention_charts  # noqa: E402

logger = get_logger("run_retention")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eda import _md_table, _pct_table  # noqa: E402


def run(db_path: Path) -> int:
    settings.ensure_directories()
    out_dir = settings.REPORTS_DIR / "retention"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with closing(get_connection(db_path)) as conn:
            clean = read_table(settings.CLEAN_TABLE, conn)
            preds = read_table(settings.PREDICTIONS_TABLE, conn)
            tables = list_tables(conn)
            explanations = (read_table(settings.EXPLANATIONS_TABLE, conn)
                            if settings.EXPLANATIONS_TABLE in tables else None)
            segments = (read_table(settings.SEGMENTS_TABLE, conn)
                        if settings.SEGMENTS_TABLE in tables else None)
    except DatabaseError as exc:
        logger.error("%s Run 'python scripts/run_modeling.py' first.", exc)
        return 1
    if explanations is None:
        logger.warning("No explanations found; actions will be based on customer attributes. "
                       "Run 'python scripts/run_explain.py' for model-based reasons.")

    plan = planner.build_retention_plan(clean, preds, explanations, segments)
    with closing(get_connection(db_path)) as conn:
        save_dataframe(plan, settings.RETENTION_TABLE, conn)

    cutoff = planner.value_cutoff(clean)
    tiers = planner.tier_summary(plan)
    scenarios = planner.impact_scenarios(plan)
    counts = planner.action_counts(plan)
    by_segment = planner.summary_by(plan, "segment_name") if "segment_name" in plan else None
    call_list = plan[plan["priority_tier"] != planner.TIER_ORDER[3]]

    call_list.to_csv(out_dir / "priority_call_list.csv", index=False)
    tiers.to_csv(out_dir / "tier_summary.csv")
    scenarios.to_csv(out_dir / "impact_scenarios.csv")
    playbook_table().to_csv(out_dir / "playbook_assumptions.csv")
    if by_segment is not None:
        by_segment.to_csv(out_dir / "value_at_risk_by_segment.csv")

    figures = {
        "retention_01_risk_value_matrix": retention_charts.plot_risk_value_matrix(plan, cutoff),
        "retention_02_actions": retention_charts.plot_action_counts(counts),
        "retention_03_impact_scenarios": retention_charts.plot_impact_scenarios(scenarios),
    }
    if by_segment is not None:
        figures["retention_04_value_at_risk_by_segment"] = (
            retention_charts.plot_expected_loss_by(by_segment, "segment"))
    saved = [eda_charts.save_figure(fig, settings.FIGURES_DIR, name)
             for name, fig in figures.items()]

    horizon = settings.RETENTION_HORIZON_MONTHS
    base = scenarios.xs("Base", level="scenario")
    total_at_risk = float(plan["expected_loss_horizon"].sum())
    top10 = call_list.head(10)[["customerID", "priority_tier", "churn_probability",
                                "MonthlyCharges", "expected_loss_horizon", "action_1",
                                "action_2"]].set_index("customerID")
    lines = [
        "# Customer360 AI – Retention intelligence report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Fictional telecom data (IBM sample)._",
        "",
        "> **All money figures below are estimates built on stated assumptions** "
        "(costs, success rates, a "
        f"{horizon}-month horizon, constant monthly charges). They show the *size* of the "
        "opportunity and how to compare options. They are not forecasts or measured results.",
        "",
        "## Headline",
        f"- Expected revenue at risk over {horizon} months across all customers: "
        f"**${total_at_risk:,.0f}** (sum of churn probability × monthly charges × {horizon}).",
        f"- **{int(tiers.loc[planner.TIER_ORDER[0], 'customers']):,}** customers are in "
        f"'{planner.TIER_ORDER[0]}' (high risk and monthly charges ≥ ${cutoff:.0f}).",
        "- Base-case estimate for contacting 'Act now + Act soon': "
        f"~{base.loc['Act now + Act soon', 'customers_saved']:.0f} customers saved, "
        f"${base.loc['Act now + Act soon', 'revenue_protected']:,.0f} revenue protected for "
        f"${base.loc['Act now + Act soon', 'campaign_cost']:,.0f} campaign cost.",
        "",
        "## Priority tiers",
        _md_table(_pct_table(tiers, ["avg_churn_probability"]), "{:,.2f}"),
        "",
        "## Top 10 customers to contact",
        _md_table(_pct_table(top10, ["churn_probability"]), "{:,.2f}"),
        "",
        "Full list: `reports/retention/priority_call_list.csv`.",
        "",
        "## Impact scenarios",
        f"Scenario multipliers on the assumed success rates: {settings.SCENARIO_MULTIPLIERS}.",
        "",
        _md_table(scenarios, "{:,.2f}"),
        "",
        "## Playbook and assumptions",
        _md_table(_pct_table(playbook_table(), ["assumed_success_rate"]), "{:,.2f}"),
        "",
        "## How actions are chosen",
        "- An action is suggested only if it applies to the customer (e.g. a contract offer "
        "only for month-to-month customers) **and** the related factor raises that "
        "customer's churn risk in the Phase 5 explanation.",
        "- Two actions at most per customer, strongest driver first. Combined success "
        "assumes the actions work independently.",
        "- Customers at Low risk are not contacted.",
        "",
        "## Limitations",
        "- Success rates and costs are illustrative; real values should come from "
        "controlled experiments (offer to a random half, compare churn).",
        "- Predictions show association, not cause, so an action that targets a risk factor "
        "may not change behaviour.",
        "- Contacting customers can itself have effects (e.g. reminding some to leave), "
        "which this simple model does not capture.",
        "",
        "## Charts",
        *[f"![{p.stem}](figures/{p.name})" for p in saved],
        "",
    ]
    if by_segment is not None:
        lines[lines.index("## Impact scenarios"):lines.index("## Impact scenarios")] = [
            "## Revenue at risk by segment",
            _md_table(_pct_table(by_segment, ["avg_churn_probability",
                                              "share_of_expected_loss"]), "{:,.2f}"),
            "",
        ]
    report = settings.REPORTS_DIR / "retention_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Phase 6 summary ===")
    print(f"  Expected revenue at risk ({horizon} months): ${total_at_risk:,.0f}  "
          "(estimate, see assumptions)")
    print("  Priority tiers:")
    for tier, row in tiers.iterrows():
        print(f"   - {tier:<14} {int(row['customers']):>5,} customers  "
              f"avg risk {row['avg_churn_probability']:.0%}  "
              f"value at risk ${row['expected_loss_horizon']:,.0f}")
    print("  Most recommended actions:")
    for name, count in counts.head(5).items():
        print(f"   - {name:<32} {count:>5,}")
    print("  Base-case impact (illustrative assumptions):")
    for target, row in base.iterrows():
        print(f"   - {target:<26} saved ~{row['customers_saved']:.0f}  protected "
              f"${row['revenue_protected']:,.0f}  cost ${row['campaign_cost']:,.0f}  "
              f"net ${row['net_value']:,.0f}")
    if len(call_list):
        print(f"\n  #1 priority: {call_list.iloc[0]['recommendation']}")
    print(f"\n  Saved table '{settings.RETENTION_TABLE}' and priority_call_list.csv")
    print(f"  Charts: {len(saved)}   Report: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=settings.DB_PATH)
    args = parser.parse_args(argv)
    return run(args.db)


if __name__ == "__main__":
    sys.exit(main())
