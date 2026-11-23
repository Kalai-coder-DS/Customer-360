"""Phase 5 pipeline: explain the churn model globally and for each customer.

Usage:
    python scripts/run_explain.py

Reads   : SQLite 'customers_clean' and 'churn_predictions', models/churn_model.joblib
Writes  : SQLite table 'churn_explanations', reports/explain_report.md,
          reports/explain/*.csv, reports/figures/explain_*.png
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow project imports

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
from src.data.schema import ID_COLUMN  # noqa: E402
from src.database.db import DatabaseError, get_connection, read_table, save_dataframe  # noqa: E402
from src.explain import explainer as ex  # noqa: E402
from src.models.features import prepare_xy  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.visualization import eda_charts, explain_charts  # noqa: E402

logger = get_logger("run_explain")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eda import _md_table  # noqa: E402


def _factors_text(factors: list[dict]) -> str:
    return "; ".join(f"{f['label']} ({f['strength']})" for f in factors)


def run(db_path: Path) -> int:
    settings.ensure_directories()
    out_dir = settings.REPORTS_DIR / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not settings.CHURN_MODEL_FILE.exists():
        logger.error("No churn model found. Run 'python scripts/run_modeling.py' first.")
        return 1
    try:
        with closing(get_connection(db_path)) as conn:
            clean = read_table(settings.CLEAN_TABLE, conn)
            preds = read_table(settings.PREDICTIONS_TABLE, conn)
    except DatabaseError as exc:
        logger.error("%s Run the earlier phases first.", exc)
        return 1

    bundle = joblib.load(settings.CHURN_MODEL_FILE)
    pipeline, model_label = bundle["pipeline"], bundle["model_label"]
    X, y = prepare_xy(clean)
    X.index = clean[ID_COLUMN].values
    y.index = X.index
    preds = preds.set_index(ID_COLUMN).reindex(X.index)
    background = X.sample(min(settings.EXPLAIN_BACKGROUND_SIZE, len(X)),
                          random_state=settings.RANDOM_STATE)

    # 1. Global: what does the model rely on?
    importance = ex.global_importance(pipeline, X, y)
    importance.to_csv(out_dir / "global_importance.csv")
    coefficients = ex.logistic_coefficients(pipeline)
    if coefficients is not None:
        coefficients.to_csv(out_dir / "logistic_coefficients.csv", index=False)

    # 2. Local: why does each customer get their score?
    if ex.is_linear(pipeline):
        to_explain = X
    else:
        riskiest = preds["churn_probability"].sort_values(ascending=False).index
        to_explain = X.loc[riskiest[: settings.SLOW_EXPLAIN_MAX_CUSTOMERS]]
        logger.info("Non-linear model: explaining the %s riskiest customers "
                    "(others can be explained on demand in the dashboard)", len(to_explain))
    contributions, bases, method = ex.explain_customers(pipeline, to_explain, background)
    logger.info("Explained %s customers with %s", len(contributions), method)

    rows = []
    for cid in contributions.index:
        prob = float(preds.loc[cid, "churn_probability"])
        risk = str(preds.loc[cid, "risk_level"])
        raising, lowering = ex.top_factors(contributions.loc[cid], X.loc[cid])
        rows.append({
            ID_COLUMN: cid, "churn_probability": prob, "risk_level": risk,
            "base_probability": round(float(ex.sigmoid(bases.loc[cid])), 4),
            "top_risk_factors": _factors_text(raising),
            "top_protective_factors": _factors_text(lowering),
            "explanation": ex.explanation_text(prob, risk, model_label, raising, lowering),
            "contributions_json": ex.contributions_json(contributions.loc[cid]),
        })
    explanations = pd.DataFrame(rows)
    with closing(get_connection(db_path)) as conn:
        save_dataframe(explanations, settings.EXPLANATIONS_TABLE, conn)

    # 3. Direction of the most important features
    top_features = list(importance.index[:4])
    directions = {}
    for feature in top_features:
        directions[feature] = ex.contribution_direction(
            contributions, X.loc[contributions.index], feature)
        directions[feature].to_csv(out_dir / f"direction_{feature}.csv")

    # 4. Example customers: highest, middle and lowest predicted risk
    probs = preds["churn_probability"]
    examples = {"highest": probs.idxmax(),
                "middle": (probs - 0.45).abs().idxmin(),
                "lowest": probs.idxmin()}
    example_data = {}
    for key, cid in examples.items():
        if cid in contributions.index:
            contrib, base = contributions.loc[cid], float(bases.loc[cid])
        else:  # non-linear model, customer outside the precomputed set
            contrib, base = ex.sampling_contributions(pipeline, X.loc[cid], background)
        raising, lowering = ex.top_factors(contrib, X.loc[cid])
        prob, risk = float(probs.loc[cid]), str(preds.loc[cid, "risk_level"])
        example_data[key] = (cid, contrib, base, prob, risk,
                             ex.explanation_text(prob, risk, model_label, raising, lowering))

    # 5. Charts
    figures = {"explain_01_global_importance": explain_charts.plot_global_importance(importance)}
    for i, (key, (cid, contrib, base, prob, risk, _)) in enumerate(example_data.items(), start=2):
        figures[f"explain_{i:02d}_customer_{key}_risk"] = explain_charts.plot_customer_explanation(
            contrib, X.loc[cid], prob, risk, base, cid)
    for i, feature in enumerate(top_features, start=5):
        figures[f"explain_{i:02d}_direction_{feature}"] = explain_charts.plot_feature_direction(
            directions[feature], feature)
    saved = [eda_charts.save_figure(fig, settings.FIGURES_DIR, name)
             for name, fig in figures.items()]

    # 6. Report
    imp_view = importance[["label", "importance", "spread"]].head(12).set_index("label")
    mean_abs = ex.mean_abs_contribution(contributions).head(12).rename("avg_size").to_frame()
    mean_abs.index = [ex.FEATURE_LABELS.get(f, f) for f in mean_abs.index]
    lines = [
        "# Customer360 AI – Explainable AI report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Model: {model_label}. "
        "Fictional telecom data (IBM sample)._",
        "",
        "## How to read this report",
        f"- **Method:** {method}, the same idea behind SHAP. Each factor's contribution "
        "shows how much it pushes this customer's churn risk up or down compared with an "
        "average customer. Contributions are in log-odds and add up exactly to the "
        "prediction.",
        "- **Global importance:** permutation importance (how much ROC-AUC drops when a "
        "feature is shuffled).",
        "- **Correlation is not causation.** These explanations describe what the model "
        "learned to associate with churn in historical data. They suggest where to look, "
        "not what will definitely work. Testing an action (e.g. a contract offer to some "
        "customers and not others) is how a business finds out what causes change.",
        "",
        "## What the model relies on most",
        _md_table(imp_view, "{:.4f}"),
        "",
        f"## Average size of each factor's push (across {len(contributions):,} explained "
        "customers)",
        _md_table(mean_abs, "{:.3f}"),
        "",
        "## How the top features move predictions",
        *[block for f in top_features for block in (
            f"### {ex.FEATURE_LABELS.get(f, f)}", _md_table(directions[f], "{:+.3f}"), "")],
        "## Example customer explanations",
        *[block for key, (cid, *_rest, text) in example_data.items() for block in (
            f"### {key.capitalize()} predicted risk: customer {cid}", text, "")],
        "## Charts",
        *[f"![{p.stem}](figures/{p.name})" for p in saved],
        "",
    ]
    report = settings.REPORTS_DIR / "explain_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    # 7. Terminal summary
    print("\n=== Phase 5 summary ===")
    print(f"  Model: {model_label}   Method: {method}")
    print(f"  Customers explained: {len(contributions):,}")
    print("  Features the model relies on most (drop in ROC-AUC when shuffled):")
    for _, row in importance.head(8).iterrows():
        print(f"   - {row['label']:<24} {row['importance']:.4f}")
    cid, *_rest, text = example_data["highest"]
    print(f"\n  Example (highest-risk customer {cid}):\n   {text}")
    print(f"\n  Saved table '{settings.EXPLANATIONS_TABLE}'")
    print(f"  Charts: {len(saved)}   Report: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=settings.DB_PATH)
    args = parser.parse_args(argv)
    return run(args.db)


if __name__ == "__main__":
    sys.exit(main())
