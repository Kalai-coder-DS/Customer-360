"""Phase 4 pipeline: train, compare and evaluate churn prediction models.

Usage:
    python scripts/run_modeling.py

Reads   : SQLite table 'customers_clean' (created by scripts/run_eda.py)
Writes  : models/churn_model.joblib, SQLite table 'churn_predictions',
          reports/model_report.md, reports/modeling/*, reports/figures/model_*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow project imports

import joblib  # noqa: E402
import sklearn  # noqa: E402

from config import settings  # noqa: E402
from src.database.db import DatabaseError, get_connection, read_table, save_dataframe  # noqa: E402
from src.models import features  # noqa: E402
from src.models.train import MODEL_LABELS, train_and_evaluate  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.visualization import eda_charts, model_charts  # noqa: E402

logger = get_logger("run_modeling")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eda import _md_table, _pct_table  # noqa: E402


def run(db_path: Path) -> int:
    settings.ensure_directories()
    out_dir = settings.REPORTS_DIR / "modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with closing(get_connection(db_path)) as conn:
            df = read_table(settings.CLEAN_TABLE, conn)
    except DatabaseError as exc:
        logger.error("%s Run 'python scripts/run_eda.py' first.", exc)
        return 1
    if df["Churn"].nunique() < 2 or len(df) < 50:
        logger.error("Need at least 50 customers with both churned and retained examples.")
        return 1

    result = train_and_evaluate(df)
    chosen, thr = result.chosen, result.threshold
    label = MODEL_LABELS[chosen]
    tuned = result.test_metrics.xs("tuned", level="threshold_type")
    best = tuned.loc[chosen]

    # Save the model bundle (everything needed to score new customers later)
    bundle = {
        "pipeline": result.final_pipeline, "model_name": chosen, "model_label": label,
        "threshold": thr, "risk_thresholds": settings.RISK_THRESHOLDS,
        "numeric_features": features.NUMERIC_FEATURES,
        "categorical_features": features.CATEGORICAL_FEATURES,
        "test_metrics": best.to_dict(), "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_customers": len(df), "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, settings.CHURN_MODEL_FILE)
    with closing(get_connection(db_path)) as conn:
        save_dataframe(result.predictions, settings.PREDICTIONS_TABLE, conn)

    result.cv_scores.to_csv(out_dir / "cv_scores.csv")
    result.test_metrics.to_csv(out_dir / "test_metrics.csv")
    result.confusion.to_csv(out_dir / "confusion_matrix.csv")
    result.risk_table.to_csv(out_dir / "risk_levels_test.csv")
    (out_dir / "summary.json").write_text(json.dumps({
        "chosen_model": chosen, "threshold": thr, "thresholds": result.thresholds,
        "split": result.split, "overfit_check": result.overfit_check,
        "test_metrics": best.to_dict(),
    }, indent=2), encoding="utf-8")

    # Charts
    churn_rate = result.split["test_churn_rate"]
    figures = {
        "model_01_cv_scores": model_charts.plot_cv_scores(result.cv_scores),
        "model_02_comparison": model_charts.plot_model_comparison(result.test_metrics),
        "model_03_roc": model_charts.plot_roc_curves(result.curves, result.test_metrics),
        "model_04_precision_recall": model_charts.plot_pr_curves(
            result.curves, result.test_metrics, churn_rate),
        "model_05_confusion": model_charts.plot_confusion(result.confusion, label, thr),
        "model_06_calibration": model_charts.plot_calibration(
            result.curves[chosen]["calibration"], label),
        "model_07_risk_levels": model_charts.plot_risk_levels(result.risk_table),
    }
    saved = [eda_charts.save_figure(fig, settings.FIGURES_DIR, name)
             for name, fig in figures.items()]

    # Report
    cm = result.confusion
    caught, missed = int(cm.iloc[1, 1]), int(cm.iloc[1, 0])
    false_alarms = int(cm.iloc[0, 1])
    cv_view = result.cv_scores[[c for c in result.cv_scores.columns if c.endswith("_mean")]]
    cv_view.columns = [c.replace("_mean", "") for c in cv_view.columns]
    cv_view.index = [MODEL_LABELS[i] for i in cv_view.index]
    test_view = result.test_metrics.copy()
    test_view.index = [f"{MODEL_LABELS[m]} ({t})" for m, t in test_view.index]
    overfit = result.overfit_check
    lines = [
        "# Customer360 AI – Churn prediction report",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}. Fictional telecom data (IBM sample)._",
        "",
        "## Result in plain words",
        f"- The chosen model is **{label}** (best cross-validated ROC-AUC).",
        f"- On {result.split['test']:,} customers it never saw, it ranks churners above "
        f"non-churners with ROC-AUC **{best['roc_auc']:.3f}** (0.5 = guessing).",
        f"- With the decision threshold at **{thr:.2f}**, it caught **{caught:,}** of "
        f"{caught + missed:,} churners (recall {best['recall']:.0%}); "
        f"{best['precision']:.0%} of the customers it flagged really churned "
        f"({false_alarms:,} false alarms).",
        f"- Accuracy is {best['accuracy']:.1%}, but remember a model that predicts "
        f"'no churn' for everyone would score {1 - churn_rate:.1%}; accuracy alone is "
        "not a fair measure here.",
        f"- Overfitting check: ROC-AUC {overfit['train_roc_auc']:.3f} on training data vs "
        f"{overfit['test_roc_auc']:.3f} on test data (gap {overfit['gap']:.3f}).",
        "",
        "## Method",
        f"- Stratified split: {result.split['train']:,} training / {result.split['test']:,} "
        f"test customers (churn rate {result.split['train_churn_rate']:.1%} / "
        f"{result.split['test_churn_rate']:.1%}).",
        f"- {settings.CV_FOLDS}-fold stratified cross-validation on the training set only.",
        "- Preprocessing (median fill, scaling, one-hot encoding) inside the pipeline, "
        "learned from training folds only, to prevent data leakage.",
        "- Engineered features: " + "; ".join(
            f"`{k}`: {v}" for k, v in features.ENGINEERED_DESCRIPTIONS.items()),
        "- Model chosen by cross-validated ROC-AUC; threshold chosen by best F1 on "
        "out-of-fold training predictions. The test set was used once, at the end.",
        "- Imbalance handled with the threshold (not class re-weighting), so probabilities "
        "stay meaningful.",
        "",
        "## Cross-validation (training data, mean of 5 folds)",
        _md_table(cv_view, "{:.3f}"),
        "",
        "## Test-set results",
        _md_table(test_view, "{:.3f}"),
        "",
        "## Risk levels on the test set",
        _md_table(_pct_table(result.risk_table, ["actual_churn_rate", "share"])),
        "",
        "## Charts",
        *[f"![{p.stem}](figures/{p.name})" for p in saved],
        "",
    ]
    report = settings.REPORTS_DIR / "model_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    # Terminal summary
    print("\n=== Phase 4 summary ===")
    print(f"  Training / test customers: {result.split['train']:,} / {result.split['test']:,}")
    print("  Cross-validated ROC-AUC (training data):")
    for name, row in result.cv_scores.iterrows():
        print(f"   - {MODEL_LABELS[name]:<26} {row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}")
    print(f"  Chosen model: {label}   decision threshold {thr:.2f}")
    print("  Test set (unseen customers):")
    for metric in ("roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"):
        print(f"   - {metric:<10} {best[metric]:.3f}")
    print(f"   - caught {caught:,} of {caught + missed:,} churners; {false_alarms:,} false alarms")
    print(f"  Overfitting check: train ROC-AUC {overfit['train_roc_auc']:.3f} vs "
          f"test {overfit['test_roc_auc']:.3f}")
    print("  Risk levels on the test set:")
    for level, row in result.risk_table.iterrows():
        print(f"   - {level:<7} {int(row['customers']):>5,} customers, "
              f"{row['actual_churn_rate']:.0%} actually churned")
    print(f"\n  Saved {settings.CHURN_MODEL_FILE.name} and table '{settings.PREDICTIONS_TABLE}'")
    print(f"  Charts: {len(saved)}   Report: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=settings.DB_PATH)
    args = parser.parse_args(argv)
    return run(args.db)


if __name__ == "__main__":
    sys.exit(main())
