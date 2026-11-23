"""Module 4 - Churn prediction."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.components import ui  # noqa: E402
from config import settings  # noqa: E402
from src.dashboard import figures, queries  # noqa: E402
from src.models.train import MODEL_LABELS  # noqa: E402

ui.page_header("Step 5 · Churn prediction", "Churn prediction model",
               "A machine learning model estimates every customer's probability of leaving. "
               "This page shows how well it works on customers it has never seen, and how it "
               "was built.")

tables = data.tables()
summary = queries.read_report_json("modeling/summary.json")
test_metrics = data.report("modeling/test_metrics.csv")
if tables["predictions"] is None or summary is None or test_metrics is None:
    ui.missing_data(queries.missing_steps(tables) or ["python scripts/run_modeling.py"])
    st.stop()

preds = tables["predictions"]
best, split, chosen = summary["test_metrics"], summary["split"], summary["chosen_model"]
label = MODEL_LABELS[chosen]
confusion = data.report("modeling/confusion_matrix.csv", index_col=0)
caught = int(confusion.iloc[1, 1]) if confusion is not None else 0
churners = int(confusion.iloc[1].sum()) if confusion is not None else 0
false_alarms = int(confusion.iloc[0, 1]) if confusion is not None else 0
n_high = int((preds["risk_level"] == "High").sum())

ui.takeaways([
    f"The **{label}** model was chosen. On {split['test']:,} customers it never saw, it ranks a "
    f"churner above a stayer **{best['roc_auc']:.0%} of the time** (ROC-AUC "
    f"{best['roc_auc']:.2f}; 50% = guessing).",
    f"It **caught {caught:,} of {churners:,} churners** ({best['recall']:.0%}). "
    f"{best['precision']:.0%} of the customers it flagged really left "
    f"({false_alarms:,} false alarms).",
    f"Accuracy is {best['accuracy']:.0%}, but predicting 'no churn' for everyone already "
    f"scores {1 - split['test_churn_rate']:.0%}, so **accuracy alone would be misleading**.",
])
ui.kpi_row([
    {"label": "Chosen model", "value": label, "note": "best cross-validated ROC-AUC",
     "help": "Picked on training data only, never by test score."},
    {"label": "ROC-AUC", "value": f"{best['roc_auc']:.3f}", "tone": "good",
     "note": "0.5 = guessing · 1 = perfect", "help": "Ranking quality on unseen customers."},
    {"label": "Recall", "value": f"{best['recall']:.0%}", "note": "churners caught",
     "help": "Share of real churners the model flagged."},
    {"label": "Precision", "value": f"{best['precision']:.0%}", "note": "flagged who left",
     "help": "Share of flagged customers who really churned."},
    {"label": "High-risk customers", "value": f"{n_high:,}", "tone": "warn",
     "note": f"≥ {settings.RISK_THRESHOLDS['High']:.0%} probability"},
])

tab_perf, tab_risk, tab_method = st.tabs([":material/analytics: Performance",
                                          ":material/warning: Risk across customers",
                                          ":material/construction: How it was built"])

with tab_perf:
    left, right = st.columns([1.2, 1], gap="medium")
    with left:
        with ui.card("Models compared on unseen customers",
                     "Higher is better. The grey baseline always guesses the average churn rate; "
                     "it is the score to beat."):
            st.plotly_chart(figures.metric_comparison(test_metrics, MODEL_LABELS),
                            width="stretch")
    with right:
        with ui.card("What the model got right and wrong",
                     f"Test set · decision threshold {summary['threshold']:.2f}."):
            if confusion is not None:
                st.plotly_chart(figures.confusion_heatmap(confusion), width="stretch")
    cols = st.columns(3, gap="medium")
    for col, (name, title, note) in zip(cols, [
        ("model_03_roc", "ROC curve", "Closer to the top-left corner is better."),
        ("model_04_precision_recall", "Precision vs recall",
         "Catching more churners usually means more false alarms."),
        ("model_06_calibration", "Are the probabilities honest?",
         "Points near the diagonal mean '70% risk' really is about 70%."),
    ]):
        with col:
            with ui.card(title, note):
                path = queries.figure_path(name)
                if path:
                    st.image(str(path), width="stretch")
    overfit = summary["overfit_check"]
    st.info(f"**Overfitting check:** ROC-AUC {overfit['train_roc_auc']:.3f} on training data vs "
            f"{overfit['test_roc_auc']:.3f} on test data. A large gap would mean the model "
            "memorised training customers instead of learning general patterns.",
            icon=":material/fact_check:")

with tab_risk:
    with ui.card("How churn risk is spread across customers",
                 "Each customer's predicted probability, coloured by risk level. Dashed lines "
                 "are the risk-level cut-offs."):
        st.plotly_chart(figures.risk_distribution(preds), width="stretch")
    risk_test = data.report("modeling/risk_levels_test.csv")
    if risk_test is not None:
        with ui.card("Do the risk levels match reality?",
                     "Share of test customers in each level who actually churned. It should "
                     "rise from Low to High."):
            st.dataframe(risk_test, hide_index=True, width="stretch", column_config={
                "risk": "Risk level", "customers": "Customers",
                "actual_churn_rate": st.column_config.ProgressColumn(
                    "Actually churned", min_value=0.0, max_value=1.0, format="%.2f"),
                "share": st.column_config.NumberColumn("Share of test set", format="%.3f"),
            })

with tab_method:
    with ui.card("How the model was built", "Six steps, designed to give an honest score."):
        st.markdown(
            f"1. **Split:** {split['train']:,} customers for training, {split['test']:,} held "
            "back for a final test, both with the same churn rate (stratified).\n"
            "2. **Prepare:** fill gaps, scale numbers and encode categories inside a pipeline "
            "learned from training data only, to **prevent data leakage**.\n"
            f"3. **Compare:** baseline, Logistic Regression and Random Forest with "
            f"{settings.CV_FOLDS}-fold cross-validation on the training data.\n"
            f"4. **Choose:** best cross-validated ROC-AUC → **{label}**.\n"
            f"5. **Threshold:** {summary['threshold']:.2f}, the cut-off that best balances "
            "precision and recall on training data.\n"
            "6. **Test once** on the held-back customers, then retrain on everyone to score the "
            "whole customer base.")
    cv = data.report("modeling/cv_scores.csv")
    if cv is not None:
        with ui.card("Cross-validation results (training data)",
                     "Average of 5 folds; a small ± spread means the model is stable."):
            view = cv[["model", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "recall_mean",
                       "precision_mean", "f1_mean"]].copy()
            view["model"] = view["model"].map(MODEL_LABELS)
            st.dataframe(view.round(3), hide_index=True, width="stretch", column_config={
                "model": "Model", "roc_auc_mean": "ROC-AUC", "roc_auc_std": "± spread",
                "pr_auc_mean": "PR-AUC", "recall_mean": "Recall",
                "precision_mean": "Precision", "f1_mean": "F1"})

ui.learn_more("prediction")
