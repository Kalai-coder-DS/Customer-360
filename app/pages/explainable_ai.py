"""Module 5 - Explainable AI."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.components import ui  # noqa: E402
from src.dashboard import figures, queries  # noqa: E402
from src.explain.explainer import FEATURE_LABELS  # noqa: E402

ui.page_header("Step 6 · Risk explanation", "Explainable AI",
               "Why does the model give customers the scores it does? Which factors matter "
               "most overall, and how does each one move a prediction up or down?")

tables = data.tables()
importance = data.report("explain/global_importance.csv", index_col=0)
if tables["explanations"] is None or importance is None:
    ui.missing_data(queries.missing_steps(tables) or ["python scripts/run_explain.py"])
    st.stop()

explanations = tables["explanations"]
high = explanations[explanations["risk_level"] == "High"]
reasons = (high["top_risk_factors"].dropna().str.split("; ").explode()
           .str.replace(r" \((strong|moderate|slight)\)$", "", regex=True))
reason_counts = (reasons.str.split(":").str[0].value_counts() if len(reasons)
                 else pd.Series(dtype=int))
top3 = list(importance["label"].head(3))

points = [f"The model relies most on **{top3[0]}**, followed by "
          f"**{', '.join(top3[1:])}**."]
if len(reason_counts):
    points.append(f"Among High-risk customers, the most common reason is **{reason_counts.index[0]}** "
                  f"({reason_counts.iloc[0] / max(len(high), 1):.0%} of them).")
points.append("These are **patterns the model learned, not proven causes**. Use them to decide "
              "where to look and what to test.")
ui.takeaways(points)

ui.kpi_row([
    {"label": "Most relied-on factor", "value": top3[0], "note": "biggest accuracy drop if removed",
     "help": "Permutation importance: the factor whose shuffling hurts ROC-AUC most."},
    {"label": "Customers explained", "value": f"{len(explanations):,}",
     "note": "with stored reasons",
     "help": "With Random Forest only the riskiest are precomputed; others are explained on "
             "demand on the Customer 360 page."},
    {"label": "Factors in the model", "value": f"{len(importance)}",
     "note": "19 original + 3 engineered"},
])

left, right = st.columns([1.1, 1], gap="medium")
with left:
    with ui.card("Which factors the model relies on most",
                 "How much accuracy (ROC-AUC) drops when a factor is shuffled. Longer bar = "
                 "more reliance."):
        st.plotly_chart(figures.importance_bars(importance), width="stretch")
with right:
    with ui.card("How a factor moves predictions",
                 "Average push on churn risk for each value. **Coral raises**, **violet lowers**."):
        options = [f for f in importance.index
                   if data.report(f"explain/direction_{f}.csv") is not None]
        if options:
            feature = st.selectbox("Choose a factor", options,
                                   format_func=lambda f: FEATURE_LABELS.get(f, f))
            direction = data.report(f"explain/direction_{feature}.csv", index_col=0)
            series = direction["avg_contribution"]
            st.plotly_chart(figures.contribution_bars(series, {v: str(v) for v in series.index},
                                                      top=len(series)), width="stretch")
            st.caption("Customers per value: " + ", ".join(
                f"{i} ({int(n):,})" for i, n in direction["customers"].items()))

left, right = st.columns([1, 1.2], gap="medium")
with left:
    with ui.card("Most common reasons for high risk",
                 "How often each factor is among the top reasons for High-risk customers."):
        if len(reason_counts):
            table = pd.DataFrame({"Factor": reason_counts.index[:10],
                                  "Customers": reason_counts.values[:10],
                                  "Share": reason_counts.values[:10] / max(len(high), 1)})
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "Share": st.column_config.ProgressColumn(
                    "Share of High-risk", min_value=0.0, max_value=1.0, format="%.2f")})
        else:
            st.caption("No High-risk customers with stored explanations.")
with right:
    with ui.card("Example explanations",
                 "Written automatically for individual customers. Open Customer 360 for any "
                 "customer's full profile."):
        ranked = explanations.sort_values("churn_probability", ascending=False)
        for _, ex_row in pd.concat([ranked.head(2), ranked.tail(1)]).iterrows():
            st.markdown(f"**{ex_row['customerID']}** &nbsp; {ui.risk_badge(ex_row['risk_level'])}",
                        unsafe_allow_html=True)
            st.write(ex_row["explanation"])

st.warning("**Correlation is not causation.** If month-to-month customers churn more, it may be "
           "because less committed customers choose that contract, not because the contract "
           "makes them leave. Test actions on a random group before rolling them out.",
           icon=":material/info:")

coefs = data.report("explain/logistic_coefficients.csv")
if coefs is not None:
    with st.expander(":material/functions: For analysts: Logistic Regression coefficients and "
                     "odds ratios"):
        st.write("Numbers are standardised, so a numeric coefficient is the effect of one "
                 "standard deviation. An odds ratio above 1 raises the odds of churn; below 1 "
                 "lowers them. For categories, compare values within the same factor.")
        st.dataframe(coefs.round(3), hide_index=True, width="stretch")

ui.learn_more("explain")
