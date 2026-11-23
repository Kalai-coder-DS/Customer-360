"""Module 2 - Customer 360 profile."""

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
from src.explain import explainer as ex  # noqa: E402
from src.models.features import prepare_xy  # noqa: E402

ui.page_header("Step 6 · Risk explanation", "Customer 360 profile",
               "Everything about one customer on one page: who they are, what they use, "
               "their churn risk, the reasons behind it and what to do next.")

tables = data.tables()
if tables["clean"] is None:
    ui.missing_data(["python scripts/ingest_data.py"] + queries.missing_steps(tables))
    st.stop()

clean = tables["clean"]
ids = queries.customer_ids_by_risk(tables)
note = " · sorted from highest to lowest churn risk" if tables["predictions"] is not None else ""
customer_id = st.selectbox(f"Select a customer (type to search by ID){note}", ids, index=0)
record = queries.customer_record(tables, customer_id)
row, pred, seg, plan = record["clean"], record["predictions"], record["segments"], record["retention"]
x_customer = prepare_xy(clean[clean["customerID"] == customer_id])[0].iloc[0]
services = queries.services_used(row)

# --- Explanation (computed first so the takeaways can use it) -----------------
explanation = record["explanations"]
contributions = (queries.parse_contributions(explanation["contributions_json"])
                 if explanation is not None else None)
bundle = data.churn_bundle()
if contributions is None and bundle is not None and pred is not None:
    with st.spinner("Calculating the explanation for this customer..."):
        X, _ = prepare_xy(clean)
        X.index = clean["customerID"].values
        background = X.sample(min(settings.EXPLAIN_BACKGROUND_SIZE, len(X)),
                              random_state=settings.RANDOM_STATE)
        contributions, _base = ex.sampling_contributions(bundle["pipeline"], x_customer,
                                                         background)
raising, lowering = (ex.top_factors(contributions, x_customer) if contributions is not None
                     else ([], []))

# --- Takeaways ----------------------------------------------------------------
if pred is not None:
    prob, level = float(pred["churn_probability"]), str(pred["risk_level"])
    points = [f"Customer **{customer_id}** has an estimated **{prob:.0%} chance of churning** "
              f"(**{level} risk**)."]
    if raising:
        points.append("Main reasons: " + ", ".join(f"**{f['label']}**" for f in raising[:3]) + ".")
    if plan is not None and isinstance(plan["action_1"], str):
        points.append(f"Suggested next step: **{plan['action_1']}** ({plan['priority_tier']}).")
    elif plan is not None:
        points.append("Low risk: no retention action needed.")
    ui.takeaways(points)

# --- Header card --------------------------------------------------------------
gauge_col, facts_col = st.columns([1, 2.2], gap="medium")
with gauge_col:
    with ui.card("Churn risk"):
        if pred is not None:
            st.markdown(ui.risk_badge(level), unsafe_allow_html=True)
            st.plotly_chart(figures.risk_gauge(prob, level), width="stretch")
            actual = "left" if int(pred["actual_churn"]) == 1 else "stayed"
            st.caption(f"Bands: Low < {settings.RISK_THRESHOLDS['Medium']:.0%} ≤ Medium < "
                       f"{settings.RISK_THRESHOLDS['High']:.0%} ≤ High. In the historical "
                       f"data this customer {actual}.")
        else:
            ui.missing_data(queries.missing_steps(tables))
with facts_col:
    with ui.card(f"Customer {customer_id}", "Profile from the cleaned customer data."):
        ui.fields({
            "Gender": row["gender"],
            "Senior citizen": "Yes" if int(row["SeniorCitizen"]) else "No",
            "Partner": row["Partner"], "Dependents": row["Dependents"],
            "Contract": row["Contract"], "Tenure": f"{int(row['tenure'])} months",
            "Monthly charges": f"${float(row['MonthlyCharges']):,.2f}",
            "Total paid": f"${float(row['TotalCharges']):,.2f}",
            "Payment method": row["PaymentMethod"],
            "Paperless billing": row["PaperlessBilling"],
            "Internet": row["InternetService"], "Phone": row["PhoneService"],
        })
        st.markdown("**Services used**")
        all_addons = ["Online security", "Online backup", "Device protection", "Tech support",
                      "Streaming TV", "Streaming movies"]
        ui.chips(services, [a for a in all_addons if a not in services])
        if seg is not None:
            st.markdown("**Segment**")
            ui.fields({
                "K-Means segment": seg["segment_name"], "RFM-style segment": seg["rfm_segment"],
                "Loyalty score": f"{int(seg['loyalty_score'])} / 5",
                "Engagement score": f"{int(seg['engagement_score'])} / 5",
                "Monetary score": f"{int(seg['monetary_score'])} / 5",
            })

# --- Reasons and action -------------------------------------------------------
reasons_col, action_col = st.columns([1.5, 1], gap="medium")
with reasons_col:
    with ui.card("Why this risk score?",
                 "Each bar is one factor's push compared with an average customer. "
                 "**Coral raises** the risk, **violet lowers** it. These are associations the model "
                 "learned, not proven causes."):
        if contributions is not None and pred is not None:
            labels = {f: ex.readable_value(f, x_customer[f]) for f in contributions.index}
            st.plotly_chart(figures.contribution_bars(contributions, labels), width="stretch")
            if explanation is not None:
                st.write(explanation["explanation"])
            else:
                st.write(ex.explanation_text(prob, level, bundle["model_label"], raising, lowering))
        else:
            ui.missing_data(queries.missing_steps(tables) or ["python scripts/run_explain.py"])
with action_col:
    with ui.card("Recommended next step", "From the retention playbook."):
        if plan is None:
            ui.missing_data(queries.missing_steps(tables) or ["python scripts/run_retention.py"])
        elif isinstance(plan["action_1"], str):
            st.markdown(f"**{plan['priority_tier']}**")
            for n in (1, 2):
                action = plan[f"action_{n}"]
                if isinstance(action, str):
                    st.markdown(f":material/check_circle: **{action}**  \n"
                                f"<span class='c360-note'>{plan[f'action_{n}_reason']}</span>",
                                unsafe_allow_html=True)
            ui.fields({
                f"Value at risk ({settings.RETENTION_HORIZON_MONTHS} mo)":
                    ui.money(float(plan["expected_loss_horizon"])),
                "Assumed cost": ui.money(float(plan["actions_cost"])),
            })
            st.caption(f"Based on {plan['action_basis']}. Costs and success rates are "
                       "illustrative assumptions.")
        else:
            st.markdown(f"**{plan['priority_tier']}**")
            st.write("Low risk: no retention action needed. Keep standard engagement.")

ui.learn_more("profile")
