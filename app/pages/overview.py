"""Module 1 - Executive overview."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.components import ui  # noqa: E402
from src.analysis import eda  # noqa: E402
from src.dashboard import content, figures, queries  # noqa: E402

ui.page_header("Executive overview", "Customer health at a glance", content.BUSINESS_PROBLEM)

tables = data.tables()
if tables["clean"] is None:
    ui.workflow()
    ui.missing_data(["python scripts/ingest_data.py"] + queries.missing_steps(tables))
    st.stop()

clean = tables["clean"]
k = queries.executive_kpis(tables)
contract = queries.churn_rate_table(clean, "Contract")
bands = queries.tenure_band_table(clean)
profiles = data.report("segmentation/cluster_profiles.csv")

# --- Takeaways ----------------------------------------------------------------
top_c, low_c = contract.iloc[0], contract.iloc[-1]
points = [
    f"**{k['churn_rate']:.1%} of customers left** ({k['churned']:,} of {k['customers']:,}), "
    f"taking **{ui.money(k['monthly_revenue_lost'])} of monthly revenue** with them.",
    f"**{top_c['Contract']} customers churn at {top_c['churn_rate']:.0%}**, versus "
    f"{low_c['churn_rate']:.0%} on {low_c['Contract']} contracts. Churn is "
    f"{bands['churn_rate'].iloc[0]:.0%} for customers in their first "
    f"{bands[eda.TENURE_BAND].iloc[0].split(' ')[0]} months vs "
    f"{bands['churn_rate'].iloc[-1]:.0%} for {bands[eda.TENURE_BAND].iloc[-1]}.",
]
if profiles is not None and "revenue_at_risk" in k:
    worst = profiles.sort_values("churn_rate", ascending=False).iloc[0]
    points.append(
        f"**{k['high_risk_active']:,} active customers are at high risk**, and "
        f"{ui.money(k['revenue_at_risk'])} of revenue is at risk over 12 months (estimate). "
        f"The segment needing most attention is **{worst['segment_name']}** "
        f"({worst['churn_rate']:.0%} churn).")
ui.takeaways(points)

# --- KPIs ---------------------------------------------------------------------
ui.kpi_row([
    {"label": "Total customers", "value": f"{k['customers']:,}", "note": "in the dataset",
     "help": "Every customer in the dataset."},
    {"label": "Active customers", "value": f"{k['active']:,}", "tone": "good",
     "note": f"{1 - k['churn_rate']:.1%} retained", "help": "Did not leave in the last month."},
    {"label": "Churned customers", "value": f"{k['churned']:,}", "tone": "bad",
     "note": "left in the last month", "help": "Customers who cancelled."},
    {"label": "Churn rate", "value": f"{k['churn_rate']:.1%}", "tone": "bad",
     "note": "churned ÷ total customers", "help": "Share of customers who left."},
])
second = [
    {"label": "Avg monthly charge", "value": f"${k['avg_monthly_charges']:,.2f}", "tone": "neutral",
     "note": f"avg tenure {k['avg_tenure']:.0f} months", "help": "Average bill per customer."},
    {"label": "Monthly revenue lost", "value": ui.money(k["monthly_revenue_lost"]), "tone": "bad",
     "note": "what churned customers paid", "help": "Sum of churned customers' monthly charges."},
]
if "high_risk_active" in k:
    second.append({"label": "Active at high risk", "value": f"{k['high_risk_active']:,}",
                   "tone": "warn", "note": "60%+ predicted churn",
                   "help": "Active customers the model scores at 60% churn probability or more."})
if "revenue_at_risk" in k:
    second.append({"label": "Revenue at risk", "value": ui.money(k["revenue_at_risk"]),
                   "tone": "amber", "note": "next 12 months, estimate",
                   "help": "Sum of churn probability × monthly charges × 12. Not a forecast."})
ui.kpi_row(second)

# --- Workflow -----------------------------------------------------------------
ui.section("How the platform works",
           "From raw data to action: each stage has its own page in the menu on the left.")
ui.workflow()

# --- Charts -------------------------------------------------------------------
left, right = st.columns(2, gap="medium")
with left:
    with ui.card("Who stayed and who left",
                 f"A model predicting 'no churn' for everyone would be "
                 f"{1 - k['churn_rate']:.1%} accurate and still useless, so later pages use "
                 "better measures."):
        st.plotly_chart(figures.stay_vs_churn(k["active"], k["churned"]), width="stretch")
    with ui.card("Churn by contract type",
                 f"**{top_c['Contract']}** customers churn at **{top_c['churn_rate']:.1%}** vs "
                 f"{low_c['churn_rate']:.1%} for {low_c['Contract']}."):
        st.plotly_chart(figures.churn_rate_bars(contract, "Contract", k["churn_rate"]),
                        width="stretch")
with right:
    with ui.card("Churn by how long customers have stayed",
                 f"**{bands['churn_rate'].iloc[0]:.1%}** churn in the first band "
                 f"({bands[eda.TENURE_BAND].iloc[0]}) vs {bands['churn_rate'].iloc[-1]:.1%} "
                 f"for {bands[eda.TENURE_BAND].iloc[-1]}."):
        st.plotly_chart(figures.churn_rate_columns(bands, eda.TENURE_BAND, k["churn_rate"]),
                        width="stretch")
    internet = queries.churn_rate_table(clean, "InternetService")
    with ui.card("Churn by internet service",
                 f"Highest: **{internet.iloc[0]['InternetService']}** "
                 f"({internet.iloc[0]['churn_rate']:.1%})."):
        st.plotly_chart(figures.churn_rate_bars(internet, "InternetService", k["churn_rate"]),
                        width="stretch")

# --- Findings and segments ----------------------------------------------------
left, right = st.columns([1.2, 1], gap="medium")
with left:
    with ui.card("Key findings", "Generated from the data. Each is an association, not proof "
                                 "of cause."):
        for i, finding in enumerate(eda.key_findings(clean), start=1):
            st.markdown(f"{i}. {finding}")
with right:
    if profiles is not None:
        with ui.card("Customer segments", "Open the Segmentation page for profiles and actions."):
            view = profiles[["segment_name", "customers", "churn_rate", "share_of_revenue"]]
            st.dataframe(view, hide_index=True, width="stretch", column_config={
                "segment_name": "Segment",
                "customers": st.column_config.NumberColumn("Customers", format="%d"),
                "churn_rate": st.column_config.ProgressColumn(
                    "Churn rate", min_value=0.0, max_value=1.0, format="%.2f"),
                "share_of_revenue": st.column_config.ProgressColumn(
                    "Revenue share", min_value=0.0, max_value=1.0, format="%.2f"),
            })

missing = queries.missing_steps(tables)
if missing:
    ui.missing_data(missing)

ui.learn_more("overview")
