"""How it works - the business problem, workflow, dataset, glossary and limitations."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import ui  # noqa: E402
from config import settings  # noqa: E402
from src.data.schema import COLUMN_DESCRIPTIONS  # noqa: E402
from src.dashboard import content  # noqa: E402

ui.page_header("Start here", "How Customer360 AI works",
               "A guide for first-time visitors: no Data Science background needed.")

ui.takeaways([
    "**The problem:** customers who leave take their future revenue with them, and keeping a "
    "customer is usually cheaper than winning a new one.",
    "**The solution:** find who is likely to leave, understand why, and act first where risk "
    "and value are highest.",
    "**How to explore:** start with the Executive overview, then follow the menu from "
    "Customer insights to Churn intelligence.",
], title="In 30 seconds")

ui.section("The workflow", "Each page of this app covers one or more of these steps.")
ui.workflow()
steps = pd.DataFrame([
    ("1-2", "Customer data → cleaning", "Reports & data", "Load, validate and clean the data."),
    ("3", "Customer analysis", "Executive overview", "Who churns, and how much it costs."),
    ("4", "Segmentation", "Segmentation", "Group similar customers; profile each group."),
    ("5", "Churn prediction", "Churn prediction", "Score every customer's churn risk."),
    ("6", "Risk explanation", "Explainable AI / Customer 360", "Why is a customer at risk?"),
    ("7", "Retention action", "Retention strategy", "Who to contact, what to offer, impact."),
], columns=["Step", "Stage", "Where in the app", "What you learn"])
st.dataframe(steps, hide_index=True, width="stretch")

left, right = st.columns(2, gap="medium")
with left, ui.card("The dataset"):
    st.markdown(
        f"- **Source:** {settings.DATASET_NAME} (IBM sample data, also on Kaggle)\n"
        f"- **Size:** {settings.EXPECTED_ROWS:,} customers × {settings.EXPECTED_COLUMNS} columns\n"
        "- **Target:** `Churn`, whether the customer left within the last month\n"
        "- **Company:** fictional telecom provider")
    with st.expander("See every column"):
        st.dataframe(pd.DataFrame(COLUMN_DESCRIPTIONS.items(), columns=["Column", "Meaning"]),
                     hide_index=True, width="stretch")
with right, ui.card("Limitations, stated honestly"):
    for item in content.LIMITATIONS:
        st.markdown(f"- {item}")

ui.section("Correlation is not causation",
           "The single most important idea for reading this app correctly.")
st.info("When the app says a factor 'raises churn risk', it means customers with that "
        "factor churned more often in the past, and the model has learned that pattern. "
        "It does not prove that changing the factor will change the customer's decision. "
        "For example, month-to-month customers may churn more because less committed "
        "customers choose that contract. The reliable way to learn what works is to test "
        "an action on a random group and compare it with a similar group.",
        icon=":material/lightbulb:")

ui.section("Glossary", "Plain meanings of the terms used across the app.")
glossary = pd.DataFrame(content.GLOSSARY.items(), columns=["Term", "Plain meaning"])
st.dataframe(glossary, hide_index=True, width="stretch")
