"""Customer360 AI - Streamlit entry point.

Run from the project folder:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.components import ui  # noqa: E402

st.set_page_config(page_title="Customer360 AI", page_icon=":material/hub:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()
data.ensure_pipeline()

PAGES = Path(__file__).parent / "pages"
navigation = st.navigation({
    "Start here": [
        st.Page(PAGES / "overview.py", title="Executive overview", icon=":material/dashboard:",
                default=True),
        st.Page(PAGES / "guide.py", title="How it works", icon=":material/school:"),
    ],
    "Customer insights": [
        st.Page(PAGES / "customer_profile.py", title="Customer 360 profile",
                icon=":material/person_search:"),
        st.Page(PAGES / "segmentation.py", title="Segmentation", icon=":material/groups:"),
    ],
    "Churn intelligence": [
        st.Page(PAGES / "churn_prediction.py", title="Churn prediction",
                icon=":material/trending_down:"),
        st.Page(PAGES / "explainable_ai.py", title="Explainable AI", icon=":material/psychology:"),
        st.Page(PAGES / "retention.py", title="Retention strategy", icon=":material/handshake:"),
    ],
    "Data": [
        st.Page(PAGES / "data_management.py", title="Reports & data", icon=":material/database:"),
    ],
})

with st.sidebar:
    ui.brand()

navigation.run()

with st.sidebar:
    st.divider()
    ui.data_notice()
