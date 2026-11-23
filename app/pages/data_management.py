"""Module 7 - Reports and data management."""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.components import ui  # noqa: E402
from config import settings  # noqa: E402
from src.dashboard import queries  # noqa: E402
from src.data.loader import DataLoadError, load_customer_csv  # noqa: E402

ui.page_header("Steps 1–2 · Data", "Reports & data management",
               "Upload and validate customer files, store them in the database, score them "
               "with the churn model, and download every report.")
ui.takeaways([
    "**Upload** a CSV in the Telco format to check its quality: missing values, duplicates and "
    "unexpected values are listed before anything else happens.",
    "**Save** valid files to the SQLite database, or **score** them with the churn model and "
    "download the results.",
    "**Download** any report or customer list, and see the database status and load history.",
], title="How to use this page")

tab_upload, tab_reports, tab_db = st.tabs([":material/upload_file: Upload & validate",
                                           ":material/description: Reports & exports",
                                           ":material/storage: Database"])

with tab_upload:
    ui.section("Upload a customer CSV",
               "The file must use the Telco column layout. It is checked before anything "
               "else happens.")
    uploaded = st.file_uploader("Customer CSV file", type=["csv"])
    raw = None
    if uploaded is not None:
        try:
            raw = load_customer_csv(uploaded)
        except DataLoadError as exc:
            st.error(str(exc), icon=":material/error:")
    if raw is not None:
        report = queries.validate_upload(raw)
        ui.kpi_row([
            {"label": "Rows", "value": f"{report.n_rows:,}"},
            {"label": "Columns", "value": f"{report.n_columns}"},
            {"label": "Missing values", "value": f"{report.total_missing:,}",
             "help": "Empty cells plus blank text."},
            {"label": "Duplicate rows", "value": f"{report.duplicate_rows:,}"},
            {"label": "Status", "value": "Valid" if report.is_valid else "Invalid"},
        ])
        for msg in report.errors:
            st.error(msg, icon=":material/error:")
        for msg in report.warnings:
            st.warning(msg, icon=":material/warning:")
        if report.is_valid and not report.warnings:
            st.success("No problems found.", icon=":material/check_circle:")

        left, right = st.columns(2, gap="large")
        with left:
            ui.section("Missing values by column")
            if report.missing_values:
                st.dataframe(pd.DataFrame(report.missing_values.items(),
                                          columns=["Column", "Missing"]),
                             hide_index=True, width="stretch")
            else:
                st.caption("None.")
        with right:
            ui.section("Duplicates")
            st.write(f"Fully duplicated rows: **{report.duplicate_rows:,}** · Repeated customer "
                     f"IDs: **{report.duplicate_ids:,}**")
            if report.duplicate_ids and "customerID" in raw:
                dupes = raw[raw["customerID"].duplicated(keep=False)].sort_values("customerID")
                st.dataframe(dupes.head(50), hide_index=True, width="stretch")

        ui.section("Preview (first 20 rows)")
        st.dataframe(raw.head(20), hide_index=True, width="stretch")

        if report.is_valid:
            st.divider()
            a, b = st.columns(2, gap="large")
            with a:
                ui.section("Store in the database",
                           "Saved as a separate table; the main analysis is not overwritten.")
                if data.demo_mode():
                    st.button(":material/save: Save to SQLite", type="primary", disabled=True,
                              help="Disabled in the public demo.")
                    st.caption("Saving is turned off in this public demo, so no visitor's data "
                               "is stored. Scoring still works and runs in memory only.")
                elif st.button(":material/save: Save to SQLite", type="primary"):
                    table = f"upload_{datetime.now():%Y%m%d_%H%M%S}"
                    load_id = queries.save_upload(raw, table)
                    st.success(f"Saved {len(raw):,} rows to table '{table}' (load #{load_id}).")
            with b:
                ui.section("Score with the churn model",
                           "Cleans the file and predicts each customer's churn risk.")
                bundle = data.churn_bundle()
                if bundle is None:
                    st.caption("Train the model first: python scripts/run_modeling.py")
                elif st.button(":material/bolt: Predict churn risk"):
                    scored, log = queries.score_customers(raw, bundle)
                    st.session_state["scored_upload"] = scored
                    st.session_state["scored_log"] = log.steps
            if "scored_upload" in st.session_state:
                scored = st.session_state["scored_upload"]
                counts = scored["risk_level"].value_counts()
                st.caption(f"Scored {len(scored):,} customers · High risk "
                           f"{int(counts.get('High', 0)):,} · Medium "
                           f"{int(counts.get('Medium', 0)):,} · Low {int(counts.get('Low', 0)):,}")
                st.dataframe(scored.head(100), hide_index=True, width="stretch")
                st.download_button(":material/download: Download scored customers (CSV)",
                                   queries.to_csv_bytes(scored),
                                   file_name="customer360_scored_upload.csv", mime="text/csv")
                with st.expander("Cleaning steps applied"):
                    for step in st.session_state.get("scored_log", []):
                        st.markdown(f"- {step}")

with tab_reports:
    ui.section("Download reports", "Every report and table produced by the pipeline.")
    files = queries.list_report_files()
    if not files:
        st.caption("No reports yet. Run the pipeline scripts first.")
    for path in files:
        rel = path.relative_to(settings.REPORTS_DIR)
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"`{rel.as_posix()}` · {path.stat().st_size / 1024:,.1f} KB")
        mime = {".md": "text/markdown", ".csv": "text/csv", ".json": "application/json"}
        c2.download_button("Download", path.read_bytes(), file_name=path.name,
                           mime=mime[path.suffix], key=f"dl_{rel.as_posix()}")

    tables = data.tables()
    ui.section("Export customer lists")
    exports = {
        "All customers with risk and segment": None,
        "High-risk active customers": "high",
    }
    if tables["clean"] is not None and tables["predictions"] is not None:
        merged = tables["clean"].merge(tables["predictions"][["customerID", "churn_probability",
                                                              "risk_level"]], on="customerID")
        if tables["segments"] is not None:
            merged = merged.merge(tables["segments"][["customerID", "segment_name",
                                                      "rfm_segment"]], on="customerID")
        choice = st.selectbox("List", list(exports))
        out = merged
        if exports[choice] == "high":
            out = merged[(merged["risk_level"] == "High") & (merged["churn_flag"] == 0)]
        st.caption(f"{len(out):,} customers")
        st.download_button(":material/download: Download list (CSV)", queries.to_csv_bytes(out),
                           file_name=f"customer360_{exports[choice] or 'all'}_customers.csv",
                           mime="text/csv")
    else:
        st.caption("Available after running the modelling step.")

with tab_db:
    tables = data.tables()
    ui.section("Database tables", f"SQLite {queries.sqlite_version()} · {settings.DB_PATH.name}")
    status = pd.DataFrame([
        {"Table": queries.TABLES[key], "Rows": len(df) if df is not None else 0,
         "Status": "Ready" if df is not None else "Not created yet"}
        for key, df in tables.items()])
    st.dataframe(status, hide_index=True, width="stretch")
    missing = queries.missing_steps(tables)
    if missing:
        ui.missing_data(missing)
    if tables["loads"] is not None:
        ui.section("Load history", "Every time data was loaded, with its quality status.")
        st.dataframe(tables["loads"].sort_values("load_id", ascending=False),
                     hide_index=True, width="stretch")

ui.learn_more("data")
