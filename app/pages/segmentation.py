"""Module 3 - Customer segmentation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.components import ui  # noqa: E402
from config import theme  # noqa: E402
from src.dashboard import figures, queries  # noqa: E402

ui.page_header("Step 4 · Segmentation", "Customer segments",
               "Customers grouped by how long they have stayed, how much they pay and how "
               "many services they use, so each group can get the right treatment.")

tables = data.tables()
profiles = data.report("segmentation/cluster_profiles.csv")
if tables["segments"] is None or profiles is None:
    ui.missing_data(queries.missing_steps(tables) or ["python scripts/run_segmentation.py"])
    st.stop()

clean, segments = tables["clean"], tables["segments"]
k_scores = data.report("segmentation/k_selection.csv")
overall = float(clean["churn_flag"].mean())
n_seg = len(profiles)
by_churn = profiles.sort_values("churn_rate", ascending=False)
riskiest, safest = by_churn.iloc[0], by_churn.iloc[-1]
richest = profiles.sort_values("share_of_revenue", ascending=False).iloc[0]
silhouette = (k_scores.set_index("k").loc[n_seg, "silhouette"] if k_scores is not None else None)

ui.takeaways([
    f"The data splits naturally into **{n_seg} customer segments**.",
    f"**{riskiest['segment_name']}** churns most (**{riskiest['churn_rate']:.0%}**) and brings "
    f"{riskiest['share_of_revenue']:.0%} of revenue. This is the priority for retention.",
    f"**{richest['segment_name']}** brings the most revenue ({richest['share_of_revenue']:.0%}); "
    f"**{safest['segment_name']}** is the most loyal ({safest['churn_rate']:.0%} churn).",
])
ui.kpi_row([
    {"label": "Segments found", "value": f"{n_seg}", "note": "chosen automatically (3–6)",
     "help": "The clearest grouping between 3 and 6 segments, by silhouette score."},
    {"label": "Separation quality", "value": f"{silhouette:.2f}" if silhouette is not None else "–",
     "note": "silhouette score (-1 to 1)",
     "help": "How clearly separated the groups are. 0.2–0.5 is typical for customer data."},
    {"label": "Highest churn", "value": f"{riskiest['churn_rate']:.0%}", "tone": "bad",
     "note": riskiest["segment_name"]},
    {"label": "Largest revenue share", "value": f"{richest['share_of_revenue']:.0%}",
     "tone": "neutral", "note": richest["segment_name"]},
])

tab_kmeans, tab_rfm = st.tabs([":material/scatter_plot: K-Means segments",
                               ":material/leaderboard: RFM-style scores"])

with tab_kmeans:
    ui.section("Segment cards", "Sorted from highest to lowest churn. Each card shows who the "
                                "customers are and an action to test.")
    cols = st.columns(min(n_seg, 3), gap="medium")
    for i, (_, seg) in enumerate(by_churn.iterrows()):
        colour = theme.CATEGORICAL[int(seg["cluster"]) % len(theme.CATEGORICAL)]
        diff = seg["churn_rate"] - overall
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(
                    f'<p class="c360-card-title"><span style="color:{colour}">●</span> '
                    f'{seg["segment_name"]}</p>', unsafe_allow_html=True)
                ui.fields({
                    "Customers": f"{int(seg['customers']):,} ({seg['share_of_customers']:.0%})",
                    "Churn rate": f"{seg['churn_rate']:.1%} ({diff:+.1%} vs avg)",
                    "Avg tenure": f"{seg['avg_tenure']:.0f} months",
                    "Avg monthly": f"${seg['avg_monthly_charges']:.0f}",
                })
                st.caption(seg["description"])
                st.markdown(f":material/lightbulb: **Action to test:** {seg['suggested_action']}")

    left, right = st.columns(2, gap="medium")
    with left:
        with ui.card("Size vs revenue",
                     "A segment whose revenue share beats its customer share is worth more per "
                     "customer."):
            st.plotly_chart(figures.segment_size_revenue(profiles), width="stretch")
    with right:
        with ui.card("Churn rate by segment", f"Dashed line = overall average ({overall:.1%})."):
            st.plotly_chart(figures.churn_rate_bars(
                profiles[["segment_name", "churn_rate", "customers"]], "segment_name", overall),
                width="stretch")

    with ui.card("Where each segment sits",
                 "Each dot is a customer (random sample), placed by tenure and monthly charges. "
                 "Hover for details."):
        st.plotly_chart(figures.segment_map(clean, segments), width="stretch")

    with ui.card("Segment profiles side by side"):
        view = profiles[["segment_name", "customers", "avg_tenure", "avg_monthly_charges",
                         "avg_services", "share_month_to_month", "share_fiber", "churn_rate",
                         "share_of_revenue"]]
        pct = dict(min_value=0.0, max_value=1.0, format="%.2f")
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "segment_name": "Segment", "customers": "Customers",
            "avg_tenure": st.column_config.NumberColumn("Avg tenure (mo)", format="%.0f"),
            "avg_monthly_charges": st.column_config.NumberColumn("Avg monthly", format="$%.2f"),
            "avg_services": st.column_config.NumberColumn("Avg services", format="%.1f"),
            "share_month_to_month": st.column_config.ProgressColumn("Month-to-month", **pct),
            "share_fiber": st.column_config.ProgressColumn("Fiber optic", **pct),
            "churn_rate": st.column_config.ProgressColumn("Churn rate", **pct),
            "share_of_revenue": st.column_config.ProgressColumn("Revenue share", **pct),
        })
        if k_scores is not None:
            with st.expander("How the number of segments was chosen"):
                st.write("K-Means was run for 2 to 8 segments. The silhouette score measures "
                         "how clearly separated the groups are; the best score between 3 and 6 "
                         "segments (a range a team can act on) was chosen.")
                st.dataframe(k_scores, hide_index=True, width="stretch")

with tab_rfm:
    rfm = data.report("segmentation/rfm_segments.csv")
    with ui.card("RFM-style segments",
                 "Loyalty = tenure, Engagement = number of services, Monetary = monthly "
                 "charges, each scored 1–5. True RFM needs purchase dates, which this data does "
                 "not have, so these are labelled substitutes."):
        if rfm is not None:
            rfm_rates = rfm.rename(columns={"rfm_segment": "segment"})
            st.plotly_chart(figures.churn_rate_bars(rfm_rates, "segment", overall),
                            width="stretch")
            st.dataframe(rfm_rates, hide_index=True, width="stretch", column_config={
                "segment": "Segment", "customers": "Customers",
                "churn_rate": st.column_config.ProgressColumn("Churn rate", min_value=0.0,
                                                              max_value=1.0, format="%.2f"),
                "avg_tenure": st.column_config.NumberColumn("Avg tenure", format="%.0f"),
                "avg_monthly_charges": st.column_config.NumberColumn("Avg monthly",
                                                                     format="$%.2f"),
                "avg_services": st.column_config.NumberColumn("Avg services", format="%.1f"),
                "share_of_customers": st.column_config.NumberColumn("Share", format="%.3f"),
                "description": "Who they are",
            })

ui.learn_more("segmentation")
