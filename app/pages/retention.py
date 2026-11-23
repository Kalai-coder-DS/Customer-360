"""Module 6 - Retention strategy."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import data  # noqa: E402
from app.components import ui  # noqa: E402
from config import settings, theme  # noqa: E402
from src.dashboard import figures, queries  # noqa: E402
from src.retention.planner import TIER_ORDER, tier_summary  # noqa: E402
from src.retention.playbook import playbook_table  # noqa: E402

TIER_COLOURS = {TIER_ORDER[0]: theme.RISK["High"], TIER_ORDER[1]: theme.RISK["Medium"],
                TIER_ORDER[2]: theme.PRIMARY, TIER_ORDER[3]: theme.TEXT_MUTED}

ui.page_header("Step 7 · Retention action", "Retention strategy",
               "Who to contact first, what to offer each customer and why, and what a "
               "campaign might be worth under clearly stated assumptions.")
tables = data.tables()
plan = tables["retention"]
if plan is None:
    ui.missing_data(queries.missing_steps(tables) or ["python scripts/run_retention.py"])
    st.stop()

tiers = tier_summary(plan)
targeted = plan[plan["priority_tier"] != TIER_ORDER[3]]
horizon = settings.RETENTION_HORIZON_MONTHS
act_now = int(tiers.loc[TIER_ORDER[0], "customers"])
base = queries.impact_estimate(plan, TIER_ORDER[:2], 1.0, horizon)
top_action = (targeted[["action_1", "action_2"]].stack().value_counts().index[0]
              if len(targeted) else None)

ui.takeaways([
    f"**{act_now:,} customers need action now**: high churn risk and above-median bills. "
    f"Start at the top of the call list below.",
    (f"The most common recommendation is **{top_action}**, matched to each customer's own risk "
     "factors." if top_action else "No customers currently need retention actions."),
    f"Contacting 'Act now' + 'Act soon' could save **~{base['customers_saved']:.0f} customers** "
    f"and protect **{ui.money(base['revenue_protected'])}** for "
    f"{ui.money(base['campaign_cost'])} (estimate under illustrative assumptions).",
])
ui.kpi_row([
    {"label": f"Revenue at risk", "value": ui.money(plan["expected_loss_horizon"].sum()),
     "tone": "amber", "note": f"next {horizon} months, estimate",
     "help": "Sum of churn probability × monthly charges × horizon. Not a forecast."},
    {"label": "Act now", "value": f"{act_now:,}", "tone": "warn",
     "note": "high risk + high value", "help": "High risk and monthly charges ≥ median."},
    {"label": "To contact (tiers 1–3)", "value": f"{len(targeted):,}",
     "note": "low-risk customers excluded"},
    {"label": "Cost to contact all", "value": ui.money(targeted["actions_cost"].sum()),
     "note": "illustrative assumption", "help": "Sum of playbook costs for tiers 1–3."},
])

tab_list, tab_impact, tab_playbook = st.tabs([":material/call: Priority call list",
                                              ":material/calculate: Impact what-if",
                                              ":material/menu_book: Playbook & assumptions"])

with tab_list:
    left, right = st.columns([1, 1.2], gap="medium")
    with left:
        with ui.card("Priority tiers", "Risk × value decides the order of contact."):
            st.plotly_chart(figures.tier_bars(tiers, TIER_COLOURS), width="stretch")
    with right:
        with ui.card("Most recommended actions",
                     "Each action is matched to a factor that raises that customer's risk."):
            counts = plan[["action_1", "action_2"]].stack().value_counts()
            st.dataframe(counts.rename_axis("Action").reset_index(name="Customers"),
                         hide_index=True, width="stretch")

    ui.section("Priority call list", "Filter, then download the list for the retention team.")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1])
    with f1:
        tier_filter = st.multiselect("Priority tier", TIER_ORDER[:3], default=TIER_ORDER[:2])
    with f2:
        seg_options = (sorted(plan["segment_name"].dropna().unique())
                       if "segment_name" in plan else [])
        seg_filter = st.multiselect("Segment", seg_options)
    with f3:
        action_options = sorted(set(plan["action_1"].dropna()) | set(plan["action_2"].dropna()))
        action_filter = st.multiselect("Recommended action", action_options)
    with f4:
        min_prob = st.slider("Minimum churn risk", 0.0, 1.0, 0.0, 0.05, format="%.2f")
    search = st.text_input("Search customer ID", placeholder="e.g. 7590-VHVEG")

    filtered = queries.filter_plan(plan, tier_filter, seg_filter, action_filter, min_prob, search)
    st.caption(f"{len(filtered):,} customers · {ui.money(filtered['expected_loss_horizon'].sum())}"
               f" revenue at risk · {ui.money(filtered['actions_cost'].sum())} assumed cost")
    view = filtered.assign(churn_risk_pct=filtered["churn_probability"] * 100)
    columns = ["priority_rank", "customerID", "priority_tier", "churn_risk_pct",
               "MonthlyCharges", "expected_loss_horizon", "segment_name", "action_1",
               "action_1_reason", "action_2"]
    st.dataframe(view[[c for c in columns if c in view]], hide_index=True, width="stretch",
                 height=420, column_config={
                     "priority_rank": st.column_config.NumberColumn("Rank", format="%d"),
                     "customerID": "Customer", "priority_tier": "Tier",
                     "churn_risk_pct": st.column_config.ProgressColumn(
                         "Churn risk", min_value=0, max_value=100, format="%.0f%%"),
                     "MonthlyCharges": st.column_config.NumberColumn("Monthly $",
                                                                     format="$%.2f"),
                     "expected_loss_horizon": st.column_config.NumberColumn(
                         f"Value at risk ({horizon} mo)", format="$%.0f"),
                     "segment_name": "Segment", "action_1": "First action",
                     "action_1_reason": "Why", "action_2": "Second action",
                 })
    st.download_button(":material/download: Download this list (CSV)",
                       queries.to_csv_bytes(filtered), file_name="customer360_call_list.csv",
                       mime="text/csv")

with tab_impact:
    ui.section("What could a campaign be worth?",
               "Move the sliders to test different assumptions. Results are **estimates**, not "
               "forecasts.")
    c1, c2, c3 = st.columns(3)
    with c1:
        target = st.selectbox("Who to contact", ["Act now only", "Act now + Act soon",
                                                 "All at-risk (tiers 1-3)"], index=1)
    with c2:
        multiplier = st.slider("Success rate vs playbook assumption", 0.25, 2.0, 1.0, 0.25,
                               format="%.2fx",
                               help="1.0x = the playbook's assumed success rates.")
    with c3:
        months = st.slider("Months of revenue counted per saved customer", 3, 36, horizon, 3)
    tier_map = {"Act now only": TIER_ORDER[:1], "Act now + Act soon": TIER_ORDER[:2],
                "All at-risk (tiers 1-3)": TIER_ORDER[:3]}
    est = queries.impact_estimate(plan, tier_map[target], multiplier, months)
    ui.kpi_row([
        {"label": "Customers contacted", "value": f"{est['customers_contacted']:,}"},
        {"label": "Expected churners among them", "value": f"{est['expected_churners']:,.0f}",
         "help": "Sum of their churn probabilities."},
        {"label": "Customers saved (est.)", "value": f"{est['customers_saved']:,.0f}"},
        {"label": "Revenue protected (est.)", "value": ui.money(est["revenue_protected"])},
        {"label": "Net value (est.)", "value": ui.money(est["net_value"]),
         "help": "Revenue protected minus campaign cost.",
         "delta": f"{est['roi']:+.0%} return on cost" if est["campaign_cost"] else None},
    ])
    scenarios = data.report("retention/impact_scenarios.csv")
    if scenarios is not None:
        with ui.card("Scenario comparison",
                     "Net value (revenue protected − cost) under conservative, base and "
                     "optimistic success rates. A plan that only pays off when optimistic is "
                     "risky."):
            st.plotly_chart(figures.scenario_bars(scenarios), width="stretch")

with tab_playbook:
    ui.section("Retention playbook",
               "Costs and success rates are **illustrative assumptions**. Replace them with "
               "real figures, ideally learned from controlled experiments.")
    st.dataframe(playbook_table().reset_index(), hide_index=True, width="stretch",
                 column_config={
                     "action": "Action", "description": "What it is", "channel": "Channel",
                     "assumed_cost_usd": st.column_config.NumberColumn("Assumed cost",
                                                                       format="$%.0f"),
                     "assumed_success_rate": st.column_config.NumberColumn(
                         "Assumed success rate", format="%.2f"),
                 })
    st.markdown(
        "**How actions are chosen:** an action is suggested only if it applies to the customer "
        "(e.g. a contract offer only for month-to-month customers) **and** the related factor "
        "raises that customer's churn risk in the model explanation. At most two actions per "
        "customer, strongest reason first.\n\n"
        "**Limitations:** predictions show association, not cause; contacting customers can "
        "have side effects; the data describes a fictional company.")

ui.learn_more("retention")
