"""Retention planning: who to contact first, what to offer, and the likely impact.

1. **Value at risk** for each customer
       expected monthly loss = churn probability x monthly charges
   The annual figure multiplies by the horizon (12 months by default).
   ASSUMPTION: a saved customer keeps paying the same amount for the horizon.

2. **Priority tiers** combine risk and value:
   | Tier            | Who                                              |
   |-----------------|--------------------------------------------------|
   | 1 - Act now     | High risk and high value                         |
   | 2 - Act soon    | High risk and lower value, or Medium risk and high value |
   | 3 - Nurture     | Medium risk and lower value                      |
   | 4 - Monitor     | Low risk                                         |
   "High value" = monthly charges at or above the median (configurable).

3. **Actions** come from the playbook (``playbook.py``), matched to the
   factors that raise each customer's risk.

4. **Impact scenarios** estimate customers saved, revenue protected, campaign
   cost and net value under conservative / base / optimistic assumptions.
   These are estimates built on stated assumptions, not forecasts of results.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import settings
from src.data.schema import ID_COLUMN
from src.retention.playbook import ACTIONS, combined_success, match_actions

TIER_ORDER = ["1 - Act now", "2 - Act soon", "3 - Nurture", "4 - Monitor"]


def value_cutoff(df: pd.DataFrame) -> float:
    if settings.HIGH_VALUE_MONTHLY is not None:
        return float(settings.HIGH_VALUE_MONTHLY)
    return float(df["MonthlyCharges"].median())


def priority_tier(risk: str, high_value: bool) -> str:
    if risk == "High":
        return TIER_ORDER[0] if high_value else TIER_ORDER[1]
    if risk == "Medium":
        return TIER_ORDER[1] if high_value else TIER_ORDER[2]
    return TIER_ORDER[3]


def _parse_contributions(value) -> dict[str, float] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return json.loads(value)


def build_retention_plan(clean: pd.DataFrame, predictions: pd.DataFrame,
                         explanations: pd.DataFrame | None = None,
                         segments: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per customer with value at risk, priority and recommended actions."""
    data = clean.merge(
        predictions[[ID_COLUMN, "churn_probability", "risk_level"]], on=ID_COLUMN, how="inner")
    if explanations is not None and len(explanations):
        data = data.merge(explanations[[ID_COLUMN, "contributions_json"]], on=ID_COLUMN,
                          how="left")
    else:
        data["contributions_json"] = None
    if segments is not None and len(segments):
        data = data.merge(segments[[ID_COLUMN, "segment_name", "rfm_segment"]], on=ID_COLUMN,
                          how="left")

    horizon = settings.RETENTION_HORIZON_MONTHS
    cutoff = value_cutoff(data)
    prob = data["churn_probability"].astype(float)
    monthly = data["MonthlyCharges"].astype(float)
    data["expected_monthly_loss"] = (prob * monthly).round(2)
    data["expected_loss_horizon"] = (prob * monthly * horizon).round(2)
    data["high_value"] = monthly >= cutoff
    data["priority_tier"] = [priority_tier(r, v) for r, v in
                             zip(data["risk_level"], data["high_value"])]

    action_cols = {k: [] for k in ("action_1", "action_1_reason", "action_2",
                                   "action_2_reason", "action_basis", "actions_cost",
                                   "success_rate", "recommendation")}
    for _, row in data.iterrows():
        if row["risk_level"] == "Low":
            chosen, basis = [], "not targeted"
        else:
            contributions = _parse_contributions(row["contributions_json"])
            chosen, basis = match_actions(row, contributions)
            if not chosen:
                chosen = [("general_checkin", "no single strong driver identified")]
        keys = [k for k, _ in chosen]
        padded = chosen + [(None, None)] * (2 - len(chosen))
        action_cols["action_1"].append(ACTIONS[padded[0][0]].name if padded[0][0] else None)
        action_cols["action_1_reason"].append(padded[0][1])
        action_cols["action_2"].append(ACTIONS[padded[1][0]].name if padded[1][0] else None)
        action_cols["action_2_reason"].append(padded[1][1])
        action_cols["action_basis"].append(basis)
        action_cols["actions_cost"].append(sum(ACTIONS[k].cost for k in keys))
        action_cols["success_rate"].append(round(combined_success(keys), 4) if keys else 0.0)
        action_cols["recommendation"].append(_recommendation_text(row, chosen, basis))
    for col, values in action_cols.items():
        data[col] = values

    data["expected_saved_value"] = (
        prob * data["success_rate"] * monthly * horizon).round(2)
    targeted = data["priority_tier"] != TIER_ORDER[3]
    data["priority_rank"] = (data["expected_loss_horizon"].where(targeted)
                             .rank(ascending=False, method="first"))
    data["priority_rank"] = data["priority_rank"].astype("Int64")

    keep = [ID_COLUMN, "priority_rank", "priority_tier", "risk_level", "churn_probability",
            "MonthlyCharges", "tenure", "Contract", "expected_monthly_loss",
            "expected_loss_horizon", "high_value", "action_1", "action_1_reason", "action_2",
            "action_2_reason", "action_basis", "actions_cost", "success_rate",
            "expected_saved_value", "recommendation"]
    keep[7:7] = [c for c in ("segment_name", "rfm_segment") if c in data.columns]
    return data[keep].sort_values(["priority_rank", "churn_probability"],
                                  ascending=[True, False], na_position="last"
                                  ).reset_index(drop=True)


def _recommendation_text(row: pd.Series, chosen: list[tuple[str, str]], basis: str) -> str:
    tier = priority_tier(row["risk_level"], bool(row["high_value"]))
    head = (f"{tier}. Estimated {float(row['churn_probability']):.0%} churn risk; pays "
            f"${float(row['MonthlyCharges']):,.2f}/month.")
    if not chosen:
        return head + " No retention action needed; keep standard engagement."
    actions = "; ".join(f"{ACTIONS[k].name} ({reason})" for k, reason in chosen)
    return f"{head} Recommended: {actions}. Based on {basis}."


def impact_scenarios(plan: pd.DataFrame,
                     targets: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Estimated campaign impact for different target groups and assumption levels."""
    targets = targets or {"Act now only": TIER_ORDER[:1],
                          "Act now + Act soon": TIER_ORDER[:2],
                          "All at-risk (tiers 1-3)": TIER_ORDER[:3]}
    horizon = settings.RETENTION_HORIZON_MONTHS
    rows = []
    for target_name, tiers in targets.items():
        group = plan[plan["priority_tier"].isin(tiers)]
        prob = group["churn_probability"].astype(float)
        monthly = group["MonthlyCharges"].astype(float)
        cost = float(group["actions_cost"].sum())
        for scenario, mult in settings.SCENARIO_MULTIPLIERS.items():
            success = np.clip(group["success_rate"].astype(float) * mult, 0, 1)
            saved = float((prob * success).sum())
            revenue = float((prob * success * monthly * horizon).sum())
            rows.append({
                "target": target_name, "scenario": scenario,
                "customers_contacted": int(len(group)),
                "expected_churners": round(float(prob.sum()), 1),
                "customers_saved": round(saved, 1),
                "revenue_protected": round(revenue, 2),
                "campaign_cost": round(cost, 2),
                "net_value": round(revenue - cost, 2),
                "roi": round((revenue - cost) / cost, 2) if cost else np.nan,
            })
    return pd.DataFrame(rows).set_index(["target", "scenario"])


def summary_by(plan: pd.DataFrame, column: str) -> pd.DataFrame:
    """Value at risk and priority counts per segment (or any column)."""
    grouped = plan.groupby(column)
    table = pd.DataFrame({
        "customers": grouped.size(),
        "avg_churn_probability": grouped["churn_probability"].mean(),
        "expected_loss_horizon": grouped["expected_loss_horizon"].sum(),
        "act_now": grouped["priority_tier"].apply(lambda s: int((s == TIER_ORDER[0]).sum())),
    })
    table["share_of_expected_loss"] = (table["expected_loss_horizon"]
                                       / table["expected_loss_horizon"].sum())
    return table.sort_values("expected_loss_horizon", ascending=False)


def tier_summary(plan: pd.DataFrame) -> pd.DataFrame:
    grouped = plan.groupby("priority_tier")
    table = pd.DataFrame({
        "customers": grouped.size(),
        "avg_churn_probability": grouped["churn_probability"].mean(),
        "expected_loss_horizon": grouped["expected_loss_horizon"].sum(),
        "campaign_cost": grouped["actions_cost"].sum(),
    }).reindex(TIER_ORDER).fillna(0)
    table["customers"] = table["customers"].astype(int)
    return table


def action_counts(plan: pd.DataFrame) -> pd.Series:
    actions = pd.concat([plan["action_1"], plan["action_2"]]).dropna()
    return actions.value_counts()
