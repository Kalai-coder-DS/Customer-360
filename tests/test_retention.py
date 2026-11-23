import json

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_customer_data
from src.retention import planner, playbook
from tests.conftest import make_random_customers


def _customer(**overrides):
    base = {"Contract": "Month-to-month", "tenure": 3, "TechSupport": "No",
            "OnlineSecurity": "No", "MonthlyCharges": 95.0, "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic", "SeniorCitizen": 0}
    base.update(overrides)
    return pd.Series(base)


# --- Playbook --------------------------------------------------------------

def test_actions_follow_the_strongest_risk_factors():
    contributions = {"tenure": 1.4, "Contract": 0.9, "TechSupport": 0.3, "gender": 0.05}
    chosen, basis = playbook.match_actions(_customer(), contributions)
    assert [k for k, _ in chosen] == ["onboarding_call", "contract_offer"]
    assert basis == "model explanation"
    assert "raises their predicted risk" in chosen[0][1]


def test_action_needs_both_applicability_and_a_risk_raising_factor():
    contributions = {"Contract": 1.0, "TechSupport": 0.8}
    # Two-year customer: contract offer not applicable even though Contract has weight
    chosen, _ = playbook.match_actions(_customer(Contract="Two year"), contributions)
    assert [k for k, _ in chosen] == ["tech_support_trial"]
    # Factor lowers risk -> no action for it
    chosen, _ = playbook.match_actions(_customer(), {"Contract": -0.9})
    assert chosen == []


def test_attribute_fallback_without_explanations():
    chosen, basis = playbook.match_actions(_customer(), None)
    assert basis == "customer attributes" and len(chosen) == 2
    assert chosen[0][0] == "contract_offer"


def test_combined_success_rate():
    p = playbook.combined_success(["contract_offer", "onboarding_call"])
    expected = 1 - (1 - 0.15) * (1 - 0.12)
    assert p == pytest.approx(expected)
    assert playbook.combined_success([]) == 0


def test_playbook_assumptions_are_sane():
    table = playbook.playbook_table()
    assert table["assumed_success_rate"].between(0, 1).all()
    assert (table["assumed_cost_usd"] > 0).all()
    for rule in playbook.RULES:
        assert rule.action in playbook.ACTIONS


# --- Planner ---------------------------------------------------------------

@pytest.mark.parametrize("risk, high_value, tier", [
    ("High", True, "1 - Act now"), ("High", False, "2 - Act soon"),
    ("Medium", True, "2 - Act soon"), ("Medium", False, "3 - Nurture"),
    ("Low", True, "4 - Monitor"),
])
def test_priority_tiers(risk, high_value, tier):
    assert planner.priority_tier(risk, high_value) == tier


@pytest.fixture(scope="module")
def plan_inputs():
    clean = clean_customer_data(make_random_customers(300, seed=12))[0]
    rng = np.random.default_rng(0)
    prob = rng.uniform(0, 1, len(clean)).round(4)
    preds = pd.DataFrame({"customerID": clean["customerID"], "churn_probability": prob,
                          "risk_level": np.where(prob >= 0.6, "High",
                                                 np.where(prob >= 0.3, "Medium", "Low"))})
    contrib = [json.dumps({"Contract": 0.8, "tenure": 0.5, "TechSupport": -0.2})] * len(clean)
    explanations = pd.DataFrame({"customerID": clean["customerID"],
                                 "contributions_json": contrib})
    return clean, preds, explanations


def test_plan_has_one_row_per_customer_and_correct_values(plan_inputs):
    clean, preds, explanations = plan_inputs
    plan = planner.build_retention_plan(clean, preds, explanations)
    assert len(plan) == len(clean) and plan["customerID"].is_unique
    row = plan.iloc[0]
    assert row["expected_monthly_loss"] == pytest.approx(
        row["churn_probability"] * row["MonthlyCharges"], abs=0.01)
    assert row["expected_loss_horizon"] == pytest.approx(row["expected_monthly_loss"] * 12,
                                                         abs=0.2)


def test_low_risk_customers_get_no_action(plan_inputs):
    clean, preds, explanations = plan_inputs
    plan = planner.build_retention_plan(clean, preds, explanations)
    low = plan[plan["risk_level"] == "Low"]
    assert low["action_1"].isna().all() and (low["actions_cost"] == 0).all()
    assert low["priority_rank"].isna().all()
    at_risk = plan[plan["risk_level"] != "Low"]
    assert at_risk["action_1"].notna().all()


def test_priority_rank_follows_value_at_risk(plan_inputs):
    clean, preds, explanations = plan_inputs
    plan = planner.build_retention_plan(clean, preds, explanations)
    ranked = plan.dropna(subset=["priority_rank"]).sort_values("priority_rank")
    assert ranked["expected_loss_horizon"].is_monotonic_decreasing
    assert ranked["priority_rank"].iloc[0] == 1


def test_recommendation_text(plan_inputs):
    clean, preds, explanations = plan_inputs
    plan = planner.build_retention_plan(clean, preds, explanations)
    first = plan.iloc[0]["recommendation"]
    assert first.startswith(plan.iloc[0]["priority_tier"])
    assert "Recommended:" in first and "Based on model explanation" in first


def test_impact_scenarios_scale_with_assumptions(plan_inputs):
    clean, preds, explanations = plan_inputs
    plan = planner.build_retention_plan(clean, preds, explanations)
    table = planner.impact_scenarios(plan)
    for target in table.index.get_level_values(0).unique():
        rows = table.loc[target]
        assert rows.loc["Conservative", "revenue_protected"] < rows.loc["Base", "revenue_protected"] \
            < rows.loc["Optimistic", "revenue_protected"]
        assert rows["campaign_cost"].nunique() == 1  # cost does not depend on success
        assert (rows["net_value"] == (rows["revenue_protected"] - rows["campaign_cost"]).round(2)
                ).all()
    contacted = table.xs("Base", level="scenario")["customers_contacted"]
    assert contacted.is_monotonic_increasing  # wider targets contact more customers


def test_plan_works_without_explanations_or_segments(plan_inputs):
    clean, preds, _ = plan_inputs
    plan = planner.build_retention_plan(clean, preds)
    at_risk = plan[plan["risk_level"] != "Low"]
    assert (at_risk["action_basis"] == "customer attributes").all()
    assert "segment_name" not in plan.columns
