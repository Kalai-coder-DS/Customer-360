"""Retention playbook: possible actions and when each one applies.

Every action has a cost per customer and an assumed success rate (the share
of would-be churners it persuades to stay). **These numbers are illustrative
assumptions**, chosen to be plausible for a telecom business. They are not
measured results. A real company should replace them with its own figures,
ideally learned from controlled experiments (A/B tests).

An action is recommended for a customer only when BOTH are true:
1. the customer's attributes make it applicable (e.g. "contract offer" only
   for month-to-month customers), and
2. the related factor actually raises that customer's churn risk according
   to the Phase 5 explanation (contribution above the "minimal" level).
When no explanation is available, condition 1 alone is used and the
recommendation says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

MIN_CONTRIBUTION = 0.1  # same "minimal" cut-off as the Phase 5 explanations


@dataclass(frozen=True)
class Action:
    key: str
    name: str
    description: str
    channel: str
    cost: float            # ASSUMPTION: cost per customer contacted (USD)
    success_rate: float    # ASSUMPTION: share of would-be churners retained


ACTIONS: dict[str, Action] = {a.key: a for a in [
    Action("contract_offer", "Contract upgrade offer",
           "Offer a discount for moving from month-to-month to a 1-year contract.",
           "Phone call", cost=60.0, success_rate=0.15),
    Action("onboarding_call", "Early-life check-in",
           "Personal call in the first year to resolve set-up issues and explain the service.",
           "Phone call", cost=15.0, success_rate=0.12),
    Action("tech_support_trial", "Free tech support trial",
           "Three months of premium tech support at no charge.",
           "Email + in-app", cost=30.0, success_rate=0.10),
    Action("security_bundle", "Online security trial",
           "Free online security add-on for three months.",
           "Email + in-app", cost=20.0, success_rate=0.10),
    Action("plan_review", "Plan and price review",
           "Review the plan to make sure the customer pays only for what they use.",
           "Phone call", cost=25.0, success_rate=0.10),
    Action("autopay_incentive", "Automatic payment incentive",
           "One-off bill credit for switching to automatic bank or card payment.",
           "Email", cost=10.0, success_rate=0.08),
    Action("service_quality_check", "Fiber service quality check",
           "Proactive line check and speed test; fix issues before the customer complains.",
           "Technician / remote", cost=40.0, success_rate=0.08),
    Action("senior_support", "Dedicated senior support",
           "Priority support line and a simple guide to the services.",
           "Phone call", cost=15.0, success_rate=0.08),
    Action("general_checkin", "Personal satisfaction check-in",
           "Short call to ask about satisfaction and resolve any open issues.",
           "Phone call", cost=10.0, success_rate=0.05),
]}


@dataclass(frozen=True)
class Rule:
    features: tuple[str, ...]                       # model factors that justify the action
    applies: Callable[[pd.Series], bool]           # customer-attribute condition
    action: str
    reason: str                                     # shown to the retention agent


RULES: list[Rule] = [
    Rule(("Contract",), lambda r: r["Contract"] == "Month-to-month", "contract_offer",
         "customer is on a month-to-month contract"),
    Rule(("tenure",), lambda r: float(r["tenure"]) <= 12, "onboarding_call",
         "customer is in their first year"),
    Rule(("TechSupport",), lambda r: r["TechSupport"] == "No", "tech_support_trial",
         "customer has no tech support"),
    Rule(("OnlineSecurity",), lambda r: r["OnlineSecurity"] == "No", "security_bundle",
         "customer has no online security"),
    Rule(("MonthlyCharges", "avg_monthly_spend"), lambda r: float(r["MonthlyCharges"]) >= 70,
         "plan_review", "customer pays high monthly charges"),
    Rule(("PaymentMethod", "auto_payment"), lambda r: r["PaymentMethod"] == "Electronic check",
         "autopay_incentive", "customer pays by electronic check"),
    Rule(("InternetService",), lambda r: r["InternetService"] == "Fiber optic",
         "service_quality_check", "customer uses fiber optic internet"),
    Rule(("SeniorCitizen",), lambda r: float(r["SeniorCitizen"]) == 1, "senior_support",
         "customer is a senior citizen"),
]


def match_actions(customer: pd.Series, contributions: dict[str, float] | None,
                  max_actions: int = 2) -> tuple[list[tuple[str, str]], str]:
    """Return up to ``max_actions`` (action_key, reason) pairs and the basis used.

    With contributions: rules are ranked by how strongly their factor raises
    this customer's risk. Without: rules are taken in playbook order.
    """
    matches: list[tuple[float, str, str]] = []
    for order, rule in enumerate(RULES):
        if not rule.applies(customer):
            continue
        if contributions is not None:
            push = max(contributions.get(f, 0.0) for f in rule.features)
            if push <= MIN_CONTRIBUTION:
                continue
            matches.append((-push, rule.action, f"{rule.reason}, which raises their predicted risk"))
        else:
            matches.append((order, rule.action, rule.reason))
    matches.sort(key=lambda m: m[0])
    chosen, seen = [], set()
    for _, action, reason in matches:
        if action not in seen:
            chosen.append((action, reason))
            seen.add(action)
        if len(chosen) == max_actions:
            break
    basis = "model explanation" if contributions is not None else "customer attributes"
    return chosen, basis


def combined_success(action_keys: list[str]) -> float:
    """Chance that at least one action works, assuming they act independently."""
    fail = 1.0
    for key in action_keys:
        fail *= 1 - ACTIONS[key].success_rate
    return 1 - fail


def playbook_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "action": a.name, "description": a.description, "channel": a.channel,
        "assumed_cost_usd": a.cost, "assumed_success_rate": a.success_rate,
    } for a in ACTIONS.values()]).set_index("action")
