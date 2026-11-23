# Phase 6: Retention intelligence

This guide is written for readers without a Data Science background. The dashboard will reuse it in Phase 7.

## 1. What is it?
A retention plan built from the earlier phases. It answers three questions: **who to contact first**, **what to offer each customer**, and **what the campaign might be worth**.

## 2. Why is it needed?
A churn score alone doesn't tell a retention team what to do on Monday morning. Budgets and call-centre time are limited, so customers must be ranked, and offers must fit the reasons each customer is at risk.

## 3. How does it work?

**Value at risk.** For each customer:

```
expected monthly loss = churn probability × monthly charges
expected loss over 12 months = expected monthly loss × 12
```

A customer with an 80% risk paying $100/month has $80/month at risk, while one with an 80% risk paying $20 has $16. The first matters more.

**Priority tiers.**

| Tier | Who |
|---|---|
| 1 - Act now | High risk (60%+) and high value (monthly charges at or above the median) |
| 2 - Act soon | High risk and lower value, or Medium risk (30–60%) and high value |
| 3 - Nurture | Medium risk and lower value |
| 4 - Monitor | Low risk (under 30%), not contacted |

Within tiers 1–3, customers are ranked by value at risk to make the **priority call list**.

**Matching actions.** The playbook (`src/retention/playbook.py`) lists 9 actions. An action is suggested only if **both** of these are true:
1. it applies to the customer (for example, a contract offer only for month-to-month customers), and
2. the related factor actually **raises this customer's risk** in the Phase 5 explanation.

Each customer gets at most two actions, strongest driver first. If no explanation exists, the plan falls back on customer attributes and says so.

**Impact scenarios.** For each target group:

```
customers saved     = Σ churn probability × assumed success rate
revenue protected   = Σ churn probability × success rate × monthly charges × 12
net value           = revenue protected − campaign cost
```

These are calculated under Conservative (0.5×), Base (1×) and Optimistic (1.5×) success rates.

## 4. What data does it use?
Clean customer data, churn predictions (Phase 4), explanations (Phase 5) and segments (Phase 3), plus the **assumptions** in the playbook and in `config/settings.py`.

## 5. What does the result mean?
- The **priority call list** is the order in which to contact customers, each with recommended actions and the reason for them.
- The **impact scenarios** show the size of the opportunity, and how results depend on the assumptions. If a campaign only pays off in the optimistic case, it's risky.

### The assumptions, stated plainly
| Assumption | Value | Where to change it |
|---|---|---|
| Revenue horizon for a saved customer | 12 months | `RETENTION_HORIZON_MONTHS` |
| Monthly charges stay the same | — | Built into the formula |
| Action costs and success rates | See `playbook_assumptions.csv` (e.g. contract offer: $60, 15%) | `playbook.py` |
| Actions work independently | Combined = 1 − (1 − s₁)(1 − s₂) | `playbook.py` |
| High-value cut-off | Median monthly charges | `HIGH_VALUE_MONTHLY` |

**These numbers are illustrative.** They aren't measured results. A real company should replace them with its own costs, and learn its success rates from controlled experiments. That means offering an action to a random half of similar customers and comparing their churn with the other half.

## 6. How does a business use it?
- **The retention team** works through the call list from the top, using the suggested actions and reasons in the conversation.
- **Managers** compare target groups. "Act now only" is cheaper but saves fewer customers, while "all at-risk" costs more and may not pay off.
- **Analysts** turn the most common actions into experiments, which then improve the assumptions and close the loop.

## Limitations
- Predictions show association, not cause, so an offer aimed at a risk factor may not change behaviour.
- Contacting customers can have side effects. For example, it can prompt some to review their contract and leave. This model doesn't capture that.
- The data is a single snapshot of a fictional company.

## Outputs
| File | Contents |
|---|---|
| SQLite `retention_actions` | Per customer: priority rank and tier, value at risk, up to 2 actions with reasons, cost, success rate, recommendation text |
| `reports/retention/priority_call_list.csv` | Tiers 1–3 in priority order, ready to export |
| `reports/retention/impact_scenarios.csv`, `tier_summary.csv`, `playbook_assumptions.csv`, `value_at_risk_by_segment.csv` | Supporting tables |
| `reports/retention_report.md` | The full report, with headline, tiers, top 10, scenarios, assumptions, limitations and charts |
| `reports/figures/retention_*.png` | 4 charts |
