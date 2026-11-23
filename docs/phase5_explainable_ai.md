# Phase 5: Explainable AI

This guide is written for readers without a Data Science background. The dashboard will reuse it in Phase 7.

## 1. What is it?
Explanations of the churn model at two levels:
- **Global:** which customer characteristics the model relies on most overall.
- **Local:** for one customer, which factors push their churn risk up and which push it down.

## 2. Why is it needed?
"Churn probability: 87%" on its own isn't actionable, and nobody should trust it blindly. Explanations let a retention agent see *why* a customer is flagged. They let a manager check that the model is sensible, and they help catch mistakes, such as a model relying on something it shouldn't.

## 3. How does it work?

**Shapley values (the idea behind SHAP).** Imagine the prediction as a team score, and the features as players. A feature's Shapley value is its fair share of the difference between *this customer's* prediction and an *average customer's* prediction. The values add up exactly:

```
average customer's log-odds + sum of all contributions = this customer's log-odds
probability = 1 / (1 + e^(-log-odds))
```

Log-odds are the scale where contributions add up. A push of +1.0 roughly multiplies the odds of churning by 2.7.

| Model | Method | Speed |
|---|---|---|
| Logistic Regression | **Exact** linear Shapley values: coefficient × (customer's value − average value) | Instant for all customers |
| Random Forest | **Sampling** Shapley values: switch features one at a time from an average customer to this customer, in 100 random orders, and average the change | About 0.25 s per customer; the riskiest 300 are precomputed |

**Why not the SHAP package?** It's heavy, and it often lags behind new Python versions (your laptop runs Python 3.14). The same calculations are implemented directly with scikit-learn and NumPy, and the tests check them: exact values add up perfectly, and the sampling method reproduces the exact values.

**Global importance (permutation importance).** Shuffle one feature's values across customers and measure how much the model's ROC-AUC drops. A big drop means the model relies heavily on that feature. This works for any model.

**Strength words** used in explanations: strong (push of 1.0 or more), moderate (0.4 or more), slight (0.1 or more). Pushes below 0.1 count as minimal and are left out of the text.

## 4. What data does it use?
The saved Phase 4 model, the `customers_clean` table, the `churn_predictions` table, and a random sample of 200 customers that defines "an average customer".

## 5. What does the result mean?
Example of a generated explanation:

> This customer has an estimated 83% probability of churning according to the trained Logistic Regression model, which places them in the High risk level. The factors that raise this estimate the most are Tenure: 0 months (strong), Contract: Month-to-month (strong) and Internet service: Fiber optic (slight). These are patterns the model learned from past customers; they show association, not proven cause.

Only factors with a real, non-minimal contribution are named, so the text never claims something the model didn't use.

## Correlation vs causation (important)
An explanation answers *"why did the model give this score?"* It does **not** answer *"what will make this customer stay?"*
- If month-to-month contracts raise the predicted risk, it may be because less committed customers *choose* month-to-month. Moving someone to a yearly contract won't necessarily change their mind.
- Features can stand in for each other. Tenure and total charges move together, so the model may split credit between them.
- To learn what *causes* change, a business tests actions. For example, it can offer a contract upgrade to a random half of at-risk customers and compare their churn with the other half.

## 6. How does a business use it?
- **Retention agents** see the top reasons on each customer's profile and tailor the conversation. For example, a customer flagged for "no tech support" can be offered support.
- **Managers** check that the main factors make business sense before trusting the model.
- **Analysts** use the global factors to choose which retention experiments to run (Phase 6).

## Outputs

| File | Contents |
|---|---|
| SQLite `churn_explanations` | Per customer: probability, risk level, average-customer baseline, top risk factors, top protective factors, explanation text, all contributions (JSON) |
| `reports/explain_report.md` | Global importance, how top features move predictions, example explanations, caveats |
| `reports/explain/*.csv` | Global importance, direction tables, Logistic Regression coefficients and odds ratios |
| `reports/figures/explain_*.png` | 8 charts |
