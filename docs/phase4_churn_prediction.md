# Phase 4: Churn prediction

This guide is written for readers without a Data Science background. The dashboard will reuse it in Phase 7.

## 1. What is it?
A model that estimates, for every customer, the **probability that they will churn**. Each customer then gets a risk level (Low / Medium / High) and a yes/no flag.

## 2. Why is it needed?
EDA shows which *groups* churn more. A model combines all 19 characteristics at once to score **each individual customer**, so the retention team knows exactly whom to contact first.

## 3. How does it work?

| Step | What happens | Why |
|---|---|---|
| Feature engineering | Add `services_count`, `avg_monthly_spend` (TotalCharges ÷ tenure) and `auto_payment` | Express useful ideas in a form the model can use |
| Stratified split | 80% of customers for training, 20% kept aside for a final test, both with the same churn rate | A fair final exam on customers the model has never seen |
| Preprocessing in a pipeline | Fill gaps (median / most common), scale numbers, one-hot encode categories | Learned from training data only, which **prevents data leakage** |
| Cross-validation | Split the training data into 5 folds; train on 4 and score the 5th, five times | A stable estimate without touching the test set |
| Compare models | A baseline (always guesses the average), Logistic Regression and Random Forest | The baseline is the score to beat |
| Select | The model with the best cross-validated ROC-AUC | Choosing by test score would secretly train on the test set |
| Threshold | Pick the probability cut-off that best balances precision and recall (F1), using training data | With 26.5% churners, 0.5 is rarely the best cut-off |
| Test once | Score the 20% held-out customers | An honest final performance figure |
| Refit and score | Retrain the chosen model on everyone and score all customers | Used by the dashboard |

**The two models in plain words:**
- **Logistic Regression** adds up weighted evidence, such as "+ for month-to-month" and "− for long tenure", and turns the total into a probability. It's transparent and fast.
- **Random Forest** builds 300 decision trees, each asking yes/no questions such as "tenure < 12?", and averages their votes. It can capture interactions between characteristics, but it's harder to explain.

**Class imbalance** is handled by choosing the threshold, not by re-weighting the classes. That keeps the probabilities honest: "70% risk" means roughly 7 in 10 similar customers churned. The calibration chart checks this.

## 4. What data does it use?
The `customers_clean` table: 19 original characteristics plus 3 engineered ones. The customer ID, the target and the Phase 3 segment labels are **never** used as inputs.

## 5. What does the result mean?

| Metric | Plain meaning |
|---|---|
| ROC-AUC | Pick one churner and one stayer at random: how often does the model give the churner a higher score? 0.5 = coin flip, 1.0 = always |
| Precision | Of the customers flagged, how many actually churned? (Low = wasted offers) |
| Recall | Of the customers who churned, how many were flagged? (Low = missed churners) |
| F1 | A single balance of precision and recall |
| PR-AUC | Precision-recall across all thresholds; random guessing scores the churn rate (~0.27) |
| Brier score | How close the probabilities are to what actually happened (lower is better) |
| Confusion matrix | Counts of correct flags, false alarms, missed churners and correct "leave alone" decisions |

**Example explanation for one customer:** "This customer has an estimated 74% probability of churning according to the trained model, which puts them in the High risk level." *Why* the model says so comes in Phase 5 (Explainable AI).

**Honest limits:**
- Scores for existing customers come from the final model, which has seen them, so the honest performance figures are the test-set ones.
- The model finds patterns, not causes.
- The data is a single snapshot of a fictional company.

## 6. How does a business use it?
- Sort customers by churn probability and contact the High-risk ones first.
- Use the threshold to match capacity. A lower threshold catches more churners but creates more false alarms, so it suits cheap actions like emails. A higher threshold suits expensive actions like discounts.
- Use the risk-level table to plan the size of retention campaigns. Phase 6 turns this into recommended actions and estimated impact.

## Outputs

| File | Contents |
|---|---|
| `models/churn_model.joblib` | The final pipeline, threshold, risk thresholds, feature lists and test metrics |
| SQLite `churn_predictions` | Per customer: probability, risk level, flag, actual churn, whether they were in the test set |
| `reports/model_report.md` | The full report, with plain-language summary, tables and charts |
| `reports/modeling/*` | CV scores, test metrics, confusion matrix, risk levels, summary.json |
| `reports/figures/model_*.png` | 7 charts |
