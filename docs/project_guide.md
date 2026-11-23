# Customer360 AI: project guide

A complete walkthrough of the project in plain language, for reviewers, interviewers and anyone new to data science. Key numbers are in [RESULTS.md](RESULTS.md).

## 1. The business problem
A subscription business (here, a telecom provider) loses revenue every time a customer leaves (**churns**). Keeping an existing customer is usually cheaper than winning a new one, but a retention team can't call everyone. It needs to know **who** is likely to leave, **why**, whether they're **worth** the effort, and **what** to offer them.

## 2. The data
The **IBM Telco Customer Churn** dataset: 7,043 customers, 21 columns covering demographics, services, contract, billing, and whether they left in the last month (`Churn`). About 26.5% of customers churned. The company is fictional, and the data is a single snapshot with no dates. → [dataset.md](dataset.md)

## 3. The workflow, step by step

### Phase 1: Load and validate
The CSV is loaded unchanged (the "raw layer") and checked: required columns, allowed values, blanks, duplicates, number formats and target balance. Problems are reported as **errors** (stop) or **warnings** (fix later). The raw data and an audit record go into **SQLite**.
*Result:* valid file; 11 blank `TotalCharges` values found.

### Phase 2: Clean and explore
Cleaning applies only rules that are certain. For example, the 11 blank charges belong to customers with 0 months' tenure who haven't been billed yet, so they become 0. Anything statistical (filling with averages, scaling) is left for the model pipeline, to prevent **data leakage**.
The EDA measures churn by group, and uses chi-square tests with Cramér's V (categories) and Mann-Whitney tests with effect sizes (numbers) to rank what goes together with churn. → [phase2](phase2_cleaning_and_eda.md)
*Result:* contract type is the strongest signal (month-to-month ~43% churn vs ~3% for two-year contracts), and churn is highest in the first year.

### Phase 3: Segment customers
- **RFM-style scores:** true RFM needs purchase dates, which this data lacks, so honest substitutes are used: **Loyalty** = tenure, **Engagement** = number of services, **Monetary** = monthly charges, each scored 1–5. Simple rules name five segments.
- **K-Means** groups customers by tenure, charges and services (standardised). The number of groups is chosen by the silhouette score, limited to 3–6 so a team can act on them. Churn is *not* used to build segments; it is only used to describe them afterwards. → [phase3](phase3_segmentation.md)
*Result:* 4 segments. "Established mid-spend" has 47% churn and 41% of revenue, making it the retention priority.

### Phase 4: Predict churn
Three models are compared: a **baseline** (always guesses the average), **Logistic Regression** and **Random Forest**.
1. A stratified 80/20 split creates a test set that is used only once.
2. 5-fold cross-validation runs on the training data.
3. The model is chosen by cross-validated ROC-AUC.
4. The decision threshold is tuned for F1 on out-of-fold predictions.
5. The model is tested once, then retrained on everyone to score the whole customer base.

Imbalance is handled with the threshold, which keeps probabilities honest. → [phase4](phase4_churn_prediction.md)
*Result:* see RESULTS.md, section 4.

### Phase 5: Explain predictions
**Shapley values** split each prediction into a fair share per factor: *this* customer compared with an average customer. They are exact for Logistic Regression and estimated by sampling for Random Forest, without the heavy SHAP package. **Permutation importance** shows which factors the model relies on overall. Every explanation is phrased as an association, not a cause. → [phase5](phase5_explainable_ai.md)

### Phase 6: Plan retention
- **Value at risk** = churn probability × monthly charges × 12 months.
- **Priority tiers:** Act now, Act soon, Nurture, Monitor.
- **Actions:** a 9-action playbook. An action is suggested only if it applies to the customer *and* targets a factor that raises *their* risk.
- **Impact estimates:** conservative, base and optimistic, with every assumption stated. → [phase6](phase6_retention_intelligence.md)
*Result:* 291 active high-risk customers; about $1.67M of revenue at risk over 12 months (estimate).

### Phase 7: Dashboard
A Streamlit app with 8 pages, a plain-language teaching panel on every page, and a dark plum design system. → [phase7](phase7_dashboard.md)

### Phase 8: Testing and deployment
150+ automated tests (all on generated data), GitHub Actions CI, a one-command pipeline, an automatic first-start build for Streamlit Community Cloud, and a demo mode that never stores visitors' uploads. → [testing](testing.md), [deployment](deployment.md)

## 4. Key design decisions, and why

| Decision | Reason |
|---|---|
| Never judge the model on accuracy alone | With 73.5% stayers, "nobody churns" scores 73.5% and catches no one |
| Statistical preprocessing only inside the model pipeline | Prevents test information leaking into training |
| Choose the model on cross-validation, test once | Keeps the test score an honest estimate |
| Tune the threshold, don't re-weight classes | Imbalance handled without distorting probabilities |
| RFM substitutes, clearly labelled | No dates in the data; honesty over a misleading label |
| Shapley values without the SHAP package | Same idea, fewer dependencies, works on Python 3.14 |
| Actions tied to each customer's own risk factors | A relevant offer instead of a generic discount |
| All money figures labelled as estimates | Assumptions are illustrative until tested |

## 5. Limitations
- Fictional company, single snapshot, one industry and region.
- Explanations show association, not cause. Retention actions should be tested with controlled experiments.
- Retention costs and success rates are illustrative.
- Scores for existing customers come from the final model, which has seen them. The honest performance figures are the test-set ones.

## 6. How to run it
See the Quick start in the [README](../README.md).
