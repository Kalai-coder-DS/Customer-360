# Interview preparation guide: Customer360 AI

Before an interview, run `python scripts/build_results.py` and read [RESULTS.md](RESULTS.md). Quote **your** numbers from there. Don't memorise the examples below as facts.

---

## 1. Pitches

### 30 seconds
> "I built Customer360 AI, an end-to-end churn analytics platform on the IBM Telco dataset. It cleans and validates the data, segments customers, predicts each customer's churn probability with a model chosen by cross-validation, explains every prediction with Shapley values, and turns that into a prioritised retention call list. It's all in a Streamlit dashboard designed so a non-technical manager can use it, with 150+ automated tests and CI."

### 2 minutes (problem → approach → result → learning)
1. **Problem:** a subscription business loses revenue when customers leave, and a retention team can't call everyone. They need to know who, why, and what to offer.
2. **Approach:** validated and cleaned the data, explored drivers with proper statistics, segmented customers, compared Logistic Regression and Random Forest against a baseline without leakage, explained predictions, and ranked customers by value at risk.
3. **Result:** *(from RESULTS.md)* e.g. "26.5% churn; contract type is the strongest driver; the chosen model reaches ROC-AUC __ on unseen customers; 291 active customers are high risk, with an estimated $1.67M of revenue at risk."
4. **Learning:** honest evaluation and clear explanations matter as much as the model. I state assumptions and the difference between correlation and causation throughout.

---

## 2. Project story (STAR)
- **Situation:** churn costs subscription businesses recurring revenue, and teams need to target their effort.
- **Task:** build a portfolio-quality platform covering the full lifecycle, understandable without data science knowledge.
- **Action:** 9 phases: validation → cleaning/EDA → segmentation → prediction → explainability → retention planning → dashboard → testing/deployment → documentation.
- **Result:** a deployed dashboard, a reproducible one-command pipeline, 150+ tests in CI, and documented results.

---

## 3. Likely questions and strong answers

### Data and cleaning
**Q: What data quality issues did you find?**
11 blank `TotalCharges`, all for customers with 0 months' tenure. They haven't been billed yet, so I set them to 0 by business rule instead of imputing a statistic. I also check duplicates, invalid categories and types, and log every change.

**Q: Why not fill missing values with the median during cleaning?**
A median computed on all rows includes the test set, which is data leakage. Statistical imputation happens inside the model pipeline, fitted on training folds only.

**Q: How did you handle categorical variables?**
One-hot encoding inside a `ColumnTransformer` (binary columns use a single column), with `handle_unknown="ignore"` so new categories in uploaded files don't crash scoring.

### EDA and statistics
**Q: How did you decide which factors matter?**
Churn rate per group plus statistical tests: chi-square with **Cramér's V** for categorical columns, **Mann-Whitney U** with rank-biserial effect size for numeric ones. With 7,000 rows almost everything is "significant", so I rank by **effect size**, not p-value.

**Q: Is contract type causing churn?**
Not necessarily. Less committed customers may *choose* month-to-month contracts. The data shows association. To prove an effect, I'd run an experiment, such as offering a contract discount to a random half of at-risk customers.

### Segmentation
**Q: This dataset has no transaction dates. How did you do RFM?**
I didn't pretend. I used labelled substitutes: Loyalty = tenure, Engagement = number of services, Monetary = monthly charges. The app and the docs say clearly that it's an adaptation.

**Q: How did you choose the number of clusters?**
I ran K-Means for K = 2–8 and chose the best silhouette score within 3–6 (a range a team can act on). I left out TotalCharges (≈ tenure × charges, so double-counted) and churn (segments should describe customers, not predict churn). The silhouette was ~0.43, which is typical for customer data, where groups overlap.

**Q: Why standardise before K-Means?**
K-Means uses distances. Without scaling, monthly charges (up to ~120) would dominate the number of services (0–9).

### Modelling
**Q: Why not just use accuracy?**
73.5% of customers stay, so "nobody churns" is 73.5% accurate and useless. I report precision, recall, F1, ROC-AUC, PR-AUC, a confusion matrix and calibration.

**Q: How did you prevent data leakage?**
1. A stratified test split, used once.
2. All preprocessing inside a scikit-learn `Pipeline`, so cross-validation refits it on each training fold.
3. The model chosen on cross-validation, not test score.
4. The threshold tuned on out-of-fold training predictions.
5. The ID, the target and the segment labels excluded from the inputs.
I also test that a model trained on random data scores near 0.5. If something leaked, it would score high.

**Q: How did you handle class imbalance?**
By tuning the decision threshold rather than re-weighting classes, so predicted probabilities stay meaningful ("70% risk" ≈ 7 in 10 similar customers churned). The calibration chart checks this.

**Q: Logistic Regression or Random Forest, and why did one win?**
*(Use RESULTS.md section 4.)* On tabular data like this, a well-regularised linear model is often as good as a forest, and it's far easier to explain. The choice was made by cross-validated ROC-AUC, and the overfitting check compares training and test ROC-AUC.

**Q: What does ROC-AUC mean in plain words?**
Pick one churner and one stayer at random. ROC-AUC is the chance the model gives the churner the higher score. 0.5 is a coin flip, 1.0 is perfect.

**Q: How would you choose the threshold for the business?**
It depends on what an action costs. Cheap actions like emails can tolerate more false alarms (lower threshold, higher recall). Expensive discounts need higher precision. The dashboard's what-if sliders show that trade-off.

### Explainability
**Q: How do you explain an individual prediction?**
Shapley values: each factor's fair share of the difference between this customer's prediction and an average customer's, in log-odds, adding up exactly. For Logistic Regression it's exact (coefficient × (value − average)). For Random Forest I estimate it by permutation sampling.

**Q: Why not use the SHAP library?**
It's heavy and often lags new Python versions (I developed on 3.14). The underlying maths is simple to implement and test: my tests check that contributions add up to the prediction, and that the sampling method matches the exact one.

**Q: Global vs local explanations?**
Global = permutation importance (how much ROC-AUC drops when a factor is shuffled). Local = Shapley values for one customer.

### Retention and business impact
**Q: How do you turn predictions into action?**
Value at risk = probability × monthly charges × 12. Priority tiers combine risk and value. Actions come from a playbook and are suggested only if they apply *and* target a factor that raises that customer's risk.

**Q: How confident are you in the $ impact?**
Not very, and I say so. Costs and success rates are illustrative assumptions, shown in the app with conservative, base and optimistic scenarios. The next step is A/B testing to learn real success rates.

### Engineering
**Q: How is the project structured?**
Layered: config → data → business logic → dashboard. Pages are layout-only; calculations live in tested modules. A single theme file defines all colours.

**Q: How did you test it?**
150+ pytest tests on generated data:
- unit tests (cleaning rules, statistics, Shapley additivity)
- "learns a planted pattern" and "near-random on random data" model tests
- end-to-end pipeline tests
- Streamlit `AppTest` page rendering
GitHub Actions runs them on every push.

**Q: How is it deployed?**
Streamlit Community Cloud from GitHub. Data isn't in the repo, so on first start the app downloads the dataset and builds everything once. Demo mode prevents storing visitors' uploads.

---

## 4. Questions that test depth (be ready)
- *What would you do with more time?* Gradient boosting with calibration; time-stamped data for true RFM and survival analysis (time-to-churn); A/B testing retention actions; monitoring for data drift.
- *What's the biggest limitation?* Fictional, single-snapshot data, and explanations that are associations, not causes.
- *What surprised you?* e.g. how much of the signal is in contract and tenure, or that the simpler model held its own *(check your results)*.
- *How would this work in production?* Scheduled re-scoring, model versioning, drift monitoring, feedback from campaign outcomes into the success-rate assumptions.

## 5. Numbers to know (fill from RESULTS.md)

| Fact | Your number |
|---|---|
| Customers / churn rate | 7,043 / 26.5% |
| Strongest driver | Contract (month-to-month vs two-year churn) |
| Segments / silhouette | 4 / 0.425 |
| Chosen model | __ |
| Test ROC-AUC / recall / precision | __ / __ / __ |
| Top 3 factors | __ |
| High-risk active customers / revenue at risk | 291 / ~$1.67M (estimate) |
| Tests | 150+ (CI on every push) |
