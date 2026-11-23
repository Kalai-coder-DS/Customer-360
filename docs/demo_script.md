# Demo presentation script: Customer360 AI

**Before you start (2 minutes):** open the live app or `streamlit run app/streamlit_app.py`, maximise the browser, close other tabs, and keep [RESULTS.md](RESULTS.md) open for numbers. Use the menu on the left to move between pages.

---

## Full demo (about 5–6 minutes)

### 0:00 · Opening (Executive overview)
**Say:** "Customer360 AI helps a subscription business decide who to keep, why they might leave and what to do. It uses the IBM Telco dataset: 7,043 customers of a fictional telecom company."
**Show:** the **"What this page tells you"** box.
**Say:** "Every page starts with the takeaways in plain words, generated from the data. Here, 26.5% of customers left, and month-to-month contracts churn far more than two-year contracts."
**Point at:** the KPI cards, especially **Active at high risk** and **Revenue at risk**. "Revenue at risk is labelled as an estimate. I'm careful to separate facts from assumptions."

### 0:45 · The workflow (scroll down)
**Show:** the 7-step workflow strip.
**Say:** "The platform follows the full lifecycle, from raw data to cleaning, analysis, segmentation, prediction, explanation and action. Each stage has its own page."

### 1:15 · Segmentation
**Show:** segment cards (highest churn first), then **Size vs revenue**.
**Say:** "K-Means found four natural groups using tenure, charges and number of services. I chose the number of groups with the silhouette score. One segment combines the highest churn with a large share of revenue, so that's where retention should focus."
**Mention:** "The dataset has no purchase dates, so I used labelled RFM substitutes instead of pretending to do classic RFM."

### 2:00 · Churn prediction
**Show:** the takeaways and KPI cards, then **Models compared** and the **confusion matrix**.
**Say:** "I compared Logistic Regression and Random Forest against a baseline that always guesses the average. Accuracy alone is misleading here, because predicting 'no churn' for everyone is already about 73% accurate, so I use ROC-AUC, recall and precision. The model was chosen by cross-validation, and the test set was used once."
**Optional:** the **How it was built** tab, to talk about data-leakage prevention.

### 3:00 · Customer 360 profile
**Show:** the top customer in the list (highest risk).
**Say:** "This is where it becomes actionable. The gauge shows this customer's churn probability. The chart shows why: coral factors push the risk up, violet ones push it down, compared with an average customer. These are Shapley values. They're exact for Logistic Regression, and I implemented them directly instead of using the SHAP library."
**Point at:** the **Recommended next step** card. "The suggested action targets this customer's own risk factors."

### 4:00 · Explainable AI
**Show:** **Which factors the model relies on most**, then pick **Contract** in the dropdown.
**Say:** "Globally, the model relies most on these factors. The warning here matters: this is correlation, not causation. To know whether an action works, you test it on a random group."

### 4:30 · Retention strategy
**Show:** priority tiers, then filter the call list (e.g. **Act now**) and point at **Download**.
**Say:** "Customers are ranked by value at risk, which is probability times monthly charges, into tiers. The team gets a filtered call list they can export."
**Show:** the **Impact what-if** tab and move the success-rate slider.
**Say:** "The impact is an estimate with visible, adjustable assumptions. If a campaign only pays off in the optimistic case, that's a warning sign."

### 5:15 · Close
**Show:** **Reports & data** (briefly: upload, validation, downloads).
**Say:** "Behind the dashboard there's a one-command pipeline, 150+ automated tests running on every push, and a cloud deployment. The next step I'd take is A/B testing the retention actions to replace assumptions with measured results. Happy to go deeper into any part."

---

## Short demo (2 minutes)
1. **Overview** (30 s): the problem, the takeaways box, revenue at risk.
2. **Customer 360** (45 s): the highest-risk customer, the gauge, *why* (the Shapley chart), the recommended action.
3. **Retention** (30 s): tiers, call list, download.
4. **Close** (15 s): no data leakage, cross-validated model choice, correlation ≠ causation, tests and CI.

---

## Tips
- **Lead with the business question, not the algorithm.** "Who should we call first?" lands better than "I used K-Means".
- **Say the honest caveats yourself:** fictional data, RFM substitutes, estimates, correlation vs causation. Interviewers trust candidates who state limitations.
- **Have one deep-dive ready:** data leakage prevention, or how Shapley values work.
- **If something breaks live,** switch to the screenshots in the README and keep talking. It shows composure.
