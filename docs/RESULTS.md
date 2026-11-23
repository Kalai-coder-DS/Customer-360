# Customer360 AI: key results

_Generated 2026-11-23 by `scripts/build_results.py` from the pipeline outputs. Every number below comes from the data; none are typed in by hand. Dataset: IBM Telco Customer Churn (fictional telecom company)._

## 1. The customer base
- **7,043 customers**, of whom **1,869 churned (26.5%)**.
- Average monthly charge **$64.76**; average tenure **32 months**.
- Monthly revenue of customers who left: **$139,131**.
- A model that always predicts 'no churn' would be 73.5% accurate, which is why accuracy alone is not used.

## 2. What goes together with churn (EDA)
- **Contract** (Cramér's V 0.41, strong): 'Month-to-month' churn 42.7% vs 'Two year' 2.8%.
- **OnlineSecurity** (Cramér's V 0.35, strong): 'No' churn 41.8% vs 'No internet service' 7.4%.
- **TechSupport** (Cramér's V 0.34, strong): 'No' churn 41.6% vs 'No internet service' 7.4%.
- Tenure: 47.4% churn in 0–12 months vs 9.5% for 49+ months.

## 3. Customer segments (K-Means, K = 4, silhouette 0.425)

| Segment | Customers | Churn rate | Revenue share |
|---|---|---|---|
| Established mid-spend | 2,410 | 47.0% | 41% |
| New low-spend | 1,509 | 23.9% | 10% |
| Long-term high-spend | 2,152 | 15.6% | 43% |
| Long-term low-spend | 972 | 4.1% | 6% |

Highest-churn RFM-style segment: **High-value newcomers** (747 customers, 67.5% churn).

## 4. Churn prediction
- Chosen model: **Logistic Regression** (best cross-validated ROC-AUC).
- Test set (1,409 unseen customers): **ROC-AUC 0.842**, PR-AUC 0.633, recall 70.9%, precision 54.4%, F1 0.616, accuracy 76.5% (threshold 0.35).
- Overfitting check: training ROC-AUC 0.849 vs test 0.842.

| Model | CV ROC-AUC (mean ± std) |
|---|---|
| Baseline | 0.500 ± 0.000 |
| Logistic Regression | 0.846 ± 0.013 |
| Random Forest | 0.844 ± 0.010 |

## 5. What the model relies on (permutation importance)
1. **Tenure** (ROC-AUC drop 0.1993)
2. **Contract** (ROC-AUC drop 0.0348)
3. **Total charges** (ROC-AUC drop 0.0297)
4. **Internet service** (ROC-AUC drop 0.0294)
5. **Monthly charges** (ROC-AUC drop 0.0135)

_Associations learned from past data, not proven causes._

## 6. Retention plan (estimates under stated assumptions)
- Revenue at risk over 12 months: **$1,669,812**.
- 1 - Act now: 857 customers
- 2 - Act soon: 1,167 customers
- 3 - Nurture: 643 customers
- 4 - Monitor: 4,376 customers
- Base case for contacting 'Act now + Act soon': ~276 customers saved, $279,844 protected for $175,505 (illustrative costs and success rates).
