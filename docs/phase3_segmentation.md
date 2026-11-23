# Phase 3: Customer segmentation

This guide is written for readers without a Data Science background. The dashboard will reuse it in Phase 7.

## Part A: RFM-style scores (`src/segmentation/rfm.py`)

**1. What is it?**
A simple 1–5 score for each customer on three dimensions, plus a named segment built from those scores.

**2. Why is it needed?**
It gives the business an easy-to-explain way to say "who are our best customers, and who is new and fragile?" without any machine learning.

**3. How does it work?**
Classic RFM stands for Recency, Frequency and Monetary, and it needs purchase dates and a list of purchases. **This dataset has neither**, so the project uses honest substitutes:

| Classic | Used here | Column | Meaning |
|---|---|---|---|
| Recency | **Loyalty** | tenure | How long the relationship has lasted |
| Frequency | **Engagement** | number of services | How much of the product they use |
| Monetary | **Monetary** | monthly charges | Revenue per month |

Each score is a percentile band. For example, a 5 means the customer is in the top 20% of all customers. The named segments then follow simple rules:

| Segment | Rule |
|---|---|
| Champions | Loyalty, Engagement and Monetary all 4 or higher |
| Loyal core | Loyalty 4 or higher |
| High-value newcomers | Loyalty 2 or lower, Monetary 4 or higher |
| New and light | Loyalty 2 or lower, Monetary 2 or lower |
| Developing | Everyone else |

**4. What data does it use?**
The `customers_clean` table. Churn is **not** used to build the scores.

**5. What does the result mean?**
A customer with scores 5-5-5 is long-standing, uses many services and pays a lot. A customer with 1-1-1 is new, uses little and pays little.

**6. How does a business use it?**
To target campaigns. For example, onboarding support for "New and light" customers, and loyalty rewards for "Champions".

## Part B: K-Means clustering (`src/segmentation/clustering.py`)

**1. What is it?**
An algorithm that groups customers who look alike, without being told what the groups should be.

**2. Why is it needed?**
Hand-written rules only find the groups we already expect. Clustering can reveal combinations we didn't think of.

**3. How does it work?**
1. Describe each customer by tenure, monthly charges and number of services.
2. Put the three on the same scale (standardise them), so a large number like charges doesn't dominate.
3. K-Means places K centre points and assigns each customer to the nearest one, adjusting until the groups stop changing.
4. Try K = 2 to 8 and pick the K between 3 and 6 with the best **silhouette score** (how clearly separated the groups are, from -1 to 1). Fewer than 3 groups is too coarse to act on; more than 6 is too many for a team to manage.

Two columns are deliberately left out:
- *TotalCharges* is roughly tenure × monthly charges, so including it would count the same information twice.
- *Churn* is left out so the segments describe **who customers are**. Churn is only used afterwards to see how each group behaves.

**4. What data does it use?**
The `customers_clean` table.

**5. What does the result mean?**
Each segment gets a name generated from the data, such as "New high-spend", by comparing its averages with the whole customer base. It also gets a description and its churn rate compared with the average. Silhouette scores for customer data are often modest (0.2–0.4), because real customers form a continuous spread rather than neat islands. That's normal, and it's worth saying so in an interview.

**6. How does a business use it?**
Each segment comes with a **suggested action**, such as priority retention for valuable at-risk groups, onboarding for new at-risk groups, or rewards for loyal high-value groups. These suggestions are *hypotheses to test*, ideally with a controlled experiment, not proven results.

## Outputs

| File | Contents |
|---|---|
| SQLite `customer_segments` | One row per customer: three scores, RFM segment, cluster, segment name |
| `models/segmentation_kmeans.joblib` | The fitted scaler and K-Means model, plus segment names |
| `reports/segmentation_report.md` | The full report, with profiles, descriptions, actions and charts |
| `reports/segmentation/*.csv` | K selection, cluster profiles, RFM summary |
| `reports/figures/segmentation_*.png` | 6 charts |
