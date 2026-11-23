# Phase 2: Data cleaning and exploratory data analysis

This guide explains Phase 2 for someone who has never studied Data Science. The same explanations will appear on the dashboard in Phase 7.

## Part A: Data cleaning (`src/data/cleaning.py`)

**1. What is it?**
Fixing problems in the raw data so every value is in a consistent, usable form.

**2. Why is it needed?**
Models and charts give wrong answers when data is messy. For example, a blank stored as `" "` can't be added up, and a duplicated customer would be counted twice.

**3. How does it work?**
Each step is recorded in a cleaning log:

1. Keep only the 21 known columns.
2. Trim stray spaces; blank text becomes "missing".
3. Remove duplicate rows and repeated customer IDs, keeping the first.
4. Convert number columns from text to numbers. Impossible values (negative charges, a `SeniorCitizen` code other than 0/1) become missing.
5. **Business rule:** customers with 0 months' tenure haven't been billed yet, so their blank `TotalCharges` is set to **0**. This is certain, not a guess.
6. Category values outside the allowed list become missing.
7. Rows with no `Churn` value are removed, because they can't be used to learn from.
8. Add `churn_flag` (1 = churned, 0 = stayed) to make calculations easier.

**4. What data does it use?**
The `customers_raw` table saved in Phase 1.

**5. What does the result mean?**
The `customers_clean` table has one row per customer and correct data types. Any missing values that remain are listed in `reports/phase2_cleaning_log.json`.

**6. How does a business use it?**
Every number in the dashboard comes from this clean table, so the numbers are consistent and can be traced back to the source file.

### What cleaning deliberately does *not* do
Filling gaps with an average, scaling numbers, and converting categories into numbers all depend on statistics from the whole dataset. They will happen inside the Phase 4 model pipeline, learned from the **training data only**. Doing them now, on all rows, would let information from the test data leak into the model. This is called **data leakage**, and it makes a model look better in testing than it will perform in real use.

## Part B: Exploratory data analysis (`src/analysis/eda.py`)

**1. What is it?**
Getting to know the data before modelling: what a typical customer looks like, and which characteristics go together with churn.

**2. Why is it needed?**
It catches data problems early, shows which patterns are worth modelling, and gives the business findings it can act on even before any machine learning.

**3. How does it work?**

| Tool | Plain meaning |
|---|---|
| Churn rate by group | The share of customers in a group who left |
| Lift | The group's churn rate divided by the average. 2.0 means the group churns twice as often as average |
| Chi-square test + Cramér's V | For text columns: is churn linked to this column, and how strongly? 0 = no link, 1 = perfect link |
| Mann-Whitney U + effect size | For number columns: do churned and retained customers differ, and by how much? -1 to +1 |
| Spearman correlation | Do two numbers rise and fall together? Based on ranks, so it handles skewed data |

**4. What data does it use?**
The `customers_clean` table.

**5. What does the result mean?**
- **p-value:** how surprising a difference would be if there were really no link. With 7,000 customers, even tiny differences produce very small p-values, so the **effect size** (Cramér's V or the rank-biserial effect) is what tells you whether a difference is big enough to matter.
- **Association is not causation.** If month-to-month customers churn more, that may be because uncommitted customers choose month-to-month contracts, not because the contract makes them leave. Proving cause needs an experiment, such as offering some customers a discount and comparing the results.

**6. How does a business use it?**
To decide where to look first. A group with high churn and many customers, such as a common contract type with a churn rate far above average, is a natural target for retention campaigns. The groups found here are the ones Phase 3 (segmentation) and Phase 4 (prediction) examine in more detail.

## Outputs

| File | Contents |
|---|---|
| `reports/eda_report.md` | The full written report, with findings, tables and charts |
| `reports/figures/*.png` | 13 charts |
| `reports/eda/*.csv` | Every summary table, for checking or reuse |
| `reports/phase2_cleaning_log.json` | Every cleaning change made |
| `data/processed/customers_clean.csv` and SQLite `customers_clean` | The clean dataset |
