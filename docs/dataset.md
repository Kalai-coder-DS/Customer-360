# Dataset: IBM Telco Customer Churn

## Source

- **Publisher:** IBM, as a sample dataset for its analytics products.
- **Download used by this project:** <https://github.com/IBM/telco-customer-churn-on-icp4d> (`data/Telco-Customer-Churn.csv`)
- **Kaggle mirror (same data):** <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>
- **Stored at:** `data/raw/Telco-Customer-Churn.csv`. The file is not committed to Git.

## Size

- **Rows:** 7,043. Each row is one customer.
- **Columns:** 21, made up of 1 ID, 19 descriptive features and 1 target.

`scripts/ingest_data.py` prints the real counts from your copy of the file every time it runs.

## Target variable

`Churn` is **Yes** if the customer left within the last month and **No** otherwise. Around a quarter of customers churned, so the classes are imbalanced. This is why Phase 4 evaluates models with precision, recall, F1 and ROC-AUC instead of accuracy alone.

## Feature dictionary

| Column | Type | Meaning | Allowed values |
|---|---|---|---|
| `customerID` | ID | Unique customer identifier. |  |
| `gender` | Category | Customer gender (Male / Female). | Female, Male |
| `SeniorCitizen` | Numeric | 1 if the customer is 65 or older, otherwise 0. | 0, 1 |
| `Partner` | Category | Whether the customer has a partner. | No, Yes |
| `Dependents` | Category | Whether the customer has dependents (e.g. children). | No, Yes |
| `tenure` | Numeric | Number of months the customer has stayed with the company. |  |
| `PhoneService` | Category | Whether the customer has a phone line. | No, Yes |
| `MultipleLines` | Category | Whether the customer has more than one phone line. | No, No phone service, Yes |
| `InternetService` | Category | Type of internet connection: DSL, Fiber optic or none. | DSL, Fiber optic, No |
| `OnlineSecurity` | Category | Whether the customer pays for an online security add-on. | No, No internet service, Yes |
| `OnlineBackup` | Category | Whether the customer pays for an online backup add-on. | No, No internet service, Yes |
| `DeviceProtection` | Category | Whether the customer pays for device protection. | No, No internet service, Yes |
| `TechSupport` | Category | Whether the customer pays for premium technical support. | No, No internet service, Yes |
| `StreamingTV` | Category | Whether the customer streams TV through the service. | No, No internet service, Yes |
| `StreamingMovies` | Category | Whether the customer streams movies through the service. | No, No internet service, Yes |
| `Contract` | Category | Contract length: Month-to-month, One year or Two year. | Month-to-month, One year, Two year |
| `PaperlessBilling` | Category | Whether the customer receives bills electronically. | No, Yes |
| `PaymentMethod` | Category | How the customer pays. | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check |
| `MonthlyCharges` | Numeric | Amount charged to the customer each month (USD). |  |
| `TotalCharges` | Numeric | Total amount charged over the customer's tenure (USD). |  |
| `Churn` | Category | **Target:** whether the customer left within the last month. | No, Yes |

The source of truth for this table is `src/data/schema.py`.

## Known data quirks (handled in Phase 2)

1. **`TotalCharges` is stored as text.** A few customers with `tenure = 0` (brand-new customers) have a blank value (`" "`) instead of a number. The validator counts these blanks as missing values, not as bad numbers.
2. **Mixed encodings.** `SeniorCitizen` uses 0/1, while the other yes/no columns use the words "Yes"/"No".
3. **Overlapping categories.** "No internet service" in the add-on columns repeats what `InternetService = "No"` already says. Feature engineering needs to account for this.

## Limitations (stated honestly in the app and README)

- **Fictional company.** IBM created the data to represent a telecom provider. Any figures the app shows, such as revenue at risk or retention impact, are illustrations, not real business results.
- **A single snapshot.** There are no dates and no transaction history. **True RFM analysis is not possible**, because there is no purchase date to measure Recency from and no list of purchases to count for Frequency. Phase 3 will use clearly labelled substitutes (for example, `tenure` for relationship length, `MonthlyCharges`/`TotalCharges` for monetary value, and number of subscribed services for engagement) and will explain this in the app.
- **One region and one industry.** Patterns may not carry over to other markets.
- **Correlation, not causation.** The data is observational. If a feature is associated with churn, that doesn't prove it causes churn.
- **Small, but enough for this project.** 7,043 rows is enough for classical machine learning and fits comfortably on a laptop.
