# Architecture

## Business workflow

```
Customer Data → Data Cleaning → Customer Analysis → Segmentation
             → Churn Prediction → Risk Explanation → Retention Action
```

## Technical layers

```
┌──────────────────────────────────────────────────────────────┐
│  Frontend                app/  Streamlit pages + components  │
├──────────────────────────────────────────────────────────────┤
│  Business logic          src/segmentation  (Phase 3)         │
│                          src/models        features, train   │
│                          src/explain       explainer         │
│                          src/retention     playbook, planner │
│  Visualisation           src/visualization (Phase 2+)        │
├──────────────────────────────────────────────────────────────┤
│  Data                    src/data   loader, schema, validator│
│                                     cleaning.py (Phase 2)    │
│  Storage                 src/database  SQLite                │
├──────────────────────────────────────────────────────────────┤
│  Configuration           config/settings.py, config/theme.py │
│  Utilities               src/utils/logger.py                 │
└──────────────────────────────────────────────────────────────┘
```

Rule: each layer only calls the layers below it. The frontend never touches SQL or model code directly. This keeps every part testable on its own.

## Phase 1 data flow

```
data/raw/Telco-Customer-Churn.csv
        │  load_customer_csv()         src/data/loader.py
        ▼
   pandas DataFrame (unchanged raw values)
        │  validate_customer_data()    src/data/validator.py
        ▼
   ValidationReport ──► reports/phase1_validation_report.json
        │  (only if there are no blocking errors)
        ▼
   SQLite database/customer360.db
     ├── customers_raw   (exact copy of the CSV)
     └── data_loads      (audit log: time, file fingerprint, quality status)
```

## Phase 2 data flow

```
SQLite customers_raw
        │  clean_customer_data()       src/data/cleaning.py  (+ CleaningLog)
        ▼
SQLite customers_clean  +  data/processed/customers_clean.csv
        │  src/analysis/eda.py          statistics, tests, findings
        │  src/visualization/eda_charts.py   13 charts
        ▼
reports/eda_report.md, reports/eda/*.csv, reports/figures/*.png
```

## Folder map

| Folder | Purpose | In Git? |
|---|---|---|
| `config/` | Settings and design tokens | Yes |
| `src/` | All reusable Python logic | Yes |
| `scripts/` | Command-line entry points | Yes |
| `tests/` | Automated tests (pytest) | Yes |
| `docs/` | Documentation | Yes |
| `data/raw`, `data/processed` | Datasets | No (placeholders only) |
| `database/` | SQLite file | No |
| `models/` | Trained models (Phase 4) | No |
| `reports/` | Generated reports | No |
| `logs/` | Run logs | No |

## Phase 3 data flow

```
SQLite customers_clean
        │  compute_rfm_scores()        src/segmentation/rfm.py
        │  fit_segments() -> profile_segments() -> name_segments()
        │                              src/segmentation/clustering.py
        ▼
SQLite customer_segments  +  models/segmentation_kmeans.joblib
reports/segmentation_report.md, reports/segmentation/*.csv, reports/figures/segmentation_*.png
```

## Phase 4 data flow

```
SQLite customers_clean
        │  prepare_xy()  engineered features, target separated   src/models/features.py
        ▼
stratified 80/20 split ──► 5-fold CV on training data (baseline, Logistic Regression, Random Forest)
        │  choose model by CV ROC-AUC; threshold by best F1 on out-of-fold training predictions
        ▼
single evaluation on the 20% test set                          src/models/evaluate.py
        │  refit chosen model on all customers
        ▼
models/churn_model.joblib  +  SQLite churn_predictions (probability, risk level, flag)
reports/model_report.md, reports/modeling/*, reports/figures/model_*.png
```

## Phase 5 data flow

```
models/churn_model.joblib + SQLite customers_clean + churn_predictions
        │  global_importance()        permutation importance (ROC-AUC drop)
        │  explain_customers()        exact linear Shapley (Logistic Regression)
        │                             or sampling Shapley (Random Forest)
        │  top_factors() -> explanation_text()
        ▼
SQLite churn_explanations, reports/explain_report.md, reports/explain/*.csv,
reports/figures/explain_*.png
```

## Phase 6 data flow

```
customers_clean + churn_predictions + churn_explanations + customer_segments
        │  value at risk = probability × monthly charges × horizon
        │  priority_tier()  risk × value
        │  match_actions()  playbook rules ∩ factors that raise this customer's risk
        │  impact_scenarios()  conservative / base / optimistic assumptions
        ▼
SQLite retention_actions, reports/retention/priority_call_list.csv,
reports/retention_report.md, reports/figures/retention_*.png
```

## Phase 7: dashboard

```
streamlit run app/streamlit_app.py
        │  st.navigation → app/pages/*.py (layout only)
        │  app/data.py   cached loading, refreshed when the database file changes
        ▼
src/dashboard/queries.py (calculations) · figures.py (Plotly) · content.py (explanations)
        ▼
SQLite tables + reports/ produced by Phases 1-6, plus models/churn_model.joblib for scoring uploads
```

## Phase 8: deployment

```
GitHub (code only: no data, database, models or secrets)
   ├── GitHub Actions: pytest on Python 3.12 and 3.14 for every push
   └── Streamlit Community Cloud: app/streamlit_app.py
          first start, no data → app/data.ensure_pipeline() → src/pipeline.run_pipeline(download=True)
          (one build at a time, guarded by a shared lock; visitors wait for it)
          C360_DEMO_MODE = "1" (secret) → uploads scored in memory, never saved
```
