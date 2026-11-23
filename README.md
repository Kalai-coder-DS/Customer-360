# Customer360 AI

**An intelligent customer analytics, segmentation and churn prediction platform.**

Customer360 AI helps a business answer six questions: who its customers are, which ones are valuable, which are likely to leave, why they might leave, which groups need attention, and what to do about it.

> **Status:** Phases 1–8 of 9 are complete: analytics pipeline, interactive dashboard, automated tests and cloud deployment. See the [roadmap](#roadmap).
>
> **Live demo:** _add your Streamlit Community Cloud link here after deploying (see [docs/deployment.md](docs/deployment.md))_

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux

# 2. Install libraries
pip install -r requirements.txt

# 3. Check the environment
python scripts/check_environment.py

# 4. Download the dataset and run the whole pipeline (Phases 1-6)
python scripts/run_all.py --download

# 5. Open the dashboard
streamlit run app/streamlit_app.py

# 6. Run the tests
python -m pytest
```

## Deployment

The app is ready for **Streamlit Community Cloud**. Main file: `app/streamlit_app.py`. On first start it downloads the dataset and builds the models automatically (about a minute). Set the secret `C360_DEMO_MODE = "1"` so the public demo never stores uploaded data. Full step-by-step guide: [docs/deployment.md](docs/deployment.md). Testing guide: [docs/testing.md](docs/testing.md).

## Project structure

```
customer360/
├── app/              Streamlit dashboard: streamlit_app.py, pages/, components/
├── .streamlit/       dashboard theme (config.toml)
├── .github/workflows/ tests.yml (runs pytest on every push)
├── config/           settings.py (paths, env vars, seed), theme.py (design tokens)
├── src/
│   ├── data/         schema.py, loader.py, validator.py, cleaning.py
│   ├── analysis/     eda.py (statistics and findings)
│   ├── segmentation/ rfm.py, clustering.py
│   ├── models/       features.py, train.py, evaluate.py
│   ├── explain/      explainer.py (Shapley values, permutation importance)
│   ├── retention/    playbook.py (actions + assumptions), planner.py
│   ├── dashboard/    queries.py, figures.py (Plotly), content.py (explanations)
│   ├── pipeline.py   runs Phases 1-6 in one call (CLI + first-start bootstrap)
│   ├── visualization/ eda_charts.py, segment_charts.py, model_charts.py, explain_charts.py, retention_charts.py
│   ├── database/     db.py (SQLite)
│   └── utils/        logger.py
├── scripts/          check_environment.py, download_data.py, ingest_data.py, run_eda.py, run_segmentation.py, run_modeling.py, run_explain.py, run_retention.py, run_all.py
├── tests/            pytest suite
├── docs/             dataset.md, architecture.md, phase2_cleaning_and_eda.md, phase3_segmentation.md, phase4_churn_prediction.md, phase5_explainable_ai.md, phase6_retention_intelligence.md, phase7_dashboard.md, deployment.md, testing.md
├── data/  database/  models/  reports/  logs/   (generated, not in Git)
├── requirements.txt  .env.example  .gitignore  pytest.ini
```

## Dataset

The project uses the [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset: 7,043 customers and 21 columns, with `Churn` as the target. Full data dictionary and limitations: [docs/dataset.md](docs/dataset.md).

*The data describes a fictional telecom company. Results shown by this project illustrate the methods and are not real business outcomes.*

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Planning, scope, dataset, architecture | Done |
| 1 | Environment, dataset loading, SQLite, validation | Done |
| 2 | Cleaning and exploratory data analysis | Done |
| 3 | Customer segmentation (RFM-style + K-Means) | Done |
| 4 | Churn prediction (Logistic Regression, Random Forest) | Done |
| 5 | Explainable AI | Done |
| 6 | Retention intelligence | Done |
| 7 | Premium Streamlit dashboard | Done |
| 8 | Testing and deployment | Done |
| 9 | Documentation and GitHub | |

## Tech stack

Python · Pandas · NumPy · SciPy · scikit-learn · Matplotlib / Seaborn / Plotly · Streamlit · SQLite · pytest. Everything runs on a CPU-only laptop.
