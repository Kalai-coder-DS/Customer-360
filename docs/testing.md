# Testing

## Automated tests
```bash
python -m pytest            # whole suite, about 1-2 minutes
python -m pytest -q -k dashboard   # only the dashboard tests
```

| Test file | What it protects |
|---|---|
| `test_loader.py`, `test_validator.py`, `test_database.py`, `test_pipeline.py` | Phase 1: reading CSVs, quality checks, SQLite storage, ingest script |
| `test_cleaning.py`, `test_eda.py`, `test_eda_pipeline.py` | Phase 2: cleaning rules (input never modified, tenure-0 rule), statistics, report |
| `test_segmentation*.py` | Phase 3: RFM scoring rules, K-Means reproducibility, segment naming |
| `test_models.py`, `test_modeling_pipeline.py` | Phase 4: no target leakage, learns a planted pattern, near-random on random data, stratified split, reproducibility |
| `test_explain*.py` | Phase 5: Shapley values add up exactly; sampled values match exact ones; planted pattern recovered; text names only real factors |
| `test_retention*.py` | Phase 6: action matching, tiers, impact maths, full Phase 1→6 run |
| `test_dashboard_queries.py` | Phase 7: KPIs, filters, live what-if matches the pipeline, scoring and saving uploads |
| `test_dashboard_app.py` | Phase 7: every page renders in Streamlit's own test runner; customer switch, sliders, empty-database guidance |
| `test_deployment.py` | Phase 8: one-call pipeline, first-start detection, required files, secrets never committed |

All tests use **generated test data**, never the real dataset. Some use a version with a churn pattern built in on purpose, so the models can be checked for learning a real signal. GitHub Actions runs the suite on every push (`.github/workflows/tests.yml`).

## Manual check before a demo (5 minutes)

| Page | Check |
|---|---|
| Executive overview | Takeaways box shows real numbers; 8 KPI cards; 4 charts; hovering a bar shows a tooltip |
| How it works | Workflow strip on one row (desktop); glossary table readable |
| Customer 360 | Pick 3 different customers from the list: gauge, profile fields, chips, "Why" chart and next step all change |
| Segmentation | Segment cards, size vs revenue, map (hover a dot), RFM tab |
| Churn prediction | All three tabs open; ROC / precision-recall / calibration images visible |
| Explainable AI | Change the factor in the dropdown and the chart updates; example explanations shown |
| Retention strategy | Change the filters and the list count changes; **Download** gives a CSV; move the what-if sliders and the KPIs change |
| Reports & data | Upload `data/raw/Telco-Customer-Churn.csv`: validation report, preview, **Predict churn risk**, download scored file; report downloads work |
| Narrow window | Shrink the browser to tablet width: cards and workflow reflow, nothing overlaps |
