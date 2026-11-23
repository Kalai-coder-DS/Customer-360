# Phase 7: The Customer360 AI dashboard

## Run it
```bash
python scripts/run_all.py              # build every table, model and report (once)
streamlit run app/streamlit_app.py     # open the dashboard in your browser
```

## Pages

| Group | Page | Brief module | What a visitor learns |
|---|---|---|---|
| Start here | **Executive overview** | 1 | The business problem, the workflow, headline KPIs, where churn concentrates, key findings |
| Start here | **How it works** | — | A guide for first-time visitors: workflow, dataset, limitations, correlation vs causation, glossary |
| Customer insights | **Customer 360 profile** | 2 | One customer: demographics, services, billing, segment, churn risk, reasons, next action |
| Customer insights | **Segmentation** | 3 | Segment cards with suggested actions, size vs revenue, churn by segment, segment map, RFM-style scores |
| Churn intelligence | **Churn prediction** | 4 | Chosen model, test metrics, model comparison, confusion matrix, ROC/PR/calibration, risk distribution, method |
| Churn intelligence | **Explainable AI** | 5 | Global importance, how each factor moves predictions, common reasons for high risk, example explanations |
| Churn intelligence | **Retention strategy** | 6 | Priority tiers, filterable call list with CSV export, live what-if impact, playbook and assumptions |
| Data | **Reports & data** | 7 | CSV upload with validation (missing values, duplicates), save to SQLite, score with the model, report downloads, database status and load history |

## Design system

**Visual hierarchy (same on every page)**
1. **Page header:** workflow step (eyebrow), title, and a one-sentence purpose.
2. **Workflow strip:** the seven stages, with the current one highlighted.
3. **KPI row:** 3–5 numbers that matter, each with a plain-language tooltip (ⓘ).
4. **Sections:** a heading plus a caption that **states the takeaway in words**, then the chart or table.
5. **"Learn more" panel:** the six standard questions (what, why, how, data, meaning, business use), collapsed so it doesn't crowd the page.

**Colour has meaning, and is never decoration**
- Theme: **Dark Plum-Charcoal + Violet/Magenta + Warm Coral/Amber**. Background #18111E, sidebar #22162B, cards #2A1C35.
- Violet and magenta = primary interactions, active states and neutral data series.
- Coral = churned / raises risk. Violet = retained / lowers risk.
- Risk levels: Low = violet, Medium = amber, High = warm orange, always with a text label.
- Customer segments: violet, pink, magenta (then tints of the same family).
- All colours are defined once in `config/theme.py` (design tokens) and mirrored in `.streamlit/config.toml`. Pink/magenta and amber/orange are close to each other, so charts always carry labels or legends.

**Charts**
- Built with Plotly and interactive: hover shows exact numbers.
- No chart titles inside the image. The page states the message in words above each chart.
- Values are printed on the bars, the average is marked on churn-rate charts, and there are no 3-D or decorative charts.

**Honesty built into the interface**
- Every money figure is labelled an *estimate* based on stated assumptions.
- The playbook's costs and success rates are shown and called illustrative.
- A correlation-vs-causation notice appears on the Explainable AI and How it works pages.
- If a pipeline step hasn't been run, the page says exactly which command to run instead of failing.

## Architecture

```
app/streamlit_app.py      navigation, theme CSS, sidebar
app/pages/*.py            one file per page: layout only
app/components/ui.py      page header, KPI row, sections, learn-more panel, badges, workflow strip
app/data.py               cached loading (refreshes automatically when the database changes)
src/dashboard/queries.py  all calculations: unit-tested, no Streamlit code
src/dashboard/figures.py  Plotly chart builders using the shared theme
src/dashboard/content.py  every explanation text in one place
```
Pages contain layout only. All calculations live in `src/dashboard/queries.py`, which has its own tests.

## Testing
- `tests/test_dashboard_queries.py` covers the data layer: KPIs, filters, live impact matching the pipeline, scoring uploads and saving uploads.
- `tests/test_dashboard_app.py` renders **every page** headlessly with Streamlit's `AppTest` against a full test pipeline, then checks for exceptions. It also changes a customer, moves the retention filters and checks the empty-database guidance.
