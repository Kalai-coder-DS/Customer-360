"""Tests for the dashboard's data layer (no Streamlit needed)."""

import pandas as pd
import pytest

from src.dashboard import content, queries
from src.retention.planner import TIER_ORDER, impact_scenarios
from tests.conftest import build_full_pipeline, make_customers_with_signal


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("dash")
    db = build_full_pipeline(tmp, mp)
    yield db, queries.load_tables(db)
    mp.undo()


def test_all_tables_load(pipeline):
    _, tables = pipeline
    for key in ("raw", "clean", "segments", "predictions", "explanations", "retention", "loads"):
        assert tables[key] is not None, key
    assert queries.missing_steps(tables) == []


def test_missing_database_gives_empty_tables(tmp_path):
    tables = queries.load_tables(tmp_path / "none.db")
    assert all(v is None for v in tables.values())
    assert queries.missing_steps(tables)[0] == "python scripts/run_eda.py"


def test_executive_kpis_are_consistent(pipeline):
    _, tables = pipeline
    k = queries.executive_kpis(tables)
    assert k["customers"] == k["active"] + k["churned"] == len(tables["clean"])
    assert k["churn_rate"] == pytest.approx(k["churned"] / k["customers"])
    assert k["revenue_at_risk"] == pytest.approx(tables["retention"]["expected_loss_horizon"].sum())
    assert 0 <= k["high_risk_active"] <= k["active"]


def test_churn_rate_tables(pipeline):
    _, tables = pipeline
    table = queries.churn_rate_table(tables["clean"], "Contract")
    assert table["customers"].sum() == len(tables["clean"])
    assert table["churn_rate"].is_monotonic_decreasing
    bands = queries.tenure_band_table(tables["clean"])
    assert list(bands.iloc[:, 0].astype(str)) == [b for b in bands.iloc[:, 0].cat.categories
                                                  if b in set(bands.iloc[:, 0].astype(str))]


def test_customer_record_and_ordering(pipeline):
    _, tables = pipeline
    ids = queries.customer_ids_by_risk(tables)
    probs = tables["predictions"].set_index("customerID").loc[ids, "churn_probability"]
    assert probs.is_monotonic_decreasing
    record = queries.customer_record(tables, ids[0])
    assert record["clean"]["customerID"] == ids[0]
    assert record["predictions"] is not None and record["retention"] is not None
    contributions = queries.parse_contributions(record["explanations"]["contributions_json"])
    assert "Contract" in contributions.index
    assert queries.customer_record(tables, "does-not-exist")["clean"] is None


def test_services_used():
    row = pd.Series({"PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
                     "OnlineSecurity": "Yes", "TechSupport": "No"})
    assert queries.services_used(row) == ["Phone (multiple lines)", "Internet (DSL)",
                                          "Online security"]


def test_filter_plan(pipeline):
    _, tables = pipeline
    plan = tables["retention"]
    act_now = queries.filter_plan(plan, tiers=[TIER_ORDER[0]])
    assert (act_now["priority_tier"] == TIER_ORDER[0]).all()
    risky = queries.filter_plan(plan, min_probability=0.5)
    assert (risky["churn_probability"] >= 0.5).all()
    one = queries.filter_plan(plan, search=plan.iloc[0]["customerID"])
    assert len(one) == 1


def test_live_impact_matches_pipeline_base_scenario(pipeline):
    _, tables = pipeline
    plan = tables["retention"]
    live = queries.impact_estimate(plan, TIER_ORDER[:2], 1.0, 12)
    saved = impact_scenarios(plan).loc[("Act now + Act soon", "Base")]
    assert live["revenue_protected"] == pytest.approx(saved["revenue_protected"], abs=0.05)
    assert live["campaign_cost"] == pytest.approx(saved["campaign_cost"])
    double = queries.impact_estimate(plan, TIER_ORDER[:2], 2.0, 12)
    assert double["revenue_protected"] >= live["revenue_protected"]


def test_score_customers_with_saved_model(pipeline):
    import joblib

    from config import settings
    bundle = joblib.load(settings.CHURN_MODEL_FILE)
    upload = make_customers_with_signal(30, seed=99)
    scored, log = queries.score_customers(upload, bundle)
    assert len(scored) == 30
    assert scored["churn_probability"].between(0, 1).all()
    assert scored["churn_probability"].is_monotonic_decreasing
    assert set(scored["risk_level"]) <= {"Low", "Medium", "High"}
    assert log.rows_after == 30


def test_save_upload_records_load(pipeline):
    db, _ = pipeline
    df = make_customers_with_signal(10, seed=5)
    load_id = queries.save_upload(df, "upload_test", db)
    tables = queries.load_tables(db)
    assert load_id >= 2 and len(tables["loads"]) >= 2


def test_report_helpers(pipeline):
    files = queries.list_report_files()
    names = {f.name for f in files}
    assert {"eda_report.md", "model_report.md", "retention_report.md"} <= names
    assert queries.read_report_json("modeling/summary.json")["chosen_model"]
    assert queries.figure_path("model_03_roc") is not None
    assert queries.read_report_csv("does/not/exist.csv") is None


def test_every_module_answers_the_six_questions():
    keys = {k for k, _ in content.SIX_QUESTIONS}
    for module, answers in content.EXPLAINERS.items():
        assert set(answers) == keys, module
        assert all(len(text) > 30 for text in answers.values()), module
