import numpy as np
import pandas as pd
import pytest

from src.analysis import eda
from src.data.cleaning import clean_customer_data


@pytest.fixture
def clean_sample(sample_df):
    return clean_customer_data(sample_df)[0]


@pytest.fixture
def clean_random(random_df):
    return clean_customer_data(random_df)[0]


def test_overview_numbers(clean_sample):
    o = eda.overview(clean_sample)
    assert (o["customers"], o["churned"], o["retained"]) == (6, 2, 4)
    assert o["churn_rate"] == pytest.approx(2 / 6)
    assert o["monthly_charges_of_churned"] == pytest.approx(70.70 + 99.65)


def test_majority_baseline(clean_sample):
    assert eda.majority_baseline_accuracy(clean_sample) == pytest.approx(4 / 6)


def test_churn_rate_by_contract(clean_sample):
    table = eda.churn_rate_by(clean_sample, "Contract")
    # Month-to-month: T-0001 (No), T-0004 (Yes), T-0005 (Yes) -> 2/3
    assert table.loc["Month-to-month", "churn_rate"] == pytest.approx(2 / 3)
    assert table.loc["Two year", "churn_rate"] == 0
    assert table["churn_rate"].is_monotonic_decreasing
    assert table["customers"].sum() == 6
    assert table.loc["Month-to-month", "lift"] == pytest.approx((2 / 3) / (2 / 6))


def test_churn_rate_by_labels_missing_values(clean_sample):
    df = clean_sample.copy()
    df.loc[0, "Contract"] = np.nan
    assert "Missing" in eda.churn_rate_by(df, "Contract").index


def test_tenure_bands_keep_natural_order(clean_random):
    table = eda.churn_rate_by(eda.add_tenure_band(clean_random), eda.TENURE_BAND)
    assert list(table.index) == eda.TENURE_LABELS


def test_tenure_band_edges():
    df = pd.DataFrame({"tenure": [0, 12, 13, 24, 25, 48, 49, 72]})
    bands = eda.add_tenure_band(df)[eda.TENURE_BAND].astype(str).tolist()
    assert bands == ["0–12 months"] * 2 + ["13–24 months"] * 2 + ["25–48 months"] * 2 + \
        ["49+ months"] * 2


def test_cramers_v_perfect_and_none():
    perfect = pd.DataFrame([[50, 0], [0, 50]])
    none = pd.DataFrame([[25, 25], [25, 25]])
    assert eda._cramers_v(perfect)[1] == pytest.approx(1.0)
    assert eda._cramers_v(none)[1] == pytest.approx(0.0)


def test_categorical_association_is_ranked_and_bounded(clean_random):
    table = eda.categorical_association(clean_random)
    assert len(table) == len(eda.CATEGORICAL_FEATURES)
    assert table["cramers_v"].between(0, 1).all()
    assert table["cramers_v"].is_monotonic_decreasing


def test_numeric_association_direction():
    df = pd.DataFrame({
        "tenure": [1, 2, 3, 40, 50, 60],
        "MonthlyCharges": [90.0, 95, 99, 20, 25, 30],
        "TotalCharges": [90.0, 190, 297, 800, 1250, 1800],
        "churn_flag": [1, 1, 1, 0, 0, 0],
    })
    table = eda.numeric_association(df)
    assert table.loc["MonthlyCharges", "effect_size"] == pytest.approx(1.0)
    assert table.loc["tenure", "effect_size"] == pytest.approx(-1.0)
    assert table.loc["tenure", "strength"] == "large"


def test_correlation_matrix_shape(clean_random):
    corr = eda.correlation_matrix(clean_random)
    assert corr.shape == (4, 4)
    assert np.allclose(np.diag(corr), 1.0)


def test_services_count():
    df = pd.DataFrame({
        "PhoneService": ["Yes", "No"], "MultipleLines": ["Yes", "No phone service"],
        "InternetService": ["Fiber optic", "No"], "OnlineSecurity": ["Yes", "No internet service"],
        "OnlineBackup": ["No", "No internet service"], "DeviceProtection": ["No", "No internet service"],
        "TechSupport": ["Yes", "No internet service"], "StreamingTV": ["No", "No internet service"],
        "StreamingMovies": ["No", "No internet service"],
    })
    assert eda.services_count(df).tolist() == [5, 0]


def test_numeric_and_categorical_summaries(clean_random):
    num = eda.numeric_summary(clean_random)
    cat = eda.categorical_summary(clean_random)
    assert list(num.index) == eda.NUMERIC_FEATURES
    assert set(cat.index) == set(eda.CATEGORICAL_FEATURES)


def test_key_findings_are_generated_from_data(clean_random):
    findings = eda.key_findings(clean_random)
    rate = eda.overall_churn_rate(clean_random)
    assert f"{rate:.1%}" in findings[0]
    assert any("not proof of cause" in f for f in findings)
