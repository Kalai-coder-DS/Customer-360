import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_customer_data
from src.segmentation import clustering, rfm
from tests.conftest import make_random_customers


@pytest.fixture(scope="module")
def clean_random():
    return clean_customer_data(make_random_customers(600, seed=4))[0]


# --- RFM-style scores ------------------------------------------------------

def test_percentile_score_range_and_order():
    scores = rfm.percentile_score(pd.Series(range(100)))
    assert scores.min() == 1 and scores.max() == 5
    assert scores.is_monotonic_increasing
    assert scores.value_counts().tolist() == [20] * 5


def test_percentile_score_ties_share_a_score():
    scores = rfm.percentile_score(pd.Series([1, 1, 1, 1, 9, 9, 9, 9]))
    assert scores.iloc[:4].nunique() == 1 and scores.iloc[4:].nunique() == 1
    assert scores.iloc[0] < scores.iloc[-1]


@pytest.mark.parametrize("l, e, m, expected", [
    (5, 5, 5, "Champions"),
    (4, 4, 4, "Champions"),
    (5, 1, 5, "Loyal core"),
    (4, 5, 1, "Loyal core"),
    (1, 3, 5, "High-value newcomers"),
    (2, 1, 1, "New and light"),
    (3, 3, 3, "Developing"),
    (1, 5, 3, "Developing"),
])
def test_rfm_segment_rules(l, e, m, expected):
    assert rfm.assign_rfm_segment(l, e, m) == expected


def test_compute_rfm_scores(clean_random):
    scores = rfm.compute_rfm_scores(clean_random)
    assert len(scores) == len(clean_random)
    for col in rfm.SCORE_COLUMNS:
        assert scores[col].between(1, 5).all()
    assert scores["rfm_score"].between(3, 15).all()
    assert set(scores["rfm_segment"]) <= set(rfm.RFM_SEGMENT_ORDER)
    assert "churn_flag" not in scores.columns  # churn is not used to segment


# --- K-Means ---------------------------------------------------------------

def test_cluster_features_exclude_churn_and_total_charges(clean_random):
    features = clustering.build_cluster_features(clean_random)
    assert list(features.columns) == clustering.CLUSTER_FEATURES
    assert "TotalCharges" not in features and "churn_flag" not in features


def test_choose_k_respects_business_range():
    scores = pd.DataFrame({"silhouette": [0.9, 0.3, 0.4, 0.35, 0.2, 0.1, 0.8]},
                          index=range(2, 9))
    assert clustering.choose_k(scores, range(3, 7)) == 4  # 2 and 8 excluded


def test_fit_is_reproducible_and_ordered_by_tenure(clean_random):
    first = clustering.fit_segments(clean_random, k=4)
    second = clustering.fit_segments(clean_random, k=4)
    assert (first.labels == second.labels).all()
    tenure_by_cluster = clean_random.groupby(first.labels.values)["tenure"].mean()
    assert tenure_by_cluster.is_monotonic_increasing
    assert sorted(first.labels.unique()) == [0, 1, 2, 3]


def test_automatic_k_is_in_business_range(clean_random):
    result = clustering.fit_segments(clean_random)
    assert result.k in clustering.settings.K_CHOICE_RANGE
    assert result.k_scores["silhouette"].between(-1, 1).all()


def test_profiles_and_names(clean_random):
    result = clustering.fit_segments(clean_random, k=5)
    profile = clustering.profile_segments(clean_random, result.labels)
    named = clustering.name_segments(profile, clean_random)
    assert profile["customers"].sum() == len(clean_random)
    assert profile["share_of_customers"].sum() == pytest.approx(1, abs=1e-3)
    assert profile["share_of_revenue"].sum() == pytest.approx(1, abs=1e-3)
    assert named["segment_name"].is_unique
    assert named["description"].str.contains("Churn rate").all()
    assert named["suggested_action"].str.len().gt(10).all()


def test_names_made_unique_when_buckets_collide(clean_random):
    profile = pd.DataFrame({"avg_services": [1.0, 8.0, 1.0]})
    names = clustering._make_unique(["New low-spend"] * 3, profile, clean_random)
    assert len(set(names)) == 3
    assert names[1] == "New low-spend, many services"


def test_risky_high_value_segment_gets_priority_action(monkeypatch):
    df = pd.DataFrame({"tenure": np.arange(90) % 72, "MonthlyCharges": np.linspace(20, 120, 90),
                       "churn_flag": [0] * 80 + [1] * 10})
    profile = pd.DataFrame({
        "customers": [10], "avg_tenure": [3.0], "avg_monthly_charges": [115.0],
        "avg_services": [2.0], "share_month_to_month": [0.9], "churn_rate": [0.8],
        "share_of_customers": [0.1],
    })
    monkeypatch.setattr(clustering, "services_count", lambda d: pd.Series(np.arange(len(d)) % 9))
    named = clustering.name_segments(profile, df)
    assert named.loc[0, "segment_name"] == "New high-spend"
    assert named.loc[0, "suggested_action"].startswith("Priority retention")
