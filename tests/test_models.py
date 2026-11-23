import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_customer_data
from src.models import evaluate, features
from src.models.train import train_and_evaluate
from tests.conftest import make_customers_with_signal, make_random_customers


@pytest.fixture(scope="module")
def signal_result():
    df = clean_customer_data(make_customers_with_signal(1500, seed=11))[0]
    return df, train_and_evaluate(df)


# --- Features --------------------------------------------------------------

def test_target_and_id_never_used_as_inputs(sample_df):
    clean = clean_customer_data(sample_df)[0]
    X, y = features.prepare_xy(clean)
    for col in features.EXCLUDED_COLUMNS + ["segment_name", "cluster", "rfm_segment"]:
        assert col not in X.columns
    assert list(X.columns) == features.MODEL_FEATURES
    assert y.tolist() == (clean["Churn"] == "Yes").astype(int).tolist()


def test_engineered_features(sample_df):
    clean = clean_customer_data(sample_df)[0]
    data = features.add_engineered_features(clean)
    row = data.set_index("customerID")
    assert row.loc["T-0002", "avg_monthly_spend"] == pytest.approx(1889.5 / 34)
    assert row.loc["T-0003", "avg_monthly_spend"] == pytest.approx(70.70)  # tenure 0
    assert row.loc["T-0003", "auto_payment"] == 1.0
    assert row.loc["T-0001", "auto_payment"] == 0.0


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_preprocessor_handles_missing_and_unseen_values(sample_df):
    clean = clean_customer_data(sample_df)[0]
    X, _ = features.prepare_xy(clean)
    prep = features.build_preprocessor().fit(X)
    new = X.iloc[[0]].copy()
    new["MonthlyCharges"] = np.nan
    new["PaymentMethod"] = "Cryptocurrency"  # never seen in training
    out = prep.transform(new)
    assert not np.isnan(out).any()


# --- Evaluation helpers ----------------------------------------------------

def test_metrics_on_perfect_predictions():
    m = evaluate.classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == m["precision"] == m["recall"] == m["f1"] == 1.0
    assert m["roc_auc"] == 1.0 and m["brier"] < 0.05


def test_confusion_table_layout():
    table = evaluate.confusion_table([0, 0, 1, 1], [0.9, 0.1, 0.2, 0.8])
    assert table.loc["Actually stayed", "Predicted churn"] == 1   # false alarm
    assert table.loc["Actually churned", "Predicted stay"] == 1   # missed churner
    assert table.to_numpy().sum() == 4


def test_risk_levels():
    levels = evaluate.risk_level([0.05, 0.29, 0.30, 0.59, 0.60, 0.99])
    assert levels.tolist() == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_best_threshold_is_found():
    y = [0, 0, 0, 1, 1]
    proba = [0.1, 0.2, 0.35, 0.4, 0.9]
    assert 0.35 < evaluate.best_f1_threshold(y, proba) <= 0.40


# --- Training --------------------------------------------------------------

def test_models_learn_a_real_pattern(signal_result):
    _, result = signal_result
    cv = result.cv_scores
    assert cv.loc["baseline", "roc_auc_mean"] == pytest.approx(0.5)
    for model in ("logistic_regression", "random_forest"):
        assert cv.loc[model, "roc_auc_mean"] > 0.70
    assert result.test_metrics.loc[(result.chosen, "tuned"), "roc_auc"] > 0.70


def test_chosen_by_cross_validation_not_test(signal_result):
    _, result = signal_result
    real = result.cv_scores.drop(index="baseline")
    assert result.chosen == real["roc_auc_mean"].idxmax()


def test_split_is_stratified(signal_result):
    _, result = signal_result
    assert abs(result.split["train_churn_rate"] - result.split["test_churn_rate"]) < 0.02
    assert result.split["test"] == pytest.approx(0.2 * (result.split["train"] +
                                                         result.split["test"]), abs=1)


def test_predictions_cover_every_customer(signal_result):
    df, result = signal_result
    preds = result.predictions
    assert len(preds) == len(df) and preds["customerID"].is_unique
    assert preds["churn_probability"].between(0, 1).all()
    assert set(preds["risk_level"]) <= {"Low", "Medium", "High"}
    assert (preds["predicted_churn"] == (preds["churn_probability"] >= result.threshold)).all()
    assert preds["in_test_set"].sum() == result.split["test"]


def test_risk_levels_order_real_churn(signal_result):
    _, result = signal_result
    rates = result.risk_table["actual_churn_rate"]
    present = rates[result.risk_table["customers"] > 20]
    assert present.is_monotonic_increasing


def test_training_is_reproducible():
    df = clean_customer_data(make_customers_with_signal(250, seed=3))[0]
    a, b = train_and_evaluate(df), train_and_evaluate(df)
    pd.testing.assert_frame_equal(a.predictions, b.predictions)


def test_no_signal_gives_near_random_scores():
    df = clean_customer_data(make_random_customers(500, seed=9))[0]
    result = train_and_evaluate(df)
    assert result.cv_scores.loc[result.chosen, "roc_auc_mean"] < 0.62
