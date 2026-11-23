import json

import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import clean_customer_data
from src.explain import explainer as ex
from src.models.features import MODEL_FEATURES, prepare_xy
from src.models.train import make_models
from tests.conftest import make_customers_with_signal


@pytest.fixture(scope="module")
def fitted():
    df = clean_customer_data(make_customers_with_signal(1200, seed=5))[0]
    X, y = prepare_xy(df)
    models = make_models()
    lr = models["logistic_regression"].fit(X, y)
    rf = models["random_forest"].fit(X, y)
    background = X.sample(100, random_state=0)
    return X, y, lr, rf, background


def test_encoded_columns_map_to_original_features(fitted):
    _, _, lr, _, _ = fitted
    mapping = ex.encoded_to_original(lr)
    assert set(mapping) == set(MODEL_FEATURES)
    names = lr.named_steps["prep"].get_feature_names_out()
    assert dict(zip(names, mapping))["Contract_Month-to-month"] == "Contract"


def test_linear_contributions_add_up_exactly(fitted):
    X, _, lr, _, background = fitted
    contrib, base = ex.linear_contributions(lr, X.head(20), background)
    expected = ex.logit(lr.predict_proba(X.head(20))[:, 1])
    assert np.allclose(base + contrib.sum(axis=1).to_numpy(), expected, atol=1e-6)
    assert list(contrib.columns) == MODEL_FEATURES


def test_sampling_agrees_with_exact_linear_values(fitted):
    X, _, lr, _, background = fitted
    exact, _ = ex.linear_contributions(lr, X.head(1), background)
    approx, _ = ex.sampling_contributions(lr, X.iloc[0], background, n_permutations=400)
    top = exact.iloc[0].abs().sort_values(ascending=False).index[:3]
    assert np.allclose(exact.iloc[0][top], approx[top], atol=0.15)


def test_sampling_contributions_add_up_for_random_forest(fitted):
    X, _, _, rf, background = fitted
    phi, base = ex.sampling_contributions(rf, X.iloc[3], background, n_permutations=30)
    expected = float(ex.logit(rf.predict_proba(X.iloc[[3]])[:, 1])[0])
    assert base + phi.sum() == pytest.approx(expected, abs=1e-6)


def test_explain_customers_chooses_method(fitted):
    X, _, lr, rf, background = fitted
    _, _, method_lr = ex.explain_customers(lr, X.head(3), background)
    contrib_rf, bases_rf, method_rf = ex.explain_customers(rf, X.head(2), background,
                                                           n_permutations=10)
    assert "linear" in method_lr and "sampling" in method_rf
    assert contrib_rf.shape == (2, len(MODEL_FEATURES)) and len(bases_rf) == 2


def test_planted_pattern_is_recovered(fitted):
    X, y, lr, _, _ = fitted
    importance = ex.global_importance(lr, X, y, n_repeats=3)
    assert set(importance.index[:2]) == {"Contract", "tenure"}


def test_readable_values():
    assert ex.readable_value("tenure", 2.0) == "Tenure: 2 months"
    assert ex.readable_value("MonthlyCharges", 95.1) == "Monthly charges: $95.10/month"
    assert ex.readable_value("SeniorCitizen", 1.0) == "Senior citizen: Yes"
    assert ex.readable_value("Contract", "Two year") == "Contract: Two year"
    assert ex.readable_value("tenure", np.nan) == "Tenure: unknown"


@pytest.mark.parametrize("value, word", [(1.2, "strong"), (-0.5, "moderate"),
                                          (0.15, "slight"), (0.02, "minimal")])
def test_strength_words(value, word):
    assert ex.strength(value) == word


def test_top_factors_and_text_only_name_supported_factors():
    contrib = pd.Series({"tenure": 1.5, "Contract": 0.6, "gender": 0.01, "TechSupport": -0.7})
    row = pd.Series({"tenure": 2, "Contract": "Month-to-month", "gender": "Male",
                     "TechSupport": "Yes"})
    raising, lowering = ex.top_factors(contrib, row)
    assert [f["feature"] for f in raising] == ["tenure", "Contract"]  # gender is minimal
    assert [f["feature"] for f in lowering] == ["TechSupport"]
    text = ex.explanation_text(0.87, "High", "Logistic Regression", raising, lowering)
    assert text.startswith("This customer has an estimated 87% probability of churning")
    assert "Tenure: 2 months (strong)" in text and "Gender" not in text
    assert "not proven cause" in text


def test_text_when_nothing_raises_risk():
    text = ex.explanation_text(0.05, "Low", "Random Forest", [], [])
    assert "No single factor noticeably raises" in text


def test_contributions_json_roundtrip():
    data = json.loads(ex.contributions_json(pd.Series({"tenure": 0.123456})))
    assert data == {"tenure": 0.1235}


def test_logistic_coefficients_only_for_linear(fitted):
    _, _, lr, rf, _ = fitted
    table = ex.logistic_coefficients(lr)
    assert {"coefficient", "odds_ratio"} <= set(table.columns)
    assert np.allclose(table["odds_ratio"], np.exp(table["coefficient"]))
    assert ex.logistic_coefficients(rf) is None
