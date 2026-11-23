import numpy as np
import pandas as pd

from src.data.validator import validate_customer_data


def test_clean_sample_is_valid(sample_df):
    report = validate_customer_data(sample_df)
    assert report.is_valid
    assert report.n_rows == 6 and report.n_columns == 21
    assert not report.missing_columns and not report.invalid_categories


def test_blank_total_charges_counted_as_missing_not_non_numeric(sample_df):
    report = validate_customer_data(sample_df)
    assert report.missing_values == {"TotalCharges": 1}
    assert report.blank_text_values == {"TotalCharges": 1}
    assert "TotalCharges" not in report.non_numeric_values


def test_target_distribution_and_churn_rate(sample_df):
    report = validate_customer_data(sample_df)
    assert report.target_distribution == {"No": 4, "Yes": 2}
    assert abs(report.churn_rate - 2 / 6) < 1e-9


def test_detects_duplicates(sample_df):
    df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    report = validate_customer_data(df)
    assert report.duplicate_rows == 1
    assert report.duplicate_ids == 1


def test_detects_repeated_id_with_different_data(sample_df):
    df = sample_df.copy()
    df.loc[1, "customerID"] = "T-0001"
    report = validate_customer_data(df)
    assert report.duplicate_rows == 0 and report.duplicate_ids == 1


def test_detects_invalid_category_and_code(sample_df):
    df = sample_df.copy()
    df.loc[0, "Contract"] = "Weekly"
    df.loc[1, "SeniorCitizen"] = 5
    report = validate_customer_data(df)
    assert report.invalid_categories["Contract"] == ["Weekly"]
    assert "SeniorCitizen" in report.invalid_categories
    assert report.is_valid  # warnings only


def test_detects_non_numeric_and_negative(sample_df):
    df = sample_df.copy()
    df.loc[0, "TotalCharges"] = "abc"
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df.loc[1, "MonthlyCharges"] = -5.0
    report = validate_customer_data(df)
    assert report.non_numeric_values == {"TotalCharges": 1}
    assert report.negative_values == {"MonthlyCharges": 1}


def test_real_nan_counted_as_missing(sample_df):
    df = sample_df.copy()
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df.loc[0, "MonthlyCharges"] = np.nan
    report = validate_customer_data(df)
    assert report.missing_values["MonthlyCharges"] == 1


def test_missing_column_is_an_error(sample_df):
    report = validate_customer_data(sample_df.drop(columns=["Contract"]))
    assert not report.is_valid
    assert report.missing_columns == ["Contract"]


def test_unexpected_column_is_a_warning(sample_df):
    report = validate_customer_data(sample_df.assign(Extra=1))
    assert report.is_valid
    assert report.unexpected_columns == ["Extra"]


def test_single_class_target_is_an_error(sample_df):
    report = validate_customer_data(sample_df.assign(Churn="No"))
    assert not report.is_valid


def test_empty_dataframe_is_an_error():
    assert not validate_customer_data(pd.DataFrame()).is_valid


def test_summary_lines_are_readable(sample_df):
    text = "\n".join(validate_customer_data(sample_df).summary_lines())
    assert "VALID" in text and "churn rate" in text
