import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import CHURN_FLAG, clean_customer_data


def test_input_is_not_modified(sample_df):
    before = sample_df.copy()
    clean_customer_data(sample_df)
    pd.testing.assert_frame_equal(sample_df, before)


def test_blank_total_charges_for_new_customer_becomes_zero(sample_df):
    clean, log = clean_customer_data(sample_df)
    assert clean.loc[clean["customerID"] == "T-0003", "TotalCharges"].item() == 0.0
    assert log.total_charges_filled_zero == 1
    assert clean["TotalCharges"].dtype == float
    assert not log.remaining_missing


def test_blank_total_charges_for_existing_customer_stays_missing(sample_df):
    df = sample_df.copy()
    df.loc[1, "TotalCharges"] = " "  # tenure 34: cannot be inferred with certainty
    clean, log = clean_customer_data(df)
    assert np.isnan(clean.loc[1, "TotalCharges"])
    assert log.remaining_missing == {"TotalCharges": 1}


def test_churn_flag_matches_target(sample_df):
    clean, _ = clean_customer_data(sample_df)
    assert (clean[CHURN_FLAG] == (clean["Churn"] == "Yes")).all()
    assert clean[CHURN_FLAG].sum() == 2


def test_removes_duplicates_and_repeated_ids(sample_df):
    df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    df.loc[len(df)] = sample_df.iloc[1].tolist()
    df.loc[len(df) - 1, "MonthlyCharges"] = 1.0  # same ID, different data
    clean, log = clean_customer_data(df)
    assert log.duplicate_rows_removed == 1
    assert log.duplicate_ids_removed == 1
    assert clean["customerID"].is_unique and len(clean) == 6


def test_trims_spaces_in_text(sample_df):
    df = sample_df.copy()
    df.loc[0, "Contract"] = "  Month-to-month "
    clean, _ = clean_customer_data(df)
    assert clean.loc[0, "Contract"] == "Month-to-month"


def test_invalid_values_become_missing(sample_df):
    df = sample_df.copy()
    df.loc[0, "Contract"] = "Weekly"
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df.loc[1, "MonthlyCharges"] = -10.0
    df.loc[2, "SeniorCitizen"] = 7
    clean, log = clean_customer_data(df)
    assert pd.isna(clean.loc[0, "Contract"])
    assert np.isnan(clean.loc[1, "MonthlyCharges"])
    assert pd.isna(clean.loc[2, "SeniorCitizen"])
    assert set(log.invalid_values_to_missing) == {"Contract", "MonthlyCharges", "SeniorCitizen"}


def test_rows_without_target_are_removed(sample_df):
    df = sample_df.copy()
    df.loc[0, "Churn"] = " "
    clean, log = clean_customer_data(df)
    assert len(clean) == 5 and log.rows_missing_target_removed == 1


def test_unexpected_columns_dropped(sample_df):
    clean, log = clean_customer_data(sample_df.assign(Extra=1))
    assert "Extra" not in clean.columns and log.dropped_columns == ["Extra"]


def test_missing_required_column_raises(sample_df):
    with pytest.raises(ValueError, match="Contract"):
        clean_customer_data(sample_df.drop(columns=["Contract"]))


def test_cleaning_is_reproducible(random_df):
    first, _ = clean_customer_data(random_df)
    second, _ = clean_customer_data(random_df)
    pd.testing.assert_frame_equal(first, second)
