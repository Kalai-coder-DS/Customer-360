import io

import pytest

from src.data.loader import DataLoadError, load_customer_csv


def test_loads_csv_with_expected_shape(sample_csv):
    df = load_customer_csv(sample_csv)
    assert df.shape == (6, 21)


def test_customer_id_is_text(sample_csv):
    df = load_customer_csv(sample_csv)
    assert df["customerID"].map(type).eq(str).all()


def test_does_not_change_blank_total_charges(sample_csv):
    df = load_customer_csv(sample_csv)
    assert df.loc[2, "TotalCharges"] == " "  # raw layer keeps the original value


def test_accepts_file_like_object():
    buffer = io.StringIO("customerID,Churn\nA1,Yes\nA2,No\n")
    assert len(load_customer_csv(buffer)) == 2


def test_strips_spaces_in_headers():
    df = load_customer_csv(io.StringIO(" customerID , Churn \nA1,Yes\n"))
    assert list(df.columns) == ["customerID", "Churn"]


def test_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(DataLoadError, match="download_data.py"):
        load_customer_csv(tmp_path / "nope.csv")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataLoadError, match="empty"):
        load_customer_csv(path)


def test_headers_only_raises(tmp_path):
    path = tmp_path / "headers.csv"
    path.write_text("customerID,Churn\n")
    with pytest.raises(DataLoadError, match="no rows"):
        load_customer_csv(path)
