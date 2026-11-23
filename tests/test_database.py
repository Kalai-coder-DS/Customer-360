from contextlib import closing

import pandas as pd
import pytest

from src.database.db import (
    DatabaseError, file_sha256, get_connection, init_db, list_tables, load_history,
    read_table, record_load, save_dataframe,
)


def test_save_and_read_roundtrip(tmp_path, sample_df):
    db = tmp_path / "test.db"
    with closing(get_connection(db)) as conn:
        assert save_dataframe(sample_df, "customers_raw", conn) == 6
        loaded = read_table("customers_raw", conn)
    assert loaded.shape == sample_df.shape
    assert list(loaded.columns) == list(sample_df.columns)
    assert loaded.loc[2, "TotalCharges"] == " "


def test_replace_does_not_duplicate_rows(tmp_path, sample_df):
    with closing(get_connection(tmp_path / "test.db")) as conn:
        save_dataframe(sample_df, "t", conn)
        save_dataframe(sample_df, "t", conn)
        assert len(read_table("t", conn)) == 6


def test_record_load_and_history(tmp_path):
    db = tmp_path / "test.db"
    with closing(get_connection(db)) as conn:
        init_db(conn)
        first = record_load(conn, source_file="a.csv", table_name="t", n_rows=1,
                            n_columns=2, is_valid=True, n_warnings=0)
        second = record_load(conn, source_file="b.csv", table_name="t", n_rows=3,
                             n_columns=2, is_valid=False, n_warnings=4)
        assert "data_loads" in list_tables(conn)
    history = load_history(db)
    assert (first, second) == (1, 2)
    assert history.iloc[0]["source_file"] == "b.csv"  # newest first


def test_rejects_unsafe_table_name(tmp_path):
    with closing(get_connection(tmp_path / "test.db")) as conn:
        with pytest.raises(DatabaseError):
            save_dataframe(pd.DataFrame({"a": [1]}), "x; DROP TABLE y", conn)


def test_reading_missing_table_raises(tmp_path):
    with closing(get_connection(tmp_path / "test.db")) as conn:
        with pytest.raises(DatabaseError, match="does not exist"):
            read_table("customers_raw", conn)


def test_sha256_is_stable(sample_csv):
    assert file_sha256(sample_csv) == file_sha256(sample_csv)
    assert len(file_sha256(sample_csv)) == 64
