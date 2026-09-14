"""Tests for dataset ingestion and raw-data validation."""

import pandas as pd
import pytest

import ingest
from ingest import DataValidationError, validate
from preprocess import RAW_DATA_PATH


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(RAW_DATA_PATH)


def test_valid_data_passes(raw):
    validate(raw)  # raises on failure


@pytest.mark.parametrize(
    "corrupt, expected_msg",
    [
        (lambda df: df.drop(columns=["Contract"]), "missing columns"),
        (lambda df: df.assign(Churn=df["Churn"].str.lower()), "unexpected values"),
        (lambda df: df.assign(PaymentMethod=" "), "empty columns"),
        (lambda df: df.head(100), "rows"),
    ],
    ids=["missing_column", "bad_target_values", "empty_column", "truncated"],
)
def test_corrupted_data_rejected(raw, corrupt, expected_msg):
    with pytest.raises(DataValidationError, match=expected_msg):
        validate(corrupt(raw.copy()))


def test_existing_file_skips_download(tmp_path, monkeypatch):
    existing = tmp_path / "telco.csv"
    existing.write_text("already here")

    def fail(*args):
        raise AssertionError("download should not run when the file exists")

    monkeypatch.setattr(ingest, "_download", fail)
    assert ingest.ensure_dataset(existing) == existing
    assert existing.read_text() == "already here"
