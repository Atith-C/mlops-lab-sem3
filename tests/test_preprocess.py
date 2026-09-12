"""Tests for data loading and the preprocessing pipeline."""

import pandas as pd
import pytest

from preprocess import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    build_preprocessor,
    load_data,
)


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_load_data_shape(data):
    X, y = data
    assert len(X) == len(y)
    assert len(X) > 0


def test_customer_id_dropped(data):
    X, _ = data
    assert "customerID" not in X.columns


def test_target_is_binary(data):
    _, y = data
    assert set(y.unique()) == {0, 1}


def test_total_charges_is_numeric(data):
    """The raw column is text with blanks; loading must coerce it to float."""
    X, _ = data
    assert pd.api.types.is_numeric_dtype(X["TotalCharges"])
    assert X["TotalCharges"].isna().sum() == 0


def test_new_customers_have_zero_total_charges(data):
    """tenure=0 customers were never billed, so TotalCharges must be 0."""
    X, _ = data
    new_customers = X[X["tenure"] == 0]
    assert (new_customers["TotalCharges"] == 0).all()


def test_expected_columns_present(data):
    X, _ = data
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        assert col in X.columns, f"missing column: {col}"


def test_preprocessor_expands_features(data):
    """One-hot encoding should produce more columns than it consumes."""
    X, _ = data
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.shape[1] > X.shape[1]


def test_preprocessor_output_has_no_nans(data):
    X, _ = data
    Xt = build_preprocessor().fit_transform(X)
    assert not pd.isna(Xt).any()