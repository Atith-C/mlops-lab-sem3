"""Tests for the pipeline's quality gate decision."""

import joblib

from pipeline import BASELINE, THRESHOLDS, check_gate
from preprocess import load_data
from train import MODEL_PATH, evaluate, split_data


def test_committed_model_passes_gate():
    """Gate the artifact that ships, not just a retrain of it.

    CI's pipeline job proves a retrain from current code passes; this proves
    the committed models/model.pkl, which Docker bakes in and Render serves,
    passes too. Only meaningful if that model was trained on split_data()'s
    training rows, as pipeline.py and train.py both do.
    """
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    metrics = evaluate(joblib.load(MODEL_PATH), X_test, y_test)
    assert all(check_gate(metrics, THRESHOLDS).values()), metrics


def test_gate_passes_at_baseline():
    assert all(check_gate(BASELINE, THRESHOLDS).values())


def test_gate_fails_when_any_metric_is_below_threshold():
    worse = {**BASELINE, "accuracy": 0.735}  # majority-class accuracy
    results = check_gate(worse, THRESHOLDS)
    assert results["accuracy"] is False
    assert results["roc_auc"] is True
