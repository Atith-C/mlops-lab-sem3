"""Tests for the FastAPI prediction service."""

import pytest
from fastapi.testclient import TestClient

from app import app

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_root(client):
    assert client.get("/").status_code == 200


def test_predict_returns_valid_probability(client):
    r = client.post("/predict", json=VALID_CUSTOMER)
    assert r.status_code == 200

    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)


def test_prediction_agrees_with_threshold(client):
    body = client.post("/predict", json=VALID_CUSTOMER).json()
    expected = body["churn_probability"] >= body["threshold"]
    assert body["churn_prediction"] == expected


def test_invalid_category_rejected(client):
    bad = {**VALID_CUSTOMER, "Contract": "Monthly"}
    assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_rejected(client):
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    assert client.post("/predict", json=bad).status_code == 422


def test_negative_tenure_rejected(client):
    bad = {**VALID_CUSTOMER, "tenure": -5}
    assert client.post("/predict", json=bad).status_code == 422


def test_prediction_is_logged(client, caplog):
    with caplog.at_level("INFO", logger="churn_api"):
        client.post("/predict", json=VALID_CUSTOMER)
    line = next(r.getMessage() for r in caplog.records if "predict id=" in r.getMessage())
    assert "proba=" in line
    assert "latency_ms=" in line


def test_metrics_endpoint(client):
    client.post("/predict", json=VALID_CUSTOMER)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text
    assert "http_request_duration_seconds" in r.text
    assert 'handler="/predict"' in r.text


def test_metrics_counts_rejections(client):
    client.post("/predict", json={**VALID_CUSTOMER, "Contract": "Monthly"})
    r = client.get("/metrics")
    assert any(
        line.startswith("http_requests_total")
        and 'handler="/predict"' in line
        and 'status="4xx"' in line
        for line in r.text.splitlines()
    )


def test_high_risk_scores_above_low_risk(client):
    """A new month-to-month customer should outrank a long-tenure two-year one."""
    low_risk = {
        **VALID_CUSTOMER,
        "tenure": 65,
        "Contract": "Two year",
        "PaymentMethod": "Credit card (automatic)",
        "TotalCharges": 4200.0,
    }
    high = client.post("/predict", json=VALID_CUSTOMER).json()["churn_probability"]
    low = client.post("/predict", json=low_risk).json()["churn_probability"]
    assert high > low