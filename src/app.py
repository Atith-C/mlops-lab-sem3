"""
FastAPI service exposing the trained churn model.

Loads models/model.pkl, which contains the full sklearn Pipeline
(preprocessing + classifier). Callers send raw feature values; the
pipeline handles encoding and scaling internally, so serving uses
exactly the same transformations as training.
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.pkl")
THRESHOLD = 0.5

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn from account and service features.",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def load_model():
    """Load the pipeline once at startup, not per request."""
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    model = joblib.load(MODEL_PATH)


class CustomerFeatures(BaseModel):
    """One customer's raw features, exactly as they appear in the dataset."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [{
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
            }]
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    threshold: float


@app.get("/")
def root():
    return {
        "service": "Telco Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=proba >= THRESHOLD,
        threshold=THRESHOLD,
    )