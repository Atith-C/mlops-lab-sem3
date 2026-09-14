"""
FastAPI service exposing the trained churn model.

Loads models/model.pkl, which contains the full sklearn Pipeline
(preprocessing + classifier). Callers send raw feature values; the
pipeline handles encoding and scaling internally, so serving uses
exactly the same transformations as training.
"""

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# Log to stdout: in a container the runtime (Docker, Kubernetes) collects
# stdout and ships it to the log backend. Files inside a container vanish
# with it and are invisible to `docker logs`.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("churn_api")

MODEL_PATH = Path("models/model.pkl")
THRESHOLD = 0.5

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline once at startup, not per request."""
    global model
    log.info("loading model from %s", MODEL_PATH)
    if not MODEL_PATH.exists():
        log.critical("model file not found at %s", MODEL_PATH)
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        log.exception("failed to load model from %s", MODEL_PATH)
        raise
    log.info("model loaded (%s) path=%s", type(model).__name__, MODEL_PATH)
    yield
    model = None
    log.info("model unloaded, shutting down")


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn from account and service features.",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware that times every request and counts it by route and status,
# plus a GET /metrics endpoint that Prometheus scrapes. /metrics itself is
# excluded so scrapes don't inflate the traffic numbers. Buckets start at
# 5 ms because /predict typically completes in a few ms; the defaults
# (0.1, 0.5, 1 s) would put every request in the first bucket.
Instrumentator(excluded_handlers=["/metrics"]).instrument(
    app, latency_lowr_buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1)
).expose(app)


@app.exception_handler(RequestValidationError)
async def log_validation_error(request: Request, exc: RequestValidationError):
    """Log rejected requests, then return FastAPI's standard 422 unchanged."""
    errors = "; ".join(
        f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
    )
    log.warning("rejected path=%s errors=[%s]", request.url.path, errors)
    return await request_validation_exception_handler(request, exc)


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
        log.warning("predict refused: model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    row = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000
    prediction = proba >= THRESHOLD

    log.info(
        "predict id=%s tenure=%d contract=%s monthly_charges=%.2f "
        "proba=%.4f pred=%s latency_ms=%.1f",
        request_id, features.tenure, features.Contract, features.MonthlyCharges,
        proba, prediction, latency_ms,
    )

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        threshold=THRESHOLD,
    )