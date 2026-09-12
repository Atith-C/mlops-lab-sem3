"""
Data loading and preprocessing for the Telco Customer Churn dataset.

Exposes an UNFITTED preprocessor. Fitting happens in train.py as part of a
full Pipeline, so preprocessing and model serialise together into one artifact.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DATA_PATH = "data/raw/telco.csv"
TARGET = "Churn"
DROP_COLS = ["customerID"]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_data(path: str = RAW_DATA_PATH):
    """Load the raw CSV and return (X, y) ready for the preprocessor."""
    df = pd.read_csv(path)

    # TotalCharges arrives as text: 11 new customers (tenure=0) have a blank
    # value because they have not been billed yet. The true value is 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = df.drop(columns=DROP_COLS)

    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the Telco features."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])


if __name__ == "__main__":
    X, y = load_data()
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    print(f"Raw features:        {X.shape}")
    print(f"Transformed features:{Xt.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")