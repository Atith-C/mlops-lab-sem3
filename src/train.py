"""
Train a churn classifier and save the full pipeline to models/model.pkl.

The saved artifact contains BOTH the preprocessor and the model, so serving
code can pass raw feature dicts straight in without reimplementing encoding.
"""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, load_data

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_PATH = Path("models/model.pkl")


def build_pipeline() -> Pipeline:
    """Preprocessor + classifier as a single fittable object."""
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved pipeline -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
