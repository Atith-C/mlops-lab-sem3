"""
Experiment tracking and model versioning with MLflow.

Trains several model configurations, logs params/metrics/artifacts for each,
and registers the best run in the MLflow Model Registry.
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import load_data, build_preprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "telco-churn"
REGISTERED_MODEL_NAME = "telco-churn-classifier"

CONFIGS = [
    {
        "run_name": "logreg-default",
        "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "params": {"model": "LogisticRegression", "class_weight": "none"},
    },
    {
        "run_name": "logreg-balanced",
        "model": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "params": {"model": "LogisticRegression", "class_weight": "balanced"},
    },
    {
        "run_name": "rf-default",
        "model": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "params": {"model": "RandomForest", "n_estimators": 200,
                   "class_weight": "none"},
    },
    {
        "run_name": "rf-balanced",
        "model": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "params": {"model": "RandomForest", "n_estimators": 200,
                   "class_weight": "balanced"},
    },
]


def evaluate(pipe, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_churn": precision_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
        "f1_churn": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def register_best_run():
    """Find the run with the highest churn F1 and register it."""
    best = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.f1_churn DESC"],
        max_results=1,
    ).iloc[0]

    result = mlflow.register_model(
        model_uri=f"runs:/{best['run_id']}/model",
        name=REGISTERED_MODEL_NAME,
    )

    print(f"\nRegistered '{REGISTERED_MODEL_NAME}' version {result.version}")
    print(f"  run: {best['tags.mlflow.runName']} ({best['run_id'][:8]})")
    print(f"  f1_churn: {best['metrics.f1_churn']:.3f}")


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    for cfg in CONFIGS:
        with mlflow.start_run(run_name=cfg["run_name"]):
            pipe = Pipeline([
                ("pre", build_preprocessor()),
                ("clf", cfg["model"]),
            ])
            pipe.fit(X_train, y_train)

            metrics = evaluate(pipe, X_test, y_test)

            mlflow.log_params(cfg["params"])
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipe,
                name="model",
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )

            print(f"{cfg['run_name']:<18} "
                  f"acc={metrics['accuracy']:.3f}  "
                  f"recall={metrics['recall_churn']:.3f}  "
                  f"f1={metrics['f1_churn']:.3f}  "
                  f"auc={metrics['roc_auc']:.3f}")

    register_best_run()


if __name__ == "__main__":
    main()