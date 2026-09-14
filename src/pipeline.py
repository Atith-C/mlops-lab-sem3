"""
End-to-end training pipeline with a quality gate.

    ingest -> validate -> preprocess -> train -> evaluate -> quality gate -> save

Every stage reuses an existing module; this file only orders them, times
them, and decides whether the new model is good enough to replace
models/model.pkl, which is the artifact production serves.

Run from the repo root:
    python src/pipeline.py                     # full run, saves only if the gate passes
    python src/pipeline.py --dry-run           # everything except the save
    python src/pipeline.py --min-accuracy 0.99 # demonstrate a gate failure
"""

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import joblib
import pandas as pd

from ingest import DataValidationError, ensure_dataset, validate
from preprocess import load_data
from train import MODEL_PATH, build_pipeline, evaluate, split_data

# Metrics of the model currently in production (Exp 4). The tolerance is
# ~2 standard errors of accuracy on the 1,409-row test set
# (sqrt(0.8 * 0.2 / 1409) ~= 0.011). A refreshed dataset can move the
# metrics that much by chance; a real regression moves them further.
BASELINE = {"accuracy": 0.8055, "roc_auc": 0.8421}
TOLERANCE = 0.02
THRESHOLDS = {name: round(value - TOLERANCE, 4) for name, value in BASELINE.items()}

STAGES = ["ingest", "validate", "preprocess", "train", "evaluate", "quality gate", "save"]

log = logging.getLogger("pipeline")


@contextmanager
def stage(name: str):
    """Log start, end and duration of a stage; log and re-raise on failure."""
    tag = f"[{STAGES.index(name) + 1}/{len(STAGES)}] {name}"
    log.info("%s started", tag)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        log.error("%s FAILED after %.2fs", tag, time.perf_counter() - start)
        raise
    log.info("%s done in %.2fs", tag, time.perf_counter() - start)


def check_gate(metrics: dict, thresholds: dict) -> dict:
    """Map each gated metric to True (meets its threshold) or False."""
    return {name: metrics[name] >= t for name, t in thresholds.items()}


def save_model(pipe, path: Path) -> None:
    # Write then rename, so a crash mid-dump can't leave production with a
    # truncated model.pkl.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    joblib.dump(pipe, tmp)
    tmp.replace(path)


def print_summary(metrics, thresholds, results, outcome, elapsed):
    print("\n" + "=" * 22 + " QUALITY GATE " + "=" * 22)
    for name, threshold in thresholds.items():
        verdict = "PASS" if results[name] else "FAIL"
        print(f"  {name:<13}{metrics[name]:.4f}  >=  {threshold:.4f}   {verdict}")
    print(f"  {'recall_churn':<13}{metrics['recall_churn']:.4f}               (info only)")
    print("-" * 58)
    print(f"RESULT: {outcome}")
    print(f"Total pipeline time: {elapsed:.2f}s")


def run(dry_run: bool, thresholds: dict) -> int:
    started = time.perf_counter()
    log.info(
        "pipeline started (dry_run=%s, gate: %s)",
        dry_run, ", ".join(f"{k} >= {v}" for k, v in thresholds.items()),
    )

    try:
        with stage("ingest"):
            path = ensure_dataset()
        with stage("validate"):
            validate(pd.read_csv(path))
    except DataValidationError as e:
        log.error("%s", e)
        print("\nRESULT: FAIL - data validation failed, no model trained")
        return 1

    with stage("preprocess"):
        X, y = load_data(path)
        X_train, X_test, y_train, y_test = split_data(X, y)
        log.info("train=%d rows, test=%d rows", len(X_train), len(X_test))

    with stage("train"):
        pipe = build_pipeline().fit(X_train, y_train)

    with stage("evaluate"):
        metrics = evaluate(pipe, X_test, y_test)
        log.info(", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    with stage("quality gate"):
        results = check_gate(metrics, thresholds)
        passed = all(results.values())
        log.info("gate %s", "PASSED" if passed else "FAILED")

    if not passed:
        log.warning("[7/7] save skipped: gate failed, %s left unchanged", MODEL_PATH)
        outcome = f"FAIL - model rejected, {MODEL_PATH} left unchanged"
    elif dry_run:
        log.info("[7/7] save skipped: --dry-run")
        outcome = f"PASS - dry run, {MODEL_PATH} left unchanged"
    else:
        with stage("save"):
            save_model(pipe, MODEL_PATH)
        outcome = f"PASS - model saved to {MODEL_PATH}"

    print_summary(metrics, thresholds, results, outcome, time.perf_counter() - started)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and gate the churn model.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run every stage but never write the model file",
    )
    parser.add_argument("--min-accuracy", type=float, default=THRESHOLDS["accuracy"])
    parser.add_argument("--min-roc-auc", type=float, default=THRESHOLDS["roc_auc"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    thresholds = {"accuracy": args.min_accuracy, "roc_auc": args.min_roc_auc}
    for name, value in thresholds.items():
        if value != THRESHOLDS[name]:
            log.warning("threshold overridden: %s >= %s (default %s)",
                        name, value, THRESHOLDS[name])

    return run(args.dry_run, thresholds)


if __name__ == "__main__":
    sys.exit(main())
