"""
Data ingestion: fetch the raw Telco CSV and validate it before anything uses it.

Validation runs on the RAW file, at the boundary where data enters the
system. Downstream cleaning makes assumptions that would otherwise fail
silently. For example, load_data() maps any Churn value that isn't exactly
"Yes" to 0, so a source that switched to "yes" would train a model that
thinks nobody churns. Checking here stops the pipeline before that happens.

Run from the repo root:  python src/ingest.py
"""

import logging
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import CATEGORICAL_COLS, DROP_COLS, NUMERIC_COLS, RAW_DATA_PATH, TARGET

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXPECTED_COLUMNS = [*DROP_COLS, *NUMERIC_COLS, *CATEGORICAL_COLS, TARGET]
EXPECTED_ROWS = 7043
ROW_TOLERANCE = 0.05  # catches truncated or double-appended files
TARGET_VALUES = {"Yes", "No"}

log = logging.getLogger("ingest")


class DataValidationError(ValueError):
    """The raw data does not match the schema the pipeline was built for."""


def _download(url: str, dest: Path) -> None:
    # Write to a .part file and rename only once complete, so an interrupted
    # download never leaves a truncated telco.csv that later runs would trust.
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)


def ensure_dataset(path: Path = Path(RAW_DATA_PATH), url: str = DATA_URL) -> Path:
    """Download the dataset to `path` unless it is already there."""
    if path.exists():
        log.info("found existing dataset at %s, skipping download", path)
        return path
    log.info("downloading dataset from %s", url)
    path.parent.mkdir(parents=True, exist_ok=True)
    _download(url, path)
    log.info("saved dataset to %s (%d bytes)", path, path.stat().st_size)
    return path


def validate(df: pd.DataFrame) -> None:
    """Raise DataValidationError listing every problem found in the raw data."""
    problems = []

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    low = int(EXPECTED_ROWS * (1 - ROW_TOLERANCE))
    high = int(EXPECTED_ROWS * (1 + ROW_TOLERANCE))
    if not low <= len(df) <= high:
        problems.append(
            f"got {len(df)} rows, expected ~{EXPECTED_ROWS} rows ({low}-{high})"
        )

    # The raw CSV encodes blanks as " " (e.g. TotalCharges), not NaN, so
    # treat whitespace-only strings as missing before checking emptiness.
    blanked = df.replace(r"^\s*$", np.nan, regex=True)

    if TARGET in df.columns:
        values = set(blanked[TARGET].dropna().unique())
        if values - TARGET_VALUES:
            problems.append(
                f"{TARGET} has unexpected values: {sorted(values - TARGET_VALUES)}"
            )
        if TARGET_VALUES - values:
            problems.append(f"{TARGET} is missing classes: {sorted(TARGET_VALUES - values)}")
        if n_blank := int(blanked[TARGET].isna().sum()):
            problems.append(f"{TARGET} has {n_blank} blank values")

    empty = [c for c in blanked.columns if blanked[c].isna().all()]
    if empty:
        problems.append(f"completely empty columns: {empty}")

    if problems:
        raise DataValidationError(
            "dataset failed validation:\n  - " + "\n  - ".join(problems)
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    path = ensure_dataset()
    df = pd.read_csv(path)
    try:
        validate(df)
    except DataValidationError as e:
        log.error("%s", e)
        sys.exit(1)
    churn_rate = (df[TARGET] == "Yes").mean()
    log.info(
        "validation passed: %d rows x %d columns, churn rate %.1f%%",
        len(df), df.shape[1], churn_rate * 100,
    )


if __name__ == "__main__":
    main()
