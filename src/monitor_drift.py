"""
Data drift detection for the churn model with Evidently.

WHAT DRIFT IS
    A model learns the patterns of the data it was trained on. Data drift is
    when the inputs it receives in production stop looking like that training
    data: the distribution of one or more features shifts. Here, a marketing
    push brings in newer, less committed customers, so tenure falls and
    month-to-month contracts become more common.

WHY IT MATTERS EVEN WHEN THE API IS HEALTHY
    Logs and /metrics only describe the service: every request returns 200 in
    a few milliseconds and nothing looks wrong. But the model is now scoring
    customers unlike those it learned from, so its probabilities can be
    quietly miscalibrated. We can't measure accuracy directly, because the
    true churn label only arrives weeks later. Comparing incoming features to
    the training distribution is the early warning we can get today.

HOW EVIDENTLY DECIDES
    Each feature is compared between a REFERENCE set (training-time data) and a
    CURRENT set (recent production data). With >1000 rows Evidently uses a
    distance, not a p-value test: normalised Wasserstein for numeric columns,
    Jensen-Shannon for categorical ones. A column drifts when its distance is
    >= 0.1. The dataset as a whole is flagged when >= 50% of columns drift.

Run from the repo root:  python src/monitor_drift.py
"""

from pathlib import Path

import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

from preprocess import load_data

RANDOM_STATE = 42
CURRENT_SIZE = 1000  # one "recent batch" of scored customers
REPORT_PATH = Path("monitoring/data_drift_report.html")


def make_reference_and_current():
    """Split the data in half, then bias the second half toward new customers."""
    X, _ = load_data()  # features only: labels aren't available in production
    reference = X.sample(frac=0.5, random_state=RANDOM_STATE)
    pool = X.drop(reference.index)

    # Resample REAL customers rather than editing columns, so every row stays
    # internally consistent (TotalCharges still matches tenure, etc.).
    # Weight decays with tenure (halving roughly every 25 months) and is 1.5x
    # for month-to-month contracts. Tuned so the shift is noticeable but
    # plausible: mean tenure ~32 -> ~19 months, month-to-month ~56% -> ~75%.
    # Sampling is without replacement (no duplicate customers), which caps the
    # sample size: n * max(normalised weight) must be <= 1, i.e. ~1650 here.
    weights = np.exp(-pool["tenure"] / 36) * np.where(
        pool["Contract"] == "Month-to-month", 1.5, 1.0
    )
    current = pool.sample(
        n=CURRENT_SIZE, weights=weights, random_state=RANDOM_STATE
    )
    return reference, current


def main():
    reference, current = make_reference_and_current()

    snapshot = Report([DataDriftPreset()], include_tests=True).run(
        current_data=current, reference_data=reference
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(REPORT_PATH))

    # snapshot.dict() -> {"metrics": [...], "tests": [...]}. metrics[0] is
    # DriftedColumnsCount; the rest are one ValueDrift per column. Each test
    # points at its metric by id and carries Evidently's verdict (FAIL = drift).
    result = snapshot.dict()
    status = {t["metric_config"]["metric_id"]: t["status"] for t in result["tests"]}
    summary, *columns = result["metrics"]

    def mtm_share(df):
        return (df["Contract"] == "Month-to-month").mean()

    print(f"Reference rows: {len(reference)}   Current rows: {len(current)}")
    print(
        f"Simulated shift: mean tenure {reference['tenure'].mean():.1f} -> "
        f"{current['tenure'].mean():.1f} months, month-to-month "
        f"{mtm_share(reference):.0%} -> {mtm_share(current):.0%}\n"
    )

    print(f"{'Feature':<18}{'Method':<32}{'Score':>7}{'Threshold':>11}  Drift")
    print("-" * 74)
    for m in sorted(columns, key=lambda m: m["value"], reverse=True):
        cfg = m["config"]
        drifted = "YES" if status[m["id"]] == "FAIL" else "no"
        print(
            f"{cfg['column']:<18}{cfg['method']:<32}"
            f"{m['value']:>7.3f}{cfg['threshold']:>11.2f}  {drifted}"
        )

    count, share = int(summary["value"]["count"]), summary["value"]["share"]
    dataset_drift = status[summary["id"]] == "FAIL"
    drift_share = summary["config"]["drift_share"]
    print("-" * 74)
    print(f"Drifted features: {count} / {len(columns)} ({share:.0%})")
    print(
        f"Dataset drift:    {'DETECTED' if dataset_drift else 'NOT DETECTED'} "
        f"(flagged when >= {drift_share:.0%} of features drift)"
    )
    print(f"\nHTML report: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
