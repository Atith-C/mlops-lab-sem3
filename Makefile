# Task runner for the churn MLOps pipeline. Run from the repo root with the
# mlops env active. Every recipe is one plain command, so any line can also be
# pasted into PowerShell directly.
PY ?= python
IMAGE = telco-churn-api

.PHONY: all install ingest train pipeline test lint drift docker-build docker-run clean

all: lint test

install:
	$(PY) -m pip install -r requirements.txt

ingest:
	$(PY) src/ingest.py

# Train, evaluate and gate, but never write models/model.pkl.
train:
	$(PY) src/pipeline.py --dry-run

# The only target that can replace the production model, and only if the gate passes.
pipeline:
	$(PY) src/pipeline.py

test:
	$(PY) -m pytest -v

lint:
	$(PY) -m ruff check src tests

drift:
	$(PY) src/monitor_drift.py

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

# Generated files only: never data, models/model.pkl, mlruns/ or mlflow.db.
# Python instead of rm -rf so it works under Windows cmd and Linux sh alike.
clean:
	$(PY) -c "import pathlib, shutil; here = pathlib.Path('.'); [shutil.rmtree(p, ignore_errors=True) for p in [*here.rglob('__pycache__'), here / '.pytest_cache', here / '.ruff_cache']]; [p.unlink() for p in [*here.glob('monitoring/*.html'), *here.rglob('*.part')]]"
