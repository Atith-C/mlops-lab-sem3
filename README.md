# mlops-lab-sem3
End-to-end MLOps pipeline for customer churn prediction (IBM Telco, 7,043 customers), covering data ingestion, validation, training with a quality gate, MLflow tracking, FastAPI serving, Docker, CI/CD, cloud deployment, and monitoring. Sem 3 lab work.

**Live API:** https://telco-churn-api-v7a9.onrender.com (`/docs`, `/health`, `/predict`, `/metrics`)

## Architecture

### How the ten practicals connect

```
                 IBM Telco CSV (GitHub raw URL)
                              |
  [10] src/ingest.py          |  download if missing, then validate the RAW schema:
                              |  columns, row count, Churn in {Yes,No}, no empty columns
                              v
  [2]  src/preprocess.py      load_data(): clean (TotalCharges -> numeric, blanks -> 0.0)
                              build_preprocessor(): unfitted impute/scale/one-hot
                              |
         +--------------------+---------------------+
         v                                          v
  [4]  src/train.py                          [3]  src/train_mlflow.py
       Pipeline(preprocessor +                    4 configs logged to MLflow,
       LogisticRegression)                        best registered as
         |                                        telco-churn-classifier
         v
  [10] src/pipeline.py  ingest > validate > preprocess > train > evaluate
                        > QUALITY GATE (accuracy >= 0.7855, ROC-AUC >= 0.8221)
                        > save models/model.pkl  (only if the gate passes)
                              |
                              v
  [5]  Dockerfile        python:3.11-slim + requirements-api.txt + app.py + model.pkl
                              |
                              v
  [7]  src/app.py        FastAPI: /predict (validated input), /health, /metrics
                              |
                              v
  [9]  Render            builds the Dockerfile from `main`, serves it on $PORT
                              |
                              v
  [8]  Monitoring        logs -> stdout (one line per prediction)
                         /metrics -> Prometheus (request counts, latency, status)
                         src/monitor_drift.py -> Evidently data drift report

  [1]  Environment: conda env + requirements files     [6] Tests + GitHub Actions CI
  [10] Automation: Makefile, CI pipeline job, docker-compose (API + MLflow server)
```

| # | Practical | Where it lives |
|---|---|---|
| 1 | Environment and project structure | `requirements.txt`, `requirements-api.txt` |
| 2 | Preprocessing and EDA | `src/preprocess.py`, `notebooks/01_eda.ipynb` |
| 3 | Experiment tracking and model registry | `src/train_mlflow.py` |
| 4 | Model training and artifact | `src/train.py`, `models/model.pkl` |
| 5 | Containerisation | `Dockerfile`, `.dockerignore` |
| 6 | Testing and CI | `tests/`, `.github/workflows/ci.yml` |
| 7 | Model serving API | `src/app.py` |
| 8 | Monitoring and logging | `src/app.py`, `src/monitor_drift.py`, `monitoring/README.md` |
| 9 | Cloud deployment | Render (auto-deploy from `main` after CI passes) |
| 10 | End-to-end pipeline | `src/ingest.py`, `src/pipeline.py`, `Makefile`, `docker-compose.yml`, CI `pipeline` job |

### Commit-to-production path

```
git push to main
      |
      v
GitHub Actions ─┬─ test      checkout > install > ingest+validate > ruff > pytest
                │            (includes: committed model.pkl must pass the gate)
                └─ pipeline  checkout > install > ingest+validate > pipeline.py
                             (a fresh retrain must pass the gate)
                        both green?
                             |  no  -> docker job skipped, commit marked red, nothing deploys
                             v  yes
                      docker    build image > run it > curl /health
                             |
                             v
Render (Auto-Deploy: "After CI Checks Pass")  builds the Dockerfile > live
```

**What production serves:** the `models/model.pkl` committed to git. Docker copies it into the image, and Render builds from the repo. The CI `pipeline` job retrains on a throwaway runner to prove the *code* still produces a passing model. That retrained model is discarded, and CI never commits anything. The *shipped* file is gated separately by `test_committed_model_passes_gate` in the `test` job. Updating the production model is therefore a deliberate act: run `make pipeline` locally (it only writes the file if the gate passes), then commit `models/model.pkl`.

## Run it from a fresh clone

Windows / PowerShell. Every `make` target is a single command, shown in the comments, so it can be pasted directly if `make` isn't installed (`winget install ezwinports.make`, then restart the terminal).

```powershell
git clone https://github.com/Atith-C/mlops-lab-sem3.git
cd mlops-lab-sem3
conda create -n mlops python=3.11 -y
conda activate mlops

make install        # pip install -r requirements.txt
make ingest         # python src/ingest.py              download + validate the dataset
make                # ruff check src tests; pytest -v
make train          # python src/pipeline.py --dry-run  full pipeline, never writes the model
make pipeline       # python src/pipeline.py            gated: saves models/model.pkl only on PASS
make drift          # python src/monitor_drift.py       writes monitoring/data_drift_report.html

docker compose up --build -d      # API on :8000, MLflow server on :5000
docker compose down               # stop (MLflow data kept in a Docker volume)
```

Demonstrate the quality gate rejecting a model (exit code 1, model file untouched):

```powershell
python src/pipeline.py --min-accuracy 0.99
```

`requirements.txt` is the full Windows development environment (it pins `pywin32`). On Linux or macOS, install `requirements-api.txt` plus `pytest httpx ruff mlflow evidently` instead. CI installs `requirements-api.txt` plus `pytest httpx ruff`.

More on the monitoring layers: [monitoring/README.md](monitoring/README.md).
