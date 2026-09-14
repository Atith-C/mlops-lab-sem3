# Monitoring (Experiment 8)

Three layers, each answering a different question. Run all commands from the repo root with the `mlops` env active.

| Layer | Question it answers | Tool | Where |
|---|---|---|---|
| Logs | What happened to *this* request? | Python `logging` → stdout | `src/app.py` |
| Metrics | Is the service healthy *right now*? | `prometheus-fastapi-instrumentator` | `GET /metrics` |
| Drift | Is the model still seeing the data it was trained on? | Evidently | `src/monitor_drift.py` |

## 1. Structured logs

Each prediction writes one `key=value` line to stdout: request id, key inputs, probability, prediction, model latency. Startup, model-load failures and 422 rejections are logged too. Stdout lets Docker/Kubernetes collect the logs (`docker logs <container>`). No log files live inside the container.

```powershell
uvicorn src.app:app --port 8000
```

```
INFO churn_api | predict id=9d799517 tenure=1 contract=Month-to-month monthly_charges=29.85 proba=0.6137 pred=True latency_ms=7.2
WARNING churn_api | rejected path=/predict errors=[body.Contract: Input should be 'Month-to-month', 'One year' or 'Two year']
```

## 2. Prometheus metrics

Middleware records every request. `GET /metrics` exposes the totals in Prometheus text format, and a Prometheus server scrapes it on an interval.

- `http_requests_total{handler, method, status}`: request count by route and status class (`2xx`, `4xx`, `5xx`)
- `http_request_duration_seconds_bucket{handler, method, le}`: latency histogram (buckets from 5 ms to 1 s)
- `http_request_duration_seconds_sum` / `_count`: total time and count, so mean = sum / count

```powershell
Invoke-WebRequest http://localhost:8000/metrics -UseBasicParsing | Select-Object -ExpandProperty Content
```

## 3. Data drift

Compares a reference set (training-time data) with a current set. The current set is resampled to simulate newer, month-to-month customers. It prints a per-feature summary and writes an HTML report.

```powershell
python src/monitor_drift.py
```

Output: terminal summary plus `monitoring/data_drift_report.html`. The report is generated, not committed; re-run the script to rebuild it.

Logs and metrics can show a perfectly healthy service while the model scores customers unlike those it learned from. True churn labels arrive weeks later, so input drift is the earliest warning available.
