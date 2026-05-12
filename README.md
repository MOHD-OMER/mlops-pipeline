<div align="center">

<br/>

# MLOps Pipeline — News Classification

**A production-grade, end-to-end MLOps system for multi-class text classification.**

Reproducible experiments · Automated CI/CD · Containerized serving · Live drift monitoring

<br/>

[![CI/CD](https://github.com/MOHD-OMER/mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/MOHD-OMER/mlops-pipeline/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/docker/v/omer022/mlops-news-classifier?label=DockerHub&logo=docker&color=2496ED)](https://hub.docker.com/r/omer022/mlops-news-classifier)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

<br/>

[Overview](#-overview) · [Architecture](#-architecture) · [Pipeline Stages](#-pipeline-stages) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Results](#-results) · [Configuration](#-configuration) · [Extending](#-extending-the-pipeline)

<br/>

</div>

---

## Overview

This project is a **complete MLOps reference implementation** — a system for taking a machine learning idea from raw data all the way to a monitored, production-serving API.

It classifies news articles into four categories (`World`, `Sports`, `Business`, `Sci/Tech`) using the [AG News](https://huggingface.co/datasets/ag_news) dataset. The classification task is intentionally simple; the focus is demonstrating every layer of modern ML engineering working together in one cohesive system.

| Capability | Details |
|---|---|
| **Reproducible experiments** | DVC + MLflow: anyone can re-run the pipeline and get identical results |
| **Automated model selection** | Best-performing model is auto-promoted to Production via an accuracy gate |
| **71 automated tests** | Covers data integrity, model behaviour, and API correctness |
| **4-job GitHub Actions CI/CD** | Every push to `main` triggers a full train → test → Docker push cycle |
| **FastAPI serving** | REST API with confidence scores, batch inference, and model metadata |
| **Evidently AI drift monitoring** | Detects distribution shift in incoming text against training reference data |

> **Who is this for?** ML engineers learning how production systems are structured, teams adopting MLOps practices, or anyone evaluating what a complete ML pipeline looks like beyond a notebook.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Architecture                          │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────────────────┐
  │  DATA LAYER  │    │  EXPERIMENT       │    │   CI/CD  (GitHub Actions)     │
  │              │    │  TRACKING         │    │                               │
  │ HuggingFace  │    │                   │    │  push to main                 │
  │ AG News ─────┼───►│  MLflow :5001     │    │       │                       │
  │ dataset      │    │                   │    │       ▼                       │
  │              │    │  Experiments      │    │  lint & test (71 tests)       │
  │ data/        │    │  ├─ run 1         │    │       │                       │
  │ ├─ raw/      │    │  ├─ run 2         │    │       ▼                       │
  │ └─ processed/│    │  └─ run 3         │    │  train & evaluate             │
  └──────┬───────┘    │                   │    │  (acc > 0.87 gate)            │
         │            │  Model Registry   │    │       │                       │
         ▼            │  ├─ Staging       │    │       ▼                       │
  ┌──────────────┐    │  └─ Production ───┼───►│  build & push Docker image    │
  │  DVC         │    └──────────────────┘    └───────────────────────────────┘
  │  VERSIONING  │
  │  dvc repro   │
  └──────────────┘

                    ┌─────────────────────────────────────────┐
                    │       MODEL SERVING  (FastAPI :8000)     │
                    │                                          │
                    │  POST /predict       label + confidence  │
                    │  POST /predict/batch bulk inference      │
                    │  GET  /model/info    version + metrics   │
                    │  GET  /health        liveness probe      │
                    └──────────────────┬───────────────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │       MONITORING  (Evidently AI)         │
                    │                                          │
                    │  Compare training dist vs incoming data  │
                    │  PSI per feature → HTML drift report     │
                    │  Alert if PSI exceeds threshold          │
                    └─────────────────────────────────────────┘
```

---

## Pipeline Stages

The pipeline consists of eight sequential stages, each with a clearly defined responsibility.

### 1 — Data Ingestion & Validation

`src/ingest.py` downloads AG News from HuggingFace and runs automated data quality checks before any training code executes.

| Check | Failure condition |
|---|---|
| Schema validation | `text`, `label`, or `label_name` columns missing |
| Null ratio | Nulls exceed 2% of dataset |
| Class imbalance | Majority/minority class ratio > 5× |
| Duplicate detection | Exact-duplicate text entries reported |

### 2 — Preprocessing

`src/preprocess.py` cleans text and produces a **stratified 70/15/15 train/val/test split**, preserving class distribution across all three sets. Outputs are DVC-tracked `.csv` files so splits are versioned alongside code.

### 3 — Experiment Tracking

`src/train.py` runs **three MLflow experiments** in a single execution, comparing TF-IDF configurations and classifiers.

| Run | Model | N-gram Range | C |
|---|---|:---:|:---:|
| `tfidf_lr_baseline` | Logistic Regression | (1,1) | 1.0 |
| `tfidf_lr_bigrams` | Logistic Regression | (1,2) | 5.0 |
| `tfidf_svm_bigrams` | Calibrated SVM | (1,2) | 1.0 |

Each run logs hyperparameters, metrics (accuracy, F1-macro, precision, recall, AUC-ROC), serialised model `.pkl`, confusion matrix, and classification report to MLflow.

### 4 — Automatic Model Promotion

The best model by validation accuracy is registered in the **MLflow Model Registry** and promoted `Staging → Production`, provided it clears the accuracy threshold in `params.yaml` (default: `0.87`). This gate prevents a degraded model from ever reaching the serving layer.

### 5 — Testing

Three test suites run on every CI push.

```
tests/
├── test_data.py   — 21 tests: schema, types, split integrity, no data leakage
├── test_model.py  — 21 tests: load/predict/shape, probability sums to 1, smoke perf
└── test_api.py    — 29 tests: all endpoints, edge cases, malformed input, batch inference

Total: 71 tests | All passing ✅
```

### 6 — CI/CD (GitHub Actions)

Every push to `main` triggers a four-job pipeline.

```
push to main
    │
    ├─► Job 1: Lint & Test            (~1m 13s)
    │         Generates synthetic CI data → runs all 71 tests
    │
    ├─► Job 2: Train & Evaluate       (~1m 45s)
    │         Ingest → preprocess → 3 MLflow runs → evaluate
    │         Accuracy gate: must exceed 0.87 to proceed
    │         Drift report generated → artifacts uploaded
    │
    ├─► Job 3: Build & Push Docker    (~5m 10s)
    │         Multi-stage build for linux/amd64
    │         Pushed to DockerHub as :latest
    │         Trivy security scan
    │
    └─► Job 4: Pipeline Summary       (~4s)
              GitHub Step Summary table with all run metrics
```

### 7 — Model Serving

`src/serve.py` exposes a FastAPI application with four endpoints. The API loads the Production model from MLflow at startup, with a local `.pkl` fallback if the tracking server is unavailable — ensuring resilience to infrastructure outages.

### 8 — Drift Monitoring

`monitoring/monitor.py` uses Evidently AI to compare the feature distribution of the training reference dataset against incoming data. It computes Population Stability Index (PSI) across six text-derived features and raises an alert if any PSI exceeds the configured threshold. An HTML report is written to `reports/drift_report.html`.

---

## Project Structure

```
mlops-pipeline/
├── data/
│   ├── raw/                    # Raw downloads (DVC tracked)
│   │   ├── train_raw.csv
│   │   └── test_raw.csv
│   └── processed/              # Cleaned, stratified splits (DVC tracked)
│       ├── train.csv           # 70%
│       ├── val.csv             # 15%
│       └── test.csv            # 15% — held out until final evaluation
├── src/
│   ├── ingest.py               # HuggingFace download + schema/quality validation
│   ├── preprocess.py           # Text cleaning + stratified split
│   ├── train.py                # 3 MLflow experiment runs + model registry promotion
│   ├── evaluate.py             # Final test set evaluation of Production model
│   └── serve.py                # FastAPI — 4 REST endpoints
├── tests/
│   ├── test_data.py            # 21 tests
│   ├── test_model.py           # 21 tests
│   └── test_api.py             # 29 tests
├── monitoring/
│   └── monitor.py              # Evidently drift report + PSI alerting
├── .github/
│   └── workflows/
│       └── ci.yml              # 4-job CI/CD pipeline
├── models/                     # Serialised .pkl files (DVC tracked)
├── reports/                    # Confusion matrices, metrics JSON, drift HTML
├── mlruns/                     # MLflow experiment tracking data
├── docker-compose.yml          # Orchestrates MLflow + FastAPI + training + monitor
├── Dockerfile                  # Multi-stage production image
├── dvc.yaml                    # DVC pipeline stage definitions
├── params.yaml                 # Single source of truth for all hyperparameters
├── pytest.ini
└── requirements.txt
```

---

## Quick Start

### Prerequisites

```
Python 3.10+   git   docker   docker-compose
```

### 1. Clone & install

```bash
git clone https://github.com/MOHD-OMER/mlops-pipeline.git
cd mlops-pipeline
pip install -r requirements.txt
```

### 2. Initialise DVC

```bash
dvc init
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "chore: track raw data with DVC"
```

### 3. Start MLflow tracking server

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns/artifacts
```

MLflow UI available at [http://localhost:5001](http://localhost:5001).

### 4. Run the full pipeline

```bash
# Option A — run stages manually (useful for debugging individual steps)
python src/ingest.py
python src/preprocess.py
python src/train.py
python src/evaluate.py

# Option B — DVC (fully reproducible; skips unchanged stages)
dvc repro
```

### 5. Run the test suite

```bash
pytest tests/ -v --tb=short
# Expected: 71 passed in ~12s
```

### 6. Launch the API server

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple stock rose 5% after strong quarterly earnings.", "top_k": 3}'
```

### 7. Generate drift report

```bash
python monitoring/monitor.py
# Output: reports/drift_report.html
```

### 8. Full Docker stack

```bash
# MLflow + FastAPI
docker-compose up mlflow api

# Training job (runs once, then exits)
docker-compose --profile train up train

# Drift monitoring job
docker-compose --profile monitor up monitor
```

---

## API Reference

Interactive docs auto-generated at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### `POST /predict` — Single prediction

**Request**

```json
{
  "text": "Scientists discover new exoplanet in the habitable zone",
  "top_k": 3
}
```

| Field | Type | Required | Description |
|---|---|:---:|---|
| `text` | string | ✅ | Article text to classify |
| `top_k` | integer | ❌ | Number of top predictions to return (default: 1) |

**Response**

```json
{
  "label": "Sci/Tech",
  "label_id": 3,
  "confidence": 0.9142,
  "top_predictions": [
    {"label": "Sci/Tech",  "probability": 0.9142},
    {"label": "World",     "probability": 0.0521},
    {"label": "Business",  "probability": 0.0337}
  ],
  "model_version": "local:tfidf_svm_bigrams",
  "latency_ms": 8.4
}
```

---

### `POST /predict/batch` — Batch prediction

More efficient than repeated single calls for bulk inference.

**Request**

```json
{
  "texts": [
    "Fed raises interest rates by 25 basis points",
    "Manchester City wins Premier League title"
  ]
}
```

**Response**

```json
{
  "predictions": [
    {"label": "Business", "label_id": 2, "confidence": 0.8831},
    {"label": "Sports",   "label_id": 1, "confidence": 0.9654}
  ],
  "count": 2,
  "latency_ms": 11.2
}
```

---

### `GET /model/info` — Model metadata

Returns the loaded model's version, training metrics, and originating MLflow run.

```bash
curl http://localhost:8000/model/info
```

```json
{
  "model_name": "tfidf_svm_bigrams",
  "version": "3",
  "stage": "Production",
  "val_accuracy": 0.8903,
  "val_f1": 0.8898,
  "run_id": "a3f2c91d..."
}
```

---

### `GET /health` — Health check

Liveness probe. Returns `200 OK` when the model is loaded and the API is ready.

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}
```

---

## Results

> Trained on 10,000 samples in CI. Running `dvc repro` on the full 120k dataset achieves ~91–92% test accuracy.

### Experiment comparison

| Run | Model | N-gram | Val Accuracy | Val F1 |
|---|---|:---:|:---:|:---:|
| `tfidf_lr_baseline` | Logistic Regression | (1,1) | 0.8861 | 0.8853 |
| `tfidf_lr_bigrams` | Logistic Regression | (1,2) | 0.8891 | 0.8885 |
| `tfidf_svm_bigrams` | **Calibrated SVM** | **(1,2)** | **0.8903** | **0.8898** |

### Production model — final evaluation

| Split | Accuracy | F1-macro | AUC-ROC |
|---|:---:|:---:|:---:|
| Validation | 0.8903 | 0.8898 | — |
| **Test (held-out)** | **0.8788** | **0.8785** | **0.9729** |

The ~1% gap between validation and test accuracy is expected — it confirms no overfitting to the validation set during model selection.

### Monitored features (Evidently AI)

| Feature | Description |
|---|---|
| `text_length` | Character count per article |
| `word_count` | Token count per article |
| `avg_word_length` | Average characters per word |
| `num_sentences` | Sentence boundary count |
| `uppercase_ratio` | Fraction of uppercase characters |
| `digit_ratio` | Fraction of digit characters |

All features are evaluated using Population Stability Index (PSI).

---

## Configuration

All hyperparameters and thresholds live in `params.yaml`. Change any value and run `dvc repro` to re-execute only the affected stages.

```yaml
data:
  dataset: "ag_news"
  max_samples: 10000      # set to null for full 120k dataset

training:
  C: 1.0
  max_iter: 1000

mlflow:
  tracking_uri: "http://localhost:5001"
  accuracy_threshold: 0.87   # model must exceed this to be registered
```

---

## Docker

The production image uses a multi-stage build: a `builder` layer installs dependencies, and a slim `runtime` layer contains only what's needed to serve.

```bash
# Pull latest image
docker pull mohd-omer/mlops-news-classifier:latest

# Run the API (falls back to local .pkl if MLflow is unavailable)
docker run -p 8000:8000 mohd-omer/mlops-news-classifier:latest

# Full stack
docker-compose up
```

A fresh image is built and pushed to DockerHub on every successful push to `main`.

### Required GitHub secrets

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | DockerHub username |
| `DOCKERHUB_TOKEN` | DockerHub access token (`Account Settings → Security → New Access Token`) |

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Dataset | AG News via HuggingFace `datasets` | 4-class news classification benchmark |
| ML | scikit-learn — TF-IDF + LR / Calibrated SVM | Feature extraction + classification |
| Experiment tracking | MLflow 3.x | Run logging, artifact storage, model registry |
| Data versioning | DVC 3 | Reproducible pipeline stages, data versioning |
| Drift monitoring | Evidently AI (PSI) | Distribution shift detection |
| Serving | FastAPI + Uvicorn | REST inference API with auto-generated docs |
| Testing | Pytest + httpx | 71 tests across data, model, and API layers |
| CI/CD | GitHub Actions | 4-job automated pipeline |
| Containerisation | Docker (multi-stage) + docker-compose | Reproducible builds, local orchestration |
| Registry | DockerHub | Public image distribution |

---

## Extending the Pipeline

### Add a DistilBERT fine-tuned model

```python
# params.yaml
model:
  type: "distilbert"

# src/train.py
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
# HuggingFace Trainer integrates with MLflow autologging out of the box
```

### Add remote DVC storage (S3 / GCS / Azure)

```bash
dvc remote add myremote s3://your-bucket/mlops-data
dvc remote default myremote
dvc push
```

### Add Prometheus metrics to the API

```python
# src/serve.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
# Scrape at: GET /metrics
```

Pair with Grafana for a real-time dashboard of prediction latency, request rate, and error rate.

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss the proposed direction. All 71 tests must pass before a PR is merged.

```bash
git checkout -b feature/your-feature
# make changes
pytest tests/ -v
git commit -m "feat: add your feature"
git push origin feature/your-feature
# open a pull request
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [MOHD-OMER](https://github.com/MOHD-OMER) · [GitHub](https://github.com/MOHD-OMER/mlops-pipeline) · [DockerHub](https://hub.docker.com/r/omer022/mlops-news-classifier)

</div>
