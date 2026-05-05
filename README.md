<div align="center">

<br/>

# MLOps Pipeline — News Classification

**A production-grade, end-to-end MLOps system for multi-class text classification.**  
Built to industry standards: reproducible experiments, automated CI/CD, containerized serving, and live drift monitoring — all in one cohesive pipeline.

<br/>

[![CI/CD](https://github.com/MOHD-OMER/mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/MOHD-OMER/mlops-pipeline/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/docker/v/omer022/mlops-news-classifier?label=DockerHub&logo=docker&color=2496ED)](https://hub.docker.com/r/omer022/mlops-news-classifier)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

<br/>

**[What is this?](#-overview) · [How it works](#-how-it-works) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Results](#-results) · [Extend it](#-extending-the-pipeline)**

<br/>

</div>

---

## 🧭 Overview

This project is a **complete MLOps reference implementation** — not just a model, but an entire system for taking a machine learning idea from raw data all the way to a monitored, production-serving API.

It classifies news articles into four categories (`World`, `Sports`, `Business`, `Sci/Tech`) using the [AG News](https://huggingface.co/datasets/ag_news) dataset. But the classification task is almost secondary — the real purpose of this project is to demonstrate **every layer of modern ML engineering** working together:

| What you get | Why it matters |
|---|---|
| **Reproducible experiments** via DVC + MLflow | Anyone can re-run the pipeline and get the same results |
| **Automated model selection** | Best-performing model is auto-promoted to Production |
| **71 automated tests** across data, model, and API | Catches regressions before they reach production |
| **4-job GitHub Actions CI/CD** | Every push to `main` triggers a full train → test → Docker push cycle |
| **FastAPI serving** with batch inference | Drop-in REST API with confidence scores, latency tracking, and model metadata |
| **Evidently AI drift monitoring** | Detects when the distribution of incoming text shifts away from training data |

> **Who is this for?** ML engineers who want to understand how production ML systems are structured, teams adopting MLOps practices for the first time, or anyone evaluating what a complete ML pipeline looks like beyond a Jupyter notebook.

---

## ⚙️ How It Works

The pipeline is composed of **seven sequential stages**, each with a clearly defined responsibility:

```
Raw Data ──► Validated Data ──► Cleaned Splits ──► Trained Models ──► Best Model
                                                                           │
                                                              ┌────────────┘
                                                              ▼
                                                     FastAPI REST API
                                                              │
                                                     Drift Monitoring
```

### Stage 1 — Data Ingestion & Validation

`src/ingest.py` downloads AG News from HuggingFace and runs automated data quality checks before a single line of training code runs:

- **Schema validation** — asserts `text`, `label`, and `label_name` columns are present
- **Null ratio check** — fails hard if nulls exceed 2% of the dataset
- **Class imbalance warning** — raises a flag if majority/minority class ratio > 5×
- **Duplicate detection** — reports exact-duplicate text entries

This guards against silent data corruption — a common source of unexplained model degradation.

### Stage 2 — Preprocessing

`src/preprocess.py` cleans text and produces a **stratified train/val/test split** (70% / 15% / 15%), ensuring class distribution is preserved across all three sets. Outputs are DVC-tracked `.csv` files, so splits are versioned alongside the code.

### Stage 3 — Experiment Tracking

`src/train.py` runs **three MLflow experiments** in a single pipeline execution, comparing different TF-IDF feature extraction configurations and classifiers:

| Run | Model | Vectorizer | N-gram Range | Reg. Strength (C) |
|---|---|---|:---:|:---:|
| `tfidf_lr_baseline` | Logistic Regression | TF-IDF | (1,1) unigrams | 1.0 |
| `tfidf_lr_bigrams` | Logistic Regression | TF-IDF | (1,2) bigrams | 5.0 |
| `tfidf_svm_bigrams` | Calibrated SVM | TF-IDF | (1,2) bigrams | 1.0 |

**Every run logs to MLflow:**
- All hyperparameters from `params.yaml`
- Metrics: accuracy, F1-macro, precision, recall, AUC-ROC
- Artifacts: serialized model `.pkl`, confusion matrix PNG, classification report
- Model signature and an input example (for MLflow Model Registry compatibility)

### Stage 4 — Automatic Model Promotion

The best model by validation accuracy is automatically registered in the **MLflow Model Registry** and promoted from `Staging → Production`, but only if it clears the accuracy threshold defined in `params.yaml` (default: 0.87). This is the gate that prevents a degraded model from ever reaching the API.

### Stage 5 — Testing

Three test suites covering the full system, run on every CI push:

```
tests/
├── test_data.py   (21 tests) — schema, types, split integrity, no leakage between sets
├── test_model.py  (21 tests) — load/predict/shape, probability sums to 1, performance smoke
└── test_api.py    (29 tests) — all endpoints, edge cases, malformed input, batch inference

Total: 71 tests | All passing ✅
```

### Stage 6 — CI/CD (GitHub Actions)

Every push to `main` triggers a 4-job pipeline:

```
push to main
    │
    ├─► Job 1: Lint & Test            (~1m 13s)
    │         Generates synthetic CI data → runs all 71 tests
    │
    ├─► Job 2: Train & Evaluate       (~1m 45s)
    │         Full ingest → preprocess → 3 MLflow runs → evaluate
    │         Accuracy gate: must exceed 0.87 to proceed
    │         Drift report generated → artifacts uploaded
    │
    ├─► Job 3: Build & Push Docker    (~5m 10s)
    │         Multi-stage build for linux/amd64
    │         Pushed to DockerHub: mohd-omer/mlops-news-classifier:latest
    │         Trivy security vulnerability scan
    │
    └─► Job 4: Pipeline Summary       (~4s)
              GitHub Step Summary table with all metrics
```

### Stage 7 — Model Serving

`src/serve.py` exposes a FastAPI application with four endpoints. The API loads the Production model from MLflow at startup (with a local `.pkl` fallback if MLflow is unavailable), making it resilient to tracking server outages.

### Stage 8 — Drift Monitoring

`monitoring/monitor.py` uses Evidently AI to compare the feature distribution of the **training reference dataset** against new incoming data. It computes Population Stability Index (PSI) across six text-derived features and raises an alert if any feature's PSI exceeds the threshold. An HTML report is written to `reports/drift_report.html`.

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MLOps Pipeline Architecture                            │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌───────────────┐    ┌─────────────────────────────────────┐
  │  DATA LAYER  │    │  EXPERIMENT   │    │          CI/CD  (GitHub Actions)    │
  │              │    │  TRACKING     │    │                                     │
  │ HuggingFace  │    │               │    │  push to main ──► run tests         │
  │ AG News      │───►│  MLflow       │    │                    │                │
  │ dataset      │    │  Tracking     │    │                    ▼                │
  │              │    │  Server       │    │                 train model          │
  │ data/        │    │  :5001        │    │                    │                │
  │ ├─ raw/      │    │               │    │                    ▼                │
  │ └─ processed/│    │  Experiments  │    │              evaluate (test acc)     │
  └──────┬───────┘    │  ├─ run 1    │    │                    │                │
         │            │  ├─ run 2    │    │              acc > threshold?        │
         ▼            │  └─ run 3    │    │                    │                │
  ┌──────────────┐    │              │    │              register to MLflow      │
  │  DVC         │    │  Model       │    │                    │                │
  │  VERSION     │    │  Registry    │    │              build Docker image      │
  │  CONTROL     │    │  ├─ Staging  │    │                    │                │
  │              │    │  └─Production│    │              push to DockerHub       │
  │  dvc repro   │    └──────┬───────┘    └─────────────────────────────────────┘
  └──────────────┘           │
                             │ best model
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │              MODEL SERVING  (FastAPI :8000)          │
  │                                                      │
  │   POST /predict      ──► label + confidence score   │
  │   GET  /model/info   ──► version + metrics          │
  │   GET  /health       ──► health status              │
  │   POST /predict/batch──► bulk inference             │
  └──────────────────────────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │          MONITORING  (Evidently AI)                  │
  │                                                      │
  │   monitor.py                                         │
  │   ├─ Compare training dist vs new data               │
  │   ├─ Generate HTML drift report                      │
  │   └─ Alert if PSI > threshold                        │
  └──────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
mlops-pipeline/
├── data/
│   ├── raw/                    # Raw downloads (DVC tracked)
│   │   ├── train_raw.csv
│   │   └── test_raw.csv
│   └── processed/              # Cleaned, stratified splits (DVC tracked)
│       ├── train.csv           # 70% of data
│       ├── val.csv             # 15% of data
│       └── test.csv            # 15% of data — held out until final eval
├── src/
│   ├── ingest.py               # HuggingFace download + schema/quality validation
│   ├── preprocess.py           # Text cleaning + stratified 70/15/15 split
│   ├── train.py                # 3 MLflow experiment runs + model registry promotion
│   ├── evaluate.py             # Final test set evaluation of Production model
│   └── serve.py                # FastAPI app — 4 REST endpoints
├── tests/
│   ├── test_data.py            # 21 tests: schema, nulls, class distribution, no leakage
│   ├── test_model.py           # 21 tests: load, predict, shape, proba sums, performance
│   └── test_api.py             # 29 tests: all endpoints, edge cases, batch predict
├── monitoring/
│   └── monitor.py              # Evidently drift report + PSI alerting
├── .github/
│   └── workflows/
│       └── ci.yml              # 4-job CI/CD: test → train → register → docker push
├── models/                     # Serialized .pkl files (DVC tracked)
├── reports/                    # Confusion matrices, metrics JSON, drift HTML reports
├── mlruns/                     # MLflow auto-generated experiment tracking data
├── docker-compose.yml          # Orchestrates MLflow server + FastAPI + training + monitor
├── Dockerfile                  # Multi-stage production image (~slim final layer)
├── dvc.yaml                    # DVC pipeline stage definitions
├── params.yaml                 # Single source of truth for all hyperparameters
├── pytest.ini                  # Pytest configuration
└── requirements.txt
```

---

## ⚡ Quick Start

### Prerequisites

```
Python 3.10+   git   docker   docker-compose
```

### 1. Clone & Install

```bash
git clone https://github.com/MOHD-OMER/mlops-pipeline.git
cd mlops-pipeline
pip install -r requirements.txt
```

### 2. Initialize DVC

```bash
dvc init
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "chore: track raw data with DVC"
```

### 3. Start MLflow Tracking Server

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns/artifacts
```

Open the MLflow UI at [http://localhost:5001](http://localhost:5001) to browse experiments, compare runs, and inspect the model registry.

### 4. Run the Full Pipeline

```bash
# Option A — Run each stage manually (useful for debugging individual steps)
python src/ingest.py        # Downloads AG News, runs data quality checks
python src/preprocess.py    # Cleans text, creates stratified splits
python src/train.py         # Runs 3 MLflow experiments, promotes best model
python src/evaluate.py      # Evaluates Production model on held-out test set

# Option B — DVC pipeline (fully reproducible, skips unchanged stages)
dvc repro
```

### 5. Run the Test Suite

```bash
pytest tests/ -v --tb=short

# Expected output: 71 passed in ~12s
```

### 6. Launch the API Server

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
```

Test a prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple stock rose 5% after strong quarterly earnings.", "top_k": 3}'
```

### 7. Generate Drift Report

```bash
python monitoring/monitor.py
# Writes: reports/drift_report.html
```

### 8. Full Docker Stack

```bash
# MLflow tracking server + FastAPI serving
docker-compose up mlflow api

# Training job (runs once, then exits)
docker-compose --profile train up train

# Drift monitoring job
docker-compose --profile monitor up monitor
```

---

## 📡 API Reference

The FastAPI server runs on port `8000` and exposes four endpoints. Interactive docs are auto-generated at [http://localhost:8000/docs](http://localhost:8000/docs).

---

### `POST /predict` — Single Prediction

Classifies a single text input and returns the top predicted label with confidence score.

**Request body:**

```json
{
  "text": "Scientists discover new exoplanet in the habitable zone",
  "top_k": 3
}
```

| Field | Type | Required | Description |
|---|---|:---:|---|
| `text` | string | ✅ | The news article text to classify |
| `top_k` | integer | ❌ | Number of top predictions to return (default: 1) |

**Response:**

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

### `POST /predict/batch` — Batch Prediction

Classifies multiple texts in a single request — more efficient than repeated single calls for bulk inference.

**Request body:**

```json
{
  "texts": [
    "Fed raises interest rates by 25 basis points",
    "Manchester City wins Premier League title"
  ]
}
```

**Response:**

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

### `GET /model/info` — Model Metadata

Returns information about the currently loaded model, including its version, training metrics, and which MLflow run produced it.

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

### `GET /health` — Health Check

Lightweight liveness probe. Returns `200 OK` when the model is loaded and the API is ready to serve.

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}
```

---

## 📊 Results

> Trained on 10,000 samples (CI/CD environment). Running `dvc repro` on the full 120k dataset achieves ~91–92% test accuracy.

### Experiment Comparison

| Run | Model | N-gram | Val Accuracy | Val F1-macro | Notes |
|---|---|:---:|:---:|:---:|---|
| `tfidf_lr_baseline` | Logistic Regression | (1,1) | 0.8861 | 0.8853 | Unigrams only |
| `tfidf_lr_bigrams` | Logistic Regression | (1,2) | 0.8891 | 0.8885 | Bigrams help slightly |
| `tfidf_svm_bigrams` | **Calibrated SVM** | **(1,2)** | **0.8903** | **0.8898** | ✅ Auto-promoted to Production |

### Production Model — Final Evaluation

| Split | Accuracy | F1-macro | AUC-ROC |
|---|:---:|:---:|:---:|
| Validation | 0.8903 | 0.8898 | — |
| **Test (held-out)** | **0.8788** | **0.8785** | **0.9729** |

The ~1% gap between validation and test accuracy is expected and healthy — it confirms no overfitting to the validation set during model selection.

### Monitored Text Features (Evidently AI)

| Feature | Description | Drift Metric |
|---|---|---|
| `text_length` | Character count per article | PSI |
| `word_count` | Token count per article | PSI |
| `avg_word_length` | Average characters per word | PSI |
| `num_sentences` | Sentence boundary count | PSI |
| `uppercase_ratio` | Fraction of uppercase characters | PSI |
| `digit_ratio` | Fraction of digit characters | PSI |

---

## 🔧 Configuration

All hyperparameters and thresholds live in `params.yaml` — a single source of truth. Change any value and run `dvc repro` to re-execute only the affected pipeline stages.

```yaml
data:
  dataset: "ag_news"
  max_samples: 10000      # set to null for full 120k dataset

training:
  C: 1.0                  # regularization strength
  max_iter: 1000

mlflow:
  tracking_uri: "http://localhost:5001"
  accuracy_threshold: 0.87   # model must exceed this to be registered
```

---

## 🐳 Docker

The production image is built in two stages: a `builder` layer installs dependencies, and a slim `runtime` layer contains only what's needed to serve — keeping the final image lean.

```bash
# Pull the latest image from DockerHub
docker pull mohd-omer/mlops-news-classifier:latest

# Run the API (falls back to local .pkl if MLflow is unavailable)
docker run -p 8000:8000 mohd-omer/mlops-news-classifier:latest

# Full stack: MLflow tracking server + API server
docker-compose up
```

A fresh Docker image is automatically built and pushed to DockerHub on every successful `main` branch push.

---

## 🔒 Required GitHub Secrets

To enable the CI/CD Docker push, add these secrets to your repository (`Settings → Secrets → Actions`):

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Your DockerHub username |
| `DOCKERHUB_TOKEN` | DockerHub access token (`Account Settings → Security → New Access Token`) |

---

## 📦 Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Dataset | AG News via HuggingFace `datasets` | 4-class news classification benchmark |
| ML Framework | scikit-learn — TF-IDF + LR / Calibrated SVM | Feature extraction + classification |
| Experiment Tracking | MLflow 3.x | Run logging, artifact storage, model registry |
| Data Versioning | DVC 3 | Reproducible pipeline stages, data version control |
| Drift Monitoring | Evidently AI (PSI fallback) | Distribution shift detection |
| API Serving | FastAPI + Uvicorn | REST inference API with auto-generated docs |
| Testing | Pytest + httpx | 71 tests across data, model, and API layers |
| CI/CD | GitHub Actions | 4-job automated pipeline |
| Containerization | Docker (multi-stage) + docker-compose | Reproducible builds, local orchestration |
| Registry | DockerHub | Public image distribution |

---

## 🧩 Extending the Pipeline

The pipeline is designed to be extended. Here are three common additions:

### Add a DistilBERT Fine-Tuned Model

```python
# params.yaml
model:
  type: "distilbert"

# src/train.py — add a training branch:
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
# HuggingFace Trainer integrates with MLflow autologging out of the box
```

### Add Remote DVC Storage (S3 / GCS / Azure)

```bash
dvc remote add myremote s3://your-bucket/mlops-data
dvc remote default myremote
dvc push    # push data artifacts to the remote
```

This enables full reproducibility across machines and teams without committing data to Git.

### Add Prometheus Metrics to the API

```python
# src/serve.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

# Scrape at: GET /metrics
```

Pair with Grafana for a real-time dashboard of prediction latency, request rate, and error rate.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss the proposed change. Please ensure all 71 tests pass before submitting a PR.

```bash
# Fork → Clone → Create branch
git checkout -b feature/your-feature

# Make changes, then verify
pytest tests/ -v

# Commit with conventional commits
git commit -m "feat: add your feature"
git push origin feature/your-feature
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built by [MOHD-OMER](https://github.com/MOHD-OMER) · [GitHub](https://github.com/MOHD-OMER/mlops-pipeline) · [DockerHub](https://hub.docker.com/r/omer022/mlops-news-classifier)

</div>
