# 🚀 MLOps Pipeline — News Classification

An **end-to-end production MLOps pipeline** for text classification on the AG News dataset.
Covers every stage from raw data ingestion to monitored model serving, with full experiment
tracking, CI/CD automation, and data drift detection.

[![CI/CD](https://github.com/MOHD-OMER/mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/MOHD-OMER/mlops-pipeline/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/docker/v/mohd-omer/mlops-news-classifier?label=DockerHub)](https://hub.docker.com/r/omer022/mlops-news-classifier)

---

## 🏗️ Pipeline Architecture

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
  │ data/        │    │  :5000        │    │                    │                │
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
│   └── processed/              # Cleaned, split data (DVC tracked)
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── ingest.py               # Download + validate AG News
│   ├── preprocess.py           # Clean text, stratified split
│   ├── train.py                # 3 MLflow experiment runs, model registry
│   ├── evaluate.py             # Production model evaluation
│   └── serve.py                # FastAPI serving (4 endpoints)
├── tests/
│   ├── test_data.py            # Schema, nulls, class distribution (21 tests)
│   ├── test_model.py           # Load, predict, shape, performance (21 tests)
│   └── test_api.py             # All FastAPI endpoints (29 tests)
├── monitoring/
│   └── monitor.py              # Evidently drift report + alerting
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD: test → train → register → docker
├── models/                     # Serialized .pkl files (DVC tracked)
├── reports/                    # Confusion matrices, metrics, drift HTML
├── mlruns/                     # MLflow auto-generated tracking data
├── docker-compose.yml          # MLflow + FastAPI + training services
├── Dockerfile                  # Multi-stage production image
├── dvc.yaml                    # DVC pipeline stages
├── params.yaml                 # All hyperparameters (single source of truth)
└── requirements.txt
```

---

## ⚡ Quick Start — Run Everything Locally

### Prerequisites
```bash
python 3.10+   git   docker   docker-compose
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
dvc add data/raw          # version the raw data
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

# Open: http://localhost:5001
```

### 4. Run the Full Pipeline
```bash
# Option A: Run stages manually
python src/ingest.py        # Download AG News, validate schema
python src/preprocess.py    # Clean text, stratified split
python src/train.py         # 3 MLflow runs, register best model
python src/evaluate.py      # Test set evaluation

# Option B: DVC pipeline (reproducible)
dvc repro
```

### 5. Run Tests
```bash
pytest tests/ -v --tb=short
# Expected: 71 tests passing
```

### 6. Start the API Server
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload

# Test the endpoints:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple stock rose 5% after strong quarterly earnings."}'

curl http://localhost:8000/model/info
curl http://localhost:8000/health
```

### 7. Generate Drift Report
```bash
python monitoring/monitor.py
# Opens: reports/drift_report.html in browser
```

### 8. Full Docker Stack
```bash
# Start MLflow + API
docker-compose up mlflow api

# Run training job
docker-compose --profile train up train

# Run drift monitoring
docker-compose --profile monitor up monitor
```

---

## 🧪 Pipeline Stages Explained

### Stage 1 — Data Layer (`src/ingest.py` + `src/preprocess.py`)

| Check | Implementation |
|-------|---------------|
| Schema validation | Ensures `text`, `label`, `label_name` columns present |
| Null check | Fails if null ratio > 2% |
| Class imbalance | Warns if majority/minority > 5× |
| Duplicate detection | Reports exact text duplicates |
| Stratified split | 70% train / 15% val / 15% test |

### Stage 2 — Experiment Tracking (`src/train.py`)

Three runs logged to MLflow:

| Run Name | Model | Max Features | N-gram | C |
|----------|-------|-------------|--------|---|
| `tfidf_lr_baseline` | LR | 30,000 | (1,1) | 1.0 |
| `tfidf_lr_bigrams` | LR | 50,000 | (1,2) | 5.0 |
| `tfidf_svm_bigrams` | Calibrated SVM | 50,000 | (1,2) | 1.0 |

**Logged per run:**
- Hyperparameters (C, ngram_range, max_features, …)
- Metrics: accuracy, F1-macro, precision, recall, AUC-ROC
- Artifacts: model pkl, confusion matrix PNG, classification report TXT
- Signature + input example for Model Registry

### Stage 3 — Model Registry

Best model is automatically promoted:
```
Staging → Production
```
Registered if `val_accuracy > params.yaml::mlflow.accuracy_threshold` (default: 0.87)

### Stage 4 — Testing (`tests/`)

```
tests/
├── test_data.py   (21 tests) — schema, types, split integrity, no leakage
├── test_model.py  (21 tests) — load, shape, proba sums, performance smoke
└── test_api.py    (29 tests) — all endpoints, edge cases, batch predict

Total: 71 tests | All passing ✅
```

### Stage 5 — CI/CD (`.github/workflows/ci.yml`)

```
push to main
    │
    ├─► Job 1: 🧪 Lint & Test  (~1m 13s)
    │   ├─ Generate synthetic CI data (no HuggingFace download)
    │   ├─ pytest tests/test_data.py  (21 tests)
    │   ├─ Train quick CI model (LogisticRegression)
    │   ├─ pytest tests/test_model.py (21 tests)
    │   └─ pytest tests/test_api.py   (29 tests)
    │
    ├─► Job 2: 🏋️ Train & Evaluate  (~1m 45s)
    │   ├─ Start MLflow server (retry loop, up in ~12s)
    │   ├─ python src/ingest.py       (10k AG News samples)
    │   ├─ python src/preprocess.py
    │   ├─ python src/train.py        (3 MLflow runs)
    │   ├─ python src/evaluate.py     (acc: 0.8788, AUC-ROC: 0.9729)
    │   ├─ Check accuracy > 0.87 ✅
    │   ├─ python monitoring/monitor.py
    │   └─ Upload ml-artifacts (~6MB)
    │
    ├─► Job 3: 🐳 Build & Push Docker  (~5m 10s)
    │   ├─ Download ml-artifacts from Job 2
    │   ├─ docker buildx build --platform linux/amd64
    │   ├─ docker push → mohd-omer/mlops-news-classifier:latest
    │   └─ trivy security scan (report only)
    │
    └─► Job 4: 📢 Pipeline Summary  (~4s)
        └─ Generate GitHub Step Summary table
```

### Stage 6 — Model Serving

```bash
# Predict
POST /predict
{
  "text": "Scientists discover new exoplanet in habitable zone",
  "top_k": 3
}
# Response:
{
  "label": "Sci/Tech",
  "label_id": 3,
  "confidence": 0.9142,
  "top_predictions": [
    {"label": "Sci/Tech", "probability": 0.9142},
    {"label": "World",    "probability": 0.0521},
    {"label": "Business", "probability": 0.0337}
  ],
  "model_version": "local:tfidf_svm_bigrams",
  "latency_ms": 8.4
}
```

### Stage 7 — Monitoring

Evidently compares numeric text features between reference and current data:

| Feature | Description |
|---------|-------------|
| `text_length` | Character count |
| `word_count` | Token count |
| `avg_word_length` | Average chars per word |
| `num_sentences` | Sentence count |
| `uppercase_ratio` | Fraction of uppercase chars |
| `digit_ratio` | Fraction of digit chars |

---

## 📊 Results — Model Comparison

> *Results on AG News dataset (10,000 training samples, 4 classes)*

| Run | Model | N-gram | Val Accuracy | Val F1 | AUC-ROC |
|-----|-------|--------|:---:|:---:|:---:|
| `tfidf_lr_baseline` | LR | (1,1) | 0.8861 | 0.8853 | — |
| `tfidf_lr_bigrams` | LR | (1,2) | 0.8891 | 0.8885 | — |
| `tfidf_svm_bigrams` | Cal-SVM | (1,2) | **0.8903** | **0.8898** | — |

**Best model:** `tfidf_svm_bigrams` (promoted to Production in MLflow)

| Split | Accuracy | F1-macro | AUC-ROC |
|-------|:---:|:---:|:---:|
| Validation | 0.8903 | 0.8898 | — |
| **Test** | **0.8788** | **0.8785** | **0.9729** |

> CI uses 10k samples (vs full 120k locally). Running `dvc repro` on the full dataset achieves ~91–92% test accuracy.

---

## 🔧 Configuration (`params.yaml`)

All hyperparameters are centralized in `params.yaml`. Change them and re-run `dvc repro`
for fully reproducible experiments:

```yaml
data:
  dataset: "ag_news"
  max_samples: 10000      # null = full 120k dataset

training:
  C: 1.0
  max_iter: 1000

mlflow:
  tracking_uri: "http://localhost:5001"
  accuracy_threshold: 0.87   # minimum to register model (0.88+ on full dataset)
```

---

## 🐳 Docker

```bash
# Pull from DockerHub
docker pull mohd-omer/mlops-news-classifier:latest

# Run API only (falls back to local pkl if MLflow unavailable)
docker run -p 8000:8000 mohd-omer/mlops-news-classifier:latest

# Full stack with MLflow
docker-compose up
```

The Docker image is automatically built and pushed to DockerHub on every successful
`main` branch push via GitHub Actions.

---

## 🔒 GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Your DockerHub username |
| `DOCKERHUB_TOKEN` | DockerHub access token (Settings → Security → New Token) |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Dataset | AG News (HuggingFace `datasets`) |
| ML Framework | scikit-learn (TF-IDF + LR / Calibrated SVM) |
| Experiment Tracking | MLflow 3.x |
| Data Versioning | DVC 3 |
| Drift Monitoring | Evidently AI (PSI fallback) |
| API Serving | FastAPI + Uvicorn |
| Testing | Pytest + httpx (71 tests) |
| CI/CD | GitHub Actions (4-job pipeline) |
| Containerization | Docker (multi-stage) + docker-compose |
| Registry | DockerHub (`mohd-omer/mlops-news-classifier`) |

---

## 🧩 Extending This Pipeline

**Add DistilBERT fine-tuning:**
```python
# In params.yaml:
model:
  type: "distilbert"

# In src/train.py — add a DistilBERT training path
# Uses HuggingFace Trainer with same MLflow autologging
```

**Add remote DVC storage (S3/GCS):**
```bash
dvc remote add myremote s3://your-bucket/mlops-data
dvc push
```

**Add Prometheus metrics to FastAPI:**
```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

---

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Make sure to update tests as appropriate.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
