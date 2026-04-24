# 🚀 MLOps Pipeline — News Classification

An **end-to-end production MLOps pipeline** for text classification on the AG News dataset.
Covers every stage from raw data ingestion to monitored model serving, with full experiment
tracking, CI/CD automation, and data drift detection.

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
│   ├── ingest.py               # Download + validate AG News / TruthLens
│   ├── preprocess.py           # Clean text, stratified split
│   ├── train.py                # 3 MLflow experiment runs, model registry
│   ├── evaluate.py             # Production model evaluation
│   └── serve.py                # FastAPI serving (3 endpoints)
├── tests/
│   ├── test_data.py            # Schema, nulls, class distribution
│   ├── test_model.py           # Load, predict, shape, performance
│   └── test_api.py             # All FastAPI endpoints
├── monitoring/
│   └── monitor.py              # Evidently drift report + alerting
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml     # CI/CD: test → train → register → docker
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
  --port 5000 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns/artifacts

# Open: http://localhost:5000
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
# Expected: 40+ tests passing
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
python monitoring/monitor.py \
  --reference data/processed/train.csv \
  --current   data/processed/test.csv \
  --report    reports/drift_report.html

# Open: reports/drift_report.html in browser
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
Registered if `val_accuracy > params.yaml::mlflow.accuracy_threshold` (default: 0.88)

### Stage 4 — Testing (`tests/`)

```
tests/
├── test_data.py   (17 tests) — schema, types, split integrity, no leakage
├── test_model.py  (15 tests) — load, shape, proba sums, performance smoke
└── test_api.py    (20 tests) — all endpoints, edge cases, batch predict
```

### Stage 5 — CI/CD (`.github/workflows/ml_pipeline.yml`)

```
push to main
    │
    ├─► Job 1: test
    │   ├─ Generate synthetic CI data (no HuggingFace download)
    │   ├─ pytest tests/test_data.py
    │   ├─ Train tiny CI model
    │   ├─ pytest tests/test_model.py
    │   └─ pytest tests/test_api.py
    │
    ├─► Job 2: train-and-evaluate  (main branch only)
    │   ├─ Start MLflow server
    │   ├─ python src/ingest.py
    │   ├─ python src/preprocess.py
    │   ├─ python src/train.py (3 MLflow runs)
    │   ├─ python src/evaluate.py
    │   ├─ Check accuracy > threshold
    │   └─ python monitoring/monitor.py
    │
    └─► Job 3: docker-build  (if train-and-evaluate passes)
        ├─ docker buildx build --platform linux/amd64,arm64
        ├─ docker push → DockerHub
        └─ trivy security scan
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
  "model_version": "3",
  "latency_ms": 8.4
}
```

### Stage 7 — Monitoring

Evidently compares these numeric text features between reference and current data:

| Feature | Description |
|---------|-------------|
| `text_length` | Character count |
| `word_count` | Token count |
| `avg_word_length` | Average chars per word |
| `num_sentences` | Sentence count |
| `uppercase_ratio` | Fraction of uppercase chars |
| `digit_ratio` | Fraction of digit chars |

---

## 📊 Results Table — Model Comparison

> *Results on AG News dataset (10,000 samples, 4 classes)*

| Run | Model | N-gram | C | Val Accuracy | Val F1 | AUC-ROC |
|-----|-------|--------|---|:---:|:---:|:---:|
| `tfidf_lr_baseline` | LR | (1,1) | 1.0 | 0.907 | 0.906 | 0.985 |
| `tfidf_lr_bigrams` | LR | (1,2) | 5.0 | **0.921** | **0.921** | **0.988** |
| `tfidf_svm_bigrams` | Cal-SVM | (1,2) | 1.0 | 0.918 | 0.917 | 0.987 |

**Best model:** `tfidf_lr_bigrams` (promoted to Production)

| Split | Accuracy | F1-macro | AUC-ROC |
|-------|:---:|:---:|:---:|
| Validation | 0.921 | 0.921 | 0.988 |
| Test | 0.919 | 0.918 | 0.987 |

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
  accuracy_threshold: 0.88  # minimum to register model
```

---

## 🐳 Docker

```bash
# Build
docker build -t mlops-api:latest .

# Run API only
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  mlops-api:latest

# Full stack
docker-compose up
```

---

## 🔒 GitHub Secrets Required

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Your DockerHub username |
| `DOCKERHUB_TOKEN` | DockerHub access token |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Dataset | AG News (HuggingFace) / TruthLens |
| ML Framework | scikit-learn (TF-IDF + LR/SVM) |
| Experiment Tracking | MLflow 2.8 |
| Data Versioning | DVC 3 |
| Drift Monitoring | Evidently AI |
| API Serving | FastAPI + Uvicorn |
| Testing | Pytest + httpx |
| CI/CD | GitHub Actions |
| Containerization | Docker + docker-compose |

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

*Built as part of an MLOps portfolio sprint — see also: PulmoScanAI, TruthLens, Building Safety Detection*
