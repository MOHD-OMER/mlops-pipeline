"""
src/serve.py
────────────
FastAPI model server for the news classifier.

Endpoints:
  POST /predict       — text → label + confidence
  GET  /model/info    — current model version & metrics
  GET  /health        — health check

Run:
  uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
  OR
  python src/serve.py
"""

import json
import logging
import os
import time
import yaml
import joblib
import numpy as np

from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

log = logging.getLogger("uvicorn.error")

ROOT   = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))

# ── Global state ──────────────────────────────────────────────────────────────
MODEL_STATE = {
    "pipeline"     : None,
    "model_version": "unknown",
    "label_names"  : [],
    "metrics"      : {},
    "loaded_at"    : None,
}


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_model():
    """Try MLflow Registry first, fall back to local pkl."""
    import mlflow
    import mlflow.sklearn

    tracking_uri = PARAMS["mlflow"]["tracking_uri"]
    model_name   = PARAMS["mlflow"]["registered_model_name"]
    stage        = PARAMS["serving"]["model_stage"]

    mlflow.set_tracking_uri(tracking_uri)

    try:
        model_uri = f"models:/{model_name}/{stage}"
        pipeline  = mlflow.sklearn.load_model(model_uri)

        # Fetch version metadata
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        version  = versions[0].version if versions else "unknown"
        run_id   = versions[0].run_id  if versions else None

        metrics = {}
        if run_id:
            run  = client.get_run(run_id)
            metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}

        MODEL_STATE.update({
            "pipeline"     : pipeline,
            "model_version": version,
            "metrics"      : metrics,
            "loaded_at"    : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        log.info(f"✅  Loaded model '{model_name}' v{version} from MLflow ({stage})")

    except Exception as e:
        log.warning(f"MLflow load failed ({e}), falling back to local pkl …")

        pkls = sorted((ROOT / "models").glob("*.pkl"))
        if not pkls:
            raise RuntimeError("No model found — run src/train.py first")

        pipeline = joblib.load(pkls[-1])
        MODEL_STATE.update({
            "pipeline"     : pipeline,
            "model_version": f"local:{pkls[-1].stem}",
            "metrics"      : {},
            "loaded_at"    : time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        log.info(f"✅  Loaded local model: {pkls[-1]}")

    # Infer label names
    try:
        MODEL_STATE["label_names"] = list(
            MODEL_STATE["pipeline"].classes_.astype(str)
        )
    except Exception:
        MODEL_STATE["label_names"] = [str(i) for i in range(10)]


# ──────────────────────────────────────────────────────────────────────────────
# App startup / shutdown
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title       = "News Classifier API — MLOps Pipeline",
    description = "Serves a TF-IDF + LR text classification model logged with MLflow.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3          # return top-k class probabilities

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("text field must not be empty")
        return v.strip()


class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    top_predictions: List[dict]
    model_version: str
    latency_ms: float


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Simple health check."""
    pipeline = MODEL_STATE["pipeline"]
    return {
        "status" : "healthy" if pipeline is not None else "model_not_loaded",
        "model_version": MODEL_STATE["model_version"],
        "loaded_at"    : MODEL_STATE["loaded_at"],
    }


@app.get("/model/info", tags=["System"])
async def model_info():
    """Returns model version, metrics, and label mapping."""
    if MODEL_STATE["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name"   : PARAMS["mlflow"]["registered_model_name"],
        "model_version": MODEL_STATE["model_version"],
        "label_names"  : MODEL_STATE["label_names"],
        "metrics"      : MODEL_STATE["metrics"],
        "loaded_at"    : MODEL_STATE["loaded_at"],
        "tracking_uri" : PARAMS["mlflow"]["tracking_uri"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Predict news category from text.

    Returns the predicted label, confidence, and top-k probabilities.
    """
    pipeline = MODEL_STATE["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()

    try:
        proba     = pipeline.predict_proba([request.text])[0]
        label_id  = int(np.argmax(proba))
        label     = MODEL_STATE["label_names"][label_id]
        confidence = float(proba[label_id])

        top_k = min(request.top_k, len(MODEL_STATE["label_names"]))
        top_indices = np.argsort(proba)[::-1][:top_k]
        top_predictions = [
            {"label": MODEL_STATE["label_names"][i], "probability": round(float(proba[i]), 4)}
            for i in top_indices
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        label           = label,
        label_id        = label_id,
        confidence      = round(confidence, 4),
        top_predictions = top_predictions,
        model_version   = MODEL_STATE["model_version"],
        latency_ms      = round(latency_ms, 2),
    )


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(texts: List[str]):
    """Batch predict for multiple texts."""
    pipeline = MODEL_STATE["pipeline"]
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Max batch size is 100")

    t0 = time.perf_counter()
    probas   = pipeline.predict_proba(texts)
    labels   = pipeline.predict(texts)
    latency  = (time.perf_counter() - t0) * 1000

    results = []
    for i, (label_id, proba) in enumerate(zip(labels, probas)):
        results.append({
            "text"      : texts[i][:100] + "…" if len(texts[i]) > 100 else texts[i],
            "label"     : MODEL_STATE["label_names"][int(label_id)],
            "confidence": round(float(proba[int(label_id)]), 4),
        })

    return {"predictions": results, "count": len(results), "latency_ms": round(latency, 2)}


@app.post("/model/reload", tags=["System"])
async def reload_model():
    """Hot-reload the Production model from MLflow Registry."""
    try:
        _load_model()
        return {"status": "reloaded", "model_version": MODEL_STATE["model_version"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serve:app",
        host    = PARAMS["serving"]["host"],
        port    = PARAMS["serving"]["port"],
        reload  = False,
        workers = 1,
    )
