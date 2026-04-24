"""
tests/test_api.py
──────────────────
Tests all FastAPI endpoints for correct status codes and response shapes.
Uses httpx AsyncClient for async testing without needing a live server.

Run: pytest tests/test_api.py -v
"""

import pytest
import pytest_asyncio
import joblib
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# Mock pipeline fixture (avoids needing MLflow running during tests)
# ──────────────────────────────────────────────────────────────────────────────

def build_mock_pipeline():
    """Build a tiny trained pipeline for testing without real data."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Use 4 classes with enough samples each
    train_texts = [
        "stock market earnings profit revenue business investor quarterly",
        "football game match score championship sports league trophy",
        "science technology research innovation discovery lab experiment",
        "politics government election world leaders summit diplomat",
        "company merger acquisition deal shareholders board profit",
        "athlete team win cup final stadium crowd performance",
        "scientists breakthrough study climate biology chemistry",
        "foreign policy minister treaty international relations",
        "bank interest rate inflation economy financial sector",
        "olympic gold medal record performance competition athlete",
        "nasa spacecraft orbit satellite astronomy telescope galaxy",
        "election campaign vote debate president senator legislation",
    ]
    train_labels = [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]
    label_names  = ["World", "Business", "Sports", "Sci/Tech"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=200)),
        ("clf",   LogisticRegression(max_iter=500)),
    ])
    pipe.fit(train_texts, train_labels)
    # classes_ is set automatically by fit — don't override it
    return pipe, label_names


@pytest.fixture(scope="module")
def mock_pipeline():
    return build_mock_pipeline()


# ──────────────────────────────────────────────────────────────────────────────
# Test client setup — patches MODEL_STATE to avoid needing MLflow
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_client(mock_pipeline):
    pipeline, label_names = mock_pipeline

    # Patch _load_model so the app doesn't try to connect to MLflow on startup
    with patch("src.serve._load_model") as mock_load:
        def side_effect():
            from src import serve
            serve.MODEL_STATE.update({
                "pipeline"     : pipeline,
                "model_version": "test-v1",
                "label_names"  : label_names,
                "metrics"      : {"val_accuracy": 0.92, "val_f1_macro": 0.91},
                "loaded_at"    : "2024-01-01T00:00:00Z",
            })
        mock_load.side_effect = side_effect

        from httpx import AsyncClient
        from src.serve import app
        import asyncio

        # Return a factory; actual client created per-test or in async fixtures
        return app


# ──────────────────────────────────────────────────────────────────────────────
# Sync convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client(mock_pipeline):
    """Sync TestClient for simple status-code tests."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch
    from src import serve

    pipeline, label_names = mock_pipeline
    serve.MODEL_STATE.update({
        "pipeline"     : pipeline,
        "model_version": "test-v1",
        "label_names"  : label_names,
        "metrics"      : {"val_accuracy": 0.92, "val_f1_macro": 0.91},
        "loaded_at"    : "2024-01-01T00:00:00Z",
    })

    with patch("src.serve._load_model"):
        with TestClient(serve.app, raise_server_exceptions=True) as c:
            yield c


# ──────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, client):
        response = client.get("/health")
        body = response.json()
        assert "status" in body

    def test_health_status_is_healthy(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "healthy"

    def test_health_includes_model_version(self, client):
        response = client.get("/health")
        assert "model_version" in response.json()


# ──────────────────────────────────────────────────────────────────────────────
# Model info endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_required_fields(self, client):
        body = response = client.get("/model/info").json()
        required = {"model_name", "model_version", "label_names", "metrics", "loaded_at"}
        assert required.issubset(body.keys()), f"Missing fields: {required - body.keys()}"

    def test_model_info_label_names_is_list(self, client):
        body = client.get("/model/info").json()
        assert isinstance(body["label_names"], list)
        assert len(body["label_names"]) >= 2

    def test_model_info_metrics_is_dict(self, client):
        body = client.get("/model/info").json()
        assert isinstance(body["metrics"], dict)


# ──────────────────────────────────────────────────────────────────────────────
# Predict endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json={"text": "Apple stocks rose 5% on strong earnings."})
        assert response.status_code == 200

    def test_predict_response_has_label(self, client):
        response = client.post("/predict", json={"text": "Football match ended in a draw."})
        body = response.json()
        assert "label" in body
        assert isinstance(body["label"], str)

    def test_predict_response_has_confidence(self, client):
        response = client.post("/predict", json={"text": "Scientists discover new galaxy."})
        body = response.json()
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_response_has_top_predictions(self, client):
        response = client.post("/predict", json={"text": "Government passes new legislation."})
        body = response.json()
        assert "top_predictions" in body
        assert isinstance(body["top_predictions"], list)
        assert len(body["top_predictions"]) >= 1

    def test_predict_top_predictions_have_probability(self, client):
        response = client.post("/predict", json={"text": "New research in artificial intelligence."})
        body = response.json()
        for pred in body["top_predictions"]:
            assert "label" in pred
            assert "probability" in pred
            assert 0.0 <= pred["probability"] <= 1.0

    def test_predict_returns_model_version(self, client):
        response = client.post("/predict", json={"text": "Market analysis today."})
        body = response.json()
        assert "model_version" in body

    def test_predict_returns_latency(self, client):
        response = client.post("/predict", json={"text": "Test latency measurement."})
        body = response.json()
        assert "latency_ms" in body
        assert body["latency_ms"] >= 0.0

    def test_predict_top_k_parameter(self, client):
        response = client.post("/predict", json={"text": "Sports news update.", "top_k": 2})
        body = response.json()
        assert len(body["top_predictions"]) <= 2

    def test_predict_empty_text_returns_422(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_missing_text_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_whitespace_text_returns_422(self, client):
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422

    @pytest.mark.parametrize("text", [
        "Breaking news: Central bank raises interest rates to combat inflation.",
        "Scientists announce major breakthrough in cancer treatment research.",
        "National team qualifies for World Cup after dramatic penalty shootout.",
        "Tech giant reports record quarterly revenue, beats analyst expectations.",
    ])
    def test_predict_various_news_texts(self, client, text):
        response = client.post("/predict", json={"text": text})
        assert response.status_code == 200
        body = response.json()
        assert "label" in body
        assert 0.0 <= body["confidence"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Batch predict endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        texts = [
            "Stock market news today.",
            "Sports championship results.",
            "Science research update.",
        ]
        response = client.post("/predict/batch", json=texts)
        assert response.status_code == 200

    def test_batch_predict_count_matches_input(self, client):
        texts = ["News item one.", "News item two.", "News item three."]
        body = client.post("/predict/batch", json=texts).json()
        assert body["count"] == len(texts)

    def test_batch_predict_results_structure(self, client):
        texts = ["Technology news.", "Political update."]
        body = client.post("/predict/batch", json=texts).json()
        for result in body["predictions"]:
            assert "label" in result
            assert "confidence" in result

    def test_batch_predict_too_many_texts_returns_400(self, client):
        texts = [f"News text {i}" for i in range(101)]
        response = client.post("/predict/batch", json=texts)
        assert response.status_code == 400


# ──────────────────────────────────────────────────────────────────────────────
# OpenAPI / docs
# ──────────────────────────────────────────────────────────────────────────────

class TestDocs:
    def test_openapi_schema_accessible(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200

    def test_docs_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
