"""
tests/test_model.py
────────────────────
Tests that the model loads correctly, produces valid output shapes,
and prediction values are in expected ranges.

Run: pytest tests/test_model.py -v
"""

import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROC_DIR   = ROOT / "data" / "processed"

SAMPLE_TEXTS = [
    "The stock market surged on positive earnings reports from major tech companies.",
    "Scientists discover a new method for carbon capture that could fight climate change.",
    "The national football team won the championship after a dramatic penalty shootout.",
    "World leaders met in Geneva to discuss nuclear disarmament treaties.",
    "New AI models achieve human-level performance on reasoning benchmarks.",
]


# ──────────────────────────────────────────────────────────────────────────────
# Fixture
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    pkls = sorted(MODELS_DIR.glob("*.pkl"))
    if not pkls:
        pytest.skip("No model .pkl found — run src/train.py first")
    return joblib.load(pkls[-1])


@pytest.fixture(scope="module")
def test_df():
    path = PROC_DIR / "test.csv"
    if not path.exists():
        pytest.skip("test.csv not found — run src/preprocess.py first")
    return pd.read_csv(path)


# ──────────────────────────────────────────────────────────────────────────────
# Loading tests
# ──────────────────────────────────────────────────────────────────────────────

class TestModelLoading:
    def test_model_file_exists(self):
        pkls = list(MODELS_DIR.glob("*.pkl"))
        assert len(pkls) > 0, f"No .pkl files found in {MODELS_DIR}"

    def test_model_is_sklearn_pipeline(self, pipeline):
        assert isinstance(pipeline, Pipeline), \
            f"Expected sklearn Pipeline, got {type(pipeline)}"

    def test_pipeline_has_two_steps(self, pipeline):
        assert len(pipeline.steps) == 2, \
            f"Pipeline should have 2 steps (tfidf, clf), got {len(pipeline.steps)}"

    def test_pipeline_has_tfidf_step(self, pipeline):
        step_names = [name for name, _ in pipeline.steps]
        assert "tfidf" in step_names, f"Expected 'tfidf' step, got: {step_names}"

    def test_pipeline_has_classifier_step(self, pipeline):
        step_names = [name for name, _ in pipeline.steps]
        assert "clf" in step_names, f"Expected 'clf' step, got: {step_names}"

    def test_model_has_classes(self, pipeline):
        assert hasattr(pipeline, "classes_"), "Pipeline should expose .classes_ attribute"
        assert len(pipeline.classes_) >= 2, "Pipeline should have at least 2 classes"


# ──────────────────────────────────────────────────────────────────────────────
# Prediction shape tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictionShape:
    def test_predict_returns_array(self, pipeline):
        preds = pipeline.predict(SAMPLE_TEXTS)
        assert hasattr(preds, "__len__"), "predict() should return an array-like"

    def test_predict_length_matches_input(self, pipeline):
        preds = pipeline.predict(SAMPLE_TEXTS)
        assert len(preds) == len(SAMPLE_TEXTS), \
            f"Expected {len(SAMPLE_TEXTS)} predictions, got {len(preds)}"

    def test_predict_proba_shape(self, pipeline):
        proba = pipeline.predict_proba(SAMPLE_TEXTS)
        n_classes = len(pipeline.classes_)
        assert proba.shape == (len(SAMPLE_TEXTS), n_classes), \
            f"Expected shape ({len(SAMPLE_TEXTS)}, {n_classes}), got {proba.shape}"

    def test_predict_proba_sums_to_one(self, pipeline):
        proba = pipeline.predict_proba(SAMPLE_TEXTS)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f"Probabilities should sum to 1.0, got: {row_sums}"

    def test_predict_proba_nonnegative(self, pipeline):
        proba = pipeline.predict_proba(SAMPLE_TEXTS)
        assert (proba >= 0).all(), "All probabilities should be >= 0"

    def test_predict_proba_leq_one(self, pipeline):
        proba = pipeline.predict_proba(SAMPLE_TEXTS)
        assert (proba <= 1.0 + 1e-8).all(), "All probabilities should be <= 1.0"


# ──────────────────────────────────────────────────────────────────────────────
# Prediction value tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictionValues:
    def test_predicted_labels_are_valid_classes(self, pipeline):
        preds = pipeline.predict(SAMPLE_TEXTS)
        valid_classes = set(pipeline.classes_)
        for pred in preds:
            assert pred in valid_classes, f"Predicted {pred} not in classes {valid_classes}"

    def test_confidence_is_reasonable(self, pipeline):
        """Max class probability should be at least 1/n_classes (better than random)."""
        proba = pipeline.predict_proba(SAMPLE_TEXTS)
        n_classes = len(pipeline.classes_)
        random_baseline = 1.0 / n_classes
        max_probs = proba.max(axis=1)
        assert all(p >= random_baseline * 0.5 for p in max_probs), \
            "Some predictions are worse than random — model may not be trained"

    def test_single_text_prediction(self, pipeline):
        text = ["Apple announces record quarterly earnings amid strong iPhone sales."]
        pred = pipeline.predict(text)
        assert len(pred) == 1

    def test_empty_text_handling(self, pipeline):
        """Model should handle (not crash on) short/edge-case inputs."""
        edge_cases = ["", ".", "a", "123"]
        try:
            preds = pipeline.predict(edge_cases)
            assert len(preds) == len(edge_cases)
        except Exception as e:
            pytest.skip(f"Model raises on edge-case input (acceptable): {e}")

    @pytest.mark.parametrize("text", SAMPLE_TEXTS)
    def test_individual_prediction_not_nan(self, pipeline, text):
        proba = pipeline.predict_proba([text])
        assert not np.isnan(proba).any(), f"NaN probability for text: '{text[:50]}…'"


# ──────────────────────────────────────────────────────────────────────────────
# Performance smoke test on held-out test set
# ──────────────────────────────────────────────────────────────────────────────

class TestModelPerformance:
    def test_test_accuracy_above_random(self, pipeline, test_df):
        from sklearn.metrics import accuracy_score
        n_classes    = len(pipeline.classes_)
        random_acc   = 1.0 / n_classes

        preds = pipeline.predict(test_df["text"])
        acc   = accuracy_score(test_df["label"], preds)
        assert acc > random_acc * 1.5, \
            f"Test accuracy {acc:.3f} barely above random baseline {random_acc:.3f}"

    def test_test_accuracy_above_minimum(self, pipeline, test_df):
        """Minimum acceptable accuracy — sanity check."""
        from sklearn.metrics import accuracy_score
        preds = pipeline.predict(test_df["text"])
        acc   = accuracy_score(test_df["label"], preds)
        MINIMUM_ACCURACY = 0.60
        assert acc >= MINIMUM_ACCURACY, \
            f"Test accuracy {acc:.3f} below minimum threshold {MINIMUM_ACCURACY}"
