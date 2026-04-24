"""
tests/test_data.py
──────────────────
Validates input schema, data types, and integrity of processed datasets.
Run: pytest tests/test_data.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT / "data" / "processed"
RAW_DIR   = ROOT / "data" / "raw"

REQUIRED_COLUMNS = {"text", "label", "label_name"}
MIN_TEXT_LENGTH  = 5


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def train_df():
    path = PROC_DIR / "train.csv"
    if not path.exists():
        pytest.skip(f"train.csv not found at {path} — run src/preprocess.py first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def val_df():
    path = PROC_DIR / "val.csv"
    if not path.exists():
        pytest.skip(f"val.csv not found at {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def test_df():
    path = PROC_DIR / "test.csv"
    if not path.exists():
        pytest.skip(f"test.csv not found at {path}")
    return pd.read_csv(path)


# ──────────────────────────────────────────────────────────────────────────────
# Schema tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSchema:
    def test_train_required_columns(self, train_df):
        missing = REQUIRED_COLUMNS - set(train_df.columns)
        assert not missing, f"Missing columns in train: {missing}"

    def test_val_required_columns(self, val_df):
        missing = REQUIRED_COLUMNS - set(val_df.columns)
        assert not missing, f"Missing columns in val: {missing}"

    def test_test_required_columns(self, test_df):
        missing = REQUIRED_COLUMNS - set(test_df.columns)
        assert not missing, f"Missing columns in test: {missing}"

    def test_text_column_is_string(self, train_df):
        assert train_df["text"].dtype == object, "text column should be string/object dtype"

    def test_label_column_is_numeric(self, train_df):
        assert pd.api.types.is_integer_dtype(train_df["label"]) or \
               pd.api.types.is_float_dtype(train_df["label"]), \
               "label column should be numeric"

    def test_label_name_is_string(self, train_df):
        assert train_df["label_name"].dtype == object


# ──────────────────────────────────────────────────────────────────────────────
# Data quality tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDataQuality:
    def test_no_null_text_train(self, train_df):
        null_count = train_df["text"].isnull().sum()
        assert null_count == 0, f"Found {null_count} null values in train text"

    def test_no_null_labels_train(self, train_df):
        null_count = train_df["label"].isnull().sum()
        assert null_count == 0, f"Found {null_count} null values in train labels"

    def test_text_minimum_length(self, train_df):
        short = (train_df["text"].str.len() < MIN_TEXT_LENGTH).sum()
        assert short == 0, f"Found {short} texts shorter than {MIN_TEXT_LENGTH} chars"

    def test_label_label_name_consistency(self, train_df):
        """Each label ID should map to exactly one label_name."""
        mapping = train_df.groupby("label")["label_name"].nunique()
        inconsistent = mapping[mapping > 1]
        assert len(inconsistent) == 0, f"Inconsistent label→name mapping: {inconsistent.to_dict()}"

    def test_no_empty_strings(self, train_df):
        empty = (train_df["text"].str.strip() == "").sum()
        assert empty == 0, f"Found {empty} empty text strings"


# ──────────────────────────────────────────────────────────────────────────────
# Split integrity tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSplitIntegrity:
    def test_train_is_largest_split(self, train_df, val_df, test_df):
        assert len(train_df) > len(val_df), "Train should be larger than val"
        assert len(train_df) > len(test_df), "Train should be larger than test"

    def test_no_overlap_train_test(self, train_df, test_df):
        """Verify no exact text duplicates between train and test."""
        train_texts = set(train_df["text"].values)
        test_texts  = set(test_df["text"].values)
        overlap = train_texts & test_texts
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts between train and test"

    def test_class_distribution_balanced(self, train_df):
        """Class imbalance ratio should be below 5×."""
        counts = train_df["label"].value_counts()
        ratio  = counts.max() / counts.min()
        assert ratio <= 5.0, f"Class imbalance ratio {ratio:.1f}× exceeds 5×"

    def test_all_classes_in_all_splits(self, train_df, val_df, test_df):
        """All classes in train should also appear in val and test."""
        train_classes = set(train_df["label"].unique())
        val_classes   = set(val_df["label"].unique())
        test_classes  = set(test_df["label"].unique())

        assert train_classes == val_classes, \
            f"Val missing classes: {train_classes - val_classes}"
        assert train_classes == test_classes, \
            f"Test missing classes: {train_classes - test_classes}"

    def test_minimum_split_sizes(self, train_df, val_df, test_df):
        assert len(train_df) >= 100, f"Train set too small: {len(train_df)}"
        assert len(val_df)   >= 50,  f"Val set too small: {len(val_df)}"
        assert len(test_df)  >= 50,  f"Test set too small: {len(test_df)}"


# ──────────────────────────────────────────────────────────────────────────────
# Type coercion tests (simulate API input)
# ──────────────────────────────────────────────────────────────────────────────

class TestInputTypes:
    def test_text_accepts_unicode(self):
        text = "Breaking news: über 100 people affected by naïve policymakers."
        assert isinstance(text, str)
        assert len(text) > 10

    def test_numeric_label_range(self, train_df):
        labels = train_df["label"].unique()
        n = len(labels)
        assert set(labels) == set(range(n)), \
            f"Labels should be 0-indexed integers 0..{n-1}, got: {sorted(labels)}"

    @pytest.mark.parametrize("text", [
        "Stock markets rallied on Friday after positive jobs report.",
        "The national team won the championship after a thrilling match.",
        "Scientists discover potential breakthrough in quantum computing.",
        "World leaders gather for climate summit in Geneva.",
    ])
    def test_valid_sample_texts(self, text):
        assert isinstance(text, str)
        assert len(text.split()) >= 3
