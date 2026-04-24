"""
src/ingest.py
─────────────
Downloads the AG News dataset from HuggingFace (or loads a local TruthLens CSV),
validates schema / nulls / class distribution, then writes:
  data/raw/train_raw.csv
  data/raw/test_raw.csv

Run: python src/ingest.py
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Download / Load
# ──────────────────────────────────────────────────────────────────────────────

def load_ag_news(max_samples: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download AG News from HuggingFace and return (train_df, test_df)."""
    from datasets import load_dataset

    log.info("📥  Downloading AG News from HuggingFace …")
    ds = load_dataset("ag_news")

    train_df = ds["train"].to_pandas()
    test_df  = ds["test"].to_pandas()

    # Subsample for faster iteration (remove for full run)
    if max_samples:
        train_df = (
            train_df.groupby("label", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), max_samples // 4), random_state=42))
                    .reset_index(drop=True)
        )
        test_df = (
            test_df.groupby("label", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), 1_000 // 4), random_state=42))
                   .reset_index(drop=True)
        )

    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    train_df["label_name"] = train_df["label"].map(label_map)
    test_df["label_name"]  = test_df["label"].map(label_map)

    log.info(f"✅  AG News loaded — train: {len(train_df):,}  test: {len(test_df):,}")
    return train_df, test_df


def load_truthlens(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load local TruthLens CSV. Expects columns: text, label (0=real,1=fake)."""
    from sklearn.model_selection import train_test_split

    log.info(f"📥  Loading TruthLens from {csv_path} …")
    df = pd.read_csv(csv_path)

    required = {"text", "label"}
    assert required.issubset(df.columns), f"CSV must contain columns: {required}"

    df["label_name"] = df["label"].map({0: "Real", 1: "Fake"})
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    log.info(f"✅  TruthLens loaded — train: {len(train_df):,}  test: {len(test_df):,}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Validation
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {"text", "label", "label_name"}
MIN_TEXT_LENGTH  = 10
MAX_NULL_RATIO   = 0.02        # 2 %
MAX_CLASS_IMBALANCE = 5.0      # majority / minority  ≤ 5×


def validate_dataframe(df: pd.DataFrame, split_name: str = "train") -> Dict[str, Any]:
    """Run data quality checks and return a report dict. Raises on hard failures."""
    report: Dict[str, Any] = {"split": split_name, "rows": len(df), "issues": []}

    # ── Column schema ─────────────────────────────────────────────────────────
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"[{split_name}] Missing columns: {missing_cols}")

    # ── Null check ────────────────────────────────────────────────────────────
    null_ratio = df["text"].isnull().mean()
    report["null_ratio"] = round(null_ratio, 4)
    if null_ratio > MAX_NULL_RATIO:
        raise ValueError(
            f"[{split_name}] Null ratio {null_ratio:.2%} exceeds threshold {MAX_NULL_RATIO:.2%}"
        )

    # ── Text length ───────────────────────────────────────────────────────────
    short_mask = df["text"].str.len() < MIN_TEXT_LENGTH
    n_short = short_mask.sum()
    report["short_texts"] = int(n_short)
    if n_short > 0:
        report["issues"].append(f"{n_short} texts shorter than {MIN_TEXT_LENGTH} chars")
        log.warning(f"[{split_name}] ⚠  {n_short} very short texts — consider filtering")

    # ── Class distribution ────────────────────────────────────────────────────
    counts = df["label"].value_counts()
    imbalance_ratio = counts.max() / counts.min()
    report["class_counts"]     = counts.to_dict()
    report["imbalance_ratio"]  = round(float(imbalance_ratio), 2)
    if imbalance_ratio > MAX_CLASS_IMBALANCE:
        report["issues"].append(f"Class imbalance ratio {imbalance_ratio:.1f}× exceeds {MAX_CLASS_IMBALANCE}×")
        log.warning(f"[{split_name}] ⚠  High class imbalance: {imbalance_ratio:.1f}×")

    # ── Duplicates ────────────────────────────────────────────────────────────
    n_dups = df["text"].duplicated().sum()
    report["duplicate_texts"] = int(n_dups)
    if n_dups > 0:
        report["issues"].append(f"{n_dups} duplicate text entries")

    log.info(
        f"[{split_name}] ✅  Validation passed | rows={len(df):,} | "
        f"nulls={null_ratio:.2%} | imbalance={imbalance_ratio:.1f}× | dups={n_dups}"
    )
    return report


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Save
# ──────────────────────────────────────────────────────────────────────────────

def save_raw(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "train_raw.csv"
    test_path  = raw_dir / "test_raw.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    log.info(f"💾  Saved → {train_path}  ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    log.info(f"💾  Saved → {test_path}  ({os.path.getsize(test_path) / 1e6:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    dataset     = PARAMS["data"].get("dataset", "ag_news")
    max_samples = PARAMS["data"].get("max_samples")

    if dataset == "ag_news":
        train_df, test_df = load_ag_news(max_samples=max_samples)
    elif dataset == "truthlens":
        csv_path = PARAMS["data"].get("truthlens_path", "data/raw/truthlens.csv")
        train_df, test_df = load_truthlens(csv_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'ag_news' or 'truthlens'.")

    # Validate both splits
    train_report = validate_dataframe(train_df, "train")
    test_report  = validate_dataframe(test_df,  "test")

    # Print summary
    log.info("─" * 60)
    log.info("VALIDATION SUMMARY")
    for report in [train_report, test_report]:
        log.info(f"  Split  : {report['split']}")
        log.info(f"  Rows   : {report['rows']:,}")
        log.info(f"  Classes: {report['class_counts']}")
        log.info(f"  Issues : {report['issues'] or 'None'}")
        log.info("─" * 60)

    save_raw(train_df, test_df)
    log.info("🎉  Ingestion complete.")


if __name__ == "__main__":
    main()
