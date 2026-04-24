"""
src/preprocess.py
─────────────────
Reads data/raw/*.csv, cleans text, performs stratified train/val/test split,
and writes data/processed/{train,val,test}.csv

Run: python src/preprocess.py
"""

import re
import logging
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))


# ──────────────────────────────────────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lightweight text normalization suitable for TF-IDF or BERT tokenizer."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize URLs
    text = re.sub(r"https?://\S+|www\.\S+", "URL", text)
    # Normalize numbers
    text = re.sub(r"\b\d+\b", "NUM", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train, val, test) with proportions from params.yaml.
    Stratifies on 'label' column.
    """
    test_size = PARAMS["data"]["test_size"]
    val_size  = PARAMS["data"]["val_size"]
    seed      = PARAMS["data"]["random_state"]

    # First carve out test
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    # Then split val from train
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label"], random_state=seed
    )

    log.info(
        f"Split → train: {len(train):,}  val: {len(val):,}  test: {len(test):,}"
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    raw_train = pd.read_csv(ROOT / "data" / "raw" / "train_raw.csv")
    raw_test  = pd.read_csv(ROOT / "data" / "raw" / "test_raw.csv")

    # Combine, clean, re-split (ensures val comes from training pool only)
    df = pd.concat([raw_train, raw_test], ignore_index=True)
    log.info(f"Total rows before cleaning: {len(df):,}")

    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 10].reset_index(drop=True)
    log.info(f"Total rows after cleaning : {len(df):,}")

    train, val, test = stratified_split(df)

    proc_dir = ROOT / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = proc_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        log.info(f"💾  Saved {split_name}.csv  ({len(split_df):,} rows)")

    log.info("🎉  Preprocessing complete.")


if __name__ == "__main__":
    main()
