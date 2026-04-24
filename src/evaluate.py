"""
src/evaluate.py
───────────────
Loads the Production model from MLflow Registry and evaluates it
on the test set. Saves a full evaluation report to reports/.

Run: python src/evaluate.py
"""

import json
import logging
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)

import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))


def load_production_model():
    """Load Production model from MLflow Registry, fallback to latest local pkl."""
    model_name = PARAMS["mlflow"]["registered_model_name"]
    tracking_uri = PARAMS["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    try:
        model_uri = f"models:/{model_name}/Production"
        pipeline = mlflow.sklearn.load_model(model_uri)
        log.info(f"✅  Loaded Production model from MLflow: {model_name}")
        return pipeline
    except Exception as e:
        log.warning(f"Could not load from MLflow ({e}), falling back to local pkl")
        pkls = sorted((ROOT / "models").glob("*.pkl"))
        if not pkls:
            raise FileNotFoundError("No local model files found in models/")
        pipeline = joblib.load(pkls[-1])
        log.info(f"✅  Loaded local model: {pkls[-1]}")
        return pipeline


def evaluate(pipeline, test: pd.DataFrame, label_names: list) -> dict:
    """Full evaluation with all metrics."""
    y_pred = pipeline.predict(test["text"])
    y_prob = pipeline.predict_proba(test["text"])

    n_classes = len(label_names)
    average   = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy"  : round(accuracy_score(test["label"], y_pred), 4),
        "f1_macro"  : round(f1_score(test["label"], y_pred, average="macro"), 4),
        "precision" : round(precision_score(test["label"], y_pred, average=average, zero_division=0), 4),
        "recall"    : round(recall_score(test["label"], y_pred, average=average, zero_division=0), 4),
    }

    try:
        if n_classes == 2:
            metrics["auc_roc"] = round(roc_auc_score(test["label"], y_prob[:, 1]), 4)
        else:
            metrics["auc_roc"] = round(
                roc_auc_score(test["label"], y_prob, multi_class="ovr", average="macro"), 4
            )
    except Exception:
        metrics["auc_roc"] = 0.0

    return metrics, y_pred, y_prob


def save_reports(test, y_pred, y_prob, label_names, metrics):
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Classification report
    report_str = classification_report(
        test["label"], y_pred, target_names=label_names, digits=4
    )
    (reports_dir / "final_classification_report.txt").write_text(report_str)
    log.info("📄  Saved final_classification_report.txt")

    # Metrics JSON
    (reports_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("📄  Saved final_metrics.json")

    # Confusion matrix
    cm = confusion_matrix(test["label"], y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Production Model — Test Set Evaluation", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalized"],
        ["d", ".2f"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="viridis",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    cm_path = reports_dir / "final_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    log.info(f"📊  Saved {cm_path}")


def main():
    pipeline = load_production_model()

    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    label_names = sorted(test["label_name"].unique().tolist())

    metrics, y_pred, y_prob = evaluate(pipeline, test, label_names)
    save_reports(test, y_pred, y_prob, label_names, metrics)

    print("\n" + "═" * 45)
    print("  PRODUCTION MODEL — TEST RESULTS")
    print("═" * 45)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v:.4f}")
    print("═" * 45)


if __name__ == "__main__":
    main()
