"""
src/train.py
────────────
Trains TF-IDF + Logistic Regression (or SVM) on AG News / TruthLens.
Performs 3 hyperparameter runs, logs everything to MLflow,
and promotes the best model to the Model Registry.

Run: python src/train.py
"""

import os
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
from typing import Dict, Any, Tuple

from sklearn.pipeline         import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import LinearSVC
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.metrics          import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proc = ROOT / "data" / "processed"
    train = pd.read_csv(proc / "train.csv")
    val   = pd.read_csv(proc / "val.csv")
    test  = pd.read_csv(proc / "test.csv")
    log.info(f"Loaded splits — train:{len(train):,}  val:{len(val):,}  test:{len(test):,}")
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Model building
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline(config: Dict[str, Any]) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features = config["max_features"],
        ngram_range  = tuple(config["ngram_range"]),
        sublinear_tf = True,
        min_df       = 2,
        strip_accents = "unicode",
        analyzer     = "word",
    )

    model_type = config.get("model_type", "tfidf_lr")

    if model_type == "tfidf_lr":
        classifier = LogisticRegression(
            C            = config["C"],
            max_iter     = config["max_iter"],
            class_weight = config.get("class_weight", "balanced"),
            solver       = "saga",
            n_jobs       = -1,
        )
    elif model_type == "tfidf_svm":
        svc = LinearSVC(
            C            = config["C"],
            max_iter     = config["max_iter"],
            class_weight = config.get("class_weight", "balanced"),
        )
        classifier = CalibratedClassifierCV(svc, cv=3)   # adds predict_proba
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, label_names) -> Dict[str, float]:
    n_classes = len(label_names)
    average   = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "f1_macro"  : f1_score(y_true, y_pred, average="macro"),
        "precision" : precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall"    : recall_score(y_true, y_pred, average=average, zero_division=0),
    }

    try:
        if n_classes == 2:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics["auc_roc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
    except Exception:
        metrics["auc_roc"] = 0.0

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Artifacts
# ──────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, label_names, path: Path) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=label_names, yticklabels=label_names, ax=ax
        )
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Single MLflow run
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    run_name: str,
    config: Dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    label_names: list,
) -> Tuple[float, str]:
    """Train, evaluate on val, log to MLflow. Returns (val_accuracy, run_id)."""

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        log.info(f"▶  Run: {run_name}  (run_id={run_id[:8]}…)")

        # ── Log params ────────────────────────────────────────────────────────
        mlflow.log_params({
            "model_type"  : config["model_type"],
            "max_features": config["max_features"],
            "ngram_range" : str(config["ngram_range"]),
            "C"           : config["C"],
            "max_iter"    : config["max_iter"],
            "class_weight": config.get("class_weight", "balanced"),
            "dataset"     : PARAMS["data"]["dataset"],
            "train_size"  : len(train),
            "val_size"    : len(val),
        })

        # ── Train ─────────────────────────────────────────────────────────────
        pipeline = build_pipeline(config)
        pipeline.fit(train["text"], train["label"])

        # ── Evaluate on validation ────────────────────────────────────────────
        y_pred = pipeline.predict(val["text"])
        y_prob = pipeline.predict_proba(val["text"])
        metrics = compute_metrics(val["label"].values, y_pred, y_prob, label_names)

        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        log.info(f"   val_accuracy={metrics['accuracy']:.4f}  val_f1={metrics['f1_macro']:.4f}")

        # ── Classification report ─────────────────────────────────────────────
        report_str = classification_report(
            val["label"], y_pred, target_names=label_names, digits=4
        )
        report_path = ROOT / "reports" / f"cls_report_{run_name}.txt"
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(report_str)
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        # ── Confusion matrix ──────────────────────────────────────────────────
        cm_path = ROOT / "reports" / f"confusion_matrix_{run_name}.png"
        save_confusion_matrix(val["label"].values, y_pred, label_names, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="reports")

        # ── Log model ─────────────────────────────────────────────────────────
        signature  = infer_signature(train["text"].head(5), pipeline.predict(train["text"].head(5)))
        input_ex   = train["text"].head(3).tolist()

        mlflow.sklearn.log_model(
            sk_model       = pipeline,
            artifact_path  = "model",
            signature      = signature,
            input_example  = input_ex,
            registered_model_name = None,  # we register the best one later
        )

        # ── Save locally too ──────────────────────────────────────────────────
        model_dir = ROOT / "models"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(pipeline, model_dir / f"{run_name}.pkl")

        return metrics["accuracy"], run_id


# ──────────────────────────────────────────────────────────────────────────────
# Register best model
# ──────────────────────────────────────────────────────────────────────────────

def register_best_model(
    best_run_id: str,
    val_accuracy: float,
    test: pd.DataFrame,
    label_names: list,
    pipeline,
) -> None:
    """Evaluate on test set and register model if above accuracy threshold."""
    threshold = PARAMS["mlflow"]["accuracy_threshold"]

    if val_accuracy < threshold:
        log.warning(
            f"Best val_accuracy {val_accuracy:.4f} < threshold {threshold}. "
            f"Model NOT registered."
        )
        return

    # Test set evaluation
    y_pred = pipeline.predict(test["text"])
    y_prob = pipeline.predict_proba(test["text"])
    metrics = compute_metrics(test["label"].values, y_pred, y_prob, label_names)

    log.info(f"📊  Test metrics: {json.dumps({k: round(v,4) for k,v in metrics.items()})}")

    # Log test metrics back to the run
    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    # Register
    model_name = PARAMS["mlflow"]["registered_model_name"]
    model_uri  = f"runs:/{best_run_id}/model"

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    log.info(f"✅  Registered '{model_name}' version {mv.version}")

    # Transition to Staging → Production
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name    = model_name,
        version = mv.version,
        stage   = "Staging",
    )
    client.transition_model_version_stage(
        name    = model_name,
        version = mv.version,
        stage   = "Production",
    )
    log.info(f"🚀  Model transitioned to Production")

    # Add description
    client.update_model_version(
        name        = model_name,
        version     = mv.version,
        description = (
            f"Best model from MLOps pipeline run. "
            f"val_acc={val_accuracy:.4f} test_acc={metrics['accuracy']:.4f}"
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main — run 3 experiments
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = [
    {
        "run_name"    : "tfidf_lr_baseline",
        "model_type"  : "tfidf_lr",
        "max_features": 30_000,
        "ngram_range" : [1, 1],
        "C"           : 1.0,
        "max_iter"    : 1000,
        "class_weight": "balanced",
    },
    {
        "run_name"    : "tfidf_lr_bigrams",
        "model_type"  : "tfidf_lr",
        "max_features": 50_000,
        "ngram_range" : [1, 2],
        "C"           : 5.0,
        "max_iter"    : 1000,
        "class_weight": "balanced",
    },
    {
        "run_name"    : "tfidf_svm_bigrams",
        "model_type"  : "tfidf_svm",
        "max_features": 50_000,
        "ngram_range" : [1, 2],
        "C"           : 1.0,
        "max_iter"    : 2000,
        "class_weight": "balanced",
    },
]


def main():
    # ── MLflow setup ──────────────────────────────────────────────────────────
    tracking_uri = PARAMS["mlflow"]["tracking_uri"]
    exp_name     = PARAMS["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    log.info(f"MLflow tracking at {tracking_uri}  experiment='{exp_name}'")

    # ── Load data ─────────────────────────────────────────────────────────────
    train, val, test = load_splits()

    label_names = sorted(train["label_name"].unique().tolist())
    log.info(f"Classes: {label_names}")

    # ── Run 3 experiments ─────────────────────────────────────────────────────
    results = []
    for cfg in EXPERIMENT_CONFIGS:
        run_name = cfg.pop("run_name")
        acc, run_id = run_experiment(run_name, cfg, train, val, label_names)
        results.append({"run_name": run_name, "val_accuracy": acc, "run_id": run_id, "config": cfg})

    # ── Find best ─────────────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["val_accuracy"])
    log.info(f"\n🏆  Best run: '{best['run_name']}'  val_accuracy={best['val_accuracy']:.4f}")

    # Reload best model
    best_pipeline = joblib.load(ROOT / "models" / f"{best['run_name']}.pkl")

    # ── Register if above threshold ───────────────────────────────────────────
    register_best_model(
        best_run_id  = best["run_id"],
        val_accuracy = best["val_accuracy"],
        test         = test,
        label_names  = label_names,
        pipeline     = best_pipeline,
    )

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  EXPERIMENT RESULTS")
    print("═" * 55)
    print(f"  {'Run Name':<30} {'Val Accuracy':>12}")
    print("─" * 55)
    for r in sorted(results, key=lambda x: -x["val_accuracy"]):
        marker = " ⬅ BEST" if r["run_id"] == best["run_id"] else ""
        print(f"  {r['run_name']:<30} {r['val_accuracy']:>12.4f}{marker}")
    print("═" * 55)


if __name__ == "__main__":
    main()
