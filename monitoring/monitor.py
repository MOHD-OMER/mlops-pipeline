"""
monitoring/monitor.py
─────────────────────
Generates an Evidently AI data drift report comparing reference (training)
data vs new incoming data. Saves HTML report to reports/drift_report.html.
Prints a DRIFT ALERT if PSI exceeds the configured threshold.

Run: python monitoring/monitor.py [--current data/processed/test.csv]
"""

import argparse
import logging
import sys
import yaml
import json
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parent.parent
PARAMS = yaml.safe_load(open(ROOT / "params.yaml"))

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ColumnDriftMetric,
        ColumnSummaryMetric,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    log.warning("Evidently not installed — pip install evidently")
    EVIDENTLY_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering for drift monitoring
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw text column into numeric features Evidently can analyze.
    This is standard practice for NLP drift detection.
    """
    feat = pd.DataFrame()
    feat["text_length"]    = df["text"].str.len()
    feat["word_count"]     = df["text"].str.split().str.len()
    feat["avg_word_length"]= df["text"].apply(
        lambda t: np.mean([len(w) for w in str(t).split()]) if str(t).split() else 0
    )
    feat["num_sentences"]  = df["text"].str.count(r"[.!?]+") + 1
    feat["uppercase_ratio"]= df["text"].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    feat["digit_ratio"]    = df["text"].apply(
        lambda t: sum(1 for c in str(t) if c.isdigit()) / max(len(str(t)), 1)
    )
    feat["label"] = df["label"].values

    if "label_name" in df.columns:
        feat["label_name"] = df["label_name"].values

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Drift detection
# ──────────────────────────────────────────────────────────────────────────────

def run_drift_report(
    reference_path: str,
    current_path: str,
    report_path: str,
) -> dict:
    """
    Build Evidently drift report and return summary dict.
    """
    log.info(f"📂  Reference : {reference_path}")
    log.info(f"📂  Current   : {current_path}")

    reference_df = pd.read_csv(reference_path)
    current_df   = pd.read_csv(current_path)

    # Feature extraction
    ref_feat = extract_text_features(reference_df)
    cur_feat = extract_text_features(current_df)

    numeric_cols = ["text_length", "word_count", "avg_word_length",
                    "num_sentences", "uppercase_ratio", "digit_ratio"]

    if EVIDENTLY_AVAILABLE:
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            DatasetMissingValuesMetric(),
            DatasetDriftMetric(),
            *[ColumnDriftMetric(column_name=col) for col in numeric_cols],
            *[ColumnSummaryMetric(column_name=col) for col in numeric_cols],
        ])

        report.run(reference_data=ref_feat, current_data=cur_feat)

        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(report_path)
        log.info(f"📊  Drift report saved → {report_path}")

        # Extract drift summary
        report_dict = report.as_dict()
        drift_metric = next(
            (m for m in report_dict["metrics"] if m["metric"] == "DatasetDriftMetric"),
            None,
        )

        if drift_metric:
            result = drift_metric.get("result", {})
            n_drifted  = result.get("number_of_drifted_columns", 0)
            n_total    = result.get("number_of_columns", len(numeric_cols))
            share      = result.get("share_of_drifted_columns", 0.0)
            dataset_drifted = result.get("dataset_drift", False)
        else:
            n_drifted = 0
            n_total   = len(numeric_cols)
            share     = 0.0
            dataset_drifted = False

        summary = {
            "timestamp"         : datetime.utcnow().isoformat(),
            "reference_rows"    : len(reference_df),
            "current_rows"      : len(current_df),
            "n_drifted_columns" : n_drifted,
            "n_total_columns"   : n_total,
            "drift_share"       : round(share, 4),
            "dataset_drifted"   : dataset_drifted,
            "report_path"       : str(report_path),
        }

    else:
        # Fallback: manual PSI-based drift detection
        log.info("⚙   Running fallback PSI drift detection …")
        summary = _psi_drift_detection(ref_feat, cur_feat, numeric_cols, report_path)

    return summary


def _psi_drift_detection(ref_feat, cur_feat, numeric_cols, report_path) -> dict:
    """Fallback drift detection using Population Stability Index (PSI)."""

    def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
        ref_counts, edges = np.histogram(ref, bins=bins)
        cur_counts, _     = np.histogram(cur, bins=edges)

        ref_pct = (ref_counts + 1e-8) / len(ref)   # +eps to avoid log(0)
        cur_pct = (cur_counts + 1e-8) / len(cur)

        return float(np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct)))

    threshold = PARAMS["monitoring"]["drift_threshold"]
    results   = {}
    drifted   = []

    for col in numeric_cols:
        score = psi(ref_feat[col].dropna().values, cur_feat[col].dropna().values)
        results[col] = {"psi": round(score, 4), "drifted": score > threshold}
        if score > threshold:
            drifted.append(col)

    # Save minimal HTML report
    html_rows = "\n".join(
        f"<tr><td>{col}</td><td>{v['psi']:.4f}</td>"
        f"<td style='color:{'red' if v['drifted'] else 'green'}'>"
        f"{'⚠ DRIFT' if v['drifted'] else '✓ OK'}</td></tr>"
        for col, v in results.items()
    )
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Data Drift Report</title>
<style>body{{font-family:monospace;margin:40px}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:8px;text-align:left}}
th{{background:#f0f0f0}}</style>
</head>
<body>
<h1>📊 Data Drift Report (PSI)</h1>
<p>Generated: {datetime.utcnow().isoformat()} UTC</p>
<p>Threshold: {threshold}</p>
<table>
<tr><th>Feature</th><th>PSI Score</th><th>Status</th></tr>
{html_rows}
</table>
<p><strong>Drifted columns:</strong> {', '.join(drifted) if drifted else 'None'}</p>
</body></html>"""

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(html)
    log.info(f"📊  Fallback drift report saved → {report_path}")

    return {
        "timestamp"         : datetime.utcnow().isoformat(),
        "reference_rows"    : len(ref_feat),
        "current_rows"      : len(cur_feat),
        "n_drifted_columns" : len(drifted),
        "n_total_columns"   : len(numeric_cols),
        "drift_share"       : round(len(drifted) / len(numeric_cols), 4),
        "dataset_drifted"   : len(drifted) > 0,
        "psi_scores"        : results,
        "report_path"       : str(report_path),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Alert
# ──────────────────────────────────────────────────────────────────────────────

def check_and_alert(summary: dict) -> bool:
    """Print a drift alert. Returns True if drift detected."""
    drifted = summary.get("dataset_drifted", False)
    share   = summary.get("drift_share", 0.0)

    print("\n" + "═" * 60)
    print("  DATA DRIFT MONITORING REPORT")
    print("═" * 60)
    print(f"  Timestamp        : {summary['timestamp']}")
    print(f"  Reference rows   : {summary['reference_rows']:,}")
    print(f"  Current rows     : {summary['current_rows']:,}")
    print(f"  Drifted columns  : {summary['n_drifted_columns']} / {summary['n_total_columns']}")
    print(f"  Drift share      : {share:.1%}")
    print(f"  Dataset drifted  : {'⚠  YES — ACTION REQUIRED' if drifted else '✅  No'}")
    print(f"  Report           : {summary['report_path']}")
    print("═" * 60)

    if drifted:
        print("\n🚨  DRIFT ALERT: Distribution shift detected!")
        print("    Recommended actions:")
        print("    1. Review the drift report HTML")
        print("    2. Check for data pipeline issues or distribution shift")
        print("    3. Consider retraining the model on fresh data")
        print("    4. Investigate upstream data sources\n")

    return drifted


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evidently data drift monitor")
    parser.add_argument(
        "--reference",
        default=str(ROOT / PARAMS["monitoring"]["reference_data"]),
        help="Path to reference (training) CSV",
    )
    parser.add_argument(
        "--current",
        default=str(ROOT / "data" / "processed" / "test.csv"),
        help="Path to current (new) data CSV",
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / PARAMS["monitoring"]["report_path"]),
        help="Output path for HTML drift report",
    )
    args = parser.parse_args()

    summary = run_drift_report(args.reference, args.current, args.report)

    # Save summary JSON
    summary_path = Path(args.report).parent / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    drifted = check_and_alert(summary)
    sys.exit(1 if drifted else 0)   # non-zero exit lets CI catch drift


if __name__ == "__main__":
    main()
