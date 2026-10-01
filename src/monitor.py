"""
Data-drift monitoring. Compares incoming/current data against the training
reference distribution and flags when drift crosses a threshold — the signal
that would trigger a retrain in production.

Uses Evidently if installed; otherwise falls back to a lightweight PSI
(Population Stability Index) check so the pipeline always runs.

Usage:  python -m src.monitor --current data/new_batch.csv
"""
import argparse
import numpy as np
import pandas as pd

from src import config


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for a single numeric feature."""
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / len(expected)
    a = np.histogram(actual, bins=cuts)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def simple_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    scores = {}
    for col in config.NUMERIC_FEATURES:
        scores[col] = psi(reference[col].values, current[col].values)
    drifted = {k: v for k, v in scores.items() if v > 0.2}  # 0.2 = moderate drift
    return {
        "psi_scores": scores,
        "drifted_features": list(drifted),
        "drift_detected": len(drifted) > 0,
    }


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame):
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    out = config.REPORT_DIR / "drift_report.html"
    report.save_html(str(out))
    result = report.as_dict()
    share = result["metrics"][0]["result"]["dataset_drift"]
    return {"drift_detected": bool(share), "report_html": str(out)}


def run(current_path: str):
    reference = pd.read_csv(config.REFERENCE_PATH)
    current = pd.read_csv(current_path)
    try:
        result = evidently_report(reference, current)
        print("[evidently] report saved:", result.get("report_html"))
    except Exception as e:  # noqa: BLE001
        print(f"[evidently] not available, using PSI fallback ({e})")
        result = simple_drift_report(reference, current)

    print("Drift detected:", result["drift_detected"])
    if result["drift_detected"]:
        print(">> TRIGGER: retrain the model (run `python -m src.train`)")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--current", required=True, help="CSV of recent/live data")
    args = p.parse_args()
    run(args.current)
