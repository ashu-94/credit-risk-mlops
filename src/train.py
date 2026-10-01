"""
Train the credit-risk model, evaluate it, log everything to MLflow, and persist
the fitted pipeline for serving.

Usage:  python -m src.train
"""
import json
import joblib
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report,
)

from src import config
from src.pipeline import load_data, split, build_full_pipeline


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
    }


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    pipe = build_full_pipeline()
    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_test, y_test)

    # ---- MLflow tracking (optional at runtime; degrades gracefully) ----
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run():
            mlflow.log_params(config.MODEL_PARAMS)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipe, "model", registered_model_name=config.REGISTERED_MODEL_NAME
            )
        print("[mlflow] run logged")
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow] skipped ({e})")

    # ---- Persist for serving + save metrics report ----
    joblib.dump(pipe, config.MODEL_PATH)
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Model saved to", config.MODEL_PATH)
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, (pipe.predict_proba(X_test)[:, 1] >= 0.5)))
    return metrics


if __name__ == "__main__":
    main()
