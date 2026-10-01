"""
Model selection — the rigor layer.

Answers the three questions a data-science interviewer will ask:
  1. What's your BASELINE?           -> Logistic Regression (the honest floor)
  2. How do you know XGBoost is better? -> stratified cross-validation, mean ± std
  3. Did you TUNE it?                 -> RandomizedSearchCV over a real grid

Every model is evaluated inside a Pipeline so preprocessing is fit *within each
CV fold* — no data leakage. Results are logged to MLflow and saved to reports/.

Usage:
  python -m src.benchmark                 # full run
  python -m src.benchmark --sample 5000   # faster, subsampled
  python -m src.benchmark --n-iter 30 --cv 5
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src import config
from src.pipeline import load_data, build_preprocessor, build_model


# ---------------------------------------------------------------- candidates
def candidate_models() -> dict:
    """The line-up: a baseline, a bagging model, and the boosted model."""
    models = {
        "Logistic Regression (baseline)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=config.RANDOM_STATE
        ),
        "XGBoost": build_model(),
    }
    return models


# ------------------------------------------------------------- CV comparison
def compare_models(X, y, cv_folds: int) -> pd.DataFrame:
    """Stratified CV for every candidate; report mean ± std on ROC-AUC and PR-AUC."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

    rows = []
    for name, model in candidate_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        cv = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        rows.append({
            "model": name,
            "roc_auc_mean": cv["test_roc_auc"].mean(),
            "roc_auc_std": cv["test_roc_auc"].std(),
            "pr_auc_mean": cv["test_pr_auc"].mean(),
            "pr_auc_std": cv["test_pr_auc"].std(),
        })
        print(f"  {name:32s}  ROC-AUC {rows[-1]['roc_auc_mean']:.3f} "
              f"± {rows[-1]['roc_auc_std']:.3f}   "
              f"PR-AUC {rows[-1]['pr_auc_mean']:.3f} ± {rows[-1]['pr_auc_std']:.3f}")
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def plot_comparison(df: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["roc_auc_mean"], xerr=df["roc_auc_std"],
            color="#4f7cff", capsize=4)
    ax.set_yticks(y_pos); ax.set_yticklabels(df["model"])
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC (5-fold CV, error bars = ±1 std)")
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Model comparison — cross-validated")
    for i, v in enumerate(df["roc_auc_mean"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


# ----------------------------------------------------------- tuning (XGBoost)
def tune_xgboost(X, y, n_iter: int, cv_folds: int) -> dict:
    param_dist = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }
    pipe = Pipeline([("preprocess", build_preprocessor()), ("model", build_model())])
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=n_iter, scoring="roc_auc",
        cv=skf, n_jobs=-1, random_state=config.RANDOM_STATE, verbose=0,
    )
    search.fit(X, y)

    # strip the 'model__' prefix so params can drop straight into config.MODEL_PARAMS
    best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    return {"best_params": best, "best_cv_roc_auc": float(search.best_score_)}


# ------------------------------------------------------------------- runner
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None, help="subsample rows for speed")
    p.add_argument("--n-iter", type=int, default=25, help="random search iterations")
    p.add_argument("--cv", type=int, default=5, help="CV folds")
    args = p.parse_args()

    df = load_data()
    if args.sample:
        df = df.sample(args.sample, random_state=config.RANDOM_STATE)
    X, y = df[config.FEATURES], df[config.TARGET]

    print(f"Comparing models on {len(df):,} rows, {args.cv}-fold CV...")
    comparison = compare_models(X, y, args.cv)
    plot_comparison(comparison, config.REPORT_DIR / "model_comparison.png")

    print(f"\nTuning XGBoost ({args.n_iter} random configs x {args.cv}-fold CV)...")
    tuning = tune_xgboost(X, y, args.n_iter, args.cv)
    print("  best CV ROC-AUC:", round(tuning["best_cv_roc_auc"], 4))
    print("  best params:", json.dumps(tuning["best_params"]))

    # -------- persist results --------
    result = {
        "comparison": comparison.to_dict(orient="records"),
        "tuning": tuning,
    }
    (config.REPORT_DIR / "benchmark.json").write_text(json.dumps(result, indent=2, default=float))

    # -------- MLflow (graceful) --------
    try:
        import mlflow
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="model_selection"):
            for row in comparison.to_dict(orient="records"):
                tag = row["model"].split()[0].lower()
                mlflow.log_metric(f"{tag}_roc_auc", row["roc_auc_mean"])
            mlflow.log_params(tuning["best_params"])
            mlflow.log_metric("tuned_cv_roc_auc", tuning["best_cv_roc_auc"])
        print("[mlflow] benchmark run logged")
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow] skipped ({e})")

    print("\nResults saved to reports/benchmark.json and reports/model_comparison.png")
    print("To use the tuned params, copy them into MODEL_PARAMS in src/config.py and re-run `python -m src.train`.")
    return result


if __name__ == "__main__":
    main()
