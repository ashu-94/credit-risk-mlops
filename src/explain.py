"""
Model explainability with SHAP.

Credit models must be explainable — regulators require lenders to state why an
application was declined. This module answers two questions:
  1. Globally: which features drive default risk across all applicants?
  2. Locally: why did THIS applicant get THIS score?

Usage:  python -m src.explain
Falls back to model feature_importances_ if the shap package isn't installed.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.pipeline import load_data, split


def _feature_names(pipeline):
    """Recover human-readable names after the ColumnTransformer."""
    pre = pipeline.named_steps["preprocess"]
    return list(pre.get_feature_names_out())


def global_importance_fallback(pipeline, out):
    """If shap is unavailable, use the model's own importances."""
    model = pipeline.named_steps["model"]
    names = _feature_names(pipeline)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print("[explain] model has no feature_importances_; skipping")
        return
    order = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([names[i] for i in order][::-1], importances[order][::-1], color="#2a9d8f")
    ax.set_title("Top feature importances (model-native)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print("[explain] saved native importances ->", out)


def run():
    pipeline = joblib.load(config.MODEL_PATH)
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    try:
        import shap
        pre = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        X_test_enc = pre.transform(X_test)
        if hasattr(X_test_enc, "toarray"):
            X_test_enc = X_test_enc.toarray()
        names = _feature_names(pipeline)

        # TreeExplainer is fast + exact for tree models (XGBoost / HistGBM)
        explainer = shap.TreeExplainer(model)
        sample = X_test_enc[:500]
        shap_values = explainer.shap_values(sample)

        # Global: summary beeswarm
        shap.summary_plot(shap_values, sample, feature_names=names, show=False)
        plt.tight_layout()
        plt.savefig(config.REPORT_DIR / "shap_global.png", dpi=110, bbox_inches="tight")
        plt.close()

        # Local: one applicant's explanation (waterfall-style bar)
        idx = 0
        contrib = shap_values[idx]
        order = np.argsort(np.abs(contrib))[::-1][:10]
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#e76f51" if contrib[i] > 0 else "#2a9d8f" for i in order]
        ax.barh([names[i] for i in order][::-1], contrib[order][::-1], color=colors[::-1])
        ax.set_title("Why this applicant scored as they did (SHAP, red = raises risk)")
        fig.tight_layout(); fig.savefig(config.REPORT_DIR / "shap_local.png", dpi=110); plt.close(fig)

        print("[explain] SHAP global + local plots saved to reports/")
    except ImportError:
        print("[explain] shap not installed; using model-native importances")
        global_importance_fallback(pipeline, config.REPORT_DIR / "feature_importances.png")


if __name__ == "__main__":
    run()
