"""
Business-cost threshold optimization.

A default 0.5 cutoff is almost never the right business decision. Approving a
borrower who defaults (false negative) costs far more than reviewing a good
borrower (false positive). This module finds the probability threshold that
MINIMIZES expected cost, and shows the trade-off curve.

This is the piece that shows an interviewer you connect a model to money, not
just to accuracy.

Usage:  python -m src.threshold
"""
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.pipeline import load_data, split

# Business assumptions (tune to the real loan economics)
COST_FALSE_NEGATIVE = 5.0   # approved a defaulter — lose ~5x a review's cost
COST_FALSE_POSITIVE = 1.0   # flagged a good borrower for review — minor cost


def expected_cost(y_true, proba, threshold):
    preds = (proba >= threshold).astype(int)
    fn = np.sum((preds == 0) & (y_true == 1))   # missed defaulters
    fp = np.sum((preds == 1) & (y_true == 0))   # unnecessary reviews
    return fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE, fn, fp


def run():
    pipeline = joblib.load(config.MODEL_PATH)
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_test = y_test.values

    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in thresholds:
        c, _, _ = expected_cost(y_test, proba, t)
        costs.append(c)
    costs = np.array(costs)

    best_i = int(np.argmin(costs))
    best_t = float(thresholds[best_i])
    best_cost, fn, fp = expected_cost(y_test, proba, best_t)
    default_cost, _, _ = expected_cost(y_test, proba, 0.5)

    result = {
        "optimal_threshold": round(best_t, 3),
        "cost_at_optimal": float(best_cost),
        "cost_at_0.5": float(default_cost),
        "savings_vs_0.5": float(default_cost - best_cost),
        "assumptions": {"cost_fn": COST_FALSE_NEGATIVE, "cost_fp": COST_FALSE_POSITIVE},
    }
    (config.REPORT_DIR / "threshold_analysis.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, costs, color="#264653")
    ax.axvline(best_t, color="#e76f51", ls="--", label=f"optimal = {best_t:.2f}")
    ax.axvline(0.5, color="#999", ls=":", label="default = 0.50")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("expected cost")
    ax.set_title("Choosing the threshold that minimizes business cost")
    ax.legend()
    fig.tight_layout(); fig.savefig(config.REPORT_DIR / "threshold_curve.png", dpi=110); plt.close(fig)

    print(json.dumps(result, indent=2))
    print("Curve saved to reports/threshold_curve.png")
    return result


if __name__ == "__main__":
    run()
