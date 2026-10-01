"""
Pure (Streamlit-free) helpers for the dashboard's insight panels.
Kept separate so they can be unit-tested without a Streamlit runtime.
"""
import numpy as np
import pandas as pd

from src import config

LABELS = {
    "credit_score": "Credit Score",
    "debt_to_income": "Debt-to-Income",
    "past_defaults": "Past Defaults",
    "annual_income": "Annual Income",
    "employment_years": "Employment Years",
    "loan_amount": "Loan Amount",
    "num_existing_loans": "Existing Loans",
    "loan_term_months": "Loan Term",
    "age": "Age",
    "home_ownership": "Home Ownership",
    "loan_purpose": "Loan Purpose",
}


def feature_importance_table(pipeline, top_n: int = 5) -> pd.DataFrame:
    """Aggregate model importances back to human-readable base features."""
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    names = pre.get_feature_names_out()
    agg: dict[str, float] = {}
    for name, w in zip(names, importances):
        key = name.split("__", 1)[1] if "__" in name else name
        for cf in config.CATEGORICAL_FEATURES:
            if key.startswith(cf + "_"):
                key = cf
                break
        agg[key] = agg.get(key, 0.0) + float(w)

    df = pd.DataFrame(
        [(LABELS.get(k, k), v) for k, v in agg.items()],
        columns=["feature", "importance"],
    )
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    total = df["importance"].sum() or 1.0
    df["importance"] = df["importance"] / total  # normalise for display
    return df


def peer_percentiles(app: dict, df: pd.DataFrame) -> list[dict]:
    """Where this applicant sits versus the population, for key features."""
    out = []
    for field in ["credit_score", "annual_income", "debt_to_income", "employment_years"]:
        col = df[field].values
        val = app[field]
        pct = float((col < val).mean() * 100)
        out.append({
            "label": LABELS[field],
            "value": val,
            "percentile": round(pct),
            # for DTI, lower is better — flag interpretation
            "higher_is_better": field != "debt_to_income",
        })
    return out


def risk_band(proba: float):
    if proba < 0.2:
        return "LOW", "APPROVE", "#22c55e"
    if proba < 0.5:
        return "MEDIUM", "REVIEW", "#f59e0b"
    return "HIGH", "DECLINE", "#ef4444"
