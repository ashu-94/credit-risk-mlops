"""Central configuration — one place to change paths, columns, and hyperparameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "model.joblib"
REFERENCE_PATH = DATA_PATH  # baseline distribution for drift checks

TARGET = "default"
NUMERIC_FEATURES = [
    "age", "annual_income", "loan_amount", "loan_term_months",
    "employment_years", "credit_score", "num_existing_loans",
    "debt_to_income", "past_defaults",
]
CATEGORICAL_FEATURES = ["home_ownership", "loan_purpose"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42

# XGBoost-style params (works with sklearn HistGradientBoosting fallback too)
MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.9,
    "random_state": RANDOM_STATE,
}

MLFLOW_EXPERIMENT = "credit-risk"
REGISTERED_MODEL_NAME = "credit-risk-classifier"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
