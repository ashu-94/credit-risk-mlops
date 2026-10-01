"""Data loading, splitting, and the preprocessing + model pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split(df: pd.DataFrame):
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def build_model():
    """Prefer XGBoost; fall back to sklearn's HistGradientBoosting if unavailable."""
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=config.MODEL_PARAMS["n_estimators"],
            learning_rate=config.MODEL_PARAMS["learning_rate"],
            max_depth=config.MODEL_PARAMS["max_depth"],
            subsample=config.MODEL_PARAMS["subsample"],
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=6,  # class imbalance: ~default rate
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            learning_rate=config.MODEL_PARAMS["learning_rate"],
            max_depth=config.MODEL_PARAMS["max_depth"],
            max_iter=config.MODEL_PARAMS["n_estimators"],
            random_state=config.RANDOM_STATE,
        )
    return clf


def build_full_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", build_model()),
    ])
