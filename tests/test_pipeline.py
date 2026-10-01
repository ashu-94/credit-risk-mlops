"""Smoke tests for data + training pipeline."""
import subprocess, sys, pandas as pd
from src.pipeline import build_full_pipeline
from src import config


def test_data_generation(tmp_path):
    out = tmp_path / "d.csv"
    subprocess.run([sys.executable, "data/generate_data.py", "--n", "500", "--out", str(out)], check=True)
    df = pd.read_csv(out)
    assert len(df) == 500
    assert config.TARGET in df.columns


def test_pipeline_fits_and_predicts():
    import numpy as np
    from data.generate_data import generate
    df = generate(1000)
    pipe = build_full_pipeline()
    pipe.fit(df[config.FEATURES], df[config.TARGET])
    proba = pipe.predict_proba(df[config.FEATURES].head(5))
    assert proba.shape == (5, 2)
    assert np.all((proba >= 0) & (proba <= 1))
