"""
FastAPI serving layer for the credit-risk model.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from src import config

app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained. Run `python -m src.train`.")
        _model = joblib.load(config.MODEL_PATH)
    return _model


class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    loan_term_months: int = Field(..., ge=1)
    employment_years: int = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=900)
    num_existing_loans: int = Field(..., ge=0)
    debt_to_income: float = Field(..., ge=0)
    past_defaults: int = Field(..., ge=0, le=1)
    home_ownership: Literal["OWN", "RENT", "MORTGAGE"]
    loan_purpose: Literal["vehicle", "personal", "business", "home_improvement", "education"]


class PredictionResponse(BaseModel):
    default_probability: float
    risk_band: str
    decision: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(app_in: LoanApplication):
    model = get_model()
    row = pd.DataFrame([app_in.model_dump()])
    proba = float(model.predict_proba(row)[:, 1][0])

    if proba < 0.2:
        band, decision = "LOW", "APPROVE"
    elif proba < 0.5:
        band, decision = "MEDIUM", "REVIEW"
    else:
        band, decision = "HIGH", "DECLINE"

    return PredictionResponse(
        default_probability=round(proba, 4),
        risk_band=band,
        decision=decision,
    )
