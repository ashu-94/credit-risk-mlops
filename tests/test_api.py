"""API contract test."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

SAMPLE = {
    "age": 35, "annual_income": 60000, "loan_amount": 200000,
    "loan_term_months": 48, "employment_years": 5, "credit_score": 700,
    "num_existing_loans": 1, "debt_to_income": 0.3, "past_defaults": 0,
    "home_ownership": "MORTGAGE", "loan_purpose": "vehicle",
}

def test_predict_contract():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code in (200, 503)
