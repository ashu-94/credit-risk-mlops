"""
Generate a realistic synthetic credit-risk dataset.

Offline + reproducible so the pipeline runs anywhere. In production you would
swap this for a real source (Home Credit Default Risk, LendingClub, or your own
loan book). The schema below mirrors a typical vehicle/consumer loan book —
close to the Shriram Transport domain.

Usage:  python data/generate_data.py --n 20000 --out data/credit.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 65, n)
    income = rng.lognormal(mean=11.0, sigma=0.5, size=n).round(0)          # annual income
    loan_amount = rng.lognormal(mean=11.5, sigma=0.6, size=n).round(0)
    loan_term = rng.choice([12, 24, 36, 48, 60, 72], n)
    employment_years = rng.integers(0, 35, n)
    credit_score = rng.normal(650, 90, n).clip(300, 900).round(0)
    num_existing_loans = rng.poisson(1.2, n)
    dti = (loan_amount / loan_term) / (income / 12 + 1)                    # debt-to-income proxy
    dti = dti.clip(0, 5).round(3)
    home_ownership = rng.choice(["OWN", "RENT", "MORTGAGE"], n, p=[0.35, 0.4, 0.25])
    loan_purpose = rng.choice(
        ["vehicle", "personal", "business", "home_improvement", "education"],
        n, p=[0.4, 0.25, 0.15, 0.1, 0.1],
    )
    past_defaults = rng.binomial(1, 0.12, n)

    # Latent risk score -> default probability (kept interpretable on purpose)
    z = (
        -3.0
        + 2.2 * dti
        + 1.5 * past_defaults
        + 0.8 * (num_existing_loans > 3)
        - 0.004 * (credit_score - 650)
        - 0.03 * employment_years
        + 0.4 * (loan_purpose == "business")
        + 0.3 * (home_ownership == "RENT")
        - 0.000002 * (income - 60000)
    )
    prob_default = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, prob_default.clip(0.01, 0.95))

    df = pd.DataFrame({
        "age": age,
        "annual_income": income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term,
        "employment_years": employment_years,
        "credit_score": credit_score,
        "num_existing_loans": num_existing_loans,
        "debt_to_income": dti,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "past_defaults": past_defaults,
        "default": default,          # target: 1 = defaulted
    })
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--out", type=str, default="data/credit.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)
    rate = df["default"].mean()
    print(f"Wrote {len(df):,} rows to {args.out}  |  default rate = {rate:.1%}")
