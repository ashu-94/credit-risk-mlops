"""
Reason codes — the plain-language "why" behind each decision.

Lenders are legally required to give applicants the main reasons for a decline
(an "adverse action notice"). This module inspects an application and returns
the factors that raised risk (decline reasons) and the factors that lowered it
(approval strengths), ordered by importance.

These rules mirror the same drivers the model learns (verified via SHAP:
debt-to-income and past defaults dominate), so the explanation is faithful to
how the model actually scores.

Usage:
    from src.reasons import explain_decision
    result = explain_decision(application_dict)
"""


def _risk_factors(app: dict) -> list[tuple[int, str]]:
    """Factors that RAISE default risk. Returns (weight, text), higher = stronger."""
    f = []
    dti = app["debt_to_income"]
    cs = app["credit_score"]

    if app["past_defaults"] == 1:
        f.append((5, "History of a previous default"))
    if dti >= 0.5:
        f.append((5, f"Very high debt-to-income ratio ({dti:.2f}) — repayments consume too much income"))
    elif dti >= 0.35:
        f.append((3, f"Elevated debt-to-income ratio ({dti:.2f})"))
    if cs < 600:
        f.append((4, f"Low credit score ({int(cs)})"))
    elif cs < 680:
        f.append((2, f"Below-average credit score ({int(cs)})"))
    if app["num_existing_loans"] >= 4:
        f.append((3, f"High number of existing loans ({app['num_existing_loans']})"))
    if app["employment_years"] < 2:
        f.append((2, f"Short employment history ({app['employment_years']} yr)"))
    # loan size relative to income
    if app["annual_income"] > 0 and app["loan_amount"] / app["annual_income"] > 4:
        f.append((3, "Loan amount is large relative to annual income"))
    if app["loan_purpose"] == "business":
        f.append((1, "Business-purpose loans carry higher historical default rates"))
    if app["home_ownership"] == "RENT":
        f.append((1, "Renting (no property collateral)"))

    return sorted(f, reverse=True)


def _strength_factors(app: dict) -> list[tuple[int, str]]:
    """Factors that LOWER default risk (approval strengths)."""
    f = []
    dti = app["debt_to_income"]
    cs = app["credit_score"]

    if app["past_defaults"] == 0:
        f.append((4, "No history of prior defaults"))
    if cs >= 750:
        f.append((5, f"Strong credit score ({int(cs)})"))
    elif cs >= 700:
        f.append((3, f"Good credit score ({int(cs)})"))
    if dti <= 0.25:
        f.append((4, f"Low debt-to-income ratio ({dti:.2f}) — comfortable repayment capacity"))
    if app["employment_years"] >= 7:
        f.append((3, f"Stable, long employment history ({app['employment_years']} yr)"))
    if app["home_ownership"] in ("OWN", "MORTGAGE"):
        f.append((2, f"Property ownership ({app['home_ownership'].title()})"))
    if app["num_existing_loans"] <= 1:
        f.append((2, "Few or no existing loan obligations"))

    return sorted(f, reverse=True)


def explain_decision(app: dict, probability: float, decision: str, top_n: int = 4) -> dict:
    """
    Build a human-readable review of the decision.
    Returns dict with 'headline', 'primary_factors', and 'other_factors'.
    """
    risks = [t for _, t in _risk_factors(app)]
    strengths = [t for _, t in _strength_factors(app)]

    if decision == "DECLINE":
        headline = (
            f"Application declined. Estimated default probability {probability:.1%} — "
            f"above the acceptable risk threshold. The main reasons:"
        )
        primary = risks[:top_n] or ["Overall risk profile exceeds lending criteria"]
        other = strengths[:2]
        other_label = "Positive factors noted (but outweighed):"
    elif decision == "REVIEW":
        headline = (
            f"Referred for manual review. Estimated default probability {probability:.1%} — "
            f"in the borderline band. Points a reviewer should weigh:"
        )
        primary = risks[:top_n] or ["No single dominant risk, but score is mid-range"]
        other = strengths[:top_n]
        other_label = "Supporting strengths:"
    else:  # APPROVE
        headline = (
            f"Application approved. Estimated default probability {probability:.1%} — "
            f"within acceptable limits. This application was supported by:"
        )
        primary = strengths[:top_n] or ["Overall risk profile within lending criteria"]
        other = risks[:2]
        other_label = "Minor risk factors present (within tolerance):"

    return {
        "headline": headline,
        "primary_label": "Key factors:",
        "primary_factors": primary,
        "other_label": other_label,
        "other_factors": other,
    }
