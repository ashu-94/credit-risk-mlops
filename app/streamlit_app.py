"""
Credit Risk Scoring — interactive dashboard.

A fintech risk-desk UI: an application goes in, a calibrated default probability,
risk gauge, model-insight panel, and an explainable decision come out — all from
the real XGBoost pipeline.

Run:  streamlit run app/streamlit_app.py
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src import config
from src.reasons import explain_decision
from src.insights import feature_importance_table, peer_percentiles, risk_band

st.set_page_config(page_title="Credit Risk Scoring", page_icon="🏦", layout="wide")

# ----------------------------- styling -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

.stApp { background: radial-gradient(1200px 600px at 20% -10%, #182034 0%, #0e131f 55%) fixed; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .disp { font-family: 'Space Grotesk', sans-serif; letter-spacing:-0.02em; }

.block-container { padding-top: 4.5rem; max-width: 1180px; }

/* header */
.hero-title { font-family:'Space Grotesk'; font-size: 2.5rem; font-weight:700; color:#f4f6fb;
  display:flex; align-items:center; gap:.6rem; margin:0; }
.hero-sub { color:#8a93a6; font-size:1rem; margin:.25rem 0 0 0; }

/* cards */
.card { background: linear-gradient(180deg, #1a2334 0%, #151d2c 100%);
  border:1px solid #26314a; border-radius:16px; padding:1.4rem 1.5rem;
  box-shadow: 0 10px 30px -18px rgba(0,0,0,.7); }
.card h3 { color:#e8ecf5; font-size:1.05rem; margin:0 0 .2rem 0; }
.card .eyebrow { color:#5f6b85; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; }

/* summary numbers */
.big-prob { font-family:'Space Grotesk'; font-size:3rem; font-weight:700; line-height:1; }
.decision-pill { display:inline-block; padding:.35rem .9rem; border-radius:999px;
  font-weight:600; font-size:.95rem; margin-top:.4rem; }

/* gauge */
.gauge-track { height:12px; border-radius:999px; margin:.8rem 0 .3rem 0;
  background:linear-gradient(90deg,#22c55e 0%,#84cc16 25%,#f59e0b 55%,#ef4444 100%); position:relative; }
.gauge-marker { position:absolute; top:-5px; width:4px; height:22px; border-radius:2px;
  background:#f4f6fb; box-shadow:0 0 0 3px rgba(15,20,30,.85); }
.gauge-scale { display:flex; justify-content:space-between; color:#5f6b85; font-size:.7rem; }

/* inputs */
.stSlider label, .stNumberInput label, .stSelectbox label { color:#aab3c7 !important; font-weight:500; }
.stButton>button { background:#4f7cff; color:white; border:0; border-radius:10px;
  padding:.6rem 1.4rem; font-weight:600; font-family:'Space Grotesk'; }
.stButton>button:hover { background:#3d68f0; }

/* factor lists */
.factor { padding:.4rem .6rem; border-left:3px solid #26314a; margin:.3rem 0;
  background:#131a28; border-radius:0 8px 8px 0; color:#cfd6e6; font-size:.92rem; }
.factor.risk { border-left-color:#ef4444; }
.factor.good { border-left-color:#22c55e; }

.footer { color:#4b556b; font-size:.8rem; margin-top:1.5rem; display:flex; gap:1.6rem; }
.footer a { color:#6b7794; text-decoration:none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(config.MODEL_PATH)

@st.cache_data
def load_reference():
    return pd.read_csv(config.DATA_PATH)


st.markdown(
    '<div><p class="hero-title">🏦 Credit Risk Scoring</p>'
    '<p class="hero-sub">Loan-default prediction — XGBoost pipeline with an explainable decision.</p></div>',
    unsafe_allow_html=True,
)
st.write("")

try:
    model = load_model()
except Exception:
    st.error("Model not found. Run `python -m src.train` in your terminal, then reload.")
    st.stop()

# ----------------------------- input form -----------------------------
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Applicant</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 75, 35)
        annual_income = st.number_input("Annual income", 10000, 1_000_000, 60000, step=5000)
        loan_amount = st.number_input("Loan amount", 10000, 2_000_000, 200000, step=10000)
        loan_term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 72], index=3)
        credit_score = st.slider("Credit score", 300, 900, 680)
    with c2:
        employment_years = st.slider("Employment (years)", 0, 40, 5)
        num_existing_loans = st.slider("Existing loans", 0, 10, 1)
        debt_to_income = st.slider("Debt-to-income", 0.0, 3.0, 0.30, step=0.05)
        past_defaults = st.selectbox("Past defaults", [0, 1])
        home_ownership = st.selectbox("Home ownership", ["OWN", "RENT", "MORTGAGE"])
        loan_purpose = st.selectbox("Loan purpose", ["vehicle", "personal", "business", "home_improvement", "education"])
    st.markdown('</div>', unsafe_allow_html=True)

application = {
    "age": age, "annual_income": annual_income, "loan_amount": loan_amount,
    "loan_term_months": loan_term_months, "employment_years": employment_years,
    "credit_score": credit_score, "num_existing_loans": num_existing_loans,
    "debt_to_income": debt_to_income, "past_defaults": past_defaults,
    "home_ownership": home_ownership, "loan_purpose": loan_purpose,
}

# ----------------------------- model insight (always shown) -----------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="eyebrow">Model Insight</span>', unsafe_allow_html=True)
    st.markdown("<h3>What drives the model</h3>", unsafe_allow_html=True)
    fi = feature_importance_table(model, top_n=5)
    if not fi.empty:
        chart_df = fi.set_index("feature")["importance"]
        st.bar_chart(chart_df, height=230, color="#4f7cff")
        st.caption("Top 5 features by global importance (share of total).")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
scored = st.button("Score application", type="primary")

# ----------------------------- results -----------------------------
if scored:
    proba = float(model.predict_proba(pd.DataFrame([application]))[:, 1][0])
    band, decision, color = risk_band(proba)
    marker_pct = min(max(proba, 0.0), 1.0) * 100

    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Prediction Summary</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-prob" style="color:{color}">{proba:.1%}</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#8a93a6;font-size:.85rem;margin-top:.2rem">estimated default probability</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="gauge-track"><div class="gauge-marker" style="left:calc({marker_pct}% - 2px)"></div></div>'
            '<div class="gauge-scale"><span>Low</span><span>Medium</span><span>High</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="margin-top:.8rem"><span style="color:#8a93a6">Risk band</span> '
            f'<span class="decision-pill" style="background:{color}22;color:{color}">{band}</span></div>'
            f'<div style="margin-top:.5rem"><span style="color:#8a93a6">Decision</span> '
            f'<span class="decision-pill" style="background:{color}22;color:{color}">{decision}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with res_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="eyebrow">Explainable Decision</span>', unsafe_allow_html=True)
        review = explain_decision(application, proba, decision)
        st.markdown(f"<h3 style='font-size:.98rem'>{review['headline']}</h3>", unsafe_allow_html=True)
        cls = "risk" if decision != "APPROVE" else "good"
        for f in review["primary_factors"]:
            st.markdown(f'<div class="factor {cls}">{f}</div>', unsafe_allow_html=True)
        if review["other_factors"]:
            st.markdown(f"<div style='color:#8a93a6;font-size:.8rem;margin-top:.6rem'>{review['other_label']}</div>", unsafe_allow_html=True)
            other_cls = "good" if decision != "APPROVE" else "risk"
            for f in review["other_factors"]:
                st.markdown(f'<div class="factor {other_cls}">{f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- expandable actions (Detailed Report + Compare with Peers) --------
    a1, a2 = st.columns(2)
    with a1:
        with st.expander("📄  View detailed report"):
            st.write("**Full feature importance**")
            st.dataframe(feature_importance_table(model, top_n=10), use_container_width=True, hide_index=True)
            st.write("**Decision thresholds**")
            st.write("• Probability < 20% → **Approve**  \n• 20–50% → **Review**  \n• ≥ 50% → **Decline**")
            st.caption("Thresholds are business-cost tunable — see src/threshold.py for the cost-optimal cutoff.")
    with a2:
        with st.expander("📊  Compare with peers"):
            df = load_reference()
            st.write("Where this applicant sits versus the population:")
            for p in peer_percentiles(application, df):
                arrow = "higher = stronger" if p["higher_is_better"] else "lower = stronger"
                st.markdown(f"**{p['label']}**: {p['value']} — {p['percentile']}th percentile  \n"
                            f"<span style='color:#5f6b85;font-size:.8rem'>({arrow})</span>", unsafe_allow_html=True)
                st.progress(p["percentile"] / 100)

st.markdown(
    '<div class="footer"><span>Credit Risk MLOps</span>'
    '<a href="#">Docs</a><a href="#">Model card</a>'
    '<span>XGBoost · SHAP · FastAPI · MLflow</span></div>',
    unsafe_allow_html=True,
)
