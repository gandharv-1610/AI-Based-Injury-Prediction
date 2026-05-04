import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import joblib
from utils.feature_extractor import extract_features
from utils.medical_engine import generate_report

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GaitScan AI — Clinical Pressure Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🦶"
)

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* ── Background ── */
.stApp {
    background: radial-gradient(ellipse at 0% 0%, #0c1445 0%, #0f172a 40%, #0a0a1a 100%);
    min-height: 100vh;
}

/* ── Animated floating orbs ── */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.12;
    animation: float 12s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
.orb1 { width: 500px; height: 500px; background: #6366f1; top: -100px; left: -100px; animation-delay: 0s; }
.orb2 { width: 400px; height: 400px; background: #0ea5e9; bottom: -80px; right: -80px; animation-delay: 4s; }
.orb3 { width: 300px; height: 300px; background: #8b5cf6; top: 50%; left: 50%; animation-delay: 8s; }

@keyframes float {
    0%, 100% { transform: translateY(0) scale(1); }
    50%       { transform: translateY(-30px) scale(1.05); }
}

/* ── Typography ── */
h1, h2, h3 { color: #f8fafc !important; font-weight: 800; letter-spacing: -0.5px; }
h1 { font-size: 2.4rem !important; }

.gradient-text {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
}

/* ── Risk severity borders ── */
.risk-low      { border-left: 5px solid #10b981; }
.risk-moderate { border-left: 5px solid #f59e0b; }
.risk-high     { border-left: 5px solid #ef4444; }
.risk-critical { border-left: 5px solid #9333ea; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    animation: pulseBadge 2.5s ease-in-out infinite;
}
@keyframes pulseBadge {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,255,255,0.1); }
    50%       { box-shadow: 0 0 18px 6px rgba(255,255,255,0.07); }
}
.badge-low      { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b981; }
.badge-moderate { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b; }
.badge-high     { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef4444; }
.badge-critical { background: rgba(147,51,234,0.15); color: #9333ea; border: 1px solid #9333ea; }

/* ── Risk gauge ring ── */
.gauge-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
}
.gauge-ring {
    position: relative;
    width: 200px;
    height: 200px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1rem;
}
.gauge-ring::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    padding: 12px;
    background: conic-gradient(VAR_COLOR VAR_DEG, rgba(255,255,255,0.06) 0deg);
    -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 22px), black calc(100% - 22px));
    mask: radial-gradient(farthest-side, transparent calc(100% - 22px), black calc(100% - 22px));
    animation: spinIn 1.2s cubic-bezier(0.25,0.46,0.45,0.94) both;
}
@keyframes spinIn {
    from { opacity: 0; transform: rotate(-90deg) scale(0.8); }
    to   { opacity: 1; transform: rotate(0deg)  scale(1); }
}
.gauge-score {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
}
.gauge-label {
    font-size: 0.8rem;
    color: #64748b;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Insight cards ── */
.insight-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.06);
    transition: background 0.2s;
}
.insight-item:hover { background: rgba(255,255,255,0.06); }
.insight-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 1px; }
.insight-text { color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; }

/* ── Exercise cards ── */
.ex-card {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.ex-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(56,189,248,0.12); }
.ex-name { color: #38bdf8; font-weight: 700; font-size: 1rem; margin-bottom: 4px; }
.ex-desc { color: #94a3b8; font-size: 0.88rem; margin-bottom: 6px; }
.ex-sets { color: #818cf8; font-size: 0.82rem; font-weight: 600; }

/* ── Metrics ── */
div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.88rem !important;
    color: #64748b !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    padding: 10px 24px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ── Info/warning boxes ── */
.info-box {
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(14,165,233,0.25);
    border-radius: 12px;
    padding: 14px 18px;
    color: #7dd3fc;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    line-height: 1.6;
}
.warning-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 12px;
    padding: 14px 18px;
    color: #fcd34d;
    font-size: 0.88rem;
    margin-bottom: 1rem;
}
.emergency-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 12px;
    padding: 16px 20px;
    color: #fca5a5;
    font-size: 0.9rem;
    margin-bottom: 1rem;
}

/* ── Section header ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

/* ── Confidence bars ── */
.conf-bar-wrap { margin-bottom: 8px; }
.conf-label { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.85rem; color: #94a3b8; }
.conf-bar-bg { background: rgba(255,255,255,0.07); border-radius: 6px; height: 8px; overflow: hidden; }
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    transition: width 1s ease;
}
</style>

<!-- Floating orbs -->
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Gauge HTML
# ─────────────────────────────────────────────────────────────────────────────
def _gauge_html(score: int, color: str) -> str:
    deg    = int(360 * score / 100)
    return f"""
    <div class="gauge-container">
      <div class="gauge-ring" style="
          background: conic-gradient({color} {deg}deg, rgba(255,255,255,0.05) {deg}deg);
          border-radius: 50%;
          padding: 14px;
          mask: radial-gradient(farthest-side, transparent calc(100% - 24px), black calc(100% - 24px));
          -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 24px), black calc(100% - 24px));
          width:200px; height:200px;
          display:flex; align-items:center; justify-content:center;">
        <div style="text-align:center;">
          <div class="gauge-score" style="color:{color};">{score}</div>
          <div class="gauge-label" style="color:#64748b;">/ 100</div>
        </div>
      </div>
    </div>"""


def _badge_html(category: str, color: str) -> str:
    cls = f"badge-{category.lower()}"
    return f'<span class="risk-badge {cls}">{category} Risk</span>'


def _confidence_bars_html(labels, probs) -> str:
    bars = ""
    for lbl, p in zip(labels, probs):
        pct = p * 100
        bars += f"""
        <div class="conf-bar-wrap">
          <div class="conf-label"><span>{lbl.title()}</span><span>{pct:.1f}%</span></div>
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct:.1f}%; background: linear-gradient(90deg, #38bdf8, #818cf8);"></div>
          </div>
        </div>"""
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    dl_path = 'models/pressure_dl_model.pth'
    rf_path = 'models/pressure_model.pkl'

    if os.path.exists(dl_path):
        try:
            import torch
            from utils.dl_model import PressureCNN
            model = PressureCNN(num_classes=6)
            state = torch.load(dl_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            return model, "DL"
        except Exception as e:
            # Old weights are incompatible with new architecture — remove stale file
            # and fall back to RF model. Re-run train_dl.py to generate new weights.
            import os as _os
            _os.remove(dl_path)
            st.warning(
                f"⚠️ Saved DL model is incompatible with the new ResNet architecture "
                f"and has been removed. Falling back to Random Forest model. "
                f"**Run `python train_dl.py` to train the new model.** ({e})",
                icon="🔄"
            )

    if os.path.exists(rf_path):
        try:
            return joblib.load(rf_path), "RF"
        except Exception as e:
            st.warning(f"⚠️ Could not load RF model: {e}", icon="🔄")
    return None, None


model, model_type = load_ml_model()

LABELS = ['normal', 'antalgic', 'lurching', 'steppage', 'stiff-legged', 'trendelenburg']

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="gradient-text" style="font-size:1.6rem;font-weight:800;margin-bottom:0;">GaitScan AI</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;font-size:0.85rem;margin-top:0;">Clinical Footstrike Diagnostics</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**📂 Upload Pressure Data**")
    st.caption("Standard 128 × 48 pressure matrix CSV")
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    st.divider()
    st.markdown("**👤 Patient Profile**")
    p_name   = st.text_input("Full Name",     "John Doe")
    col_a, col_b = st.columns(2)
    p_age    = col_a.number_input("Age",      1, 120, 30)
    p_sex    = col_b.selectbox("Sex",         ["Male", "Female"])
    p_height = st.number_input("Height (cm)", 50.0, 250.0, 170.0, step=0.5)
    p_weight = st.number_input("Weight (kg)", 10.0, 300.0, 70.0, step=0.5)

    st.divider()
    if model_type:
        st.success(f"🟢  Model Active — **{model_type}** backend")
    else:
        st.error("⚠️  No trained model found. Run `train_dl.py` first.")

# ─────────────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1>🦶 GaitScan <span class="gradient-text">AI Diagnostics</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b;font-size:1.05rem;margin-top:-10px;">Clinical Plantar Pressure Analysis • Deep Learning Gait Pathology Detection</p>', unsafe_allow_html=True)

if uploaded_file is None:
    # ── Landing State ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)
    cards = [
        ("🧠", "AI-Powered Classification", "ResNet CNN trained on 1,440 pressure samples across 6 gait pathologies"),
        ("🏥", "Clinical Medical Reports",   "Evidence-informed risk scoring, anatomical insights, and specialist referrals"),
        ("📊", "3D Biomechanics",            "Interactive heatmaps, 3D topographic surfaces, and foot-zone analysis"),
    ]
    for col, (icon, title, desc) in zip(cols, cards):
        col.markdown(f"""
        <div class="glass-card" style="text-align:center;">
          <div style="font-size:2.5rem;margin-bottom:0.5rem;">{icon}</div>
          <div style="color:#f1f5f9;font-weight:700;font-size:1.05rem;margin-bottom:8px;">{title}</div>
          <div style="color:#64748b;font-size:0.88rem;line-height:1.5;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-box">⬅️  Upload a <strong>128×48 pressure matrix CSV</strong> using the sidebar to begin analysis. Sample files are available in the <code>Pressure_Data/</code> folder.</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Load & Validate Matrix
# ─────────────────────────────────────────────────────────────────────────────
try:
    matrix = np.loadtxt(uploaded_file, delimiter=',')
    if matrix.ndim != 2:
        st.error("❌  Format Error: Expected a 2D pressure matrix.")
        st.stop()
except Exception as e:
    st.error(f"❌  Read Error: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────
features = extract_features(matrix)

# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
probabilities = None

if model and model_type == "DL":
    import torch
    # Normalise per-sample (same as training)
    mn, mx = matrix.min(), matrix.max()
    norm_matrix = (matrix - mn) / (mx - mn + 1e-8)

    tensor = torch.tensor(norm_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_idx      = int(np.argmax(probs))
    prediction    = LABELS[pred_idx]
    probabilities = probs
    confidence    = float(probs[pred_idx])

elif model and model_type == "RF":
    try:
        # RF was trained on original 6 features — use only those columns
        RF_FEATURES = ['total_pressure', 'balance', 'cop_x', 'cop_y', 'max_pressure', 'variance']
        rf_available = [f for f in RF_FEATURES if f in features]
        feature_df = pd.DataFrame([{k: features[k] for k in rf_available}])
        prediction = model.predict(feature_df)[0]
        confidence = 1.0
        if hasattr(model, 'predict_proba'):
            rf_probs      = model.predict_proba(feature_df)[0]
            probabilities = rf_probs
            confidence    = float(max(rf_probs))
    except Exception as e:
        st.warning(
            f"⚠️ Random Forest model failed: {e}. "
            f"Run `python train.py` and/or `python train_dl.py` to rebuild both models.",
            icon="🔄"
        )
        prediction = "unknown"
        confidence = 0.0
else:
    prediction  = "unknown"
    confidence  = 0.0

# If no model could run, show a clear message and stop
if prediction == "unknown" and (model is None or model_type is None):
    st.error(
        "❌ **No functional model available.** "
        "Both the DL and RF models are missing or incompatible.\n\n"
        "**To fix this:**\n"
        "1. Open a terminal in the project directory\n"
        "2. Run: `python train_dl.py`  (trains the new ResNet model — ~15–30 min)\n"
        "3. Refresh this page after training completes"
    )
    st.stop()

report = generate_report(
    gait_type  = prediction,
    features   = features,
    age        = p_age,
    weight_kg  = p_weight,
    height_cm  = p_height,
    confidence = confidence,
)

risk_css = report.risk_category.lower()  # low / moderate / high / critical

# ─────────────────────────────────────────────────────────────────────────────
# BMI & BMR
# ─────────────────────────────────────────────────────────────────────────────
bmi = p_weight / ((p_height / 100) ** 2)
if p_sex == "Male":
    bmr = (10 * p_weight) + (6.25 * p_height) - (5 * p_age) + 5
else:
    bmr = (10 * p_weight) + (6.25 * p_height) - (5 * p_age) - 161

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Clinical Report",
    "🔬 Biomechanics",
    "📊 Data Analysis",
    "💊 Medical Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Clinical Report
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Patient Header ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="glass-card">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:1rem;">
        <div style="width:54px;height:54px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#38bdf8);
                    display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">👤</div>
        <div>
          <div style="color:#f1f5f9;font-size:1.25rem;font-weight:700;">{p_name}</div>
          <div style="color:#475569;font-size:0.88rem;">{p_sex} &nbsp;·&nbsp; {p_age} yrs &nbsp;·&nbsp; {p_height:.0f} cm &nbsp;·&nbsp; {p_weight:.1f} kg &nbsp;·&nbsp; BMI {bmi:.1f}</div>
        </div>
        <div style="margin-left:auto;text-align:right;">
          <div style="color:#475569;font-size:0.78rem;letter-spacing:1px;text-transform:uppercase;">Backend</div>
          <div style="color:#38bdf8;font-weight:700;">{model_type or "—"} Model</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Risk Gauge + Primary Result ───────────────────────────────────────────
    c_gauge, c_result = st.columns([1, 2])

    with c_gauge:
        st.markdown(_gauge_html(report.injury_risk_score, report.risk_color), unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;">{_badge_html(report.risk_category, report.risk_color)}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center;color:#475569;font-size:0.78rem;margin-top:8px;">Injury Risk Score</p>', unsafe_allow_html=True)

    with c_result:
        st.markdown(f"""
        <div class="glass-card risk-{risk_css}" style="height:100%;box-sizing:border-box;">
          <p class="section-header">AI Gait Classification</p>
          <h2 style="margin:0;font-size:2rem;color:#f8fafc;">{prediction.upper()}</h2>
          <p style="color:#94a3b8;font-size:1rem;margin-top:4px;">{report.primary_condition}</p>
          <hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:12px 0;">
          <p style="color:#64748b;font-size:0.88rem;margin-bottom:6px;">
            Model Confidence: <strong style="color:#38bdf8;">{confidence*100:.1f}%</strong>
            &nbsp;·&nbsp; L/R Asymmetry: <strong style="color:#f59e0b;">{features['balance']*100:.1f}%</strong>
          </p>
          <p style="color:#94a3b8;font-size:0.9rem;line-height:1.6;margin:0;">{report.short_description}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Biometric Metrics Row ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BMI",         f"{bmi:.1f}")
    m2.metric("BMR",         f"{bmr:.0f} kcal")
    m3.metric("Peak Pressure", f"{features['max_pressure']:.1f} N")
    m4.metric("Contact Area",  f"{features['contact_area']*100:.1f}%")

    st.divider()

    # ── Clinical Insights ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Clinical Insights</p>', unsafe_allow_html=True)
    icons = ["🔍", "⚠️", "📍", "🧬", "📋"]
    for i, insight in enumerate(report.clinical_insights):
        icon = icons[i % len(icons)]
        st.markdown(f"""
        <div class="insight-item">
          <span class="insight-icon">{icon}</span>
          <span class="insight-text">{insight}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Referral + Urgency ────────────────────────────────────────────────────
    c_ref, c_urg = st.columns(2)
    with c_ref:
        st.markdown(f"""
        <div class="glass-card" style="padding:1.2rem 1.5rem;">
          <p class="section-header" style="margin-bottom:6px;">Specialist Referral</p>
          <p style="color:#f1f5f9;font-weight:600;font-size:0.95rem;margin:0;">{report.specialist_referral}</p>
        </div>""", unsafe_allow_html=True)
    with c_urg:
        st.markdown(f"""
        <div class="glass-card" style="padding:1.2rem 1.5rem;">
          <p class="section-header" style="margin-bottom:6px;">Follow-Up Urgency</p>
          <p style="color:#f59e0b;font-weight:600;font-size:0.95rem;margin:0;">{report.follow_up_urgency}</p>
        </div>""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(f'<div class="warning-box">{report.disclaimer}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Biomechanics
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col2a, col2b = st.columns(2)

    with col2a:
        st.subheader("2D Pressure Heatmap")
        st.caption("Plantar pressure distribution — light = high pressure")

        fig_2d = px.imshow(
            matrix, color_continuous_scale='Plasma', aspect='auto',
            labels=dict(color="Pressure (N)")
        )
        fig_2d.add_trace(go.Scatter(
            x=[features['cop_y']], y=[features['cop_x']],
            mode='markers+text',
            marker=dict(color='white', size=14, symbol='x-thin', line=dict(width=3, color='white')),
            text=["COP"], textposition="top right",
            textfont=dict(color='white', size=11),
            name='Centre of Pressure'
        ))
        fig_2d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), margin=dict(l=0,r=0,b=0,t=20), height=480,
            coloraxis_colorbar=dict(tickfont=dict(color='white'), title=dict(font=dict(color='white')))
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    with col2b:
        st.subheader("3D Topographic Surface")
        st.caption("Interactive 3D pressure topology — drag to rotate")

        fig_3d = go.Figure(data=[go.Surface(
            z=matrix, colorscale='Plasma',
            lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.6, specular=0.8, fresnel=0.4),
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor='white', project_z=True))
        )])
        fig_3d.update_layout(
            height=480, autosize=True,
            scene=dict(
                xaxis=dict(title='Width (cols)', showbackground=False, gridcolor='rgba(255,255,255,0.06)'),
                yaxis=dict(title='Length (rows)', showbackground=False, gridcolor='rgba(255,255,255,0.06)'),
                zaxis=dict(title='Pressure (N)', showbackground=False, gridcolor='rgba(255,255,255,0.06)'),
                camera=dict(eye=dict(x=-1.4, y=-1.4, z=1.1)),
                bgcolor='rgba(0,0,0,0)',
            ),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
            margin=dict(l=0,r=0,b=0,t=20)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # ── Foot Zone Pressure Breakdown ──────────────────────────────────────────
    st.subheader("Foot Zone Pressure Distribution")
    zone_cols = st.columns(3)

    zones = [
        ("🦶 Forefoot",  features['forefoot_ratio'] * 100, "#38bdf8"),
        ("🔷 Midfoot",   features['midfoot_ratio']  * 100, "#818cf8"),
        ("👟 Heel",      features['heel_ratio']     * 100, "#c084fc"),
    ]
    for col, (name, pct, clr) in zip(zone_cols, zones):
        col.markdown(f"""
        <div class="glass-card" style="text-align:center;padding:1.5rem;">
          <div style="font-size:1.8rem;margin-bottom:4px;">{name.split()[0]}</div>
          <div style="color:{clr};font-size:2.2rem;font-weight:800;">{pct:.1f}%</div>
          <div style="color:#64748b;font-size:0.85rem;">{name.split()[1]}</div>
        </div>""", unsafe_allow_html=True)

    # ── Left / Right Distribution Donut ──────────────────────────────────────
    st.subheader("Bilateral Load Distribution")
    c_donut, c_bar = st.columns(2)

    with c_donut:
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Left Foot', 'Right Foot'],
            values=[features['left_ratio'] * 100, features['right_ratio'] * 100],
            hole=0.65,
            marker=dict(colors=['#3b82f6', '#f43f5e'],
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            textinfo='label+percent',
            textfont=dict(color='white', size=12),
            hovertemplate='%{label}: %{value:.1f}%<extra></extra>'
        )])
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
            showlegend=True, height=300, margin=dict(l=0,r=0,b=0,t=10),
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c_bar:
        midpoint = matrix.shape[1] // 2
        left_sum  = float(np.sum(matrix[:, :midpoint]))
        right_sum = float(np.sum(matrix[:, midpoint:]))
        fig_bar = px.bar(
            pd.DataFrame({'Foot': ['Left', 'Right'], 'Pressure': [left_sum, right_sum]}),
            x='Foot', y='Pressure', color='Foot',
            color_discrete_map={'Left': '#3b82f6', 'Right': '#f43f5e'},
            text_auto='.2s',
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), showlegend=False,
            height=300, margin=dict(l=0,r=0,b=0,t=10)
        )
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Confidence Distribution")
    st.caption("Softmax probability across all 6 gait classes")

    if probabilities is not None:
        sorted_pairs = sorted(zip(LABELS, probabilities), key=lambda x: -x[1])
        s_labels, s_probs = zip(*sorted_pairs)
        fig_conf = go.Figure(go.Bar(
            x=list(s_probs),
            y=[l.title() for l in s_labels],
            orientation='h',
            marker=dict(
                color=list(s_probs),
                colorscale='Viridis',
                showscale=False,
                line=dict(width=0)
            ),
            text=[f"{p*100:.1f}%" for p in s_probs],
            textposition='outside',
            textfont=dict(color='white', size=11),
        ))
        fig_conf.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'), height=340, margin=dict(l=0,r=60,b=0,t=10),
            xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.15]),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("Confidence breakdown unavailable for this model type.")

    st.divider()

    # ── Extended Feature Metrics ──────────────────────────────────────────────
    st.subheader("Biomechanical Feature Metrics")
    feat_col1, feat_col2 = st.columns(2)
    feat_items = list(features.items())
    half = len(feat_items) // 2
    for label, val in feat_items[:half]:
        feat_col1.metric(label.replace('_', ' ').title(), f"{val:.4f}")
    for label, val in feat_items[half:]:
        feat_col2.metric(label.replace('_', ' ').title(), f"{val:.4f}")

    st.divider()

    # ── Raw Matrix Viewer ─────────────────────────────────────────────────────
    with st.expander("🔢 Expand Raw 128×48 Pressure Matrix"):
        st.dataframe(pd.DataFrame(matrix), use_container_width=True)

    # ── Download Report ───────────────────────────────────────────────────────
    report_json = json.dumps({
        "patient":          {"name": p_name, "age": p_age, "sex": p_sex,
                             "height_cm": p_height, "weight_kg": p_weight, "bmi": round(bmi, 2)},
        "gait_type":        report.gait_type,
        "risk_score":       report.injury_risk_score,
        "risk_category":    report.risk_category,
        "condition":        report.primary_condition,
        "affected_regions": report.affected_regions,
        "description":      report.short_description,
        "insights":         report.clinical_insights,
        "recommendations":  report.recommendations,
        "exercises":        report.exercises,
        "referral":         report.specialist_referral,
        "urgency":          report.follow_up_urgency,
        "prognosis":        report.prognosis,
        "features":         {k: round(v, 6) for k, v in features.items()},
        "confidence":       round(confidence, 4),
        "disclaimer":       report.disclaimer,
    }, indent=2)

    st.download_button(
        label      = "⬇️  Download Clinical Report (JSON)",
        data       = report_json,
        file_name  = f"gaitscan_report_{p_name.replace(' ','_').lower()}.json",
        mime       = "application/json",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Medical Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    # ── Condition Overview ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="glass-card risk-{risk_css}">
      <p class="section-header">Detected Condition</p>
      <h2 style="margin:0 0 8px 0;font-size:1.8rem;">{report.primary_condition}</h2>
      <p style="color:#94a3b8;font-size:0.95rem;line-height:1.7;margin:0;">{report.short_description}</p>
    </div>""", unsafe_allow_html=True)

    # ── Affected Regions ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Affected Anatomical Regions</p>', unsafe_allow_html=True)
    reg_html = " &nbsp; ".join(
        f'<span style="background:rgba(99,102,241,0.15);color:#818cf8;border:1px solid rgba(99,102,241,0.3);'
        f'border-radius:30px;padding:5px 16px;font-size:0.85rem;font-weight:600;">{r}</span>'
        for r in report.affected_regions
    )
    st.markdown(f'<div style="margin-bottom:1.5rem;">{reg_html}</div>', unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Clinical Recommendations</p>', unsafe_allow_html=True)
    rec_icons = ["✅", "💊", "🩹", "👟", "🚫"]
    for i, rec in enumerate(report.recommendations):
        ico = rec_icons[i % len(rec_icons)]
        st.markdown(f"""
        <div class="insight-item">
          <span class="insight-icon">{ico}</span>
          <span class="insight-text">{rec}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Exercises ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Prescribed Physiotherapy Exercises</p>', unsafe_allow_html=True)
    for ex in report.exercises:
        st.markdown(f"""
        <div class="ex-card">
          <div class="ex-name">🏋️ {ex['name']}</div>
          <div class="ex-desc">{ex['desc']}</div>
          <div class="ex-sets">📅 {ex['sets']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prognosis ─────────────────────────────────────────────────────────────
    c_prog, c_blank = st.columns([2, 1])
    with c_prog:
        st.markdown(f"""
        <div class="glass-card">
          <p class="section-header">Prognosis</p>
          <p style="color:#a3e635;font-size:0.95rem;line-height:1.6;margin:0;">{report.prognosis}</p>
        </div>""", unsafe_allow_html=True)

    # ── Emergency Warning ─────────────────────────────────────────────────────
    if report.risk_category in ("High", "Critical"):
        st.markdown("""
        <div class="emergency-box">
          <strong>🚨 When to Seek Emergency Care</strong><br>
          Seek immediate medical attention if you experience: sudden loss of motor function,
          severe unilateral limb weakness, loss of bladder/bowel control, or acute inability
          to bear weight. These may indicate a neurological emergency requiring urgent intervention.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
          ℹ️ Your current risk level does not indicate an emergency. Continue monitoring with
          regular follow-ups and adhere to the recommended exercise programme.
        </div>""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(f'<div class="warning-box" style="margin-top:1rem;">{report.disclaimer}</div>', unsafe_allow_html=True)
