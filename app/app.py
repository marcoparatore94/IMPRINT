import os
import sys
import streamlit as st
import numpy as np
from joblib import load

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.info_panel import get_info_html

# --- Page config ---
st.set_page_config(page_title="IMPRINT Risk Calculator", page_icon="🧮", layout="centered")

# --- Central login box ---
st.title("🔒 IMPRINT access")
PASSWORD = "Imprintfpg2025"
password = st.text_input("Enter the password", type="password")
if password != PASSWORD:
    st.warning("Enter the password to access the application.")
    st.stop()

# --- App title ---
st.title("IMPRINT malignancy risk calculator")

# --- Model path ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "imprint_risk_model.joblib"))

# --- Info panel ---
with st.expander("Information and cut-offs"):
    st.markdown(get_info_html(), unsafe_allow_html=True)

# --- Form (rows: clinical, ultrasound; lab separated) ---
with st.form("risk_form"):
    # Row 1: Clinical data
    st.subheader("Clinical data")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=15, max_value=95, value=50)
    with c2:
        diameter = st.number_input("Target diameter (mm)", min_value=10, max_value=250, value=75)

    # Row 2: Ultrasound data
    st.subheader("Ultrasound data")
    u1, u2, u3 = st.columns(3)
    with u1:
        margins = st.selectbox("Margins", ["Regular", "Irregular"])
    with u2:
        color_doppler = st.selectbox("Color Doppler (score)", ["Absent/1-3", "4"])
    with u3:
        acoustic_shadows = st.selectbox("Acoustic shadows", ["Present", "Absent"])

    # Separate section: Laboratory data
    st.subheader("Laboratory markers")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        neutrophils = st.number_input("Neutrophils", min_value=0.0, max_value=30.0, value=4.0)
    with l2:
        lymphocytes = st.number_input("Lymphocytes", min_value=0.0, max_value=10.0, value=2.0)
    with l3:
        monocytes = st.number_input("Monocytes", min_value=0.0, max_value=5.0, value=0.5)
    with l4:
        platelets = st.number_input("Platelets", min_value=10.0, max_value=1000.0, value=250.0)

    submitted = st.form_submit_button("Compute risk")

# --- Helper functions ---
def classify_risk(p: float) -> str:
    if p < 0.004:
        return "Low"
    elif p < 0.023:
        return "Intermediate"
        # 0.4% and 2.3% thresholds aligned with prior app logic
    else:
        return "High"

def risk_color(cls: str) -> str:
    return "green" if cls == "Low" else ("orange" if cls == "Intermediate" else "red")

# Youden cut-offs for visualization (from latest analysis)
CUTOFFS = {
    "NLR": 2.39,
    "SII": 655.08,
    "SIRI": 0.90,
    "PIV": 216.42,
}

# --- Compute and predict ---
if submitted:
    # Derived hematologic markers (safe divide)
    if lymphocytes and lymphocytes > 0:
        NLR = neutrophils / lymphocytes
        SII = (neutrophils * platelets) / lymphocytes
        SIRI = (neutrophils * monocytes) / lymphocytes
        PIV = (neutrophils * platelets * monocytes) / lymphocytes
    else:
        NLR = SII = SIRI = PIV = 0.0

    # Ultrasound and diameter encoding (aligned with current deployed model)
    diameter_gt8 = 1 if diameter > 80 else 0
    margins_irregular = 1 if margins == "Irregular" else 0
    color4 = 1 if color_doppler == "4" else 0
    shadows_absent = 1 if acoustic_shadows == "Absent" else 0

    # Input vector (keep order consistent with the current model)
    # ["NLR","SII","SIRI","PIV","Age","diameter_gt8","Margini_irregolari","Color4","Ombre_assenti"]
    x = np.array([[NLR, SII, SIRI, PIV,
                   age, diameter_gt8,
                   margins_irregular, color4, shadows_absent]])

    # Load model
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Check path: data/imprint_risk_model.joblib")
        st.stop()

    model = load(MODEL_PATH)

    # Predict probability
    try:
        p = float(model.predict_proba(x)[0, 1])
    except Exception as e:
        st.error("Prediction error. Ensure feature order matches the trained model.")
        st.caption(str(e))
        st.stop()

    cls = classify_risk(p)
    color = risk_color(cls)

    # --- Risk output ---
    st.markdown(f"### Malignancy probability: {p:.3%}")
    st.progress(min(int(p * 100), 100))
    st.markdown(
        f"### Risk class: <span style='color:{color}'>{cls}</span>",
        unsafe_allow_html=True
    )

    # --- Numeric summary of markers vs cut-offs ---
    st.subheader("Markers vs cut-offs")
    st.write("Each marker is shown separately with its Youden cut-off.")

    import matplotlib.pyplot as plt

    markers = {
        "NLR": (NLR, CUTOFFS["NLR"]),
        "SII": (SII, CUTOFFS["SII"]),
        "SIRI": (SIRI, CUTOFFS["SIRI"]),
        "PIV": (PIV, CUTOFFS["PIV"]),
    }

    # Separate boxes: one chart per marker
    for name, (val, cutoff) in markers.items():
        with st.container():
            st.markdown(f"**{name}**: {val:.2f} (cut-off {cutoff:.2f})")

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar([name], [val], color="#4C78A8", alpha=0.85)
            ax.hlines(y=cutoff, xmin=-0.5, xmax=0.5, colors="red", linestyles="--", linewidth=1.5)
            ax.text(0, cutoff, f"Cut-off {cutoff:.2f}", va="bottom", ha="center", fontsize=9)

            status = "↑ above" if val >= cutoff else "↓ below"
            ax.text(0, val, f"{val:.2f}\n({status})", ha="center", va="bottom", fontsize=9)

            ax.set_ylim(bottom=0)
            ax.set_ylabel("Value")
            ax.set_title(f"{name} vs cut-off")
            st.pyplot(fig)

    st.caption("Model trained on center data. Clinical validation ongoing.")
