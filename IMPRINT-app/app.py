import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from model_loader import load_model
from preprocessing import prepare_input

st.set_page_config(page_title="IMPRINT Clinical Decision Support", page_icon="🩺", layout="centered")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Clinical guidance")
    st.markdown("""
- **Ultrasound anchors triage**; IMPRINT refines probabilities in borderline cases.  
- **Thresholds** informed by decision curves (10% and 30%).  
- **Postmenopause**: enhanced discrimination; SIMs reduce overtreatment.  
- **STUMP exclusion** can improve performance when suspicion is low.
    """)
    st.header("Acronyms used in Extended model")
    st.markdown("""
- **NLR**: Neutrophil-to-Lymphocyte Ratio  
- **SII**: Systemic Immune-Inflammation Index  
- **PIV**: Pan-Immune-Inflammation Value  
- **CS**: Color Score (1–4)
    """)

st.title("IMPRINT Clinical Decision Support")
st.caption("Enter ultrasound and lab data to estimate malignancy risk (IMPRINT Extended).")

# -----------------------------
# Input form
# -----------------------------
with st.form("imprint_form"):
    st.subheader("Clinical data")
    age_str = st.text_input("Age (years)", "50")
    try:
        age = int(age_str)
    except ValueError:
        age = 50
    menopause = st.checkbox("Postmenopausal", value=False)

    st.subheader("Ultrasound data")
    u1, u2 = st.columns(2)
    with u1:
        diameter_mm = st.number_input("Maximum diameter (mm)", min_value=0, max_value=300, value=90)
        # ✅ mettiamo le due checkbox sulla stessa riga
        c1, c2 = st.columns(2)
        with c1:
            margins_irregular = st.checkbox("Irregular margins", value=False)
        with c2:
            cystic_areas = st.checkbox("Cystic areas", value=False)
    with u2:
        cs = st.slider("Color score (1–4)", min_value=1, max_value=4, value=2)

    st.subheader("Laboratory data (absolute counts, x10^9/L)")
    l1, l2, l3, l4, l5 = st.columns(5)
    with l1: platelets = float(st.text_input("Platelets", "260.0"))
    with l2: neutrophils = float(st.text_input("Neutrophils", "4.7"))
    with l3: lymphocytes = float(st.text_input("Lymphocytes", "1.7"))
    with l4: monocytes = float(st.text_input("Monocytes", "0.4"))
    with l5: eosinophils = float(st.text_input("Eosinophils", "0.1"))

    submitted = st.form_submit_button("Estimate risk")

# -----------------------------
# Compute and display results
# -----------------------------
if submitted:
    try:
        model, predictors, transforms = load_model()
    except Exception as e:
        st.error(f"Model load failed. Error: {e}")
        st.stop()

    df_row = prepare_input(
        age=age, aub=False, pain=False, abdominal_distress=False,
        diameter_mm=diameter_mm, margins_irregular=margins_irregular,
        cystic_areas=cystic_areas, cs=cs,
        platelets=platelets, neutrophils=neutrophils, lymphocytes=lymphocytes,
        monocytes=monocytes, eosinophils=eosinophils,
        transforms=transforms
    )

    df_row["Menopausal_status"] = 1 if menopause else 0

    # Align predictors
    for c in predictors:
        if c not in df_row.columns:
            df_row[c] = 0
    X = df_row[predictors]

    # ➡️ Add constant to match training
    X = sm.add_constant(X, has_constant="add")

    # Predict probability with statsmodels
    proba = float(model.predict(X)[0])

    # Risk bands
    if proba < 0.10:
        band, color, guidance = "Low risk", "#239a3b", "Consider conservative management; monitor or complementary imaging."
    elif proba < 0.30:
        band, color, guidance = "Intermediate risk", "#f59e0b", "Consider MRI or referral; integrate clinical context."
    else:
        band, color, guidance = "High risk", "#dc2626", "Refer to sarcoma center; plan oncologic surgical approach."

    # Display probability in %
    percent = proba * 100
    st.markdown("### Estimated malignancy probability")
    st.markdown(f"<span style='color:{color}; font-size:32px; font-weight:700'>{percent:.1f}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Risk band:** <span style='color:{color}; font-weight:600'>{band}</span>", unsafe_allow_html=True)
    st.info(guidance)

    # Show key markers
    st.markdown("### Key systemic inflammatory markers")
    cutoff = {"NLR": 3.0, "SII": 600.0, "PIV": 400.0}
    markers = {
        "NLR": float(df_row.get("NLR", np.nan)),
        "SII": float(df_row.get("SII", np.nan)),
        "PIV": float(df_row.get("PIV", np.nan)),
    }
    for name, value in markers.items():
        if np.isnan(value):
            st.write(f"{name}: n/a")
        elif value > cutoff[name]:
            st.markdown(f"**{name}:** <span style='color:red'>{value:.2f}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{name}:** {value:.2f}")