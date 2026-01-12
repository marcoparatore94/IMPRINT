import streamlit as st
import numpy as np
import math

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="IMPRINT Risk Calculator",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------------
# DATI E COEFFICIENTI (ESTRATTI DAI FILE UPLOADATI)
# ---------------------------------------------------------

# 1. PARAMETRI DI STANDARDIZZAZIONE (Supplementary Table 1)
# Format: 'Variable': {'mean': float, 'sd': float, 'log': bool}
Z_PARAMS = {
    'diameter': {'mean': 4.377, 'sd': 0.466, 'log': True},  # ln(Diameter)
    'cs':       {'mean': 2.500, 'sd': 0.894, 'log': False}, # Raw
    'nlr':      {'mean': 0.917, 'sd': 0.584, 'log': True},  # ln(NLR)
    'mlr':      {'mean': -1.512,'sd': 0.385, 'log': True},  # ln(MLR)
    'piv':      {'mean': 5.537, 'sd': 0.975, 'log': True}   # ln(PIV)
}

# 2. COEFFICIENTI IMPRINT EXTENDED (Intercept: -1.0345)
COEFF_EXTENDED = {
    'intercept': -1.034533,
    'menopause': 1.731603,
    'irregular_margins': 2.044660,
    'cystic_areas': 0.613274,   # Nota: Cystic Areas è nel modello Extended
    'z_cs': 0.608670,
    'z_diameter': 0.509756,
    'z_nlr': -0.739700,         # Coefficiente negativo (aggiustamento matematico)
    'z_mlr': 0.448500,
    'z_piv': 1.222700
}

# 3. COEFFICIENTI IMPRINT CORE (Intercept: -0.9611)
COEFF_CORE = {
    'intercept': -0.961059,
    'menopause': 1.238229,
    'irregular_margins': 2.061382,
    'cystic_areas': 0.832600,   # Nota: Cystic Areas c'è anche nel Core
    'z_cs': 0.626500,
    'z_diameter': 0.529300
}

# 4. COEFFICIENTI MYLUNAR (Replicato: Intercept -1.3203)
COEFF_MYLUNAR = {
    'intercept': -1.320251,
    'age': 0.023323,            # Raw Age
    'diameter_gt80': 0.912094,  # Binary > 80mm
    'irregular_margins': 1.790647,
    'cs_4': 1.425928,           # Color Score = 4
    'shadows': -1.032002        # Acoustic Shadows (Protective)
}

# ---------------------------------------------------------
# FUNZIONI DI CALCOLO
# ---------------------------------------------------------

def calculate_zscore(value, param_key):
    """Calcola lo z-score applicando log se necessario."""
    params = Z_PARAMS[param_key]
    
    if params['log']:
        if value <= 0: return 0 # Evita errori log
        val_to_std = np.log(value)
    else:
        val_to_std = value
        
    z = (val_to_std - params['mean']) / params['sd']
    return z

def sigmoid(logit):
    """Converte logit in probabilità."""
    return 1 / (1 + math.exp(-logit))

# ---------------------------------------------------------
# INTERFACCIA UTENTE (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("Patient Data")

# Dati Clinici
age = st.sidebar.number_input("Age (years)", min_value=18, max_value=100, value=50)
menopause = st.sidebar.selectbox("Menopause", ["Pre-menopausal", "Post-menopausal"])
menopause_val = 1 if menopause == "Post-menopausal" else 0

# Dati Ecografici
st.sidebar.subheader("Ultrasound Features")
diameter = st.sidebar.number_input("Max Lesion Diameter (mm)", min_value=10, max_value=300, value=60)
color_score = st.sidebar.slider("Color Score (1-4)", 1, 4, 2)
irregular_margins = st.sidebar.checkbox("Irregular Margins")
cystic_areas = st.sidebar.checkbox("Cystic Areas")
shadows = st.sidebar.checkbox("Acoustic Shadows") # Solo per MyLunar

# Dati Laboratorio (Input Grezzi)
st.sidebar.subheader("Complete Blood Count")
st.sidebar.info("Enter absolute counts (e.g., 4500 for 4.5k)")
neutrophils = st.sidebar.number_input("Neutrophils (/µL)", min_value=1, value=4000)
lymphocytes = st.sidebar.number_input("Lymphocytes (/µL)", min_value=1, value=2000)
monocytes = st.sidebar.number_input("Monocytes (/µL)", min_value=1, value=500)
platelets = st.sidebar.number_input("Platelets (x10^3/µL)", min_value=1, value=250) # Input normale es 250

# Calcolo Marker Derivati
try:
    nlr = neutrophils / lymphocytes
    mlr = monocytes / lymphocytes
    # PIV = (Neutrophils * Platelets * Monocytes) / Lymphocytes
    # Attenzione: Platelets nel PIV sono spesso usate come conteggio assoluto o x10^9/L.
    # Nel paper standard PIV = Neutrophils * Monocytes * Platelets / Lymphocytes.
    # Assumiamo che l'utente inserisca i dati coerenti con la tua validazione.
    piv = (neutrophils * platelets * monocytes) / lymphocytes 
except ZeroDivisionError:
    nlr, mlr, piv = 0, 0, 0

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Calculated Markers:**")
st.sidebar.code(f"NLR: {nlr:.2f}\nMLR: {mlr:.2f}\nPIV: {piv:.0f}")

# ---------------------------------------------------------
# CALCOLO RISCHI
# ---------------------------------------------------------

# 1. PREPARA VARIABILI
# Variabili binarie/numeriche
irr_val = 1 if irregular_margins else 0
cyst_val = 1 if cystic_areas else 0
shadow_val = 1 if shadows else 0
diam_gt80 = 1 if diameter > 80 else 0
cs4_val = 1 if color_score == 4 else 0

# Z-Scores (per IMPRINT)
z_diam = calculate_zscore(diameter, 'diameter')
z_cs = calculate_zscore(color_score, 'cs')
z_nlr = calculate_zscore(nlr, 'nlr')
z_mlr = calculate_zscore(mlr, 'mlr')
z_piv = calculate_zscore(piv, 'piv')

# 2. LOGIT IMPRINT EXTENDED
logit_ext = (
    COEFF_EXTENDED['intercept'] +
    (COEFF_EXTENDED['menopause'] * menopause_val) +
    (COEFF_EXTENDED['irregular_margins'] * irr_val) +
    (COEFF_EXTENDED['cystic_areas'] * cyst_val) +
    (COEFF_EXTENDED['z_cs'] * z_cs) +
    (COEFF_EXTENDED['z_diameter'] * z_diam) +
    (COEFF_EXTENDED['z_nlr'] * z_nlr) +
    (COEFF_EXTENDED['z_mlr'] * z_mlr) +
    (COEFF_EXTENDED['z_piv'] * z_piv)
)
prob_ext = sigmoid(logit_ext)

# 3. LOGIT IMPRINT CORE
logit_core = (
    COEFF_CORE['intercept'] +
    (COEFF_CORE['menopause'] * menopause_val) +
    (COEFF_CORE['irregular_margins'] * irr_val) +
    (COEFF_CORE['cystic_areas'] * cyst_val) +
    (COEFF_CORE['z_cs'] * z_cs) +
    (COEFF_CORE['z_diameter'] * z_diam)
)
prob_core = sigmoid(logit_core)

# 4. LOGIT MYLUNAR
logit_mylunar = (
    COEFF_MYLUNAR['intercept'] +
    (COEFF_MYLUNAR['age'] * age) +
    (COEFF_MYLUNAR['diameter_gt80'] * diam_gt80) +
    (COEFF_MYLUNAR['irregular_margins'] * irr_val) +
    (COEFF_MYLUNAR['cs_4'] * cs4_val) +
    (COEFF_MYLUNAR['shadows'] * shadow_val)
)
prob_mylunar = sigmoid(logit_mylunar)

# ---------------------------------------------------------
# MAIN DISPLAY
# ---------------------------------------------------------
st.title("IMPRINT Risk Calculator")
st.markdown("""
This tool integrates **Standardized Ultrasound Features** and **Systemic Inflammatory Markers (SIMs)** to predict the risk of uterine sarcoma/STUMP in patients with myometrial lesions.
""")

tabs = st.tabs(["🚀 IMPRINT Extended", "🔹 IMPRINT Core", "🌙 MYLUNAR Comparison"])

# --- TAB 1: EXTENDED ---
with tabs[0]:
    st.subheader("IMPRINT Extended Model (Recommended)")
    st.caption("Integrates US features + PIV, NLR, MLR. Best calibration and robustness.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="Malignancy Probability", value=f"{prob_ext*100:.1f}%")
        
        if prob_ext < 0.10:
            st.success("Low Risk (<10%)")
        elif prob_ext < 0.50:
            st.warning("Intermediate Risk")
        else:
            st.error("High Risk (>50%)")
            
    with col2:
        st.progress(prob_ext)
        st.markdown(f"""
        **Driver Analysis:**
        * **PIV Contribution:** Z-score {z_piv:.2f} ({(z_piv*COEFF_EXTENDED['z_piv']):.2f} logit)
        * **Morphology:** {diameter}mm, CS {color_score}
        """)

# --- TAB 2: CORE ---
with tabs[1]:
    st.subheader("IMPRINT Core Model")
    st.caption("Ultrasound + Clinical features only. Use if blood tests are unavailable.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Core Probability", value=f"{prob_core*100:.1f}%")
    with col2:
        st.progress(prob_core)

# --- TAB 3: MYLUNAR ---
with tabs[2]:
    st.subheader("MYLUNAR Model (Replicated)")
    st.caption("Based on: Age, Diameter>8cm, Irreg. Margins, CS=4, Shadows.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="MYLUNAR Probability", value=f"{prob_mylunar*100:.1f}%")
    with col2:
        st.progress(prob_mylunar)
    
    st.markdown("---")
    st.info("Note: MYLUNAR uses dichotomized variables (e.g., >8cm), which may result in step-changes in risk compared to IMPRINT's continuous assessment.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
**Disclaimer:** This tool is for **research and educational purposes only**. 
It is not intended to replace clinical judgment. 
*Derived from the IMPRINT Study (Paratore et al., 2025).*
""")
