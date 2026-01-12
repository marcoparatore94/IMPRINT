import streamlit as st
import numpy as np
import math

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="IMPRINT Risk Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS per pulire l'interfaccia e centrare i box
st.markdown("""
<style>
    .stNumberInput > label {font-weight: bold;}
    .stSelectbox > label {font-weight: bold;}
    .stSlider > label {font-weight: bold;}
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATI E COEFFICIENTI (AGGIORNATI)
# ---------------------------------------------------------

Z_PARAMS = {
    'diameter': {'mean': 4.377, 'sd': 0.466, 'log': True},
    'cs':       {'mean': 2.500, 'sd': 0.894, 'log': False},
    'nlr':      {'mean': 0.917, 'sd': 0.584, 'log': True},
    'mlr':      {'mean': -1.512,'sd': 0.385, 'log': True},
    'piv':      {'mean': 5.537, 'sd': 0.975, 'log': True}
}

COEFF_EXTENDED = {
    'intercept': -1.034533,
    'menopause': 1.731603,
    'irregular_margins': 2.044660,
    'cystic_areas': 0.848012,
    'z_cs': 0.608570,
    'z_diameter': 0.504175,
    'z_nlr': -0.722877,
    'z_mlr': -0.138242,
    'z_piv': 1.225346
}

COEFF_CORE = {
    'intercept': -0.961059,
    'menopause': 1.238229,
    'irregular_margins': 2.061382,
    'cystic_areas': 0.832600,
    'z_cs': 0.626500,
    'z_diameter': 0.529300
}

COEFF_MYLUNAR = {
    'intercept': -1.320251,
    'age': 0.023323,
    'diameter_gt80': 0.912094,
    'irregular_margins': 1.790647,
    'cs_4': 1.425928,
    'shadows': -1.032002
}

# ---------------------------------------------------------
# FUNZIONI UTILI
# ---------------------------------------------------------
def calculate_zscore(value, param_key):
    params = Z_PARAMS[param_key]
    if params['log']:
        if value <= 0: return 0
        val_to_std = np.log(value)
    else:
        val_to_std = value
    return (val_to_std - params['mean']) / params['sd']

def sigmoid(logit):
    return 1 / (1 + math.exp(-logit))

def get_risk_style(prob):
    """Restituisce colore, label e bg_color in base alla probabilità"""
    pct = prob * 100
    if pct < 10:
        return "#28a745", "LOW RISK", "rgba(40, 167, 69, 0.1)", "Conservative management / Follow-up"
    elif pct < 50:
        return "#ffc107", "INTERMEDIATE RISK", "rgba(255, 193, 7, 0.1)", "MRI / Referral to expert center"
    else:
        return "#dc3545", "HIGH RISK", "rgba(220, 53, 69, 0.1)", "Planned oncologic surgery"

def display_risk_card(prob, model_name):
    """Crea il box colorato standard"""
    color, label, bg, guidance = get_risk_style(prob)
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: {bg}; border-radius: 10px; border: 2px solid {color}; margin-bottom: 20px;">
        <h4 style="color: {color}; margin:0; text-transform: uppercase; letter-spacing: 1px;">{model_name}</h4>
        <h1 style="color: {color}; margin:10px 0; font-size: 3.5rem;">{prob*100:.1f}%</h1>
        <h3 style="color: {color}; margin:0;">{label}</h3>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"**Guidance:** {guidance}")

# ---------------------------------------------------------
# INTERFACCIA PRINCIPALE
# ---------------------------------------------------------

st.title("🏥 IMPRINT Risk Calculator")
st.markdown("### Preoperative Risk Assessment for Uterine Sarcoma")
st.markdown("---")

col_input, col_spacer, col_output = st.columns([1, 0.1, 1])

with col_input:
    st.subheader("1. Patient & Ultrasound Data")
    
    with st.form("risk_assessment_form"):
        # CLINICA
        c1, c2 = st.columns(2)
        with c1: age = st.number_input("Age (years)", 18, 100, 50)
        with c2: 
            menopause = st.radio("Menopause", ["Pre-menopausal", "Post-menopausal"], horizontal=True)
            menopause_val = 1 if menopause == "Post-menopausal" else 0

        st.markdown("---")
        # ECOGRAFIA
        st.markdown("##### 🖥️ Ultrasound Features")
        diameter = st.number_input("Max Lesion Diameter (mm)", 10, 300, 80)
        color_score = st.slider("Color Score (1-4)", 1, 4, 2)
        
        u1, u2, u3 = st.columns(3)
        with u1: irregular_margins = st.toggle("Irregular Margins")
        with u2: cystic_areas = st.toggle("Cystic Areas")
        with u3: shadows = st.toggle("Acoustic Shadows")

        st.markdown("---")
        # LABORATORIO
        st.markdown("##### 🩸 Complete Blood Count")
        st.info("Enter absolute counts (e.g., 4000 for 4.0 x10⁹/L)")
        
        l1, l2 = st.columns(2)
        with l1:
            neutrophils_abs = st.number_input("Neutrophils (/µL)", 1, 20000, 3000)
            monocytes_abs = st.number_input("Monocytes (/µL)", 1, 5000, 400)
        with l2:
            lymphocytes_abs = st.number_input("Lymphocytes (/µL)", 1, 10000, 2000)
            platelets_abs = st.number_input("Platelets (x10^3/µL)", 1, 1000, 200)

        st.markdown("###")
        submit_btn = st.form_submit_button("🚀 ESTIMATE RISK", use_container_width=True, type="primary")

# ---------------------------------------------------------
# CALCOLO
# ---------------------------------------------------------
if submit_btn:
    # Conversione unità
    neutrophils = neutrophils_abs / 1000.0
    lymphocytes = lymphocytes_abs / 1000.0
    monocytes = monocytes_abs / 1000.0
    platelets = platelets_abs

    # Markers Derivati
    try:
        nlr = neutrophils_abs / lymphocytes_abs
        mlr = monocytes_abs / lymphocytes_abs
        piv = (neutrophils * platelets * monocytes) / lymphocytes 
    except:
        nlr, mlr, piv = 0, 0, 0

    # Variabili Binarie
    irr_val = 1 if irregular_margins else 0
    cyst_val = 1 if cystic_areas else 0
    shadow_val = 1 if shadows else 0
    diam_gt80 = 1 if diameter > 80 else 0
    cs4_val = 1 if color_score == 4 else 0

    # Z-Scores
    z_diam = calculate_zscore(diameter, 'diameter')
    z_cs = calculate_zscore(color_score, 'cs')
    z_nlr = calculate_zscore(nlr, 'nlr')
    z_mlr = calculate_zscore(mlr, 'mlr')
    z_piv = calculate_zscore(piv, 'piv')

    # Calcolo Probabilità
    # EXTENDED
    logit_ext = (COEFF_EXTENDED['intercept'] + 
                 (COEFF_EXTENDED['menopause'] * menopause_val) +
                 (COEFF_EXTENDED['irregular_margins'] * irr_val) + 
                 (COEFF_EXTENDED['cystic_areas'] * cyst_val) +
                 (COEFF_EXTENDED['z_cs'] * z_cs) + 
                 (COEFF_EXTENDED['z_diameter'] * z_diam) +
                 (COEFF_EXTENDED['z_nlr'] * z_nlr) + 
                 (COEFF_EXTENDED['z_mlr'] * z_mlr) +
                 (COEFF_EXTENDED['z_piv'] * z_piv))
    prob_ext = sigmoid(logit_ext)

    # CORE
    logit_core = (COEFF_CORE['intercept'] + 
                  (COEFF_CORE['menopause'] * menopause_val) +
                  (COEFF_CORE['irregular_margins'] * irr_val) + 
                  (COEFF_CORE['cystic_areas'] * cyst_val) +
                  (COEFF_CORE['z_cs'] * z_cs) + 
                  (COEFF_CORE['z_diameter'] * z_diam))
    prob_core = sigmoid(logit_core)

    # MYLUNAR
    logit_mylunar = (COEFF_MYLUNAR['intercept'] + 
                     (COEFF_MYLUNAR['age'] * age) +
                     (COEFF_MYLUNAR['diameter_gt80'] * diam_gt80) + 
                     (COEFF_MYLUNAR['irregular_margins'] * irr_val) +
                     (COEFF_MYLUNAR['cs_4'] * cs4_val) + 
                     (COEFF_MYLUNAR['shadows'] * shadow_val))
    prob_mylunar = sigmoid(logit_mylunar)

    # --------------------------------
    # DISPLAY OUTPUT
    # --------------------------------
    with col_output:
        st.subheader("2. Risk Analysis Results")
        
        tab1, tab2, tab3 = st.tabs(["🚀 IMPRINT Extended", "🔹 IMPRINT Core", "🌙 MYLUNAR"])
        
        # --- TAB EXTENDED ---
        with tab1:
            st.info("💡 **Recommended Model**: Uses Imaging + Biomarkers")
            display_risk_card(prob_ext, "Extended Risk")
            
            with st.expander("🔍 Biomarker Details"):
                st.write(f"**NLR:** {nlr:.2f}")
                st.write(f"**PIV:** {piv:.0f}")

        # --- TAB CORE ---
        with tab2:
            st.info("Morphology Only (No Blood Tests)")
            display_risk_card(prob_core, "Core Risk")

        # --- TAB MYLUNAR ---
        with tab3:
            st.info("External Benchmark Model")
            display_risk_card(prob_mylunar, "MYLUNAR Risk")
            if shadows:
                st.success("✅ Acoustic Shadows detected: Strong protective factor in MYLUNAR.")

else:
    with col_output:
        st.info("👈 Enter patient data on the left and click **ESTIMATE RISK**.")
