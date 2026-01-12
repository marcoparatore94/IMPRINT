import streamlit as st
import numpy as np
import math

# ---------------------------------------------------------
# CONFIGURAZIONE PAGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="IMPRINT Risk Calculator",
    page_icon="🏥",
    layout="wide", # Layout ampio per affiancare input e output
    initial_sidebar_state="collapsed"
)

# CSS per pulire l'interfaccia
st.markdown("""
<style>
    .stNumberInput > label {font-weight: bold;}
    .stSelectbox > label {font-weight: bold;}
    .stSlider > label {font-weight: bold;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATI E COEFFICIENTI (Hardcoded per sicurezza)
# ---------------------------------------------------------

# 1. PARAMETRI DI STANDARDIZZAZIONE (Supplementary Table 1)
Z_PARAMS = {
    'diameter': {'mean': 4.377, 'sd': 0.466, 'log': True},
    'cs':       {'mean': 2.500, 'sd': 0.894, 'log': False},
    'nlr':      {'mean': 0.917, 'sd': 0.584, 'log': True},
    'mlr':      {'mean': -1.512,'sd': 0.385, 'log': True},
    'piv':      {'mean': 5.537, 'sd': 0.975, 'log': True}
}

# 2. COEFFICIENTI IMPRINT EXTENDED
COEFF_EXTENDED = {
    'intercept': -1.034533,
    'menopause': 1.731603,
    'irregular_margins': 2.044660,
    'cystic_areas': 0.613274,
    'z_cs': 0.608670,
    'z_diameter': 0.509756,
    'z_nlr': -0.739700,
    'z_mlr': 0.448500,
    'z_piv': 1.222700
}

# 3. COEFFICIENTI IMPRINT CORE
COEFF_CORE = {
    'intercept': -0.961059,
    'menopause': 1.238229,
    'irregular_margins': 2.061382,
    'cystic_areas': 0.832600,
    'z_cs': 0.626500,
    'z_diameter': 0.529300
}

# 4. COEFFICIENTI MYLUNAR
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

# ---------------------------------------------------------
# INTERFACCIA PRINCIPALE
# ---------------------------------------------------------

st.title("🏥 IMPRINT Risk Calculator")
st.markdown("### Preoperative Risk Assessment for Uterine Sarcoma")
st.markdown("---")

# Creiamo due colonne principali: Input (Sinistra) e Output (Destra)
col_input, col_spacer, col_output = st.columns([1, 0.1, 1])

with col_input:
    st.subheader("1. Patient & Ultrasound Data")
    
    # Usiamo un FORM per evitare ricaricamenti continui
    with st.form("risk_assessment_form"):
        
        # --- SEZIONE CLINICA ---
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 18, 100, 50)
        with c2:
            menopause = st.radio("Menopause Status", ["Pre-menopausal", "Post-menopausal"], horizontal=True)
            menopause_val = 1 if menopause == "Post-menopausal" else 0

        st.markdown("---")
        
        # --- SEZIONE ECOGRAFIA ---
        st.markdown("##### 🖥️ Ultrasound Features")
        diameter = st.number_input("Max Lesion Diameter (mm)", 10, 300, 80)
        color_score = st.slider("Color Score (1-4)", 1, 4, 2, help="1: No flow, 4: Abundant flow")
        
        u1, u2, u3 = st.columns(3)
        with u1:
            irregular_margins = st.toggle("Irregular Margins")
        with u2:
            cystic_areas = st.toggle("Cystic Areas")
        with u3:
            shadows = st.toggle("Acoustic Shadows") # Per MyLunar

        st.markdown("---")

        # --- SEZIONE LABORATORIO ---
        st.markdown("##### 🩸 Complete Blood Count")
        st.caption("Enter absolute counts (e.g., 4500 for 4.5k)")
        
        l1, l2 = st.columns(2)
        with l1:
            neutrophils = st.number_input("Neutrophils (/µL)", 1, 20000, 4000)
            monocytes = st.number_input("Monocytes (/µL)", 1, 5000, 500)
        with l2:
            lymphocytes = st.number_input("Lymphocytes (/µL)", 1, 10000, 2000)
            platelets = st.number_input("Platelets (x10^3/µL)", 1, 1000, 250)

        # Bottone Calcolo
        st.markdown("###")
        submit_btn = st.form_submit_button("🚀 ESTIMATE RISK", use_container_width=True, type="primary")

# ---------------------------------------------------------
# CALCOLO E VISUALIZZAZIONE RISULTATI
# ---------------------------------------------------------
if submit_btn:
    # 1. Calcolo Markers Derivati
    try:
        nlr = neutrophils / lymphocytes
        mlr = monocytes / lymphocytes
        piv = (neutrophils * platelets * monocytes) / lymphocytes 
    except:
        nlr, mlr, piv = 0, 0, 0

    # 2. Preparazione Variabili Modello
    irr_val = 1 if irregular_margins else 0
    cyst_val = 1 if cystic_areas else 0
    shadow_val = 1 if shadows else 0
    diam_gt80 = 1 if diameter > 80 else 0
    cs4_val = 1 if color_score == 4 else 0

    # 3. Z-Scores
    z_diam = calculate_zscore(diameter, 'diameter')
    z_cs = calculate_zscore(color_score, 'cs')
    z_nlr = calculate_zscore(nlr, 'nlr')
    z_mlr = calculate_zscore(mlr, 'mlr')
    z_piv = calculate_zscore(piv, 'piv')

    # 4. Calcolo Logits e Probabilità
    # Extended
    logit_ext = (COEFF_EXTENDED['intercept'] + (COEFF_EXTENDED['menopause'] * menopause_val) +
                 (COEFF_EXTENDED['irregular_margins'] * irr_val) + (COEFF_EXTENDED['cystic_areas'] * cyst_val) +
                 (COEFF_EXTENDED['z_cs'] * z_cs) + (COEFF_EXTENDED['z_diameter'] * z_diam) +
                 (COEFF_EXTENDED['z_nlr'] * z_nlr) + (COEFF_EXTENDED['z_mlr'] * z_mlr) +
                 (COEFF_EXTENDED['z_piv'] * z_piv))
    prob_ext = sigmoid(logit_ext)

    # Core
    logit_core = (COEFF_CORE['intercept'] + (COEFF_CORE['menopause'] * menopause_val) +
                  (COEFF_CORE['irregular_margins'] * irr_val) + (COEFF_CORE['cystic_areas'] * cyst_val) +
                  (COEFF_CORE['z_cs'] * z_cs) + (COEFF_CORE['z_diameter'] * z_diam))
    prob_core = sigmoid(logit_core)

    # MyLunar
    logit_mylunar = (COEFF_MYLUNAR['intercept'] + (COEFF_MYLUNAR['age'] * age) +
                     (COEFF_MYLUNAR['diameter_gt80'] * diam_gt80) + (COEFF_MYLUNAR['irregular_margins'] * irr_val) +
                     (COEFF_MYLUNAR['cs_4'] * cs4_val) + (COEFF_MYLUNAR['shadows'] * shadow_val))
    prob_mylunar = sigmoid(logit_mylunar)

    # --------------------------------
    # DISPLAY OUTPUT (Colonna Destra)
    # --------------------------------
    with col_output:
        st.subheader("2. Risk Analysis Results")
        
        # Tabs per i modelli
        tab1, tab2, tab3 = st.tabs(["IMPRINT Extended", "IMPRINT Core", "External (MYLUNAR)"])
        
        # --- TAB EXTENDED ---
        with tab1:
            st.info("💡 **Recommended Model**: Uses Imaging + Biomarkers")
            
            # Grafico a "Semaforo"
            risk_pct = prob_ext * 100
            if risk_pct < 10:
                color = "green"
                risk_label = "LOW RISK"
                rec = "Conservative management / Follow-up"
            elif risk_pct < 50:
                color = "orange"
                risk_label = "INTERMEDIATE RISK"
                rec = "MRI / Referral to expert center"
            else:
                color = "red"
                risk_label = "HIGH RISK"
                rec = "Planned oncologic surgery"

            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                <h2 style="color: {color}; margin:0;">{risk_pct:.1f}%</h2>
                <p style="font-weight:bold; margin:0;">{risk_label}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.markdown(f"**Guidance:** {rec}")
            
            with st.expander("🔍 See Biomarker Details"):
                st.write(f"**NLR:** {nlr:.2f} (z={z_nlr:.2f})")
                st.write(f"**MLR:** {mlr:.2f} (z={z_mlr:.2f})")
                st.write(f"**PIV:** {piv:.0f} (z={z_piv:.2f})")
                st.caption("Standardized against IMPRINT cohort data.")

        # --- TAB CORE ---
        with tab2:
            st.write("Morphology Only (No Blood Tests)")
            st.metric("Core Probability", f"{prob_core*100:.1f}%")
            st.progress(prob_core)

        # --- TAB MYLUNAR ---
        with tab3:
            st.write("External Benchmark Model")
            st.metric("MYLUNAR Probability", f"{prob_mylunar*100:.1f}%")
            st.progress(prob_mylunar)
            st.caption("Note: Uses dichotomized variables (>8cm, CS=4).")

# Stato Iniziale (Prima del click)
else:
    with col_output:
        st.info("👈 Enter patient data on the left and click **ESTIMATE RISK**.")
        st.image("https://cdn-icons-png.flaticon.com/512/2823/2823795.png", width=100) # Icona placeholder generica
