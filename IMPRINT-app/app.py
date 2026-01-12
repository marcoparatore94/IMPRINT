import streamlit as st
import numpy as np
import math

# ---------------------------------------------------------
# 1. CONFIGURAZIONE PAGINA & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="IMPRINT Risk Calculator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS per styling avanzato
st.markdown("""
<style>
    .stNumberInput > label {font-weight: bold; font-size: 1.1em;}
    .stSelectbox > label {font-weight: bold; font-size: 1.1em;}
    .stSlider > label {font-weight: bold; font-size: 1.1em;}
    .stToggle > label {font-weight: bold;}
    .block-container {padding-top: 1.5rem;}
    
    /* Box Risultati */
    .risk-card {
        text-align: center; 
        padding: 20px; 
        border-radius: 12px; 
        margin-bottom: 20px;
    }
    
    /* SIMs Box */
    .sim-container {
        display: flex;
        justify-content: space-around;
        margin-top: 10px;
    }
    .sim-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
        width: 30%;
    }
    .sim-value {
        font-size: 1.8em;
        font-weight: bold;
    }
    .sim-label {
        font-size: 0.9em;
        color: #6c757d;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATI E COEFFICIENTI
# ---------------------------------------------------------

# Parametri Z-Score (Supplementary Table 1)
Z_PARAMS = {
    'diameter': {'mean': 4.377, 'sd': 0.466, 'log': True},
    'cs':       {'mean': 2.500, 'sd': 0.894, 'log': False},
    'nlr':      {'mean': 0.917, 'sd': 0.584, 'log': True},
    'mlr':      {'mean': -1.512,'sd': 0.385, 'log': True},
    'piv':      {'mean': 5.537, 'sd': 0.975, 'log': True}
}

# Coefficienti Modelli
COEFF_EXTENDED = {
    'intercept': -1.034533, 'menopause': 1.731603, 'irregular_margins': 2.044660,
    'cystic_areas': 0.848012, 'z_cs': 0.608570, 'z_diameter': 0.504175,
    'z_nlr': -0.722877, 'z_mlr': -0.138242, 'z_piv': 1.225346
}

COEFF_CORE = {
    'intercept': -0.961059, 'menopause': 1.238229, 'irregular_margins': 2.061382,
    'cystic_areas': 0.832600, 'z_cs': 0.626500, 'z_diameter': 0.529300
}

COEFF_MYLUNAR = {
    'intercept': -1.320251, 'age': 0.023323, 'diameter_gt80': 0.912094,
    'irregular_margins': 1.790647, 'cs_4': 1.425928, 'shadows': -1.032002
}

# Cutoffs indicativi per colorazione
SIM_CUTOFFS = {
    "NLR": 3.0,
    "PIV": 400.0,
    "MLR": 0.35 
}

# ---------------------------------------------------------
# 3. FUNZIONI DI CALCOLO
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

def get_risk_info(prob):
    """Restituisce colore, classe e raccomandazione (ENG)."""
    pct = prob * 100
    if pct < 10:
        return "#28a745", "LOW RISK", "Conservative management / Follow-up", "rgba(40, 167, 69, 0.1)"
    elif pct < 50:
        return "#ffc107", "INTERMEDIATE RISK", "MRI / Referral to expert center", "rgba(255, 193, 7, 0.1)"
    else:
        return "#dc3545", "HIGH RISK", "Planned oncologic surgery", "rgba(220, 53, 69, 0.1)"

def get_sim_style(value, name):
    cutoff = SIM_CUTOFFS.get(name, 9999)
    color = "#dc3545" if value >= cutoff else "#212529"
    return f"color: {color};"

def generate_report_text(lang_selection, inputs, results):
    """Genera il testo per il copia-incolla. Usa .get() per evitare crash."""
    
    # Estrazione sicura valori
    p_ext = results.get('prob_ext', 0) * 100
    p_core = results.get('prob_core', 0) * 100
    p_mylunar = results.get('prob_mylunar', 0) * 100
    
    risk_label = results.get('risk_label', 'N/A')
    
    # Recupero valori SIMs sicuri
    v_piv = results.get('piv', 0)
    v_nlr = results.get('nlr', 0)
    v_mlr = results.get('mlr', 0)

    # Determina se Italiano (Controlla se la stringa contiene 'Italiano')
    is_ita = "Italiano" in lang_selection
    
    if is_ita:
        # Traduzione dinamica della raccomandazione
        if p_ext < 10:
            rec_ita = "Gestione conservativa / Follow-up ecografico"
        elif p_ext < 50:
            rec_ita = "Risonanza Magnetica (RM) / Riferimento a centro esperto"
        else:
            rec_ita = "Chirurgia oncologica pianificata presso centro di riferimento"

        text = f"""**REFERTO IMPRINT RISK ASSESSMENT**
------------------------------------------------
**Paziente:** {int(inputs.get('age',0))} anni, {inputs.get('menopause','')}
**Ecografia:** Massa di {inputs.get('diameter',0)} mm, CS {inputs.get('cs',0)}
Caratteristiche: Margini {"Irregolari" if inputs.get('irr') else "Regolari"}, {"Aree Cistiche" if inputs.get('cyst') else "Solida"}, {"Ombre Acustiche" if inputs.get('shadows') else "Nessuna Ombra"}.

**Profilo Infiammatorio (Drivers):**
- PIV: {v_piv:.0f}
- NLR: {v_nlr:.2f}
- MLR: {v_mlr:.2f}

**RISULTATI MODELLI:**
1. IMPRINT Extended (Ref): {p_ext:.1f}% - {risk_label}
   *Raccomandazione: {rec_ita}*

2. IMPRINT Core (Morphology only): {p_core:.1f}%
3. MYLUNAR (External): {p_mylunar:.1f}%
------------------------------------------------
*Calcolato tramite IMPRINT App - Solo a scopo di ricerca.*"""
    
    else: # English
        rec_eng = results.get('rec', '')
        text = f"""**IMPRINT RISK ASSESSMENT REPORT**
------------------------------------------------
**Patient:** {int(inputs.get('age',0))} y.o., {inputs.get('menopause','')}
**Ultrasound:** Mass {inputs.get('diameter',0)} mm, CS {inputs.get('cs',0)}
Features: {"Irregular" if inputs.get('irr') else "Regular"} margins, {"Cystic areas" if inputs.get('cyst') else "Solid"}, {"Acoustic Shadows" if inputs.get('shadows') else "No Shadows"}.

**Inflammatory Profile (Drivers):**
- PIV: {v_piv:.0f}
- NLR: {v_nlr:.2f}
- MLR: {v_mlr:.2f}

**MODEL RESULTS:**
1. IMPRINT Extended (Ref): {p_ext:.1f}% - {risk_label}
   *Guidance: {rec_eng}*

2. IMPRINT Core (Morphology only): {p_core:.1f}%
3. MYLUNAR (External): {p_mylunar:.1f}%
------------------------------------------------
*Calculated via IMPRINT App - For research use only.*"""
    
    return text

# ---------------------------------------------------------
# 4. INTERFACCIA UTENTE
# ---------------------------------------------------------

st.title("🏥 IMPRINT Risk Calculator")
st.markdown("### Preoperative Risk Assessment for Uterine Sarcoma")
st.markdown("---")

col_input, col_spacer, col_output = st.columns([1, 0.05, 1.1])

# --- COLONNA INPUT (SINISTRA) ---
with col_input:
    c_head1, c_head2 = st.columns([3, 1])
    with c_head1: st.subheader("1. Patient Data")
    with c_head2: 
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()

    with st.form("risk_assessment_form"):
        
        # CLINICA
        st.markdown("##### 👤 Clinical")
        c1, c2 = st.columns(2)
        with c1: age = st.number_input("Age (years)", 18, 100, 50)
        with c2: 
            menopause = st.radio("Menopause", ["Pre-menopausal", "Post-menopausal"], horizontal=True, 
                                 help="Post-menopausal: >12 months amenorrhea.")
            menopause_val = 1 if menopause == "Post-menopausal" else 0

        st.markdown("---")
        
        # ECOGRAFIA (MUSA Definitions)
        st.markdown("##### 🖥️ Ultrasound (MUSA Criteria)")
        diameter = st.number_input("Max Lesion Diameter (mm)", 10, 300, 80)
        
        color_score = st.slider("Color Score (1-4)", 1, 4, 2, 
            help="**MUSA Definitions:**\n1: No flow\n2: Minimal flow\n3: Moderate flow\n4: Abundant flow")
        
        u1, u2, u3 = st.columns(3)
        with u1: 
            irregular_margins = st.toggle("Irreg. Margins", 
                help="**MUSA:** Non-circumscribed margins or infiltrative growth pattern.")
        with u2: 
            cystic_areas = st.toggle("Cystic Areas", 
                help="Presence of anechoic cystic spaces (necrosis/degeneration).")
        with u3: 
            shadows = st.toggle("Shadows", 
                help="**MYLUNAR:** Internal acoustic shadowing (fan-shaped attenuation). Suggestive of benignity.")

        st.markdown("---")
        
        # LABORATORIO
        st.markdown("##### 🩸 Complete Blood Count")
        st.caption("Please enter absolute counts (e.g., 4000 for 4.0 x10⁹/L).")
        
        l1, l2 = st.columns(2)
        with l1:
            neutrophils_abs = st.number_input("Neutrophils (/µL)", 0, 50000, 3000)
            monocytes_abs = st.number_input("Monocytes (/µL)", 0, 10000, 400)
        with l2:
            lymphocytes_abs = st.number_input("Lymphocytes (/µL)", 0, 20000, 2000)
            platelets_abs = st.number_input("Platelets (x10^3/µL)", 0, 1000, 200)

        st.markdown("###")
        submit_btn = st.form_submit_button("🚀 ESTIMATE RISK", use_container_width=True, type="primary")

# ---------------------------------------------------------
# 5. LOGICA DI CALCOLO
# ---------------------------------------------------------
if submit_btn:
    # Preparazione variabili
    # Conversione unità per formule SIMs
    N = neutrophils_abs / 1000.0
    L = lymphocytes_abs / 1000.0
    M = monocytes_abs / 1000.0
    P = platelets_abs 

    # Calcolo SIMs
    try:
        nlr = N / L
        mlr = M / L
        piv = (N * P * M) / L
    except:
        nlr, mlr, piv = 0, 0, 0

    # Variabili binarie
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

    # Info Rischio (Colore/Label)
    color_ext, label_ext, rec_ext, bg_ext = get_risk_info(prob_ext)
    
    # DIZIONARIO RISULTATI (Completo per evitare crash)
    results_data = {
        'prob_ext': prob_ext,
        'prob_core': prob_core,     
        'prob_mylunar': prob_mylunar,
        'risk_label': label_ext, 
        'rec': rec_ext,
        'nlr': nlr, 'mlr': mlr, 'piv': piv
    }
    inputs_data = {
        'age': age, 'menopause': menopause, 'diameter': diameter, 'cs': color_score,
        'irr': irregular_margins, 'cyst': cystic_areas, 'shadows': shadows
    }

    # --------------------------------
    # 6. OUTPUT (COLONNA DESTRA)
    # --------------------------------
    with col_output:
        st.subheader("2. Risk Analysis Results")
        
        tab1, tab2, tab3 = st.tabs(["🚀 IMPRINT Extended", "🔹 IMPRINT Core", "🌙 MYLUNAR"])
        
        # --- TAB EXTENDED ---
        with tab1:
            st.info("💡 **Recommended Model**: Integrates Ultrasound + Inflammatory Markers")
            
            # Risk Card
            st.markdown(f"""
            <div class="risk-card" style="background-color: {bg_ext}; border: 2px solid {color_ext};">
                <h4 style="color: {color_ext}; margin:0;">EXTENDED MODEL RISK</h4>
                <h1 style="color: {color_ext}; margin:10px 0; font-size: 3.5rem;">{prob_ext*100:.1f}%</h1>
                <h3 style="color: {color_ext}; margin:0;">{label_ext}</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Guidance:** {rec_ext}")

            # SIMs Grid semplificata
            st.markdown("#### 🔬 Biomarker Profile (Drivers)")
            
            html_sims = f"""
            <div class="sim-container">
                <div class="sim-box">
                    <div class="sim-label">PIV</div>
                    <div class="sim-value" style="{get_sim_style(piv, 'PIV')}">{piv:.0f}</div>
                    <div style="font-size:0.8em; color:gray;">(Cutoff 400)</div>
                </div>
                <div class="sim-box">
                    <div class="sim-label">NLR</div>
                    <div class="sim-value" style="{get_sim_style(nlr, 'NLR')}">{nlr:.2f}</div>
                    <div style="font-size:0.8em; color:gray;">(Cutoff 3.0)</div>
                </div>
                <div class="sim-box">
                    <div class="sim-label">MLR</div>
                    <div class="sim-value" style="{get_sim_style(mlr, 'MLR')}">{mlr:.2f}</div>
                    <div style="font-size:0.8em; color:gray;">(Cutoff 0.35)</div>
                </div>
            </div>
            """
            st.markdown(html_sims, unsafe_allow_html=True)

        # --- TAB CORE ---
        with tab2:
            st.info("Morphology Only (No Blood Tests)")
            c_core, l_core, r_core, bg_core = get_risk_info(prob_core)
            st.markdown(f"""
            <div class="risk-card" style="background-color: {bg_core}; border: 2px solid {c_core};">
                <h4 style="color: {c_core}; margin:0;">CORE RISK</h4>
                <h1 style="color: {c_core}; font-size: 3rem;">{prob_core*100:.1f}%</h1>
                <h3 style="color: {c_core};">{l_core}</h3>
            </div>
            """, unsafe_allow_html=True)

        # --- TAB MYLUNAR ---
        with tab3:
            st.info("External Benchmark (MYLUNAR)")
            c_my, l_my, r_my, bg_my = get_risk_info(prob_mylunar)
            st.markdown(f"""
            <div class="risk-card" style="background-color: {bg_my}; border: 2px solid {c_my};">
                <h4 style="color: {c_my}; margin:0;">MYLUNAR RISK</h4>
                <h1 style="color: {c_my}; font-size: 3rem;">{prob_mylunar*100:.1f}%</h1>
                <h3 style="color: {c_my};">{l_my}</h3>
            </div>
            """, unsafe_allow_html=True)
            if shadows:
                st.success("✅ Acoustic Shadows detected: Strong protective factor applied.")

        # --- SEZIONE REPORT GENERATORE ---
        st.markdown("---")
        st.subheader("📋 Generate Report")
        lang_choice = st.radio("Language", ["🇬🇧 English", "🇮🇹 Italiano"], horizontal=True, label_visibility="collapsed")
        
        # Generazione report sicura
        report_text = generate_report_text(lang_choice, inputs_data, results_data)
        
        st.code(report_text, language='markdown')
        st.caption("Click the 'Copy' button in the top-right of the box above.")

else:
    with col_output:
        st.info("👈 Enter patient data on the left and click **ESTIMATE RISK** to generate the analysis.")
        st.markdown("""
        <div style="text-align: center; opacity: 0.5; margin-top: 50px;">
            <div style="font-size: 4rem;">🏥</div>
            <h3>Ready to calculate</h3>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 7. FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.8em; color: gray;">
    <b>DISCLAIMER:</b> This tool is intended for <b>research and educational purposes only</b>. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    <br>
    <b>Privacy Notice:</b> No patient data is stored on external servers. All calculations are performed in your browser session.
    <br>
    <i>Derived from the IMPRINT Study (Bruno et al., 2025).</i>
</div>
""", unsafe_allow_html=True)
