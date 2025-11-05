import os
import sys
import streamlit as st
import numpy as np
from joblib import load

# Permetti import dalla root del progetto
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.info_panel import get_info_html

# --- Configurazione pagina ---
st.set_page_config(page_title="IMPRINT Risk Calculator", page_icon="🧮", layout="centered")

# --- Login box centrale ---
st.title("🔒 Accesso IMPRINT")
PASSWORD = "Imprintfpg2025"
password = st.text_input("Inserisci la password", type="password")
if password != PASSWORD:
    st.warning("Inserisci la password per accedere all'applicazione.")
    st.stop()

# --- Titolo app ---
st.title("Calcolatore rischio malignità (IMPRINT)")

# --- Percorso modello ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "imprint_risk_model.joblib"))

# --- Pannello informazioni ---
with st.expander("Informazioni e cut-off"):
    st.markdown(get_info_html(), unsafe_allow_html=True)

# --- Form di input (clinico, ecografico, ematologico) ---
with st.form("risk_form"):
    # Clinico
    age = st.number_input("Età (anni)", min_value=15, max_value=95, value=50)
    diameter = st.number_input("Diametro target (mm)", min_value=10, max_value=250, value=75)

    # Ecografico (coerente con le feature del training)
    margini = st.selectbox("Margini", ["Regolari", "Irregolari"])
    color_doppler = st.selectbox("Color Doppler (score)", ["Assente/1-3", "4"])
    ombre = st.selectbox("Ombre acustiche", ["Presenti", "Assenti"])

    # Ematologico
    neutrofili = st.number_input("Neutrofili", min_value=0.0, max_value=30.0, value=4.0)
    linfociti = st.number_input("Linfociti", min_value=0.0, max_value=10.0, value=2.0)
    monociti = st.number_input("Monociti", min_value=0.0, max_value=5.0, value=0.5)
    piastrine = st.number_input("Piastrine", min_value=10.0, max_value=1000.0, value=250.0)

    submitted = st.form_submit_button("Calcola rischio")

# --- Checkbox FUORI dal form ---
show_details = st.checkbox("Mostra dettagli indici ematologici")
show_radar = st.checkbox("Mostra grafico radar indici ematologici")

# --- Funzioni di supporto ---
def classify_risk(p: float) -> str:
    if p < 0.004:
        return "Basso"
    elif p < 0.023:
        return "Intermedio"
    else:
        return "Alto"

def risk_color(cls: str) -> str:
    return "green" if cls == "Basso" else ("orange" if cls == "Intermedio" else "red")

# --- Calcolo e predizione ---
if submitted:
    # Marker ematologici (gestione divisione per zero)
    if linfociti and linfociti > 0:
        NLR = neutrofili / linfociti
        SII = (neutrofili * piastrine) / linfociti
        SIRI = (neutrofili * monociti) / linfociti
        PIV = (neutrofili * piastrine * monociti) / linfociti
    else:
        NLR = SII = SIRI = PIV = 0.0

    # Codifica ecografica e diametro come nel training
    diameter_gt8 = 1 if diameter > 80 else 0
    margini_irregolari = 1 if margini == "Irregolari" else 0
    color4 = 1 if color_doppler == "4" else 0
    ombre_assenti = 1 if ombre == "Assenti" else 0

    # Vettore X nello stesso ordine del training:
    # ["NLR","SII","SIRI","PIV","Age","diameter_gt8","Margini_irregolari","Color4","Ombre_assenti"]
    x = np.array([[NLR, SII, SIRI, PIV,
                   age, diameter_gt8,
                   margini_irregolari, color4, ombre_assenti]])

    # Caricamento modello
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Modello non trovato. Verifica il percorso: data/imprint_risk_model.joblib")
        st.stop()

    model = load(MODEL_PATH)

    # Predizione
    try:
        p = float(model.predict_proba(x)[0, 1])
    except Exception as e:
        st.error("Si è verificato un errore nella predizione. Verifica che le feature siano nell'ordine corretto.")
        st.caption(str(e))
        st.stop()

    cls = classify_risk(p)
    color = risk_color(cls)

    # Output rischio con colore dinamico
    st.markdown(f"### Probabilità di malignità: {p:.3%}")
    st.progress(min(int(p * 100), 100))
    st.markdown(
        f"### Classe di rischio: <span style='color:{color}'>{cls}</span>",
        unsafe_allow_html=True
    )

    # Dettagli opzionali
    if show_details:
        st.subheader("Dettagli indici ematologici")
        st.write(f"NLR: {NLR:.2f}")
        st.write(f"SII: {SII:.2f}")
        st.write(f"SIRI: {SIRI:.2f}")
        st.write(f"PIV: {PIV:.2f}")

    # Radar opzionale
    if show_radar:
        import matplotlib.pyplot as plt
        labels = ["NLR", "SII", "SIRI", "PIV"]
        values = [NLR, SII, SIRI, PIV]

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        values_r = values + values[:1]
        angles_r = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.plot(angles_r, values_r, "o-", linewidth=2)
        ax.fill(angles_r, values_r, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles), labels)
        st.pyplot(fig)

    st.caption("Modello addestrato sui dati del centro. Validazione clinica in corso.")
