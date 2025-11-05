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
st.title("Calcolatore rischio malignità (IMPRINT)")

# --- Login ---
PASSWORD = "Imprintfpg2025"
st.sidebar.subheader("🔒 Accesso")
password = st.sidebar.text_input("Inserisci la password", type="password")
if password != PASSWORD:
    st.error("Accesso negato. Inserisci la password corretta.")
    st.stop()

# --- Percorso modello ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "imprint_risk_model.joblib"))

# --- Info ---
with st.expander("Informazioni e cut-off"):
    st.markdown(get_info_html(), unsafe_allow_html=True)

# --- Form di input ---
with st.form("risk_form"):
    age = st.number_input("Età (anni)", min_value=15, max_value=95, value=50)
    diameter = st.number_input("Diametro target (mm)", min_value=10, max_value=250, value=75)

    neutrofili = st.number_input("Neutrofili", min_value=0.0, max_value=30.0, value=4.0)
    linfociti = st.number_input("Linfociti", min_value=0.0, max_value=10.0, value=2.0)
    monociti = st.number_input("Monociti", min_value=0.0, max_value=5.0, value=0.5)
    piastrine = st.number_input("Piastrine", min_value=10.0, max_value=1000.0, value=250.0)

    submitted = st.form_submit_button("Calcola rischio")

# --- Checkbox fuori dal form ---
show_details = st.checkbox("Mostra dettagli indici ematologici")
show_radar = st.checkbox("Mostra grafico radar indici ematologici")

# --- Funzioni ---
def classify_risk(p: float) -> str:
    if p < 0.004:
        return "Basso"
    elif p < 0.023:
        return "Intermedio"
    else:
        return "Alto"

# --- Calcolo ---
if submitted:
    if linfociti > 0:
        NLR = neutrofili / linfociti
        SII = (neutrofili * piastrine) / linfociti
        SIRI = (neutrofili * monociti) / linfociti
        PIV = (neutrofili * piastrine * monociti) / linfociti
    else:
        NLR = SII = SIRI = PIV = 0.0

    x = np.array([[NLR, SII, SIRI, PIV, age, 1 if diameter > 80 else 0]])

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Modello non trovato.")
        st.stop()

    model = load(MODEL_PATH)
    p = float(model.predict_proba(x)[0, 1])
    cls = classify_risk(p)

    # Barra di rischio
    st.markdown(f"### Probabilità di malignità: {p:.3%}")
    st.progress(min(int(p * 100), 100))
    st.write(f"Classe di rischio: **{cls}**")

    # Dettagli opzionali
    if show_details:
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
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        ax.plot(angles, values, "o-", linewidth=2, label="Paziente")
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        st.pyplot(fig)

    st.caption("Modello addestrato sui dati del centro. Validazione clinica in corso.")
