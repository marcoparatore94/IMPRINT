import os
import sys
import streamlit as st
import numpy as np
from joblib import load

# Permetti import dalla root del progetto
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.info_panel import get_info_html  # pannello info

# --- Configurazione pagina ---
st.set_page_config(page_title="IMPRINT Risk Calculator", page_icon="🧮", layout="centered")
st.title("Calcolatore rischio malignità (IMPRINT)")

# --- Login semplice ---
PASSWORD = "Imprintfpg2025"
st.sidebar.subheader("🔒 Accesso")
password = st.sidebar.text_input("Inserisci la password", type="password")
if password != PASSWORD:
    st.error("Accesso negato. Inserisci la password corretta.")
    st.stop()

# --- Percorso modello ---
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "imprint_risk_model.joblib"))

# --- Pannello informativo ---
with st.expander("Informazioni e cut-off"):
    st.markdown(get_info_html(), unsafe_allow_html=True)

# --- Form di input ---
with st.form("risk_form"):
    col1, col2 = st.columns(2)
    age = col1.number_input("Età (anni)", min_value=15, max_value=95, value=50)
    diameter = col2.number_input("Diametro target (mm)", min_value=10, max_value=250, value=75)

    st.subheader("Ecografia (MUSA/MYLUNAR)")
    c1, c2, c3 = st.columns(3)
    irregular_margins = c1.selectbox("Margini irregolari?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sì")
    color4 = c2.selectbox("Color score 4?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sì")
    shadows_absent = c3.selectbox("Assenza di ombre acustiche?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sì")

    st.subheader("Emocromo (valori assoluti, x10^9/L)")
    neutrofili = st.number_input("Neutrofili", min_value=0.0, max_value=30.0, value=4.0)
    linfociti = st.number_input("Linfociti", min_value=0.0, max_value=10.0, value=2.0)
    monociti = st.number_input("Monociti", min_value=0.0, max_value=5.0, value=0.5)
    eosinofili = st.number_input("Eosinofili", min_value=0.0, max_value=5.0, value=0.2)
    piastrine = st.number_input("Piastrine", min_value=10.0, max_value=1000.0, value=250.0)

    show_details = st.checkbox("Mostra dettagli indici ematologici")
    show_radar = st.checkbox("Mostra grafico radar indici ematologici")

    submitted = st.form_submit_button("Calcola rischio")

# --- Funzioni di supporto ---
def classify_risk(p: float) -> str:
    if p < 0.004:
        return "Basso"
    elif p < 0.023:
        return "Intermedio"
    else:
        return "Alto"

def suggest_action(cls: str) -> str:
    if cls == "Alto":
        return "Imaging avanzato/MDT; considerare chirurgia con approccio oncologico."
    elif cls == "Intermedio":
        return "Rivalutazione ecografica, ripetere markers; considerare MRI."
    return "Gestione conservativa come leiomioma, follow-up."

# --- Calcolo e output ---
if submitted:
    # Calcolo indici ematologici
    if linfociti > 0:
        NLR = neutrofili / linfociti
        SII = (neutrofili * piastrine) / linfociti
        SIRI = (neutrofili * monociti) / linfociti
        PIV = (neutrofili * piastrine * monociti) / linfociti
    else:
        NLR = SII = SIRI = PIV = 0.0

    diameter_gt8 = 1 if diameter > 80 else 0

    x = np.array([[NLR, SII, SIRI, PIV, age, diameter_gt8,
                   irregular_margins, color4, shadows_absent]])

    # Carica modello
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Modello non trovato. Rigenera il modello con train_model.py e riprova.")
        st.stop()

    model = load(MODEL_PATH)
    p = float(model.predict_proba(x)[0, 1])  # probabilità di classe positiva

    # Testi
    cls = classify_risk(p)
    recommendation = suggest_action(cls)

    # Probabilità
    st.markdown(f"### Probabilità di malignità: {p:.3%}")

    # Barra di rischio (progress bar)
    st.progress(min(int(p * 100), 100))

    # Classe di rischio con colore
    color = "green" if cls == "Basso" else ("orange" if cls == "Intermedio" else "red")
    st.markdown(
        f"### Classe di rischio: <span style='color:{color}'>{cls}</span>",
        unsafe_allow_html=True
    )

    # Raccomandazione clinica
    st.write(recommendation)

    # Dettagli indici ematologici
    if show_details:
        st.write(f"NLR: {NLR:.2f}")
        st.write(f"SII: {SII:.2f}")
        st.write(f"SIRI: {SIRI:.2f}")
        st.write(f"PIV: {PIV:.2f}")

    # Grafico radar dei marker con media coorte reale (se i file Excel sono disponibili)
    if show_radar:
        import matplotlib.pyplot as plt
        import pandas as pd

        benign_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DB Imprint_benign.xlsx"))
        malignant_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DB_Imprint_malignant.xlsx"))

        try:
            if os.path.exists(benign_path) and os.path.exists(malignant_path):
                benign = pd.read_excel(benign_path)
                malignant = pd.read_excel(malignant_path)
                df = pd.concat([benign, malignant], ignore_index=True)

                # Conversione a numerico
                for col in ["Neutrophils", "Lymphocytes", "Monocytes", "Platelets"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                # Calcolo indici medi della coorte
                df = df.dropna(subset=["Neutrophils", "Lymphocytes"])
                df["NLR"] = df["Neutrophils"] / df["Lymphocytes"].replace(0, np.nan)
                df["SII"] = (df["Neutrophils"] * df["Platelets"]) / df["Lymphocytes"].replace(0, np.nan)
                df["SIRI"] = (df["Neutrophils"] * df["Monocytes"]) / df["Lymphocytes"].replace(0, np.nan)
                df["PIV"] = (df["Neutrophils"] * df["Platelets"] * df["Monocytes"]) / df["Lymphocytes"].replace(0, np.nan)

                reference_means = [
                    float(df["NLR"].mean()),
                    float(df["SII"].mean()),
                    float(df["SIRI"].mean()),
                    float(df["PIV"].mean())
                ]
            else:
                st.info("Valori medi coorte non disponibili (file Excel non trovati). Uso medie di riferimento generiche.")
                reference_means = [3.0, 600.0, 1.5, 800.0]  # fallback generico

            # Valori del paziente
            labels = ["NLR", "SII", "SIRI", "PIV"]
            patient_values = [NLR, SII, SIRI, PIV]

            # Preparazione angoli per radar
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            pv = patient_values + patient_values[:1]
            rm = reference_means + reference_means[:1]
            ang = angles + angles[:1]

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            ax.plot(ang, pv, "o-", linewidth=2, label="Paziente")
            ax.fill(ang, pv, alpha=0.25)
            ax.plot(ang, rm, "o--", linewidth=1, label="Media coorte")
            ax.set_thetagrids(np.degrees(angles), labels)
            ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
            st.pyplot(fig)

        except Exception as e:
            st.warning("Impossibile calcolare/mostrare il grafico radar: " + str(e))

    # Disclaimer
    st.caption("Modello addestrato sui dati del centro. Validazione clinica in corso.")
