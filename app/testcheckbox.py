import streamlit as st

st.set_page_config(page_title="Test Checkbox", layout="centered")
st.title("Demo con form e checkbox")

# --- Form ---
with st.form("myform"):
    nome = st.text_input("Nome")
    submitted = st.form_submit_button("Invia")

# --- Checkbox FUORI dal form ---
show_details = st.checkbox("Mostra dettagli")
show_radar = st.checkbox("Mostra grafico radar")

# --- Output ---
if submitted:
    st.write(f"Hai inserito: {nome}")

if show_details:
    st.write("✅ Dettagli visibili!")

if show_radar:
    st.write("📊 Grafico radar visibile!")
