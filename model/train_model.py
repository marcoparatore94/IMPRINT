import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import dump

# --- PATH FILE DATI ---
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
benign_path = os.path.join(ROOT_DIR, "DB Imprint_benign.xlsx")
malignant_path = os.path.join(ROOT_DIR, "DB_Imprint_malignant.xlsx")

# --- CARICAMENTO DATI ---
print("📂 Caricamento dati...")
benign = pd.read_excel(benign_path)
malignant = pd.read_excel(malignant_path)

benign["label"] = 0
malignant["label"] = 1
df = pd.concat([benign, malignant], ignore_index=True)

# --- DEBUG: stampa intestazioni ---
print("Colonne disponibili:", df.columns.tolist())

# --- CONVERSIONE A NUMERICO ---
for col in ["Neutrophils", "Lymphocytes", "Monocytes", "Platelets"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Rimuovi righe senza valori validi
df = df.dropna(subset=["Neutrophils", "Lymphocytes"])

# --- CALCOLO MARKER ---
df["NLR"] = df["Neutrophils"] / df["Lymphocytes"].replace(0, np.nan)
df["SII"] = (df["Neutrophils"] * df["Platelets"]) / df["Lymphocytes"].replace(0, np.nan)
df["SIRI"] = (df["Neutrophils"] * df["Monocytes"]) / df["Lymphocytes"].replace(0, np.nan)
df["PIV"] = (df["Neutrophils"] * df["Platelets"] * df["Monocytes"]) / df["Lymphocytes"].replace(0, np.nan)

# --- VARIABILI CLINICHE ---
df["Age"] = pd.to_numeric(df["Age at diagnosis"], errors="coerce")

# NB: placeholder per variabili ecografiche non presenti nei tuoi file
df["diameter_gt8"] = 0
df["Margini_irregolari"] = 0
df["Color4"] = 0
df["Ombre_assenti"] = 0

# --- FEATURE SELEZIONATE (coerenti con app.py) ---
features = [
    "NLR", "SII", "SIRI", "PIV",
    "Age", "diameter_gt8",
    "Margini_irregolari", "Color4", "Ombre_assenti"
]

X = df[features].fillna(0)
y = df["label"]

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- MODELLO ---
print("⚙️ Addestramento modello...")
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train, y_train)

# --- VALUTAZIONE ---
y_pred_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"✅ Modello addestrato. AUC su test: {auc:.3f}")

# --- SALVATAGGIO ---
model_path = os.path.join(ROOT_DIR, "data", "imprint_risk_model.joblib")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
dump(model, model_path)
print(f"💾 Modello salvato in: {model_path}")