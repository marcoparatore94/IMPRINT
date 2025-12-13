import os
import numpy as np
import pandas as pd
from scipy.stats import skew
import statsmodels.api as sm
import joblib

BASE_DIR = os.path.dirname(__file__)
DATA_BEN = os.path.join(BASE_DIR, "data", "IMPRINT_ben.xlsx")
DATA_MAL = os.path.join(BASE_DIR, "data", "IMPRINT_mal.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def to_binary(s: pd.Series) -> pd.Series:
    s = s.copy().astype(str).str.strip().str.lower()
    mapping = {"1": 1, "true": 1, "yes": 1, "si": 1, "sì": 1, "post": 1,
               "0": 0, "false": 0, "no": 0, "pre": 0}
    out = s.map(mapping)
    mask_num = s.str.fullmatch(r"^-?\d+(\.\d+)?$")
    out[mask_num] = s[mask_num].astype(float).apply(lambda x: 1 if x == 1 else (0 if x == 0 else np.nan))
    return out

def zscore(s: pd.Series) -> pd.Series:
    s = ensure_numeric(s)
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd

def maybe_log_transform(s: pd.Series) -> (pd.Series, bool):
    s = ensure_numeric(s)
    s_nonan = s.dropna()
    if s_nonan.empty:
        return s, False
    if (s_nonan > 0).all():
        sk = skew(s_nonan)
        if np.isfinite(sk) and (sk > 1.0):
            return np.log(s), True
    return s, False

# Load data
ben = pd.read_excel(DATA_BEN); ben["Malignant"] = 0
mal = pd.read_excel(DATA_MAL); mal["Malignant"] = 1
df = pd.concat([ben, mal], ignore_index=True)

# Feature engineering
df["Menopausal_status"] = to_binary(df["Menopause"])
df["Margins_bin"] = to_binary(df["Irregular margins"])
df["Cystic_areas"] = to_binary(df["Cystic areas"])

# CS e diametro
df["_cs_z"] = zscore(df["CS"])
df["_diam_z"] = zscore(df["Max diameter (mm)"])

# Trasformazioni da salvare
transforms = {
    "CS": {"mu": float(df["CS"].mean()), "sd": float(df["CS"].std(ddof=0))},
    "Max diameter (mm)": {"mu": float(df["Max diameter (mm)"].mean()), "sd": float(df["Max diameter (mm)"].std(ddof=0))}
}

# SIMs
for sim in ["NLR", "SII", "PIV"]:
    transformed, log_flag = maybe_log_transform(df[sim])
    df[f"_{sim}_z"] = zscore(transformed)
    transforms[sim] = {
        "log": log_flag,
        "mu": float(transformed.mean()),
        "sd": float(transformed.std(ddof=0))
    }

pred_cols = ["Menopausal_status","Margins_bin","_cs_z","_diam_z","Cystic_areas","_NLR_z","_SII_z","_PIV_z"]
D = df[["Malignant"] + pred_cols].dropna().reset_index(drop=True)
Y = D["Malignant"].astype(int)
X = sm.add_constant(D[pred_cols], has_constant="add")

model = sm.Logit(Y, X)
res = model.fit(disp=0)

bundle = {"model": res, "predictors": pred_cols, "transforms": transforms}
out_path = os.path.join(OUTPUT_DIR, "imprint_extended.pkl")
joblib.dump(bundle, out_path)

print("✅ Extended model saved to:", out_path)